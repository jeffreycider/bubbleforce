"""Compute Muon displacements and their response to weight perturbations."""

import torch

from configs.measurement_config import MuonStepConfig
from bubbleforce.analysis.loss_hessian import LossHessian
from bubbleforce.analysis.muon_jvp import compute_newton_schulz_jvp
from bubbleforce.analysis.sharded_hessian import compute_distributed_hvp
from bubbleforce.analysis.sharded_vector import pack_tensor, unpack_vector
from bubbleforce.load_checkpoint import MeasurementCheckpoint
from optimizers.ringed.muon import Muon


def compute_negative_update_jvp(
    current_weight: torch.Tensor,
    saved_momentum: torch.Tensor,
    muon_step_config: MuonStepConfig,
    weight_perturbation: torch.Tensor,
    gradient_response_to_weight_perturbation: torch.Tensor,
) -> torch.Tensor:
    """Compute the negative Jacobian-vector product of the normalized Muon update.

    Propagate the weight perturbation and its gradient response through the
    saved update while holding older momentum gradients fixed. A Jacobian-vector
    product (JVP) measures the first-order response along this perturbation.
    Add the weight-decay response to the scaled Newton-Schulz response.

    Args:
        current_weight: Saved weight matrix, shape (rows, columns).
        saved_momentum: Momentum including the current gradient, before the
            matrix-sign calculation. Same shape as current_weight.
        muon_step_config: Saved optimizer settings and gradient scaling.
        weight_perturbation: Direction of change in the weights. Same shape
            as current_weight.
        gradient_response_to_weight_perturbation: This matrix's block of the
            validation-mean loss Hessian applied to the weight perturbation.
            Includes cross-matrix effects; same shape as current_weight.

    Returns:
        (weight_perturbation - next_weight_response) / learning_rate,
        using the shape and dtype of weight_perturbation. This is the negative
        directional derivative of (next_weight - current_weight) / learning_rate.
    """
    momentum_response = (
        muon_step_config.current_gradient_coefficient
        * muon_step_config.training_batch_tokens
        * gradient_response_to_weight_perturbation
    ) # Hv * N * c_0

    _, newton_schulz_response = compute_newton_schulz_jvp(
        saved_momentum, momentum_response
    )
    matrix_shape_scale = Muon.shape_scale(current_weight)
    scaled_update_response = matrix_shape_scale * newton_schulz_response

    return (
        muon_step_config.weight_decay * weight_perturbation.double()
        + scaled_update_response.double()
    ).to(weight_perturbation.dtype)


def compute_optimizer_jvp(
    checkpoint: MeasurementCheckpoint,
    hessian: LossHessian,
    direction_shard: torch.Tensor,
) -> torch.Tensor:
    """Compute the Muon update's response across weight matrices.

    Compute H v across the model, then propagate each matrix's weight
    perturbation and gradient response through its saved Muon update.
    Hold older gradients and Adam parameters fixed.

    Read assigned matrices from the loaded model and momentum by parameter name.
    Run on every rank. Return this rank's flattened
    -D[(W_next - W) / eta][v], with zero padding.
    """
    vector_config = checkpoint.vector_config
    gradient_perturbations = unpack_vector(
        compute_distributed_hvp(vector_config, hessian, direction_shard),
        config=vector_config,
    )
    weight_perturbations = unpack_vector(direction_shard, config=vector_config)

    return pack_tensor(
        {
            name: compute_negative_update_jvp(
                checkpoint.model.get_parameter(name).detach(),
                momentum, checkpoint.step_configs[name],
                weight_perturbations[name], gradient_perturbations[name],
            )
            for name, momentum in checkpoint.momenta.items()
        },
        config=vector_config,
    )