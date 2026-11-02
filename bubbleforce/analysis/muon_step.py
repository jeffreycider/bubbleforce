"""Compute Muon displacements and their response to weight perturbations."""

from collections.abc import Mapping
from functools import partial

import torch
from torch.func import jvp

from configs.measurement_config import MuonStepConfig, ShardedVectorConfig
from bubbleforce.analysis.loss_hessian import LossHessian
from bubbleforce.analysis.muon_jvp import linearize_newton_schulz
from bubbleforce.analysis.sharded_hessian import compute_distributed_hvp
from bubbleforce.analysis.sharded_vector import pack_tensor, unpack_vector
from optimizers.ringed.muon import (
    Muon, compute_weight_after_muon_step, compute_weight_after_muonh_step,
)

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
    Compute how the next updated weights respond by differentiating the next weights
    with respect to both current weights and the scaled update direction.

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
    )

    newton_schulz_output, apply_newton_schulz_jacobian = linearize_newton_schulz(saved_momentum)
    matrix_shape_scale = Muon.shape_scale(current_weight)
    scaled_update_direction = newton_schulz_output * matrix_shape_scale
    scaled_update_response = matrix_shape_scale * apply_newton_schulz_jacobian(momentum_response)

    if muon_step_config.optimizer == "muonh":
        compute_next_weight = compute_weight_after_muonh_step
    else:
        compute_next_weight = partial(
            compute_weight_after_muon_step,
            weight_decay=muon_step_config.weight_decay,
        )

    _, next_weight_response = jvp(
        partial(compute_next_weight, learning_rate=muon_step_config.learning_rate),
        (current_weight.double(), scaled_update_direction.double()),
        (weight_perturbation.double(), scaled_update_response.double()),
    )
    return (
        (weight_perturbation.double() - next_weight_response)
        / muon_step_config.learning_rate
    ).to(weight_perturbation.dtype)



def compute_muon_response(
    vector_config: ShardedVectorConfig,
    hessian: LossHessian,
    weights: Mapping[str, torch.Tensor],
    momenta: Mapping[str, torch.Tensor],
    muon_step_config: Mapping[str, MuonStepConfig],
    direction_shard: torch.Tensor,
) -> torch.Tensor:
    """Compute the Muon update's response across weight matrices.

    Compute H v across the model, then propagate each matrix's weight
    perturbation and gradient response through its saved Muon update.
    Hold older gradients and Adam parameters fixed.

    Supply local checkpoint tensors and config by parameter name.
    Run on every rank. Return this rank's flattened
    -D[(W_next - W) / eta][v], with zero padding.
    """
    gradient_perturbations = unpack_vector(
        compute_distributed_hvp(vector_config, hessian, direction_shard),
        config=vector_config,
    )
    weight_perturbations = unpack_vector(direction_shard, config=vector_config)

    return pack_tensor(
        {
            name: compute_negative_update_jvp(
                weight, momenta[name], muon_step_config[name],
                weight_perturbations[name], gradient_perturbations[name],
            )
            for name, weight in weights.items()
        },
        config=vector_config,
    )
