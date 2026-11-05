"""Apply the historical Muon response to a weight perturbation."""

import torch
from torch import Tensor

from bubbleforce.analysis.loss_hessian import LossHessian
from bubbleforce.analysis.sharded_hessian import compute_distributed_hvp
from bubbleforce.analysis.sharded_vector import pack_tensor, unpack_vector
from bubbleforce.load_checkpoint import MeasurementCheckpoint
from optimizers.ringed.constants import (
    NEWTON_SCHULZ_STEPS, NEWTON_SCHULZ_COEFFICIENTS,
)
from optimizers.ringed.muon import (
    Muon, normalize_for_newton_schulz, newton_schulz_iteration,
)


def compute_newton_schulz_jvp(momentum_matrix: Tensor, momentum_response: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate the BF16 polar approximation and its response to a perturbation.

    Propagate tangents in FP32 (FP64 for FP64 tangents), treating casts and rounding
    as identity derivatives. Tangents are not rounded at each intermediate.

    Args:
        momentum_matrix: Momentum matrix, shaped (rows, columns), before msign.
        momentum_response: Input perturbation with the same shape and device as momentum_matrix.

    Returns:
        Return the BF16 polar approximation and its Jacobian-vector product.
        Compute the product in FP64 for FP64 tangents, otherwise in FP32,
        differentiating smooth arithmetic at rounded BF16 intermediates.
    """
    tall = momentum_matrix.shape[0] > momentum_matrix.shape[1]
    g, denominator, x = normalize_for_newton_schulz(momentum_matrix.detach())
    if not bool(g.norm() > 0):
        raise ValueError("NS norm derivative is undefined at zero momentum")
    dtype = torch.float64 if momentum_response.dtype == torch.float64 else torch.float32
    e = momentum_response.to(dtype)
    if tall:
        e = e.mT
    gg, dd = g.to(dtype), denominator.to(dtype)
    # Differentiate the norm using its unrounded value and the BF16 denominator.
    dnorm = (gg * e).sum() / gg.norm()
    dx = e / dd - gg * dnorm / dd.square()
    a, b, c = NEWTON_SCHULZ_COEFFICIENTS
    for _ in range(NEWTON_SCHULZ_STEPS):
        following, gram, polynomial = newton_schulz_iteration(x)
        xx, aa, bb = x.to(dtype), gram.to(dtype), polynomial.to(dtype)
        da = dx @ xx.mT + xx @ dx.mT
        db = b * da + c * (da @ aa + aa @ da)
        dx = a * dx + db @ xx + bb @ dx
        x = following

    return (x.mT, dx.mT) if tall else (x, dx)


def compute_optimizer_jvp(
    checkpoint: MeasurementCheckpoint,
    hessian: LossHessian,
    direction_shard: torch.Tensor,
) -> torch.Tensor:
    """Apply the historical response J_P(M)[H v] + (weight_decay / N) v.

    Compute the validation-mean Hessian product H v across Muon matrices.
    Pass each matrix's product through the Newton-Schulz derivative at its
    saved summed-gradient momentum, then apply the matrix-size scale.
    Add weight_decay / N times the weight direction, using the saved training
    token count N. Hold older gradients and Adam parameters fixed.

    Run on every rank. Return the flattened local response with zero padding.
    """
    vector_config = checkpoint.vector_config
    gradient_perturbations = unpack_vector(
        compute_distributed_hvp(vector_config, hessian, direction_shard),
        config=vector_config,
    )
    weight_perturbations = unpack_vector(direction_shard, config=vector_config)

    update_responses = {}
    for name, momentum in checkpoint.momenta.items():
        step_config = checkpoint.step_configs[name]
        weight_perturbation = weight_perturbations[name]
        momentum_response = gradient_perturbations[name]
        _, newton_schulz_response = compute_newton_schulz_jvp(
            momentum, momentum_response,
        )
        scaled_update_response = (
            Muon.shape_scale(momentum) * newton_schulz_response
        )
        update_responses[name] = (
            scaled_update_response.double()
            + (
                step_config.weight_decay / step_config.training_batch_tokens
            ) * weight_perturbation.double()
        ).to(weight_perturbation.dtype)

    return pack_tensor(update_responses, config=vector_config)
