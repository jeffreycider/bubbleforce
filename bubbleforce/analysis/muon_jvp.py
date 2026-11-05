"""Apply the canonical Muon pathspring operator to a weight perturbation."""

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


def compute_mean_momentum_jvp(
    saved_momentum: Tensor, mean_momentum_response: Tensor, *, training_batch_tokens: int,
) -> Tensor:
    """Adapt summed-gradient checkpoint momentum to mean-loss coordinates.

    Return N * J_P(M_sum)[u], the response to a mean-momentum perturbation u.
    Keep the saved momentum's BF16 primals and normalization epsilon unchanged;
    rescale the tangent output instead of renormalizing the saved momentum.
    N is the training batch size, not the validation HVP token count.
    """
    _, response = compute_newton_schulz_jvp(saved_momentum, mean_momentum_response)
    return response * training_batch_tokens


def compute_optimizer_jvp(
    checkpoint: MeasurementCheckpoint,
    hessian: LossHessian,
    direction_shard: torch.Tensor,
) -> torch.Tensor:
    """Apply J_T / c0 in mean-loss coordinates, excluding decoupled decay.

    Compute the validation-mean Hessian product H v across Muon matrices.
    Pass each matrix's product through the mean-momentum adapter, then apply
    the matrix-size scale. Hold older gradients and Adam parameters fixed.
    Weight decay enters the stability polynomial, not this operator.

    Run on every rank. Return the flattened local response with zero padding.
    """
    vector_config = checkpoint.vector_config
    gradient_perturbations = unpack_vector(
        compute_distributed_hvp(vector_config, hessian, direction_shard),
        config=vector_config,
    )
    update_responses = {}
    for name, momentum in checkpoint.momenta.items():
        step_config = checkpoint.step_configs[name]
        momentum_response = gradient_perturbations[name]
        newton_schulz_response = compute_mean_momentum_jvp(
            momentum, momentum_response, training_batch_tokens=step_config.training_batch_tokens,
        )
        scaled_update_response = (
            Muon.shape_scale(momentum) * newton_schulz_response
        )
        update_responses[name] = scaled_update_response

    return pack_tensor(update_responses, config=vector_config)
