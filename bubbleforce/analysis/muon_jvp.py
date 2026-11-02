import torch
from torch import Tensor

from optimizers.ringed.muon import (
    normalize_for_newton_schulz, newton_schulz_iteration,
)
from optimizers.ringed.constants import (
    NEWTON_SCHULZ_STEPS, NEWTON_SCHULZ_COEFFICIENTS,
)


def compute_newton_schulz_jvp(momentum_matrix: Tensor, momentum_response: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate the BF16 polar approximation and its response to a perturbation.

    Propagate tangents in FP32 (FP64 for FP64 tangents), treating casts and rounding
    as identity derivatives. Tangents are not rounded at each intermediate.

    Args:
        momentum: Reference matrix, shaped (rows, columns), before msign.
        tangent: Momentum perturbation with the same shape and device.

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
