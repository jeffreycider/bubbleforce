from collections.abc import Callable

import torch
from torch import Tensor

from optimizers.ringed.muon import (
    normalize_for_newton_schulz, newton_schulz_iteration,
)
from optimizers.ringed.constants import (
    NEWTON_SCHULZ_STEPS, NEWTON_SCHULZ_COEFFICIENTS,
)


def linearize_newton_schulz(momentum: Tensor) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
    """Evaluate the BF16 polar approximation and cache its linearization.

    Propagate tangents in FP32 (FP64 for FP64 tangents), treating casts and rounding
    as identity derivatives. Tangents are not rounded at each intermediate.

    Args:
        momentum: Reference matrix, shaped (rows, columns), before msign. Its
            twelve Newton-Schulz steps are evaluated and cached in BF16.

    Returns:
        The BF16 polar approximation, followed by a callable mapping a same-shaped
        momentum perturbation to Dmsign(M)[v]. The callable returns a tensor on
        the same device, in FP64 if the perturbation is
        FP64 and FP32 otherwise. No Hessian, shape scale or learning rate is
        applied. This differentiates smooth arithmetic at rounded intermediates,
        not the discontinuous quantization map.
    """
    tall = momentum.shape[0] > momentum.shape[1]
    g, denominator, x = normalize_for_newton_schulz(momentum.detach())
    if not bool(g.norm() > 0):
        raise ValueError("NS norm derivative is undefined at zero momentum")
    primals = []
    for _ in range(NEWTON_SCHULZ_STEPS):
        following, gram, polynomial = newton_schulz_iteration(x)
        primals.append((x, gram, polynomial))
        x = following

    def jvp(tangent: Tensor) -> Tensor:
        dtype = torch.float64 if tangent.dtype == torch.float64 else torch.float32
        e = tangent.to(dtype)
        if tall:
            e = e.mT
        gg, dd = g.to(dtype), denominator.to(dtype)
        # Norm's identity-rounding AD derivative uses the unrounded real norm.
        # The denominator's PRIMAL remains the exact recorded BF16 value.
        dnorm = (gg * e).sum() / gg.norm()
        dx = e / dd - gg * dnorm / dd.square()
        a, b, c = NEWTON_SCHULZ_COEFFICIENTS
        for x, gram, polynomial in primals:
            xx, aa, bb = x.to(dtype), gram.to(dtype), polynomial.to(dtype)
            da = dx @ xx.mT + xx @ dx.mT
            db = b * da + c * (da @ aa + aa @ da)
            dx = a * dx + db @ xx + bb @ dx
        return dx.mT if tall else dx

    return x.mT if tall else x, jvp