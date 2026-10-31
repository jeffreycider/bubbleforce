from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from optimizers.ringed.momentum import GradientRing, MomentumKernel, MomentumScheduler, apply
from optimizers.ringed.state import FiniteHistoryOptimizer
from optimizers.ringed.constants import (
    NEWTON_SCHULZ_STEPS, NEWTON_SCHULZ_NORM_EPSILON, NEWTON_SCHULZ_COEFFICIENTS,
)


def normalize_for_newton_schulz(momentum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orient and normalize BF16 momentum for the matrix-sign iteration.
    Uses the Frobenius norm which equally downscales all singular values/

    Args:
        momentum: Tensor [..., rows, columns]; tall matrices are transposed.

    Returns:
        Oriented BF16 momentum, its rounded Frobenius-norm denominator
        [..., 1, 1], and the normalized matrix. The denominator includes epsilon.
        The first two values are also needed to differentiate normalization.
    """
    matrix = momentum.bfloat16()
    if momentum.size(-2) > momentum.size(-1):
        matrix = matrix.mT
    denominator = matrix.norm(dim=(-2, -1), keepdim=True) + NEWTON_SCHULZ_NORM_EPSILON
    return matrix, denominator, matrix / denominator


def newton_schulz_iteration(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate one polar-approximation polynomial without changing its input.

    Args:
        matrix: Current BF16 iterate X, shaped [..., rows, columns].

    Returns:
        Next iterate, Gram matrix A = X @ X.T, and polynomial B = b*A + c*A@A.
        The latter two are intermediates already required by the update, retained
        by analysis when differentiating at rounded BF16 values.
    """
    a, b, c = NEWTON_SCHULZ_COEFFICIENTS
    gram = matrix @ matrix.mT
    polynomial = b * gram + c * gram @ gram
    return a * matrix + polynomial @ matrix, gram, polynomial


def zeropower_via_newtonschulz5(G: torch.Tensor) -> torch.Tensor:
    """Approximate the matrix polar factor with twelve BF16 Newton-Schulz iterations."""
    assert G.ndim >= 2
    _, _, X = normalize_for_newton_schulz(G)
    for _ in range(NEWTON_SCHULZ_STEPS):
        X, _, _ = newton_schulz_iteration(X)
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def compute_weight_after_muon_step(
    weight: torch.Tensor, momentum_direction: torch.Tensor, *,
    learning_rate: float, weight_decay: float,
) -> torch.Tensor:
    """Compute W_next = (1 - learning_rate * weight_decay) * W - learning_rate * D.

    Args:
        weight: Current weight matrix W, shaped (rows, columns).
        momentum_direction: D = shape_scale(W) * msign(M), shaped like W.
            The caller applies the matrix-sign approximation and shape scaling;
            this input does not include the learning rate or weight decay.
        learning_rate: Step size applied to D and decoupled weight decay.
        weight_decay: Decoupled decay coefficient.

    Returns:
        The updated weight matrix W_next, not its displacement. Inputs are not
        mutated. The caller controls input dtypes and any cast when storing W_next.
    """
    return torch.add(weight * (1 - learning_rate * weight_decay),
                     momentum_direction, alpha=-learning_rate)


def compute_weight_after_muonh_step(
    weight: torch.Tensor, momentum_direction: torch.Tensor, *, learning_rate: float,
) -> torch.Tensor:
    """Take a MuonH step and rescale the weight to its original Frobenius norm.

    Args:
        weight: Current weight matrix W, shaped (rows, columns).
        momentum_direction: Shape-scaled msign of MuonH's momentum,
            shaped like W, before multiplying by the learning rate. This function
            rescales the direction to W's Frobenius norm before stepping.
        learning_rate: Step size for the norm-matched direction.

    Returns:
        The updated weight matrix, not its displacement, rescaled to W's norm
        (with a 1e-10 denominator floor). Neither input is mutated.
    """
    p_norm = weight.norm()
    candidate = weight - learning_rate * momentum_direction * p_norm / momentum_direction.norm().clamp_min(1e-10)
    return candidate / candidate.norm().clamp_min(1e-10) * p_norm


@dataclass(frozen=True)
class PolarObservation:
    """Actual polar-map inputs/output, before shape scaling and weight decay.

    Tensor references are live and read-only; observers own any saved copies.
    """
    parameter: torch.Tensor
    gradient: torch.Tensor
    momentum: torch.Tensor
    polar: torch.Tensor
    kernel: MomentumKernel
    iteration: int
    history_filled: int


class Muon(FiniteHistoryOptimizer):

    """
    Muon inherits the FiniteHistoryOptimizer class which runs checks before loading state dict
    Ensures the desired optimizer state matches the optimizer config
    """

    def __init__(self, params: Iterable[torch.Tensor], momentum_scheduler: MomentumScheduler,
                 lr: float = 0.02, weight_decay: float = 0.0) -> None:
        params = sorted(params, key=lambda parameter: parameter.size(), reverse=True)
        # Native optimizer loading preserves FP32 ring state only for FP32 weights.
        if not params or any(p.ndim != 2 or p.dtype != torch.float32 for p in params):
            raise ValueError("Muon requires nonempty FP32 matrix parameters")
        self.momentum_scheduler = momentum_scheduler
        self.polar = torch.compile(zeropower_via_newtonschulz5) if torch.cuda.is_available() else zeropower_via_newtonschulz5
        self.polar_observer: Callable[[PolarObservation], None] | None = None
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, iteration=0))

    @staticmethod
    def shape_scale(parameter: torch.Tensor) -> float:
        """Rectangular-matrix multiplier applied after the polar map."""
        return max(1, parameter.size(-2) / parameter.size(-1)) ** 0.5

    def direction(self, parameter: torch.Tensor, parameter_state: dict[str, Any],
                  kernel: MomentumKernel, iteration: int) -> torch.Tensor:
        ring = GradientRing()
        if "gradient_newest" in parameter_state:
            ring.load_state_dict(parameter_state)
        ring.push(parameter.grad)
        parameter_state.update(ring.state_dict())
        return apply(kernel, ring).view_as(parameter)

    def apply_update(self, parameter: torch.Tensor, update: torch.Tensor,
                     group: dict[str, Any]) -> None:
        parameter.copy_(compute_weight_after_muon_step(
            parameter, update, learning_rate=group["lr"], weight_decay=group["weight_decay"]))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            iteration = group["iteration"]
            kernel = self.momentum_scheduler(iteration)
            params = group["params"]
            # Same ownership/sort/all-gather grouping as the record.
            padded = params + [torch.empty_like(params[-1])] * (world - len(params) % world)
            for base in range(0, len(params), world):
                if base + rank < len(params):
                    p = params[base + rank]
                    if p.grad is None:
                        # This ring advances in optimizer iterations, not observed-gradient count.
                        ring = GradientRing()
                        if "gradient_newest" in self.state[p]:
                            ring.load_state_dict(self.state[p])
                        ring.push(torch.zeros_like(p))
                        self.state[p].update(ring.state_dict())
                        apply(kernel, ring)
                    else:
                        if p.grad.is_sparse:
                            raise ValueError("Muon does not support sparse gradients")
                        direction = self.direction(p, self.state[p], kernel, iteration)
                        update = self.polar(direction)
                        if self.polar_observer is not None:
                            self.polar_observer(PolarObservation(
                                p, p.grad, direction, update, kernel, iteration,
                                self.state[p]["gradient_filled"]))
                        update *= self.shape_scale(p)
                        self.apply_update(p, update, group)
                if distributed:
                    dist.all_gather(padded[base:base + world], padded[base + rank])
            group["iteration"] += 1
        return loss


class MuonH(Muon):
    """Take Muon steps, rescaling each weight matrix to its original Frobenius norm."""

    def __init__(self, params: Iterable[torch.Tensor], momentum_scheduler: MomentumScheduler,
                 lr: float = 0.02) -> None:
        super().__init__(params, momentum_scheduler, lr=lr, weight_decay=0.0)

    def apply_update(self, parameter: torch.Tensor, update: torch.Tensor,
                     group: dict[str, Any]) -> None:
        parameter.copy_(compute_weight_after_muonh_step(parameter, update, learning_rate=group["lr"]))
