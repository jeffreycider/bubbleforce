from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch

HISTORY_LENGTH = 256
@dataclass
class MomentumKernel:
    """Store newest-first weights for the gradient sum."""
    weights: torch.Tensor
    normalize_available: bool = False

    def __post_init__(self):
        # Copy weights and detach them from the computational graph.
        self.weights = torch.as_tensor(self.weights, dtype=torch.float32).detach().clone()
        if self.weights.shape != (HISTORY_LENGTH,) or not torch.isfinite(self.weights).all():
            raise ValueError(f"momentum requires exactly {HISTORY_LENGTH} finite weights, newest first")
        if self.normalize_available and (self.weights < 0).any():
            raise ValueError("available-mass correction requires nonnegative weights")

class MomentumScheduler(Protocol):
    def __call__(self, iteration: int) -> MomentumKernel: ...

class GradientRing:
    """Store recent gradients in a circular FP32 buffer."""

    def __init__(
        self, parameter: torch.Tensor | None = None, *, start_step: int = 0,
    ) -> None:
        self.buffer: torch.Tensor | None = None
        self.newest = (start_step - 1) % HISTORY_LENGTH
        self.filled = 0
        if parameter is not None:
            self.buffer = torch.zeros(
                HISTORY_LENGTH, parameter.numel(),
                dtype=torch.float32, device=parameter.device,
            )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load the buffer and counters from saved optimizer state."""
        self.buffer = state.get("gradient_history")
        self.newest = state["gradient_newest"]
        self.filled = state["gradient_filled"]

    def state_dict(self, include_history: bool = True) -> dict[str, Any]:
        """Export the buffer and counters for checkpoint storage in an organized dict."""
        state: dict[str, Any] = {
            "gradient_newest": self.newest,
            "gradient_filled": self.filled,
        }
        if include_history and self.buffer is not None:
            state["gradient_history"] = self.buffer
        return state

    def validate_ring_history_against_checkpoint_step(self, checkpoint_step: int) -> None:
        """Check the saved position and count against the checkpoint step."""
        if type(checkpoint_step) is not int or checkpoint_step < 0:
            raise ValueError("checkpoint step must be a nonnegative integer")

        if type(self.newest) is not int or self.newest != (checkpoint_step - 1) % HISTORY_LENGTH:
            raise ValueError("gradient ring newest index differs from checkpoint step")

        if type(self.filled) is not int or self.filled != min(checkpoint_step, HISTORY_LENGTH):
            raise ValueError("gradient ring fill count differs from checkpoint step")

    def get_enumerated_gradients_from_ring(
        self, checkpoint_step: int, shape: torch.Size
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield saved gradients in chronological order using views of the buffer."""
        self.validate_ring_history_against_checkpoint_step(checkpoint_step)
        for step in range(checkpoint_step - self.filled, checkpoint_step):
            yield step, self.buffer[step % HISTORY_LENGTH].view(shape)

    def push(self, sample: torch.Tensor) -> None:
        """Copy a flattened gradient into the next buffer row."""
        if self.buffer is None:
            self.buffer = torch.zeros(
                HISTORY_LENGTH, sample.numel(),
                dtype=torch.float32, device=sample.device,
            )
        if self.buffer.shape != (HISTORY_LENGTH, sample.numel()):
            raise ValueError("checkpoint history shape does not match the sample")
        # Wrap the next insertion index into the buffer's row range.
        self.newest = (self.newest + 1) % HISTORY_LENGTH
        self.buffer[self.newest].copy_(sample.detach().flatten().float())
        self.filled = min(HISTORY_LENGTH, self.filled + 1)


def apply(kernel: MomentumKernel, ring: GradientRing) -> torch.Tensor:
    """Normalize and reorder the weights, then sum the stored gradients."""
    if ring.buffer is None:
        raise ValueError("insert a gradient before applying a momentum kernel")
    device = ring.buffer.device
    weights = kernel.weights.to(device=device)
    if kernel.normalize_available:
        # Normalize so the weights for available gradients sum to one.
        mass = weights[:ring.filled].sum()
        if mass <= 0:
            raise ValueError("the available kernel mass is not positive")
        weights = weights / mass
    ages = (ring.newest - torch.arange(HISTORY_LENGTH, device=device)) % HISTORY_LENGTH
    return weights[ages] @ ring.buffer


def reconstruct_ring_from_gradients(
    gradient_history: Iterable[tuple[int, torch.Tensor]],
    parameter: torch.Tensor,
    checkpoint_step: int,
) -> GradientRing:
    """Restore recent gradients from chronological (step, gradient) records.

    Use the parameter's shape and device to allocate the buffer, then insert
    saved gradients through the step immediately before the checkpoint.
    """
    if type(checkpoint_step) is not int or checkpoint_step < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    start = max(0, checkpoint_step - HISTORY_LENGTH)
    ring = GradientRing(parameter=parameter, start_step=start)
    count = 0
    for step, gradient in gradient_history:
        if step != start + count or step >= checkpoint_step:
            raise ValueError("gradient history steps must be chronological and complete")
        if gradient.shape != parameter.shape:
            raise ValueError("gradient history parameter or shape changed")
        ring.push(gradient)
        count += 1
    if count != checkpoint_step - start:
        raise ValueError("checkpoint gradient history is incomplete")
    return ring
