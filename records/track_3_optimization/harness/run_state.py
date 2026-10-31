from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from records.track_3_optimization.harness.optimizers import GroupedOptimizers

Config = dict[str, Any]
Batch = tuple[Tensor, Tensor]


@dataclass
class NanoGPTRunState:
    """Live training context; setup installs the model, optimizer and data."""

    rank: int
    world_size: int
    model: nn.Module = field(init=False, repr=False)
    optimizer: GroupedOptimizers = field(init=False, repr=False)
    validation_data: Batch = field(init=False, repr=False)
    print_log: Callable[..., None] = field(init=False, repr=False)
    train_batches: Iterator[Batch] | None = None
    step: int = -1  # Zero-based current update; checkpoint N restores N - 1.
    setup_finished: bool = False
    training_time: float = 0.0
    clock_start: float = 0.0
    capture_root: Path | None = None
    gradient_references: dict[int, dict[str, Any]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


Hook = Callable[[Config, NanoGPTRunState], None]
