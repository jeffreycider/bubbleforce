"""Configure checkpoint measurements, eigenpair estimation, and vector shards."""

from collections.abc import Sequence
from dataclasses import InitVar, asdict, dataclass, field
from math import isfinite, prod
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
import torch


@dataclass
class EigenSolveConfig:
    """Choose eigenpair counts, basis checkpoints, comparison tolerances.

    Attributes:
        num_requested_eigenpairs: Number of positive real eigenpairs to select.
        krylov_basis_vector_counts: Basis sizes at which to estimate eigenpairs.
            Process these checkpoints in sorted, distinct order.
        comparison_window: Number of recent checkpoints to compare.
        eigenvalue_rtol: Maximum relative range across the comparison window.
            For each sorted eigenvalue position, divide its range by its
            largest absolute value; require every position to pass.
        subspace_tolerance: Maximum sine of the largest principal angle between
            successive selected eigenvector spans. Apply when requesting
            multiple eigenpairs. Zero means identical spans; one means at
            least one direction is orthogonal to the other span.
        breakdown_rtol: Stop basis extension when the component outside the
            Krylov span is this small relative to the original matrix-vector
            product.
        symmetric: Use a symmetric projected eigensolve for a symmetric operator.
        start_seed: Random starting-vector seed.
    """

    num_requested_eigenpairs: int
    krylov_basis_vector_counts: tuple[int, ...] = (8, 16, 24, 32, 48, 64, 96, 128)
    comparison_window: int = 2
    eigenvalue_rtol: float = 0.001
    subspace_tolerance: float = 0.01
    breakdown_rtol: float = 1e-12
    symmetric: bool = False
    start_seed: int = 921

    def __post_init__(self) -> None:
        self.krylov_basis_vector_counts = tuple(self.krylov_basis_vector_counts)
        if not self.krylov_basis_vector_counts or min(self.krylov_basis_vector_counts) < 1:
            raise ValueError("krylov_basis_vector_counts must contain positive integers")
        if not 1 <= self.num_requested_eigenpairs <= max(self.krylov_basis_vector_counts):
            raise ValueError(
                "num_requested_eigenpairs must be between 1 and the largest basis size"
            )
        if self.comparison_window < 2:
            raise ValueError("comparison_window must be at least 2")
        if not isfinite(self.breakdown_rtol) or self.breakdown_rtol < 0:
            raise ValueError("breakdown_rtol must be finite and nonnegative")


@dataclass(frozen=True)
class MuonStepConfig:
    """Configure a matrix's Muon step at a saved training update.

    Attributes:
        optimizer: Optimizer variant: "muon" or "muonh".
        learning_rate: Learning rate at this update.
        weight_decay: Decoupled decay coefficient; zero for MuonH.
        current_gradient_coefficient: Lag-zero kernel coefficient c0 after any
            normalization over the available gradient history.
        training_batch_tokens: Token count converting validation-mean gradient
            changes to the training-sum gradient convention.
    """
    optimizer: Literal["muon", "muonh"]
    learning_rate: float
    weight_decay: float
    current_gradient_coefficient: float
    training_batch_tokens: int

    def __post_init__(self) -> None:
        if self.optimizer not in ("muon", "muonh"):
            raise ValueError("expected Muon or MuonH")
        if self.optimizer == "muonh" and self.weight_decay != 0:
            raise ValueError("MuonH requires zero weight decay")
        if self.learning_rate <= 0:
            raise ValueError("normalized displacement requires a positive learning rate")


@dataclass
class ShardedVectorConfig:
    """Assign named tensors to ranks, then calculate vector lengths and padding.

    Attributes:
        parameter_names_and_shapes_by_rank: Parameter tensor names and shapes
            for each rank, in local packing order.
        rank: Current process index.
        device: Local tensor device.
        dtype: Local tensor dtype.
        parameter_counts_by_rank: Count scalar parameter entries on each rank,
            excluding padding.
        parameter_names: Collect tensor names in rank order, preserving each
            rank's local order.
        local_vector_size: Pad every local shard to the largest rank's
            parameter count.
        world_size: Number of ranks in the supplied layout.
    """

    parameter_names_and_shapes_by_rank: Sequence[
        Sequence[tuple[str, Sequence[int]]]
    ]
    rank: int = 0
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32

    parameter_counts_by_rank: list[int] = field(init=False)
    parameter_names: list[str] = field(init=False)
    local_vector_size: int = field(init=False)
    world_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.parameter_counts_by_rank = [
            sum(prod(shape) for _, shape in rank_parameters)
            for rank_parameters in self.parameter_names_and_shapes_by_rank
        ]
        self.parameter_names = [
            name
            for rank_parameters in self.parameter_names_and_shapes_by_rank
            for name, _ in rank_parameters
        ]

        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("Assign each parameter name to a single rank.")

        self.local_vector_size = max(self.parameter_counts_by_rank)
        self.world_size = len(self.parameter_names_and_shapes_by_rank)
        self.device = torch.device(self.device)


@dataclass(frozen=True)
class LossHessianConfig:
    """Choose how to execute Hessian-vector products and multiply matrices."""

    mode: Literal["trace", "compile"] = "compile"
    gemm_precision: Literal["fp32", "bf16_fp32"] = "fp32"


@dataclass(frozen=True)
class FixedLMHeadConfig:
    """Split output-head projection into token chunks."""

    chunk_size: int = 32768

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("positive token chunk required")


@dataclass(frozen=True)
class AddmmConfig:
    """Scale the matrix product and bias before adding them."""

    alpha: float = 1
    beta: float = 1


@dataclass
class MeasurementConfig:
    """Measure selected saved updates with shared data and eigensolver settings."""

    validation: Path
    output_dir: Path
    capture_dir: Path
    updates: tuple[int, ...] | None
    measure_every: int
    token_counts: tuple[int, ...]
    microbatch_sequences: int
    sequence_length: int
    hvp_mode: Literal["trace", "compile"]
    gemm_precision: Literal["fp32", "bf16_fp32"]
    matmul_precision: Literal["highest", "high"]
    cpu_threads: int
    warm_start: Path | None
    reuse_previous_vector: bool
    eigensolver: EigenSolveConfig
    end_step: InitVar[int | None] = None

    def __post_init__(self, end_step: int | None) -> None:
        repository = Path(__file__).resolve().parents[1]
        for name in ("output_dir", "capture_dir", "validation", "warm_start"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, (repository / value).resolve())
        if type(self.measure_every) is not int or self.measure_every < 1:
            raise ValueError("measure_every must be a positive integer")
        if self.updates is None:
            if end_step is None:
                raise ValueError("measurement-only runs require explicit measurement.updates")
            if type(end_step) is not int or end_step < 1:
                raise ValueError("end_step must be a positive integer")
            self.updates = tuple(sorted(set(range(0, end_step, self.measure_every)) | {end_step - 1}))
        if (not self.updates
                or any(type(update) is not int or update < 0 for update in self.updates)
                or len(self.updates) != len(set(self.updates))):
            raise ValueError("updates must be nonnegative, distinct integers")
        self.updates = tuple(sorted(self.updates))
        self.token_counts = tuple(self.token_counts)
        for name in ("microbatch_sequences", "sequence_length", "cpu_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (not self.token_counts
                or any(type(count) is not int or count < 1 for count in self.token_counts)
                or list(self.token_counts) != sorted(set(self.token_counts))):
            raise ValueError("token_counts must be positive, distinct, increasing integers")
        if type(self.reuse_previous_vector) is not bool:
            raise ValueError("reuse_previous_vector must be a boolean")
        if self.reuse_previous_vector and self.eigensolver.num_requested_eigenpairs != 1:
            raise ValueError("reuse_previous_vector requires a single requested eigenpair")

    @classmethod
    def from_dict(cls, values: dict, *, end_step: int | None = None) -> "MeasurementConfig":
        values = dict(values)
        values["eigensolver"] = EigenSolveConfig(**values["eigensolver"])
        return cls(**values, end_step=end_step)

    @classmethod
    def from_yaml(cls, path: Path) -> "MeasurementConfig":
        return cls.from_dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))

    def to_dict(self) -> dict:
        """Convert paths and sequences before saving JSON or YAML settings."""
        values = asdict(self)
        for name in ("validation", "output_dir", "capture_dir", "warm_start"):
            if values[name] is not None:
                values[name] = str(values[name])
        values["token_counts"] = list(self.token_counts)
        values["updates"] = list(self.updates)
        values["eigensolver"]["krylov_basis_vector_counts"] = list(
            self.eigensolver.krylov_basis_vector_counts,
        )
        return values
