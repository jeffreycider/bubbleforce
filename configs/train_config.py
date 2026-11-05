"""Validate training settings and assign GPT parameters to optimizer groups."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GPT_PARAMETER_PATTERNS = {
    "embed": r"^embed\.weight$",
    "head": r"^proj\.weight$",
    "block_matrices": r"^blocks\..*\.weight$",
    "biases_and_norms": r"^(?!embed\.weight$|proj\.weight$|blocks\..*\.weight$).*",
}


def resolve_optimizer_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate named GPT groups for the training and checkpoint code.

    Assign embedding weights to embed, output projection weights to head,
    attention and MLP weight matrices to block_matrices. Assign biases and
    normalization gains to biases_and_norms. Preserve the supplied group order
    when constructing optimizer state.
    """
    resolved = []
    seen = set()
    for group in groups:
        values = deepcopy(group)
        block_type = values.pop("block_type")
        if block_type not in GPT_PARAMETER_PATTERNS:
            raise ValueError(f"unknown GPT block_type: {block_type}")
        if block_type in seen:
            raise ValueError(f"duplicate GPT block_type: {block_type}")
        if "pattern" in values:
            raise ValueError("select GPT parameters with block_type")
        seen.add(block_type)
        resolved.append(dict(pattern=GPT_PARAMETER_PATTERNS[block_type], **values))
    if missing := set(GPT_PARAMETER_PATTERNS) - seen:
        raise ValueError(f"missing GPT block types: {', '.join(sorted(missing))}")
    return resolved


def describe_optimizer_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Display the named model groups when printing resolved training settings."""
    block_types = {pattern: name for name, pattern in GPT_PARAMETER_PATTERNS.items()}
    result = []
    for group in groups:
        values = deepcopy(group)
        block_type = block_types[values.pop("pattern")]
        result.append(dict(block_type=block_type, **values))
    return result


@dataclass
class TrainConfig:
    """Validate the training settings supplied by a resolved experiment YAML."""

    optimizer_groups: list[dict[str, Any]]
    loop: str
    seed: int
    minibatch_seed: int
    batch_tokens: int
    microbatch_sequences: int
    train_data: str
    val_data: str
    val_tokens: int
    run_id: str | None
    output_dir: str | None
    resume_checkpoint: str | None
    model: dict[str, int]
    train_steps: int
    stop_step: int | None
    wandb_offline: bool
    initialization: dict[str, Any]
    seed_offset: int
    reseed_before_initialization: bool
    checkpoint: dict[str, Any]
    metrics_snapshot: dict[str, Any]
    setup: list[dict[str, Any]] | None = None
    pre_optimizer: list[dict[str, Any]] | None = None
    post_optimizer: list[dict[str, Any]] | None = None
    teardown: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        for name in ("output_dir", "resume_checkpoint"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, str((repository / value).resolve()))
        # The fixed training checklist; order matters (e.g. optimizer before checkpoint restore).
        if self.setup is None:
            self.setup = [{"name": name} for name in (
                "open_rank_zero_log", "load_validation_tokens", "build_compiled_gpt",
                "seed_then_initialize_parameters", "assemble_grouped_optimizer", "restore_checkpoint",
                "open_training_batches", "broadcast_initial_parameters", "validate_at_step_boundaries",
            )]
        if self.pre_optimizer is None:
            self.pre_optimizer = [{"name": "schedule_optimizer_groups"}]
        if self.post_optimizer is None:
            self.post_optimizer = [
                {"name": "print_training_progress"},
                {"name": "validate_at_step_boundaries", "hyperparams": {"every": 125, "final_tenth_every": 25}},
            ]
        if self.teardown is None:
            self.teardown = [{"name": "mark_log_finished"}]
        if type(self.train_steps) is not int or self.train_steps < 1:
            raise ValueError("train_steps must be a positive integer")
        if self.stop_step is not None and (
            type(self.stop_step) is not int or not 1 <= self.stop_step <= self.train_steps
        ):
            raise ValueError("stop_step must be within the training schedule")
        for name in ("seed", "minibatch_seed", "seed_offset"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if any(type(step) is not int or step < 0
               for step in self.metrics_snapshot.get("steps", [])):
            raise ValueError("training snapshot steps must be nonnegative integers")
        resolve_optimizer_groups(self.optimizer_groups)

    def to_training_dict(self) -> dict[str, Any]:
        """Expand optimizer groups and omit unset execution options for the worker."""
        config = asdict(self)
        config["optimizer_groups"] = resolve_optimizer_groups(self.optimizer_groups)
        for name in ("run_id", "output_dir", "stop_step"):
            if config[name] is None:
                del config[name]
        return config
