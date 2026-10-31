"""Load model weights, optimizer settings, and validation tokens for checkpoint analysis."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from records.track_3_optimization.configs.measurement_config import MuonStepConfig, ShardedVectorConfig
from records.track_3_optimization.bubbleforce.analysis.fixed_lm_head_loss import GPTWithFixedHeadLoss
from records.track_3_optimization.harness.recording.store import get_formatted_path_to_capture_file
from records.track_3_optimization.optimizers.ringed.momentum.config import momentum_scheduler
from records.track_3_optimization.optimizers.ringed.momentum.kernel import HISTORY_LENGTH


@dataclass
class MeasurementCheckpoint:
    snapshot: dict[str, Any]
    momenta: dict[str, torch.Tensor]
    step_configs: dict[str, MuonStepConfig]
    vector_config: ShardedVectorConfig
    paths: list[Path]


def load_checkpoint_for_analysis(
    capture: Path, update: int, rank: int, world_size: int,
    device: torch.device,
) -> MeasurementCheckpoint:
    """Read rank-zero weights, then recover each rank's matrix assignments.

    Read metrics snapshots after the current training gradient has entered
    momentum. To measure from a full restart checkpoint, first resume training
    through the requested update using harness.checkpoint.load_checkpoint.
    """
    paths = [get_formatted_path_to_capture_file(capture, 0, "metrics_snapshot", update)]
    if (capture / "COMPLETE.json").exists():
        raise ValueError("use --resume-checkpoint for a full restart bundle; --capture expects metrics snapshots")
    snapshot = torch.load(paths[0], map_location="cpu", weights_only=True, mmap=True)
    if snapshot.get("format") != 3 or snapshot.get("update") != update:
        raise ValueError("snapshot format or update differs from the requested measurement")
    weights = snapshot["model_before"]
    parameter_names = sorted(
        name for name, weight in weights.items()
        if name.startswith("blocks.") and weight.ndim == 2
    )
    if not parameter_names:
        raise ValueError("snapshot has no hidden matrix weights")
    momenta = {}
    step_configs = load_muon_step_configs(capture, snapshot, parameter_names)
    layouts = []
    paths = [
        get_formatted_path_to_capture_file(capture, checkpoint_rank, "metrics_snapshot", update)
        for checkpoint_rank in range(world_size)
    ]
    for checkpoint_rank, path in enumerate(paths):
        rank_snapshot = snapshot if checkpoint_rank == 0 else torch.load(
            path, map_location="cpu", weights_only=True, mmap=True,
        )
        if rank_snapshot.get("update") != update or rank_snapshot.get("rank") != checkpoint_rank:
            raise ValueError(f"snapshot update/rank mismatch: {path}")
        local_momenta = rank_snapshot["post_insertion_prepolar_momentum"]
        for name, momentum in local_momenta.items():
            if name not in weights or momentum.shape != weights[name].shape:
                raise ValueError(f"saved momentum and weight shapes differ: {name}")
        layouts.append([(name, tuple(weights[name].shape)) for name in sorted(local_momenta)])
        if checkpoint_rank == rank:
            momenta = {name: value.to(device) for name, value in local_momenta.items()}
    vector_config = ShardedVectorConfig(layouts, rank=rank, device=device)
    if set(vector_config.parameter_names) != set(parameter_names):
        raise ValueError("launch one process per capture rank to include every hidden matrix")
    return MeasurementCheckpoint(snapshot, momenta, step_configs, vector_config, paths)


def load_model_from_checkpoint(checkpoint: MeasurementCheckpoint) -> GPTWithFixedHeadLoss:
    """Construct the saved GPT architecture and copy its weights and buffers."""
    weights = checkpoint.snapshot["model_before"]
    vocab_size, model_dim = weights["embed.weight"].shape
    num_layers = 1 + max(int(name.split(".")[1]) for name in checkpoint.vector_config.parameter_names)
    model = GPTWithFixedHeadLoss(vocab_size, num_layers, model_dim).to(checkpoint.vector_config.device)
    model.load_state_dict({**weights, **checkpoint.snapshot["buffers"]}, strict=True)
    return model.eval()


def load_initial_eigenvector(
    directory: Path | None, vector_config: ShardedVectorConfig,
) -> torch.Tensor | None:
    """Read this rank's starting direction and check its size against the saved layout."""
    if directory is None:
        return None
    initial_vector = torch.load(
        directory / f"current_vector_rank{vector_config.rank}.pt",
        map_location=vector_config.device, weights_only=True,
    )
    if initial_vector.shape != (vector_config.local_vector_size,):
        raise ValueError("warm-start vector size differs from the checkpoint layout")
    return initial_vector


def load_muon_step_configs(
    capture_root: Path,
    snapshot: dict[str, Any],
    parameter_names: Sequence[str],
) -> dict[str, MuonStepConfig]:
    """Read scheduled lr/decay from the snapshot and reconstruct its kernel.

    The capture stores its resolved recipe once in capture_config.json; the
    momentum kernel is evaluated from it at the snapshot's update. Gradients and
    momentum use training-token sums, while the analysis Hessian uses a mean.

    Args:
        capture_root: Capture directory containing capture_config.json.
        snapshot: Loaded schema-3 metrics snapshot, including update,
            optimizer configuration and training batch token count.
        parameter_names: Select Muon/MuonH matrices for configuration recovery.

    Returns:
        Return each matrix's configuration by parameter name, including scheduled
        learning rate, decay, and the normalized current-gradient coefficient.

    Raises:
        ValueError: The snapshot cannot identify the optimizer settings or
            the saved training batch token count is invalid.
    """
    update = snapshot["update"]
    if snapshot["format"] != 3:
        raise ValueError("optimizer step requires a schema-3 metrics snapshot")
    training_tokens = snapshot["training_batch_tokens"]
    capture_config = json.loads((capture_root / "capture_config.json").read_text())
    if training_tokens <= 0:
        raise ValueError("optimizer step requires positive saved training batch tokens")
    config = {}
    current_gradient_coefficients = {}
    for name in parameter_names:
        matches = [row for row in snapshot["settings"] if re.fullmatch(row["pattern"], name)]
        if len(matches) != 1 or matches[0]["optimizer"] not in {"muon", "muonh"}:
            raise ValueError(f"snapshot does not identify one Muon optimizer for {name}")
        saved = matches[0]
        if len(saved["groups"]) != 1:
            raise ValueError(f"snapshot does not identify the parameter group for {name}")
        group = saved["groups"][0]
        pattern = saved["pattern"]
        if pattern not in current_gradient_coefficients:
            specs = [row for row in capture_config["optimizer_groups"]
                     if re.fullmatch(row["pattern"], name)]
            if len(specs) != 1 or specs[0]["optimizer"] != saved["optimizer"]:
                raise ValueError(f"capture recipe and saved optimizer disagree for {name}")
            kernel = momentum_scheduler(specs[0]["hyperparams"]["momentum"])(update)
            filled = min(update + 1, HISTORY_LENGTH)
            current_coefficient = kernel.weights[0]
            if kernel.normalize_available:
                mass = kernel.weights[:filled].sum()
                if mass <= 0:
                    raise ValueError("the available kernel mass is not positive")
                current_coefficient = current_coefficient / mass
            current_gradient_coefficients[pattern] = float(current_coefficient)
        config[name] = MuonStepConfig(
            optimizer=saved["optimizer"], learning_rate=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            current_gradient_coefficient=current_gradient_coefficients[pattern],
            training_batch_tokens=training_tokens,
        )
    return config


def load_validation_token_batch(
    path: Path, token_count: int, *, rank: int, world_size: int,
    device: torch.device, sequence_length: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a fixed contiguous slice of FineWeb tokens for this rank.

    Shift targets by one token. Divide the global prefix across ranks without
    shuffling, then reshape each slice into complete sequences.
    """
    if token_count < 1 or token_count % (world_size * sequence_length):
        raise ValueError("token_count must provide whole validation sequences on every rank")
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype=np.int32, count=256)
    if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
        raise ValueError("expected a FineWeb version-1 token shard")
    available = int(header[2])
    if available < token_count + 1 or path.stat().st_size != 1024 + available * 2:
        raise ValueError("validation shard is truncated or has too few tokens")
    local_tokens = token_count // world_size
    offset = rank * local_tokens
    tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=1024, shape=(available,))
    local = torch.tensor(np.array(tokens[offset:offset + local_tokens + 1]), dtype=torch.int64, device=device)
    return local[:-1].reshape(-1, sequence_length), local[1:].reshape(-1, sequence_length)
