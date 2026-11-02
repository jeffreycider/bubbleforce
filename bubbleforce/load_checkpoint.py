"""Load model weights, optimizer settings, and validation tokens for checkpoint analysis."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from configs.measurement_config import MuonStepConfig, ShardedVectorConfig
from bubbleforce.analysis.fixed_lm_head_loss import GPTWithFixedHeadLoss
from harness.recording.store import get_formatted_path_to_capture_file
from optimizers.ringed.momentum.config import momentum_scheduler
from optimizers.ringed.momentum.kernel import HISTORY_LENGTH


@dataclass
class MeasurementCheckpoint:
    """Collect the loaded model and per-matrix inputs for analysis.

    Attributes:
        update: Training update number selected for measurement.
        model: Reconstruct the full GPTWithFixedHeadLoss on the requested
            device, load pre-update weights and buffers, then use eval mode.
        momenta: Map this rank's parameter names to saved momentum tensors.
            Each tensor has the corresponding weight's two-dimensional shape
            and saved dtype, transferred to the requested device. Read these
            tensors after inserting the current gradient, before applying
            Newton-Schulz. Momentum uses training-token sums.
        step_configs: Map every Muon matrix name across all ranks to a
            MuonStepConfig containing learning_rate, weight_decay,
            current_gradient_coefficient, and training_batch_tokens.
        vector_config: Record matrix names and shapes for every rank, sorting
            names within each rank. Calculate local vector padding from the
            largest rank's parameter count; use float32 for analysis vectors.
        paths: List the metrics snapshot paths in rank order.
    """

    update: int
    model: GPTWithFixedHeadLoss
    momenta: dict[str, torch.Tensor]
    step_configs: dict[str, MuonStepConfig]
    vector_config: ShardedVectorConfig
    paths: list[Path]


def load_checkpoint_for_analysis(
    capture: Path, update: int, rank: int, world_size: int,
    device: torch.device,
) -> MeasurementCheckpoint:
    """Read saved weights and momentum, then construct the model for analysis.

    Read metrics snapshots after the current training gradient has entered
    momentum. To measure from a full restart checkpoint, first resume training
    through the requested update using harness.checkpoint.load_checkpoint.

    Args:
        capture: Directory containing capture_config.json and rank folders.
        update: Zero-based training update to load.
        rank: Current worker index, from zero through world_size minus one.
        world_size: Number of workers, matching the saved snapshot ranks.
        device: Device for the model, local momentum, and analysis vectors.

    Returns:
        MeasurementCheckpoint

    Raises:
        ValueError: A restart bundle was supplied, snapshot metadata or tensor
            shapes disagree, matrix assignments are incomplete or repeated,
            or saved optimizer settings are invalid.
        FileNotFoundError: A snapshot or capture_config.json is missing.
    """
    paths = [get_formatted_path_to_capture_file(capture, 0, "metrics_snapshot", update)]
    if (capture / "COMPLETE.json").exists():
        raise ValueError("use --resume-checkpoint for a full restart bundle; --capture expects metrics snapshots")
    saved_capture = torch.load(paths[0], map_location="cpu", weights_only=True, mmap=True)
    if saved_capture.get("format") != 3 or saved_capture.get("update") != update:
        raise ValueError("snapshot format or update differs from the requested measurement")
    weights = saved_capture["model_before"]
    parameter_names = sorted(
        name for name, weight in weights.items()
        if name.startswith("blocks.") and weight.ndim == 2
    )
    if not parameter_names:
        raise ValueError("snapshot has no hidden matrix weights")
    momenta = {}
    step_configs = load_muon_step_configs(capture, saved_capture, parameter_names)
    layouts = []
    paths = [
        get_formatted_path_to_capture_file(capture, checkpoint_rank, "metrics_snapshot", update)
        for checkpoint_rank in range(world_size)
    ]
    for checkpoint_rank, path in enumerate(paths):
        rank_capture = saved_capture if checkpoint_rank == 0 else torch.load(
            path, map_location="cpu", weights_only=True, mmap=True,
        )
        if rank_capture.get("update") != update or rank_capture.get("rank") != checkpoint_rank:
            raise ValueError(f"snapshot update/rank mismatch: {path}")
        local_momenta = rank_capture["post_insertion_prepolar_momentum"]
        for name, momentum in local_momenta.items():
            if name not in weights or momentum.shape != weights[name].shape:
                raise ValueError(f"saved momentum and weight shapes differ: {name}")
        layouts.append([(name, tuple(weights[name].shape)) for name in sorted(local_momenta)])
        if checkpoint_rank == rank:
            momenta = {name: value.to(device) for name, value in local_momenta.items()}
    vector_config = ShardedVectorConfig(layouts, rank=rank, device=device)
    if set(vector_config.parameter_names) != set(parameter_names):
        raise ValueError("launch one process per capture rank to include every hidden matrix")
    return MeasurementCheckpoint(
        update=update,
        model=load_model_from_weights(weights, saved_capture["buffers"], device),
        momenta=momenta,
        step_configs=step_configs,
        vector_config=vector_config,
        paths=paths,
    )


def load_model_from_weights(
    weights: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    device: torch.device,
) -> GPTWithFixedHeadLoss:
    """Construct the saved GPT architecture, then load its weights and buffers.

    Args:
        weights: Map parameter names to saved tensors. Infer vocabulary size
            and model width from embed.weight, shaped (vocab_size, model_dim),
            and count layers from the blocks.<index> parameter names.
        buffers: Map buffer names to saved tensors, including rotary
            frequencies for each attention layer.
        device: Device on which to construct and load the model.

    Returns:
        GPTWithFixedHeadLoss containing the loaded parameters and buffers on
        device, with evaluation mode enabled. Copy saved values into the
        dtypes chosen by the model constructor.

    Raises:
        RuntimeError: Saved keys or tensor shapes disagree with the model.
    """
    vocab_size, model_dim = weights["embed.weight"].shape
    num_layers = 1 + max(
        int(name.split(".")[1]) for name in weights if name.startswith("blocks.")
    )
    model = GPTWithFixedHeadLoss(vocab_size, num_layers, model_dim).to(device)
    model.load_state_dict({**weights, **buffers}, strict=True)
    return model.eval()


def load_initial_eigenvector(
    directory: Path | None, vector_config: ShardedVectorConfig,
) -> torch.Tensor | None:
    """Read this rank's starting direction and check its size against the layout.

    Args:
        directory: Directory containing current_vector_rank<R>.pt, or None
            to leave starting-vector selection to the eigensolver.
        vector_config: Provide the rank, target device, and padded vector size.

    Returns:
        Tensor shaped (vector_config.local_vector_size,) on vector_config.device,
        preserving its saved dtype, or None when directory is None.

    Raises:
        ValueError: The saved vector shape differs from the padded local shape.
        FileNotFoundError: The requested rank's vector file is missing.
    """
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
        parameter_names: Select Muon matrices for configuration recovery.

    Returns:
        dict[str, MuonStepConfig] keyed by the supplied parameter names. For
        each matrix, recover learning_rate (float) and weight_decay (float)
        from the snapshot, calculate current_gradient_coefficient (float)
        from the scheduled kernel with any available-history normalization,
        and retain training_batch_tokens (int) from the snapshot.

    Raises:
        ValueError: The snapshot format, optimizer settings, training batch
            token count, available kernel mass, or learning rate is invalid.
        FileNotFoundError: capture_config.json is missing.
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
        if len(matches) != 1 or matches[0]["optimizer"] != "muon":
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
            learning_rate=float(group["lr"]),
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

    Args:
        path: FineWeb version-1 binary token shard.
        token_count: Number of input tokens across all ranks. Read an extra
            token to form the final target.
        rank: Current worker index, from zero through world_size minus one.
        world_size: Number of workers sharing the token prefix.
        device: Device for the returned input and target tensors.
        sequence_length: Number of tokens per sequence. Defaults to 1024.

    Returns:
        Tuple (inputs, targets) of torch.int64 tensors on device, each shaped
        (token_count // (world_size * sequence_length), sequence_length).
        Pair each input token with the next token in the shard as its target.

    Raises:
        ValueError: token_count is nonpositive or does not divide into whole
            sequences on every rank, the shard header is invalid, or the
            shard size differs from its header or supplies too few tokens.
        FileNotFoundError: The token shard is missing.
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
