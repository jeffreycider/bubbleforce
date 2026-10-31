import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from harness.run_state import Config, NanoGPTRunState

from harness.recording.store import CaptureWriter, file_sha
from optimizers.ringed.momentum.kernel import (
    HISTORY_LENGTH,
    GradientRing,
    reconstruct_ring_from_gradients,
)


def training_config_fingerprint(config: Config) -> str:
    """Resume-compatibility fingerprint, not a source hash or correctness check.

    Keep the resolved recipe fixed; recording and execution locations may change.
    Unknown keys remain bound rather than silently accepting a training change.
    """
    execution = {
        "run_id",
        "output_dir",
        "checkpoint_dir",
        "capture_dir",
        "checkpoint",
        "metrics_snapshot",
        "resume_checkpoint",
        "start_step",
        "stop_step",
    }
    protocol = {k: v for k, v in config.items() if k not in execution}
    if "setup" in protocol:
        # Restore ordering is execution control, not a change to the recipe.
        protocol["setup"] = [
            hook for hook in protocol["setup"] if hook["name"] != "restore_checkpoint"
        ]
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def _saved_parameter_names(
    run_state: NanoGPTRunState, saved_optimizers: list[dict[str, Any]]
) -> dict[tuple[int, int], str]:
    """Map (optimizer index, PyTorch parameter ID) to its model parameter name.

    PyTorch restores optimizer state by parameter position, not by name. Both
    saving and reconstruction must use the same positional binding.
    """
    names = {id(parameter): name for name, parameter in run_state.model.named_parameters()}
    mapping = {}
    for index, ((_, optimizer), saved) in enumerate(
        zip(run_state.optimizer.groups, saved_optimizers, strict=True)
    ):
        saved_ids = [pid for group in saved["param_groups"] for pid in group["params"]]
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        for pid, parameter in zip(saved_ids, parameters, strict=True):
            mapping[index, pid] = names[id(parameter)]
    return mapping


def use_capture_directory(run_state: NanoGPTRunState, store: CaptureWriter) -> None:
    """Bind a writer to this trajectory; refuse adopting or rewinding old output."""
    root = store.root.parent.resolve()
    if run_state.capture_root != root and any(store.root.iterdir()):
        raise ValueError("new training trajectories require a new capture_dir")
    checkpoint_step = run_state.step + 1
    if any(step > checkpoint_step if kind == "checkpoint" else step >= checkpoint_step
           for kind, step in store.receipts):
        raise ValueError("capture_dir has later outputs; use a new capture_dir for this fork")
    run_state.capture_root = root


def _store_gradient_history(
    run_state: NanoGPTRunState,
    optimizer_state_dict: dict[str, Any],
    store: CaptureWriter,
    rank_zero_references: dict[int, dict[str, Any]],
    checkpoint_step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Store raw-gradient files and replace saved buffers with their references."""
    parameters = dict(run_state.model.named_parameters())
    parameter_names = _saved_parameter_names(run_state, optimizer_state_dict["optimizers"])
    rings, ring_metadata = {}, []
    for group_index, saved_optimizer in enumerate(optimizer_state_dict["optimizers"]):
        for saved_parameter_id, parameter_state in saved_optimizer["state"].items():
            if "gradient_newest" not in parameter_state:
                continue
            ring = GradientRing()
            ring.load_state_dict(parameter_state)
            name = parameter_names[group_index, saved_parameter_id]
            rings[name] = ring
            ring_metadata.append(dict(
                group=group_index, parameter=saved_parameter_id, name=name,
            ))
            saved_parameter_state = parameter_state.copy()
            saved_parameter_state.pop("gradient_history", None)
            saved_optimizer["state"][saved_parameter_id] = saved_parameter_state
    names = set(rings)
    dependencies = []
    gradient_streams = [
        ring.get_enumerated_gradients_from_ring(checkpoint_step, parameters[name].shape)
        for name, ring in rings.items()
    ]
    for rows in zip(*gradient_streams, strict=True):
        step = rows[0][0]
        reference = run_state.gradient_references.get(step)
        if reference is None or not names <= set(reference["parameters"]):
            reference = rank_zero_references.get(step)
        if reference is None or not names <= set(reference["parameters"]):
            receipt = store.save(
                "checkpoint_gradient",
                step,
                {name: gradient for name, (_, gradient) in zip(rings, rows, strict=True)},
            )
            reference = dict(receipt, file=str((store.root / receipt["file"]).resolve()))
        run_state.gradient_references[step] = reference
        dependencies.append(dict(reference, file=os.path.relpath(reference["file"], store.root.parent)))
    run_state.gradient_references = {
        step: reference for step, reference in run_state.gradient_references.items()
        if step >= checkpoint_step - HISTORY_LENGTH
    }
    return ring_metadata, dependencies


def _save_rank_checkpoint(
    config: Config, run_state: NanoGPTRunState, store: CaptureWriter,
    rank_zero_references: dict[int, dict[str, Any]],
) -> dict:
    """Save this rank's gradient dependencies and training state; return a receipt."""
    checkpoint_step = run_state.step + 1
    rank, world = run_state.rank, run_state.world_size
    root = store.root.parent
    use_capture_directory(run_state, store)
    optimizer = run_state.optimizer
    optimizer_state_dict = optimizer.state_dict()
    # A custom hook must not save weights from one update with another clock.
    if any(
        group.get("iteration", checkpoint_step) != checkpoint_step
        for entry in optimizer_state_dict["optimizers"] for group in entry["param_groups"]
    ):
        raise ValueError("checkpoint step differs from scheduler clock")
    # Replace ring entries only in the saved ID map, never in live state.
    # Tensor ownership is handled by the writer at serialization.
    optimizer_state_dict["optimizers"] = [
        dict(sd, state=sd["state"].copy())
        for sd in optimizer_state_dict["optimizers"]
    ]
    ring_metadata, dependencies = _store_gradient_history(
        run_state, optimizer_state_dict, store, rank_zero_references, checkpoint_step
    )
    payload = dict(
        format=3,
        boundary=checkpoint_step,
        world_size=world,
        rank=rank,
        protocol_sha256=training_config_fingerprint(config),
        model=run_state.model.state_dict(),
        **optimizer_state_dict,
        gradient_ring_storage="references",
        histories=ring_metadata,
        gradients=dependencies,
        rng=rng_state(),
    )
    receipt = store.save("checkpoint", checkpoint_step, payload)
    receipt = dict(receipt, file=str((store.root / receipt["file"]).relative_to(root)))
    return receipt


def save_checkpoint(
    config: Config, run_state: NanoGPTRunState, store: CaptureWriter
) -> None:
    """Publish a complete bundle only after every rank reports a saved shard.

    A local write error is collected before publication, so other ranks refuse
    the incomplete checkpoint instead of hanging at an unmatched barrier.
    """
    rank, world = run_state.rank, run_state.world_size
    checkpoint_step = run_state.step + 1
    # Share only references owned by the active trajectory, never its manifest.
    rank_zero_references = [
        {step: reference for step, reference in run_state.gradient_references.items()
         if checkpoint_step - HISTORY_LENGTH <= step < checkpoint_step}
        if rank == 0 else None
    ]
    if dist.is_initialized():
        dist.broadcast_object_list(rank_zero_references, src=0)
    receipt, error = None, None
    try:
        receipt = _save_rank_checkpoint(config, run_state, store, rank_zero_references[0])
    except Exception as failure:
        if not dist.is_initialized():
            raise
        error = f"rank {rank}: {type(failure).__name__}: {failure}"
    results = [(receipt, error)]
    if dist.is_initialized():
        results = [None] * world
        dist.all_gather_object(results, (receipt, error))
    failures = [failure for _, failure in results if failure]
    if failures:
        raise RuntimeError("checkpoint incomplete: " + "; ".join(failures))
    if rank == 0:
        try:
            folder = store.root.parent / f"checkpoint_{checkpoint_step:06d}"
            folder.mkdir(exist_ok=True)
            temporary = folder / "COMPLETE.json.tmp"
            with temporary.open("w") as stream:
                json.dump(
                    dict(
                        format=3,
                        boundary=checkpoint_step,
                        world_size=world,
                        protocol_sha256=training_config_fingerprint(config),
                        shards=[r for r, _ in results],
                    ),
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, folder / "COMPLETE.json")
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
    errors = [error]
    if dist.is_initialized():
        dist.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise RuntimeError("checkpoint completion failed: " + errors[0])


def load_checkpoint(config: Config, run_state: NanoGPTRunState) -> int:
    """Reconstruct omitted gradient rings, then restore model/optimizer/RNG.

    File hashes protect against partial copyback. Recipe/world-size checks prevent
    resuming with a different schedule or distributed parameter ownership. Native
    model/optimizer loaders own their state compatibility checks.
    """
    folder = Path(config["resume_checkpoint"])
    if run_state.train_batches is not None:
        raise ValueError("restore checkpoint before opening training data")
    if not folder.is_dir():
        raise ValueError(
            "resume_checkpoint must be a schema-3 checkpoint bundle directory"
        )
    complete = json.loads((folder / "COMPLETE.json").read_text())
    if complete.get("format") != 3:
        raise ValueError("unsupported checkpoint format; expected schema 3")
    rank, world = run_state.rank, run_state.world_size
    if complete["world_size"] != world or complete[
        "protocol_sha256"
    ] != training_config_fingerprint(config):
        raise ValueError("resume protocol or world size changed")
    if len(complete["shards"]) != world:
        raise ValueError("checkpoint completion receipt has missing ranks")
    root = folder.parent
    receipt = complete["shards"][rank]
    path = root / receipt["file"]
    if file_sha(path) != receipt["sha256"]:
        raise ValueError("checkpoint shard hash mismatch")
    # Map the archive rather than allocating another full host-side checkpoint.
    # torch.load's default private mapping keeps CPU optimizer writes off disk.
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    # The existing schema calls the checkpoint step "boundary".
    checkpoint_step = payload["boundary"]
    identity = dict(
        format=3,
        boundary=complete["boundary"],
        rank=rank,
        world_size=world,
        protocol_sha256=complete["protocol_sha256"],
    )
    if (
        type(checkpoint_step) is not int
        or checkpoint_step < 0
        or any(payload.get(key) != value for key, value in identity.items())
    ):
        raise ValueError("checkpoint step/rank metadata differs")
    references = (_restore_gradient_history(payload, run_state, root, checkpoint_step)
                  if payload.get("gradient_ring_storage") != "inline" else {})
    run_state.model.load_state_dict(payload["model"])
    run_state.optimizer.load_state_dict(payload)
    run_state.step = checkpoint_step - 1
    restore_rng(payload["rng"])
    run_state.capture_root = root.resolve()
    run_state.gradient_references = references
    return checkpoint_step


def _restore_gradient_history(
    payload: dict[str, Any], run_state: NanoGPTRunState, root: Path, checkpoint_step: int
) -> dict[int, dict[str, Any]]:
    """Replace omitted rings using verified raw gradients, before live loading."""
    parameters = dict(run_state.model.named_parameters())
    ring_states = {}
    for group, optimizer in enumerate(payload["optimizers"]):
        for parameter, entry in optimizer["state"].items():
            if "gradient_newest" in entry:
                ring = GradientRing()
                ring.load_state_dict(entry)
                ring_states[group, parameter] = ring
    references = {(row["group"], row["parameter"]): row for row in payload["histories"]}
    if ring_states.keys() != references.keys() or len(references) != len(payload["histories"]):
        raise ValueError("checkpoint history references are incomplete")

    if not ring_states:
        if payload["gradients"]:
            raise ValueError("checkpoint gradients have no associated rings")
        return {}
    names = _saved_parameter_names(run_state, payload["optimizers"])
    if any(reference["name"] != names[coordinate] for coordinate, reference in references.items()):
        raise ValueError("checkpoint history differs from optimizer parameter order")
    # Map each file once. Per-matrix reconstruction can revisit these tensors
    # without rereading archives or allocating a second full history in RAM.
    gradient_records: list[tuple[int, dict[str, torch.Tensor]]] = []
    owned_references = {}
    for reference in payload["gradients"]:
        source = root / reference["file"]
        if file_sha(source) != reference["sha256"]:
            raise ValueError("checkpoint gradient hash mismatch")
        gradients = torch.load(
            source, map_location="cpu", weights_only=True, mmap=True,
        )
        gradient_records.append((reference["step"], gradients))
        owned_references[reference["step"]] = dict(
            reference, file=str(source.resolve()), parameters=sorted(gradients),
        )
    for coordinate, ring in ring_states.items():
        reference = references[coordinate]
        name = reference["name"]
        ring.validate_ring_history_against_checkpoint_step(checkpoint_step)
        reconstructed = reconstruct_ring_from_gradients(
            ((step, gradients[name]) for step, gradients in gradient_records),
            parameters[name],
            checkpoint_step,
        )
        group, parameter = coordinate
        payload["optimizers"][group]["state"][parameter].update(reconstructed.state_dict())
    return owned_references


def rng_state() -> dict[str, Any]:
    """Save every training RNG using only weights_only-loadable values."""
    kind, keys, position, has_gauss, cached = np.random.get_state()
    saved = dict(
        torch=torch.get_rng_state(),
        python=random.getstate(),
        numpy=(kind, keys.tolist(), position, has_gauss, cached),
    )
    if torch.cuda.is_available():
        saved["cuda"] = torch.cuda.get_rng_state_all()
    return saved


def restore_rng(saved: dict[str, Any]) -> None:
    """Restore RNGs after loading the saved training state."""
    torch.set_rng_state(saved["torch"])
    random.setstate(saved["python"])
    kind, keys, position, has_gauss, cached = saved["numpy"]
    np.random.set_state(
        (kind, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached)
    )
    if "cuda" in saved and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda"])
