import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from run import prepare_experiment
from harness.optimizers import GroupedOptimizers
from harness.run_state import NanoGPTRunState
from harness.recording.snapshots import Recording
from harness.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    training_config_fingerprint,
)
from tests.recording_helpers import (
    Model, configuration, state_for, gradients, muon_ring_digests,
)
from harness.recording.store import CaptureWriter, file_sha


class TwoBlockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(3, 2, bias=False) for _ in range(2)]
        )
        self.head = torch.nn.Linear(2, 1, bias=False)


def open_test_batches(state, start=0, *, training=False):
    """Open an observable toy stream at the update selected by run_state.step."""
    state.train_batches = (
        iter([(torch.arange(6).reshape(2, 3), torch.arange(6))] * (600 - start))
        if training
        else iter(range(start, 600))
    )


def advance(state, step):
    if state.train_batches is None:
        open_test_batches(state, state.step + 1)
    state.step = step
    next(state.train_batches)
    gradients(state, step)
    state.optimizer.step()
    state.optimizer.zero_grad()


def bundle(tmp_path, steps=4):
    cfg = configuration(tmp_path)
    original = state_for(cfg)
    for step in range(steps):
        advance(original, step)
    save_checkpoint(cfg, original, CaptureWriter(cfg["capture_dir"], original.rank))
    cfg["resume_checkpoint"] = str(tmp_path / f"checkpoint_{steps:06d}")
    return cfg, original


def test_experiment_records_requested_updates_with_its_optimizer_settings():
    training, measurement, world_size = prepare_experiment("configs/maxentslop.yaml", ["measurement.measured_updates=[10,11,12]"])
    config = dict(training.to_training_dict(), capture_dir=str(measurement.capture_dir),
                  require_world_size=world_size)
    assert config["require_world_size"] == 8
    assert config["optimizer_groups"][2]["hyperparams"]["momentum"]["start_step"] == 1000
    assert Recording(config).snapshot_steps == {10, 11, 12}
    assert Recording(config).quantities == {"weights", "momentum"}
    assert config["stop_step"] == 13
    assert config["train_steps"] == 3250


def test_resume_fingerprint_ignores_restore_hook_but_binds_numeric_hook_values():
    config = {
        "setup": [{"name": "seed_process"}],
        "post_optimizer": [{"name": "print_training_progress", "hyperparams": {"every": 1}}],
    }
    restored = copy.deepcopy(config)
    restored["setup"].insert(0, {"name": "restore_checkpoint"})
    assert training_config_fingerprint(restored) == training_config_fingerprint(config)

    changed = copy.deepcopy(restored)
    changed["post_optimizer"][0]["hyperparams"]["every"] = 2
    assert training_config_fingerprint(changed) != training_config_fingerprint(restored)


def test_snapshot_records_actual_update_not_weights(tmp_path):
    cfg = configuration(tmp_path)
    state = state_for(cfg)
    recording = Recording(cfg)
    recording.setup(state)
    open_test_batches(state)
    for step in range(4):
        state.step = step
        next(state.train_batches)
        gradients(state, step)
        before = {n: p.detach().clone() for n, p in state.model.named_parameters()}
        recording.before_optimizer(state)
        state.optimizer.step()
        state.optimizer.zero_grad()
        recording.after_optimizer(state)
        recording.after_step(state)
        if step >= 2:
            payload = torch.load(
                tmp_path / "rank0" / f"metrics_snapshot_{step:06d}.pt",
                weights_only=True,
            )
            momentum = payload["post_insertion_prepolar_momentum"]["blocks.weight"]
            torch.testing.assert_close(
                payload["polar_output"]["blocks.weight"],
                recording.optimizer.polar(momentum),
                rtol=0,
                atol=0,
            )
            for n, p in state.model.named_parameters():
                torch.testing.assert_close(
                    payload["executed_update"][n], before[n] - p, rtol=0, atol=0
                )
            assert payload["update"] == step
    cfg["resume_checkpoint"] = str(tmp_path / "checkpoint_000004")
    recovered = state_for(cfg)
    assert load_checkpoint(cfg, recovered) == 4
    assert muon_ring_digests(recovered) == muon_ring_digests(state)
    open_test_batches(recovered, recovered.step + 1)
    advance(state, 4)
    advance(recovered, 4)
    for a, b in zip(state.model.parameters(), recovered.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("steps", [1, 4, 255, 256, 260])
def test_checkpoint_without_dense_interval_reconstructs_exact_state(tmp_path, steps):
    cfg, state = bundle(tmp_path, steps)
    shard = tmp_path / "rank0" / f"checkpoint_{steps:06d}.pt"
    saved_sha = file_sha(shard)
    payload = torch.load(shard, weights_only=True)
    assert "gradient_history" not in payload["optimizers"][0]["state"][0]
    assert len(payload["gradients"]) == min(steps, 256)
    assert len(list((tmp_path / "rank0").glob("checkpoint_gradient_*.pt"))) == min(steps, 256)
    restored = state_for(cfg)
    load_checkpoint(cfg, restored)
    assert restored.step == steps - 1
    assert muon_ring_digests(restored) == muon_ring_digests(state)
    open_test_batches(restored, restored.step + 1)
    advance(state, steps)
    advance(restored, steps)
    for a, b in zip(state.model.parameters(), restored.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Adam is restored directly, not approximated by replaying raw gradients.
    a = state.optimizer.groups[1][1].state_dict()
    b = restored.optimizer.groups[1][1].state_dict()
    for key in a["state"][0]:
        assert torch.equal(a["state"][0][key], b["state"][0][key])
    # Restored CPU Adam tensors may use private mapped storage. Optimizer steps
    # must not mutate the checkpoint file those tensors came from.
    assert file_sha(shard) == saved_sha


def test_overlapping_checkpoints_save_gradients_once(tmp_path):
    cfg, state = bundle(tmp_path, 4)
    hashes = {p.name: file_sha(p) for p in (tmp_path / "rank0").glob("checkpoint_gradient_*.pt")}
    advance(state, 4)
    save_checkpoint(cfg, state, CaptureWriter(cfg["capture_dir"], state.rank))
    assert len(list((tmp_path / "rank0").glob("checkpoint_gradient_*.pt"))) == 5
    assert all(file_sha(tmp_path / "rank0" / name) == sha for name, sha in hashes.items())
    assert len((tmp_path / "rank0/manifest.jsonl").read_text().splitlines()) == 7
    with pytest.raises(FileExistsError):
        save_checkpoint(cfg, state, CaptureWriter(cfg["capture_dir"], state.rank))


def test_another_trajectory_cannot_adopt_existing_capture_files(tmp_path):
    """Same config/step names never grant ownership of an existing capture."""
    cfg = configuration(tmp_path)
    captured = state_for(cfg)
    recording = Recording(cfg)
    recording.setup(captured)
    for step in range(4):
        captured.step = step
        gradients(captured, step)
        recording.before_optimizer(captured)
        captured.optimizer.step()
        captured.optimizer.zero_grad()
        recording.after_optimizer(captured)
    archived = {p.name: file_sha(p) for p in (tmp_path / "rank0").glob("gradient_*.pt")}

    # Same policy and file names, but a different actual training history.
    changed = state_for(cfg)
    for step in range(4):
        changed.step = step
        gradients(changed, step)
        for parameter in changed.model.parameters():
            parameter.grad.add_(0.5)
        changed.optimizer.step()
        changed.optimizer.zero_grad()
    with pytest.raises(ValueError, match="new training trajectories"):
        save_checkpoint(cfg, changed, recording.writer)
    assert all(file_sha(tmp_path / "rank0" / name) == digest
               for name, digest in archived.items())
    assert not (tmp_path / "checkpoint_000004/COMPLETE.json").exists()
    assert not changed.gradient_references


def test_checkpoint_reuses_owned_receipts_not_writer_manifest(tmp_path):
    cfg = configuration(tmp_path)
    state = state_for(cfg)
    recording = Recording(cfg)
    recording.setup(state)
    for step in range(4):
        state.step = step
        gradients(state, step)
        recording.before_optimizer(state)
        state.optimizer.step()
        state.optimizer.zero_grad()
        recording.after_optimizer(state)
    owned = copy.deepcopy(state.gradient_references)
    # These are lookup candidates, not receipts owned by this run state.
    for (kind, step), receipt in recording.writer.receipts.items():
        if kind == "gradient":
            receipt["file"], receipt["sha256"] = "unrelated.pt", "wrong"
    save_checkpoint(cfg, state, recording.writer)
    payload = torch.load(tmp_path / "rank0/checkpoint_000004.pt", weights_only=True)
    assert all("sha256" not in history for history in payload["histories"])
    assert all(reference["sha256"] == owned[reference["step"]]["sha256"]
               for reference in payload["gradients"])
    cfg["resume_checkpoint"] = str(tmp_path / "checkpoint_000004")
    restored = state_for(cfg)
    load_checkpoint(cfg, restored)
    assert muon_ring_digests(restored) == muon_ring_digests(state)


def test_latest_checkpoint_continues_in_same_directory_without_resaving_history(tmp_path):
    cfg, original = bundle(tmp_path)
    restored = state_for(cfg)
    load_checkpoint(cfg, restored)
    inherited = copy.deepcopy(restored.gradient_references)
    Recording(cfg).setup(restored)
    advance(restored, 4)
    save_checkpoint(cfg, restored, CaptureWriter(tmp_path, restored.rank))
    assert all(restored.gradient_references[step] == reference
               for step, reference in inherited.items())
    assert len(list((tmp_path / "rank0").glob("checkpoint_gradient_*.pt"))) == 5
    advance(original, 4)
    assert muon_ring_digests(restored) == muon_ring_digests(original)


def test_checkpoint_fork_inherits_only_selected_references_in_new_directory(tmp_path):
    source_root, fork_root = tmp_path / "source", tmp_path / "fork"
    cfg, original = bundle(source_root)
    advance(original, 4)
    save_checkpoint(cfg, original, CaptureWriter(source_root, original.rank))
    later = torch.load(source_root / "rank0/checkpoint_000005.pt", weights_only=True)
    fork = state_for(cfg)
    load_checkpoint(cfg, fork)  # checkpoint 4, not checkpoint 5's later files
    assert set(fork.gradient_references) == set(range(4))
    with pytest.raises(ValueError, match="later outputs"):
        Recording(cfg).setup(fork)
    fork_config = dict(cfg, capture_dir=str(fork_root))
    Recording(fork_config).setup(fork)
    fork.step = 4
    gradients(fork, 4)
    for parameter in fork.model.parameters():
        parameter.grad.add_(0.5)
    fork.optimizer.step()
    fork.optimizer.zero_grad()
    save_checkpoint(fork_config, fork, CaptureWriter(fork_root, fork.rank))
    payload = torch.load(fork_root / "rank0/checkpoint_000005.pt", weights_only=True)
    for reference in payload["gradients"]:
        origin = (fork_root / reference["file"]).resolve()
        assert origin.is_relative_to(source_root if reference["step"] < 4 else fork_root)
    assert payload["gradients"][-1]["sha256"] != later["gradients"][-1]["sha256"]
    restored_config = dict(fork_config, resume_checkpoint=str(fork_root / "checkpoint_000005"))
    restored = state_for(restored_config)
    load_checkpoint(restored_config, restored)
    assert muon_ring_digests(restored) == muon_ring_digests(fork)


def test_new_trajectory_refuses_existing_files_without_a_manifest(tmp_path):
    cfg = configuration(tmp_path)
    root = tmp_path / "rank0"
    root.mkdir()
    torch.save({"blocks.weight": torch.ones(2, 3)}, root / "gradient_000000.pt")
    with pytest.raises(ValueError, match="new training trajectories"):
        Recording(cfg).setup(state_for(cfg))


def rewrite_payload(tmp_path, change):
    path = tmp_path / "rank0/checkpoint_000004.pt"
    payload = torch.load(path, weights_only=True)
    change(payload)
    torch.save(payload, path)
    complete = tmp_path / "checkpoint_000004/COMPLETE.json"
    receipt = json.loads(complete.read_text())
    receipt["shards"][0]["sha256"] = file_sha(path)
    complete.write_text(json.dumps(receipt))


@pytest.mark.parametrize(
    "defect",
    [
        "boundary",
        "rank",
        "world_size",
        "format",
        "gradient_newest",
        "gradient_filled",
        "missing_sample",
        "duplicate_sample",
        "missing_history",
        "history_group",
        "history_parameter",
        "history_name",
        "swapped_history_coordinates",
    ],
)
def test_bad_checkpoint_refused_before_live_mutation(tmp_path, defect):
    if defect == "swapped_history_coordinates":
        cfg = configuration(tmp_path)
        cfg["optimizer_groups"][0]["pattern"] = r"blocks\..*weight"
        source = state_for(cfg, TwoBlockModel())
        for step in range(4):
            advance(source, step)
        save_checkpoint(cfg, source, CaptureWriter(cfg["capture_dir"], source.rank))
        cfg["resume_checkpoint"] = str(tmp_path / "checkpoint_000004")
    else:
        cfg, _ = bundle(tmp_path)

    def change(p):
        if defect in {"boundary", "rank", "world_size", "format"}:
            p[defect] += 1
        elif defect in {"gradient_newest", "gradient_filled"}:
            p["optimizers"][0]["state"][0][defect] += 1
        elif defect == "missing_sample":
            p["gradients"].pop()
        elif defect == "duplicate_sample":
            p["gradients"].append(p["gradients"][-1])
        elif defect == "missing_history":
            p["histories"].clear()
        elif defect == "history_group":
            p["histories"][0]["group"] += 100
        elif defect == "history_parameter":
            p["histories"][0]["parameter"] += 100
        elif defect == "history_name":
            p["histories"][0]["name"] = "missing.parameter"
        elif defect == "swapped_history_coordinates":
            first, second = p["histories"][:2]
            assert first["group"] == second["group"]
            first["parameter"], second["parameter"] = second["parameter"], first["parameter"]
    rewrite_payload(tmp_path, change)
    restored = state_for(cfg, TwoBlockModel()) if defect == "swapped_history_coordinates" else state_for(cfg)
    before = copy.deepcopy(restored.model.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises((ValueError, KeyError)):
        load_checkpoint(cfg, restored)
    for name, p in restored.model.state_dict().items():
        assert torch.equal(p, before[name])
    assert all(not optimizer.state for _, optimizer in restored.optimizer.groups)
    assert restored.step == -1
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("fault", ["missing", "corrupt"])
def test_gradient_dependency_failure_refused(tmp_path, fault):
    cfg, _ = bundle(tmp_path)
    path = tmp_path / "rank0/checkpoint_gradient_000000.pt"
    if fault == "missing":
        path.unlink()
    else:
        with path.open("ab") as stream:
            stream.write(b"corrupt")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_checkpoint(cfg, state_for(cfg))


def test_schema3_gradient_references_load_legacy_gradient_filenames(tmp_path):
    cfg, original = bundle(tmp_path)
    checkpoint = tmp_path / "rank0/checkpoint_000004.pt"
    payload = torch.load(checkpoint, weights_only=True)
    for reference in payload["gradients"]:
        source = tmp_path / reference["file"]
        legacy_name = source.name.replace("checkpoint_gradient_", "gradient_")
        destination = source.with_name(legacy_name)
        source.rename(destination)
        reference["file"] = destination.relative_to(tmp_path).as_posix()
    torch.save(payload, checkpoint)

    complete = tmp_path / "checkpoint_000004/COMPLETE.json"
    receipt = json.loads(complete.read_text())
    receipt["shards"][0]["sha256"] = file_sha(checkpoint)
    complete.write_text(json.dumps(receipt))

    restored = state_for(cfg)
    assert load_checkpoint(cfg, restored) == 4
    assert muon_ring_digests(restored) == muon_ring_digests(original)


@pytest.mark.parametrize("change", ["recipe", "world_size"])
def test_incompatible_resume_refused_before_loading_weights(tmp_path, change):
    cfg, _ = bundle(tmp_path)
    target = state_for(cfg)
    before = copy.deepcopy(target.model.state_dict())
    if change == "recipe":
        cfg["optimizer_groups"][0]["hyperparams"]["lr"] *= 2
    else:
        target.world_size = 2
    with pytest.raises(ValueError, match="protocol or world size"):
        load_checkpoint(cfg, target)
    for name, value in target.model.state_dict().items():
        assert torch.equal(value, before[name])


def test_native_model_loader_refuses_incompatible_shapes(tmp_path):
    cfg, _ = bundle(tmp_path)
    rewrite_payload(tmp_path, lambda p: p["model"].update({"blocks.weight": torch.zeros(3, 2)}))
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_checkpoint(cfg, state_for(cfg))


def test_recording_preserves_nonfinite_state_for_diagnosis(tmp_path):
    cfg = configuration(tmp_path)
    source = state_for(cfg)
    advance(source, 0)
    with torch.no_grad():
        source.model.head.weight[0, 0] = float("nan")
    save_checkpoint(cfg, source, CaptureWriter(cfg["capture_dir"], source.rank))
    cfg["resume_checkpoint"] = str(tmp_path / "checkpoint_000001")
    target = state_for(cfg)
    load_checkpoint(cfg, target)
    assert torch.isnan(target.model.head.weight[0, 0])


def test_partial_checkpoint_is_not_resumable(tmp_path):
    cfg, _ = bundle(tmp_path)
    (tmp_path / "checkpoint_000004/COMPLETE.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(cfg, state_for(cfg))


def test_checkpoint_rng_and_loader_restored(tmp_path):
    cfg, _ = bundle(tmp_path)
    expected = (random.random(), float(np.random.random()), torch.rand(3))
    restored = state_for(cfg)
    open_test_batches(restored)
    with pytest.raises(ValueError, match="before opening training data"):
        load_checkpoint(cfg, restored)
    restored.train_batches = None
    load_checkpoint(cfg, restored)
    assert restored.step == 3
    assert random.random() == expected[0]
    assert float(np.random.random()) == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    open_test_batches(restored, restored.step + 1)
    assert next(restored.train_batches) == 4


def test_reconstruction_maps_each_gradient_file_once_for_multiple_matrices(tmp_path, monkeypatch):
    from harness import checkpoint

    cfg = configuration(tmp_path)
    cfg["optimizer_groups"][1].update(
        optimizer="muon",
        hyperparams=dict(lr=.01, momentum=dict(kind="ema", decay=.95)),
    )
    original = state_for(cfg)
    for step in range(4):
        advance(original, step)
    save_checkpoint(cfg, original, CaptureWriter(cfg["capture_dir"], original.rank))
    cfg["resume_checkpoint"] = str(tmp_path / "checkpoint_000004")
    restored = state_for(cfg)
    load = torch.load
    mapped = []

    def track_load(path, **kwargs):
        if Path(path).name.startswith("checkpoint_gradient_"):
            assert kwargs["mmap"] is True
            mapped.append(Path(path))
        return load(path, **kwargs)

    monkeypatch.setattr(checkpoint.torch, "load", track_load)
    load_checkpoint(cfg, restored)
    open_test_batches(restored, restored.step + 1)
    assert len(mapped) == len(set(mapped)) == 4
    assert muon_ring_digests(restored) == muon_ring_digests(original)


def test_update_and_scheduler_clock_refused_at_save(tmp_path):
    cfg, state = bundle(tmp_path)
    state.step = 4
    state.optimizer.groups[0][1].param_groups[0]["iteration"] = 3
    with pytest.raises(ValueError, match="scheduler clock"):
        save_checkpoint(cfg, state, CaptureWriter(cfg["capture_dir"], state.rank))



def test_checkpoint_only_needs_no_muon_observer_or_snapshot_hooks(tmp_path):
    cfg = configuration(tmp_path)
    cfg["metrics_snapshot"] = None
    cfg["optimizer_groups"] = [dict(
        pattern=".*", optimizer="adamw", hyperparams=dict(lr=0.01, weight_decay=0.0))]
    state = state_for(cfg)
    recording = Recording(cfg)
    assert not recording.snapshot_steps
    assert Recording(cfg).gradient_steps == set()
    recording.setup(state)
    open_test_batches(state)
    for step in range(4):
        advance(state, step)
        recording.after_step(state)
    cfg["resume_checkpoint"] = str(tmp_path / "checkpoint_000004")
    restored = state_for(cfg)
    assert load_checkpoint(cfg, restored) == 4
    open_test_batches(restored, restored.step + 1)
    advance(state, 4)
    advance(restored, 4)
    for actual, expected in zip(state.model.parameters(), restored.model.parameters()):
        assert torch.equal(actual, expected)
    assert not list((tmp_path / "rank0").glob("checkpoint_gradient_*.pt"))


def test_restore_then_open_stream_starts_at_saved_boundary(tmp_path):
    cfg = configuration(tmp_path)
    cfg["setup"] = [{"name": "model_and_optimizer"}, {"name": "open_training_batches"}]
    saved = state_for(cfg)
    for step in range(4):
        advance(saved, step)
    save_checkpoint(cfg, saved, CaptureWriter(cfg["capture_dir"], saved.rank))
    cfg["resume_checkpoint"] = str(tmp_path / "checkpoint_000004")

    def build(config, state):
        fixture = state_for(config)
        state.model = fixture.model
        state.optimizer = fixture.optimizer
        state.train_batches = None

    restored = NanoGPTRunState(rank=0, world_size=1)
    build(cfg, restored)
    load_checkpoint(cfg, restored)
    assert restored.step == 3
    assert restored.train_batches is None
    open_test_batches(restored, restored.step + 1)
    assert next(restored.train_batches) == 4
    assert muon_ring_digests(restored) == muon_ring_digests(saved)


@pytest.mark.parametrize("receipt", [{}, {"format": 2}, {"format": 4}])
def test_unsupported_checkpoint_format_refused_before_state_changes(tmp_path, receipt):
    cfg = configuration(tmp_path)
    folder = tmp_path / "checkpoint_000004"
    folder.mkdir()
    (folder / "COMPLETE.json").write_text(json.dumps(receipt))
    cfg["resume_checkpoint"] = str(folder)
    state = state_for(cfg)
    before = copy.deepcopy(state.model.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(cfg, state)
    for name, value in state.model.state_dict().items():
        assert torch.equal(value, before[name])
    assert torch.equal(torch.get_rng_state(), rng)
    assert state.step == -1
    assert all(not optimizer.state for _, optimizer in state.optimizer.groups)


def test_checkpoint_requires_bundle_directory(tmp_path):
    cfg = configuration(tmp_path)
    path = tmp_path / "snapshot.pt"
    torch.save({"format": 2}, path)
    cfg["resume_checkpoint"] = str(path)
    with pytest.raises(ValueError, match="checkpoint bundle directory"):
        load_checkpoint(cfg, state_for(cfg))


def test_capture_writer_durable_receipts_and_no_overwrite(tmp_path):
    writer = CaptureWriter(tmp_path, 0)
    receipt = writer.save("gradient", 4, {"blocks.weight": torch.arange(6).reshape(2, 3)})
    path = writer.root / receipt["file"]
    assert receipt["sha256"] == file_sha(path)
    assert receipt["bytes"] == path.stat().st_size
    assert CaptureWriter(tmp_path, 0).receipts["gradient", 4] == receipt
    with pytest.raises(FileExistsError):
        writer.save("gradient", 4, {})
    assert not list(writer.root.glob(".pending-*"))


def test_checkpoint_gradient_does_not_block_full_gradient_same_step(tmp_path):
    writer = CaptureWriter(tmp_path, 0)
    partial = {"blocks.weight": torch.ones(2, 3)}
    full = {**partial, "head.weight": torch.full((1, 2), 2.0)}

    checkpoint_receipt = writer.save("checkpoint_gradient", 4, partial)
    gradient_receipt = writer.save("gradient", 4, full)

    assert checkpoint_receipt["file"] == "checkpoint_gradient_000004.pt"
    assert gradient_receipt["file"] == "gradient_000004.pt"
    assert torch.load(writer.root / checkpoint_receipt["file"], weights_only=True).keys() == partial.keys()
    assert torch.load(writer.root / gradient_receipt["file"], weights_only=True).keys() == full.keys()


def test_shared_loop_runs_capture_and_checkpoint(tmp_path, monkeypatch):
    import torch.distributed as dist
    from harness import hooks as lifecycle, loop

    cfg = configuration(tmp_path / "capture")
    cfg.update(loop="gpt_record", require_world_size=1, microbatch_sequences=1)
    holder = {}

    class LossModel(Model):
        def forward(self, x, y):
            return self.head(self.blocks(x.float())).square().sum()

    def fixture_setup():
        def setup(config, state):
            built = state_for(config)
            state.model = LossModel()
            state.optimizer = GroupedOptimizers(state.model, config["optimizer_groups"])
            open_test_batches(state, training=True)
            state.print_log = built.print_log
            holder["state"] = state

        return setup

    monkeypatch.setattr(lifecycle, "build_compiled_gpt", fixture_setup)
    cfg["setup"] = [dict(name="build_compiled_gpt")]
    hooks, recording = lifecycle.prepare_training(cfg)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(tmp_path / "rendezvous"),
        rank=0,
        world_size=1,
    )
    try:
        loop.run(cfg, hooks, recording)
        assert holder["state"].step == 3
        assert (tmp_path / "capture/checkpoint_000004/COMPLETE.json").is_file()
    finally:
        dist.destroy_process_group()


def test_snapshot_does_not_reschedule_or_serialize_kernel(tmp_path):
    cfg = configuration(tmp_path)
    cfg["checkpoint"] = None
    state = state_for(cfg)
    optimizer = state.optimizer.groups[0][1]
    scheduler = optimizer.momentum_scheduler
    calls = []

    def schedule(step):
        calls.append(step)
        return scheduler(step)

    optimizer.momentum_scheduler = schedule
    recording = Recording(cfg)
    recording.setup(state)
    open_test_batches(state)
    for step in range(4):
        state.step = step
        gradients(state, step)
        recording.before_optimizer(state)
        optimizer.step()
        recording.after_optimizer(state)
    assert calls == [0, 1, 2, 3]
    payload = torch.load(tmp_path / "rank0/metrics_snapshot_000003.pt", weights_only=True)
    assert "kernel_weights_used" not in payload
    assert "normalize_available" not in payload


def test_custom_setup_restores_before_opening_iterator(tmp_path):
    cfg, original = bundle(tmp_path)
    cfg["metrics_snapshot"] = None
    # Use the same saved recipe: policies aren't part of resume compatibility.
    restored = NanoGPTRunState(rank=0, world_size=1)
    def build(config, state):
        fixture = state_for(config)
        state.model = fixture.model
        state.optimizer = fixture.optimizer
        state.train_batches = None

    build(cfg, restored)
    load_checkpoint(cfg, restored)
    assert restored.step == 3
    assert restored.train_batches is None
    assert muon_ring_digests(restored) == muon_ring_digests(original)


@pytest.mark.parametrize("policy", [[], "dense", {"quantities": "weights"}, {"quantities": [None]}])
def test_snapshot_policy_validation_uses_one_schema(policy):
    with pytest.raises(ValueError):
        Recording(dict(train_steps=10, metrics_snapshot=policy))


def test_grouped_optimizer_round_trip_loads_each_inner_once(tmp_path, monkeypatch):
    source = state_for(configuration(tmp_path)).optimizer
    saved = copy.deepcopy(source.state_dict())
    assert list(saved) == ["optimizers"]
    target = state_for(configuration(tmp_path)).optimizer
    calls = []
    for _, optimizer in target.groups:
        load = optimizer.load_state_dict

        def restore(entry, load=load):
            calls.append(load)
            load(entry)

        monkeypatch.setattr(optimizer, "load_state_dict", restore)
    target.load_state_dict(saved)
    assert len(calls) == len(target.groups)


def test_gradient_encoding_does_not_remove_buffers_from_live_optimizer(tmp_path):
    cfg = configuration(tmp_path)
    source = state_for(cfg)
    advance(source, 0)
    optimizer = source.optimizer.groups[0][1]
    original = optimizer.state[source.model.blocks.weight]["gradient_history"]
    expected = original.clone()
    save_checkpoint(cfg, source, CaptureWriter(cfg["capture_dir"], source.rank))
    assert optimizer.state[source.model.blocks.weight]["gradient_history"] is original
    assert torch.equal(original, expected)


def test_snapshot_save_failure_releases_observer_and_copies(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    state = state_for(cfg)
    recording = Recording(cfg)
    recording.setup(state)
    open_test_batches(state)
    for step in range(2):
        advance(state, step)
    state.step = 2
    gradients(state, 2)
    recording.before_optimizer(state)
    state.optimizer.step()

    def failed_write(*args):
        raise OSError("fixture storage failure")

    monkeypatch.setattr(recording.writer, "save", failed_write)
    recorder = recording
    with pytest.raises(OSError):
        recording.after_optimizer(state)
    assert state.optimizer.groups[0][1].polar_observer is None
    assert recorder.pending is None
