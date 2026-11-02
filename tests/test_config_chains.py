from dataclasses import replace
import json

import pytest

import run
from harness import hooks as lifecycle
from harness.hooks import prepare_training
from harness.recording.snapshots import Recording
from optimizers import Muon, MuonH
from run import Experiment, prepare_experiment


@pytest.mark.parametrize("name", Experiment)
def test_experiment_records_selected_measurements_without_default_restart_checkpoints(name):
    training = prepare_experiment(name)[0]
    config = training.to_training_dict()
    recording = Recording(config)
    assert config["minibatch_seed"] == 0
    assert recording.snapshot_steps == set(range(0, 3250, 50)) | {3249}
    assert recording.checkpoint_steps == set()
    assert recording.gradient_steps == set()
    assert recording.quantities == {"weights", "momentum"}
    assert "extends" not in config and "optimizer_group_overrides" not in config


def test_explicit_checkpoint_cadence_saves_restart_boundaries():
    training = prepare_experiment("original_implementation")[0]
    training = replace(training, checkpoint=dict(every=500))
    recording = Recording(training.to_training_dict())
    assert recording.checkpoint_steps == set(range(500, 3251, 500)) | {3250}


def test_all_optimizer_implementations_are_classified():
    for cls in (Muon, MuonH):
        assert ".ringed." in cls.__module__


def test_cli_parses_types_and_preserves_equals_in_strings():
    training, _, world_size = prepare_experiment("original_implementation", [
        "training.seed=1", "training.seed=2", "training.wandb_offline=false",
        "training.run_id=trial=a", "training.stop_step=10", "world_size=2",
    ])
    assert training.seed == 2
    assert training.wandb_offline is False
    assert training.run_id == "trial=a"
    assert training.stop_step == 10
    assert world_size == 2


def test_resume_origin_is_resolved_before_training(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint_000321"
    checkpoint.mkdir()
    (checkpoint / "COMPLETE.json").write_text(json.dumps(dict(boundary=321)))
    training, measurement, world_size = prepare_experiment("original_implementation", [
        f"training.resume_checkpoint={checkpoint}", "measurement.measured_updates=[321]",
    ])
    received = []
    def stop_before_cuda(config):
        received.append(config)
        raise RuntimeError("checked before CUDA")
    monkeypatch.setattr(run, "prepare_training", stop_before_cuda)
    with pytest.raises(RuntimeError, match="checked before CUDA"):
        run.run_training(training, measurement.capture_dir, world_size)
    assert received[0]["start_step"] == 321
    assert "start_step" not in training.to_training_dict()


@pytest.mark.parametrize("names", [
    ["assemble_grouped_optimizer", "open_training_batches"],
    ["restore_checkpoint", "assemble_grouped_optimizer", "open_training_batches"],
    ["assemble_grouped_optimizer", "open_training_batches", "restore_checkpoint"],
])
def test_resume_cannot_silently_train_fresh_weights_at_checkpoint_step(names):
    config = prepare_experiment("original_implementation")[0].to_training_dict()
    config.update(resume_checkpoint="checkpoint_000321", setup=[dict(name=name) for name in names])
    with pytest.raises(ValueError, match="before opening training data"):
        prepare_training(config)


@pytest.mark.parametrize("name", Experiment)
def test_resume_order_in_experiment_recipes(name):
    config = prepare_experiment(name)[0].to_training_dict()
    config["resume_checkpoint"] = "checkpoint_000321"
    hooks, recording = prepare_training(config)
    assert len(hooks["setup"]) >= 3
    assert recording.snapshot_steps == set(config["metrics_snapshot"]["steps"])


@pytest.mark.parametrize("change, error, message", [
    ({"loop": "unknown"}, SystemExit, "loop: gpt_record"),
    ({"capture": {}}, ValueError, "use metrics_snapshot"),
    ({"setup": [{"name": "unknown"}]}, ValueError, "unsupported selected lifecycle hooks"),
    ({"metrics_snapshot": {"quantities": ["unknown"]}}, ValueError, "snapshot quantities"),
    ({"optimizer_groups": [{"pattern": ".*", "optimizer": "unknown"}]}, KeyError, "unknown"),
    ({"optimizer_groups": [{"pattern": ".*", "optimizer": "adamw", "hyperparams": {
        "lr": -1, "weight_decay": 0,
    }}]}, ValueError, "Invalid learning rate"),
])
def test_training_preparation_rejects_invalid_settings_before_cuda(change, error, message, monkeypatch):
    training, measurement, world_size = prepare_experiment("original_implementation")
    config = training.to_training_dict()
    config.update(change)
    monkeypatch.setattr(training, "to_training_dict", lambda: config)

    def reject_cuda(*args, **kwargs):
        raise AssertionError("CUDA setup started before settings were checked")

    monkeypatch.setattr(run.torch.cuda, "set_device", reject_cuda)
    with pytest.raises(error, match=message):
        run.run_training(training, measurement.capture_dir, world_size)


@pytest.mark.parametrize("configured, environment, expected", [
    (False, "0", []),
    (True, "0", ["apply", "verify"]),
    (False, "1", ["apply", "verify"]),
])
def test_training_preparation_applies_determinism_before_cuda(configured, environment, expected, monkeypatch):
    config = prepare_experiment("original_implementation")[0].to_training_dict()
    config["determinism"] = configured
    calls = []
    monkeypatch.setenv("CAPTURE_DETERMINISM", environment)
    monkeypatch.setattr(lifecycle.determinism_policy, "apply", lambda: calls.append("apply"))

    def verify(*, require_launch_env):
        assert require_launch_env is True
        calls.append("verify")

    def reject_cuda(*args, **kwargs):
        raise AssertionError("training preparation initialized CUDA")

    monkeypatch.setattr(lifecycle.determinism_policy, "verify", verify)
    monkeypatch.setattr(lifecycle.torch.cuda, "set_device", reject_cuda)
    hooks, recording = prepare_training(config)
    assert calls == expected
    assert hooks["setup"]
    assert recording.snapshot_steps == set(config["metrics_snapshot"]["steps"])
