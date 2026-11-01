from dataclasses import replace
import json

import pytest

import run
from harness.hooks import bind_sites
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
        f"training.resume_checkpoint={checkpoint}", "measurement.updates=[321]",
    ])
    received = []
    def stop_before_cuda(config):
        received.append(config)
        raise RuntimeError("checked before CUDA")
    monkeypatch.setattr(run, "bind_sites", stop_before_cuda)
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
    config = dict(resume_checkpoint="checkpoint_000321", setup=[dict(name=name) for name in names])
    with pytest.raises(ValueError, match="before opening training data"):
        bind_sites(config)


@pytest.mark.parametrize("name", Experiment)
def test_resume_order_in_experiment_recipes(name):
    config = prepare_experiment(name)[0].to_training_dict()
    config["resume_checkpoint"] = "checkpoint_000321"
    hooks = bind_sites(config)
    assert len(hooks["setup"]) >= 3  # binding never opens checkpoint files or CUDA
