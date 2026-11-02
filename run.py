"""Train a named experiment and measure snapshots, or measure an existing capture."""

from __future__ import annotations

import argparse
import datetime
from dataclasses import asdict
from enum import Enum
import glob
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from bubbleforce.logging import write_predicted_equilibrium
from bubbleforce.run_analysis import predict_equilibrium, run_measurements
from harness import loop
from harness.checkpoint import training_config_fingerprint
from harness.hooks import prepare_training
from harness.recording.store import get_formatted_path_to_capture_file
from configs.measurement_config import MeasurementConfig
from configs.train_config import TrainConfig


class Experiment(str, Enum):
    original_implementation = "original_implementation"
    increased_momentum = "increased_momentum"
    decreased_momentum = "decreased_momentum"
    lr_half = "lr_half"
    lr_double = "lr_double"
    maxentslop = "maxentslop"


def prepare_experiment(
    experiment_name: Experiment | str, config_overrides: list[str] | None = None,
) -> tuple[TrainConfig | None, MeasurementConfig, int]:
    """Merge an experiment YAML with CLI arg overrides and build the training and measurement configs.

    Args:
        experiment_name: Experiment whose YAML file in configs/ is loaded.
        config_overrides: CLI key=value settings that replace YAML values.

    Returns:
        The training config (None for measurement-only runs), the measurement
        config, and the number of worker processes.
    """
    experiment_config_path = Path(__file__).parent / "configs" / f"{Experiment(experiment_name).value}.yaml"
    experiment_config = OmegaConf.to_container(OmegaConf.merge(
        OmegaConf.load(experiment_config_path),
        OmegaConf.from_dotlist(config_overrides or []),
    ), resolve=True) # merge the configs
    training_config = (
        TrainConfig(**experiment_config["training"])
        if experiment_config["training"] is not None else None
    )
    measurement_config = MeasurementConfig.from_dict(
        experiment_config["measurement"],
        end_step=(training_config.stop_step or training_config.train_steps) if training_config is not None else None,
    )
    check_experiment_config(experiment_config, training_config, measurement_config)
    if training_config is not None:
        if experiment_config["measurement"]["measured_updates"] is not None and training_config.stop_step is None:
            training_config.stop_step = measurement_config.measured_updates[-1] + 1
        training_config.metrics_snapshot["steps"] = list(measurement_config.measured_updates)
    return training_config, measurement_config, experiment_config["world_size"]


def check_experiment_config(
    experiment_config: dict, training_config: TrainConfig | None, measurement_config: MeasurementConfig,
) -> None:
    """Check the training and measurement configs to ensure that they don't contradict each other.

    Args:
        experiment_config: Merged experiment settings as a plain dict.
        training_config: Built training config, or None for measurement-only runs.
        measurement_config: Built measurement config.

    Raises:
        ValueError: If a top-level setting is unknown or invalid, or if the
            training and measurement settings disagree.
    """
    if unknown_experiment_settings := set(experiment_config) - {"world_size", "training", "measurement"}:
        raise ValueError(f"unknown experiment settings: {sorted(unknown_experiment_settings)}")
    world_size = experiment_config["world_size"]
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if training_config is not None and measurement_config.measured_updates[-1] >= (training_config.stop_step or training_config.train_steps):
        raise ValueError("measurement updates must precede stop_step and train_steps")
    if any(token_count % (world_size * measurement_config.sequence_length) for token_count in measurement_config.token_counts):
        raise ValueError("token_counts must provide whole validation sequences on every rank")


def check_experiment_checkpoint_config(
    training_config: TrainConfig, measurement_config: MeasurementConfig, world_size: int,
) -> None:
    """Check that the restart checkpoint config fits this experiment's config before any worker starts.

    Args:
        training_config: Training config whose resume_checkpoint is set.
        measurement_config: Measurement config whose updates must follow the restart.
        world_size: Number of worker processes for this run.

    Raises:
        ValueError: If the checkpoint has the wrong format or process count, starts
            after the first requested update, or was made with a different training recipe.
    """
    training_settings = training_config.to_training_dict()
    training_settings.update(
        capture_dir=str(measurement_config.capture_dir), require_world_size=world_size,
    )
    checkpoint_metadata = json.loads(
        (Path(training_config.resume_checkpoint) / "COMPLETE.json").read_text(),
    )
    if checkpoint_metadata.get("format") != 3 or checkpoint_metadata["world_size"] != world_size:
        raise ValueError("resume requires a schema-3 bundle with the selected process count")
    if measurement_config.measured_updates[0] < checkpoint_metadata["boundary"]:
        raise ValueError("requested updates precede the restart; select later measurement.measured_updates")
    if checkpoint_metadata["protocol_sha256"] != training_config_fingerprint(training_settings):
        raise ValueError("restart recipe differs; use a checkpoint created with this experiment's training recipe")


def run_experiment(
    training_config: TrainConfig | None, measurement_config: MeasurementConfig, world_size: int,
) -> None:
    """Check input files and pass config objects directly to distributed workers."""
    if measurement_config.output_dir.exists():
        raise FileExistsError(f"output_dir already exists; choose a new one: {measurement_config.output_dir}")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    repository_root = Path(__file__).resolve().parent

    # check configs are aligned
    if training_config is not None and training_config.resume_checkpoint:
        check_experiment_checkpoint_config(training_config, measurement_config, world_size)
    if not measurement_config.validation.is_file():
        raise FileNotFoundError(f"validation shard missing: {measurement_config.validation}")
    if training_config is not None:
        if not glob.glob(str(repository_root / training_config.train_data)):
            raise FileNotFoundError(f"training shards missing: {training_config.train_data}")
    else:
        for measurement_update in measurement_config.measured_updates:
            for worker_rank in range(world_size):
                snapshot_path = get_formatted_path_to_capture_file(
                    measurement_config.capture_dir, worker_rank, "metrics_snapshot", measurement_update,
                )
                if not snapshot_path.is_file():
                    raise FileNotFoundError(f"measurement snapshot missing: {snapshot_path}")
    if os.environ.get("CAPTURE_ROOT"):
        raise ValueError("unset CAPTURE_ROOT so training uses measurement.capture_dir")
    measurement_config.output_dir.mkdir(parents=True, exist_ok=False)
    distributed_launch_config = LaunchConfig(
        min_nodes=1, max_nodes=1, nproc_per_node=world_size,
        run_id=str(uuid4()), rdzv_backend="c10d", rdzv_endpoint="localhost:0",
        local_addr="127.0.0.1",
        max_restarts=0,
    )
    if training_config is not None:
        elastic_launch(distributed_launch_config, run_training)(
            training_config, measurement_config.capture_dir, world_size,
        )
    write_prediction_curve(
        training_config, capture_dir=measurement_config.capture_dir,
        output_dir=measurement_config.output_dir,
    )
    elastic_launch(distributed_launch_config, run_measurements)(measurement_config)


def write_prediction_curve(
    training_config: TrainConfig | None, *, capture_dir: Path, output_dir: Path,
) -> None:
    """Use current or saved training settings to predict equilibrium at each update."""
    if training_config is None:
        saved_training_config_path = capture_dir / "capture_config.json"
        if not saved_training_config_path.is_file():
            print("No capture_config.json: predict at saved snapshots only; a full schedule curve requires the saved recipe.")
            return
        training_settings = json.loads(saved_training_config_path.read_text())
    else:
        training_settings = training_config.to_training_dict()
    print("Computing equilibrium predictions for every training update.", flush=True)
    write_predicted_equilibrium(output_dir, predict_equilibrium(training_settings))


def dist_timeout():
    """Return the distributed timeout from DIST_TIMEOUT_S, defaulting to 1800 seconds."""
    return datetime.timedelta(seconds=int(os.environ.get("DIST_TIMEOUT_S", "1800")))


def run_training(training_config: TrainConfig, capture_dir: Path, world_size: int) -> None:
    """Prepare the supplied training config, then execute a distributed worker."""
    os.chdir(Path(__file__).resolve().parent)
    training_settings = training_config.to_training_dict()
    training_settings.update(capture_dir=str(capture_dir), require_world_size=world_size)
    if training_settings.get("resume_checkpoint"):
        checkpoint_metadata_path = Path(training_settings["resume_checkpoint"]) / "COMPLETE.json"
        training_settings["start_step"] = json.loads(checkpoint_metadata_path.read_text())["boundary"]
    training_hooks, training_recording = prepare_training(training_settings)

    worker_device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(worker_device)
    dist.init_process_group(backend="nccl", device_id=worker_device, timeout=dist_timeout())
    dist.barrier()
    loop.run(training_settings, training_hooks, training_recording)
    dist.destroy_process_group()


def main() -> None:
    def key_value_override(override: str) -> str:
        if "=" not in override:
            raise argparse.ArgumentTypeError("experiment overrides must use key=value")
        return override

    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "experiment", choices=[experiment_option.value for experiment_option in Experiment],
    )
    argument_parser.add_argument("overrides", nargs="*", metavar="key=value", type=key_value_override)
    argument_parser.add_argument(
        "--print-settings", action="store_true", help="print merged settings without running workers",
    )
    cli_arguments = argument_parser.parse_args()
    training_config, measurement_config, world_size = prepare_experiment(
        cli_arguments.experiment, cli_arguments.overrides,
    )
    if cli_arguments.print_settings:
        print(OmegaConf.to_yaml(dict(
            world_size=world_size, training=asdict(training_config) if training_config is not None else None,
            measurement=measurement_config.to_dict(),
        ), sort_keys=False))
    else:
        run_experiment(training_config, measurement_config, world_size)


if __name__ == "__main__":
    main()
