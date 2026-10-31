"""Record measurement inputs, solver readouts, and eigenvector files."""

import csv
from dataclasses import asdict
from enum import Enum
import json
from pathlib import Path
import subprocess

import torch
import torch.distributed as dist

from harness.recording.store import file_sha


def serialize_measurement_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def create_measurement_output_directory(output: Path, rank: int, world_size: int) -> None:
    """Create a new directory on rank zero and share any failure with workers."""
    errors = [None]
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=False)
        except OSError as failure:
            errors[0] = f"choose a new output directory: {failure}"
    if world_size > 1:
        dist.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise FileExistsError(errors[0])


def create_measurement_report(
    config, checkpoint, world_size: int, stability: list[dict], *,
    update: int, output: Path, warm_start: Path | None,
) -> dict:
    """Hash measurement inputs and record the saved optimizer state before solving."""
    input_paths = list(checkpoint.paths) + [config.validation]
    capture_config = config.capture_dir / "capture_config.json"
    if capture_config.exists():
        input_paths.append(capture_config)
    if warm_start is not None:
        input_paths.extend(warm_start / f"current_vector_rank{rank}.pt" for rank in range(world_size))
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True,
    )
    changes = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository, capture_output=True, text=True,
    )
    return dict(
        format=1, stage="loaded", update=update, output=output, warm_start=warm_start,
        operator="restoring-response",
        operator_definition="-D[(W_next-W)/eta], older gradients and Adam frozen",
        world_size=world_size,
        source_revision=revision.stdout.strip() if revision.returncode == 0 else None,
        source_dirty=bool(changes.stdout) if changes.returncode == 0 else None,
        inputs=[dict(path=str(path.resolve()), sha256=file_sha(path)) for path in input_paths],
        optimizer_step_configs={name: asdict(value) for name, value in checkpoint.step_configs.items()},
        matrix_layouts=checkpoint.vector_config.parameter_names_and_shapes_by_rank,
        stability=stability, measurements=[],
    )


def write_measurement_report(output: Path, measurement_report: dict) -> None:
    """Save the report and write a CSV row for each completed eigenpair estimate."""
    temporary = output / "measurement_results.json.tmp"
    temporary.write_text(json.dumps(
        measurement_report, indent=2, default=serialize_measurement_value, allow_nan=False,
    ) + "\n")
    temporary.replace(output / "measurement_results.json")
    with (output / "eigenvalues.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "update", "tokens", "mode", "eigenvalue", "residual", "stopping_reason",
        ))
        writer.writeheader()
        for measurement in measurement_report["measurements"]:
            final_readout = measurement["readouts"][-1]
            for mode, (eigenvalue, residual) in enumerate(zip(
                final_readout["eigenvalues"], final_readout["residuals"], strict=True,
            ), 1):
                writer.writerow(dict(
                    update=measurement_report["update"], tokens=measurement["tokens"], mode=mode,
                    eigenvalue=eigenvalue, residual=residual,
                    stopping_reason=final_readout["stopping_reason"].value,
                ))


def record_eigensolver_readout(
    readout: dict, *, output: Path, measurement_report: dict, rank: int, token_count: int,
) -> None:
    """Write the current solver estimate on rank zero during basis expansion."""
    if rank == 0:
        measurement_report.update(stage="eigensolver", current_readout=dict(tokens=token_count, **readout))
        write_measurement_report(output, measurement_report)


def save_eigenvector_shards(
    output: Path, token_count: int, rank: int, eigenvectors: torch.Tensor,
) -> None:
    """Save this rank's eigenvectors and its first direction for a later solve."""
    torch.save(eigenvectors.cpu(), output / f"eigenvectors_N{token_count}_rank{rank}.pt")
    if len(eigenvectors):
        torch.save(eigenvectors[0].cpu(), output / f"current_vector_rank{rank}.pt")


def write_scheduled_bubbleforce(output: Path, predictions) -> None:
    """Write one prediction per training update and Muon parameter group."""
    temporary = output / "predicted_stability.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "update", "parameter_group", "learning_rate", "weight_decay",
            "current_gradient_coefficient", "decay_step", "curvature_coupling_boundary",
            "bubble_pressure", "predicted_zeta", "reason",
        ))
        writer.writeheader()
        writer.writerows(predictions)
    temporary.replace(output / "predicted_stability.csv")
