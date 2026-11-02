"""Save completed solver readouts and eigenvector files."""

import csv
from enum import Enum
import json
from pathlib import Path

import torch
import torch.distributed as dist


def serialize_eigenpair_readout_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def prepare_bubbleforce_output_directory(output: Path, rank: int, world_size: int) -> None:
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


def save_bubbleforce_eigenpair_readouts(output: Path, bubbleforce_eigenpair_readouts_at_update: dict) -> None:
    """Save an update's solver readouts as JSON and its final eigenpairs as CSV."""
    temporary = output / "measurement_results.json.tmp"
    temporary.write_text(json.dumps(
        bubbleforce_eigenpair_readouts_at_update, indent=2, default=serialize_eigenpair_readout_value, allow_nan=False,
    ) + "\n")
    temporary.replace(output / "measurement_results.json")
    with (output / "eigenvalues.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "update", "tokens", "mode", "eigenvalue", "residual", "stopping_reason",
        ))
        writer.writeheader()
        for token_count_readouts in bubbleforce_eigenpair_readouts_at_update["measurements"]:
            final_readout = token_count_readouts["readouts"][-1]
            for mode, (eigenvalue, residual) in enumerate(zip(
                final_readout["eigenvalues"], final_readout["residuals"], strict=True,
            ), 1):
                writer.writerow(dict(
                    update=bubbleforce_eigenpair_readouts_at_update["update"],
                    tokens=token_count_readouts["tokens"], mode=mode,
                    eigenvalue=eigenvalue, residual=residual,
                    stopping_reason=final_readout["stopping_reason"].value,
                ))


def save_eigenvector_shards(
    output: Path, token_count: int, rank: int, eigenvectors: torch.Tensor,
) -> None:
    """Save this rank's eigenvectors and its first direction for a later solve."""
    torch.save(eigenvectors.cpu(), output / f"eigenvectors_N{token_count}_rank{rank}.pt")
    if len(eigenvectors):
        torch.save(eigenvectors[0].cpu(), output / f"current_vector_rank{rank}.pt")
