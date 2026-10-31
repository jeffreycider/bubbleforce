"""Load saved training state and measure its restoring-response spectrum.

Run with torchrun and a measurement YAML path. Use
records.track_3_optimization.run to select a training recipe or measure several
saved updates.
"""

import argparse
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from records.track_3_optimization.bubbleforce.analysis.eigensolver import solve_eigenpairs
from records.track_3_optimization.bubbleforce.analysis.loss_hessian import LossHessian
from records.track_3_optimization.bubbleforce.analysis.momentum_stability import compute_bubbleforce
from records.track_3_optimization.bubbleforce.analysis.muon_step import compute_muon_response
from records.track_3_optimization.configs.measurement_config import LossHessianConfig, MeasurementConfig
from records.track_3_optimization.bubbleforce.load_checkpoint import (
    MeasurementCheckpoint, load_checkpoint_for_analysis, load_model_from_checkpoint,
    load_initial_eigenvector, load_validation_token_batch,
)
from records.track_3_optimization.harness.schedules import learning_rate_factor
from records.track_3_optimization.optimizers.ringed.momentum.config import momentum_scheduler
from records.track_3_optimization.optimizers.ringed.momentum.kernel import HISTORY_LENGTH
from records.track_3_optimization.bubbleforce.logging import (
    create_measurement_output_directory, create_measurement_report, record_eigensolver_readout,
    save_eigenvector_shards, write_measurement_report,
)


@contextmanager
def initialize_analysis_process_group(config: MeasurementConfig):
    """Set numerical options, initialize workers, release communication on exit."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if any(count % (world_size * config.sequence_length) for count in config.token_counts):
        raise ValueError("token_counts must provide whole validation sequences on every rank")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_num_threads(config.cpu_threads)
    torch.set_float32_matmul_precision(config.matmul_precision)
    initialize = world_size > 1 and not dist.is_initialized()
    if initialize:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        yield rank, world_size, device
    finally:
        if initialize:
            dist.destroy_process_group()


def predict_bubbleforce(training_config: dict, updates=None):
    """Predict the edge of stability at each update from the training schedules.

    Evaluate the learning-rate and momentum schedules at the absolute update
    number and truncate the kernel to the available gradient history, as in
    training. Yield one row per update and Muon parameter group; unsupported
    rows carry predicted_zeta=None and a reason.
    """
    train_steps = training_config["train_steps"]
    if updates is None:
        updates = range(min(train_steps, training_config.get("stop_step", train_steps)))
    optimizer_groups = [
        (group, momentum_scheduler(group["hyperparams"]["momentum"]))
        for group in training_config["optimizer_groups"]
        if group["optimizer"] in {"muon", "muonh"}
    ]
    for update in updates:
        if type(update) is not int or not 0 <= update < train_steps:
            raise ValueError("prediction update must be within the training schedule")
        for group, schedule in optimizer_groups:
            learning_rate = group["hyperparams"]["lr"] * learning_rate_factor(
                group["lr_schedule"], update, train_steps,
            )
            weight_decay = group["hyperparams"].get("weight_decay", 0.)
            history_length = min(update + 1, HISTORY_LENGTH)
            kernel = schedule(update)
            momentum_kernel = kernel.weights[:history_length].cpu()
            if kernel.normalize_available:
                mass = momentum_kernel.sum()
                if mass <= 0:
                    raise ValueError("the available kernel mass is not positive")
                momentum_kernel = momentum_kernel / mass
            decay_step = learning_rate * weight_decay
            current_gradient_coefficient = float(momentum_kernel[0])
            row = dict(
                update=update, parameter_group=group["pattern"],
                learning_rate=learning_rate, weight_decay=weight_decay, decay_step=decay_step,
                current_gradient_coefficient=current_gradient_coefficient, predicted_zeta=None,
            )
            if group["optimizer"] == "muonh":
                row["reason"] = "MuonH has no scalar kernel stability law here"
            elif learning_rate <= 0 or current_gradient_coefficient <= 0:
                row["reason"] = "positive learning rate and current-gradient coefficient required"
            else:
                try:
                    boundary = compute_bubbleforce(momentum_kernel.numpy(), decay_step=decay_step)
                except ValueError as stability_error:
                    row["reason"] = str(stability_error)
                else:
                    bubble_pressure = decay_step + current_gradient_coefficient * boundary
                    row.update(
                        curvature_coupling_boundary=boundary, bubble_pressure=bubble_pressure,
                        predicted_zeta=bubble_pressure / learning_rate,
                    )
            yield row


def compute_bubbleforce_eigenpairs(
    config: MeasurementConfig,
    checkpoint: MeasurementCheckpoint,
    model: torch.nn.Module,
    validation_batch: tuple[torch.Tensor, torch.Tensor],
    token_count: int,
    initial_vector: torch.Tensor | None = None,
    on_solver_readout: Callable[[dict], None] | None = None,
) -> tuple[dict, torch.Tensor]:
    """Apply validation Hessian products through Muon and estimate eigenpairs.

    Hold older gradients and Adam parameters fixed while differentiating the
    normalized Muon displacement. Return solver readouts and this rank's
    eigenvector shards, releasing compiled Hessian programs after the solve.
    """
    vector_config = checkpoint.vector_config
    hessian = LossHessian(
        model, vector_config.parameter_names, validation_batch, config.microbatch_sequences,
        global_tokens=token_count,
        config=LossHessianConfig(mode=config.hvp_mode, gemm_precision=config.gemm_precision),
    )
    try:
        local_weights = {
            name: checkpoint.snapshot["model_before"][name].to(vector_config.device)
            for name in checkpoint.momenta
        }
        apply_restoring_response = partial(
            compute_muon_response, vector_config, hessian,
            local_weights, checkpoint.momenta, checkpoint.step_configs,
        )
        readouts, eigenvectors = solve_eigenpairs(
            apply_restoring_response, vector_config, config.eigensolver,
            initial_basis_vector=initial_vector, save_readouts=on_solver_readout,
        )
    finally:
        hessian.clear_program_cache()
    measurement = dict(tokens=token_count, readouts=readouts)
    return measurement, eigenvectors


def run_analysis(
    config: MeasurementConfig,
    *,
    update: int,
    output: Path,
    warm_start: Path | None = None,
) -> list[dict]:
    """Restore a saved update, measure validation prefixes, save reports and vectors."""
    with initialize_analysis_process_group(config) as (rank, world_size, device):
        checkpoint = load_checkpoint_for_analysis(config.capture_dir, update, rank, world_size, device)
        model = load_model_from_checkpoint(checkpoint)
        initial_vector = load_initial_eigenvector(warm_start, checkpoint.vector_config)
        create_measurement_output_directory(output, rank, world_size)
        measurement_report = dict(stage="loaded", update=update, measurements=[])
        if rank == 0:
            recipe_path = config.capture_dir / "capture_config.json"
            stability = list(predict_bubbleforce(
                json.loads(recipe_path.read_text()), [checkpoint.snapshot["update"]],
            )) if recipe_path.is_file() else [
                dict(predicted_zeta=None, reason="capture has no capture_config.json recipe"),
            ]
            measurement_report = create_measurement_report(
                config, checkpoint, world_size, stability,
                update=update, output=output, warm_start=warm_start,
            )
            write_measurement_report(output, measurement_report)

        for token_count in config.token_counts:
            validation_batch = load_validation_token_batch(
                config.validation, token_count, rank=rank, world_size=world_size,
                device=device, sequence_length=config.sequence_length,
            )
            on_solver_readout = partial(
                record_eigensolver_readout, output=output, measurement_report=measurement_report,
                rank=rank, token_count=token_count,
            )
            measurement, eigenvectors = compute_bubbleforce_eigenpairs(
                config, checkpoint, model, validation_batch, token_count, initial_vector, on_solver_readout,
            )
            measurement_report["measurements"].append(measurement)
            save_eigenvector_shards(output, token_count, rank, eigenvectors)
            initial_vector = (
                eigenvectors[0].clone() if config.reuse_previous_vector and len(eigenvectors) else None
            )
            if rank == 0:
                write_measurement_report(output, measurement_report)

        if world_size > 1:
            dist.barrier()
        if rank == 0:
            measurement_report.update(stage="complete")
            measurement_report.pop("current_readout", None)
            write_measurement_report(output, measurement_report)
        return measurement_report["measurements"]


def run_measurements(config: MeasurementConfig) -> None:
    """Measure each selected update using the supplied measurement config."""
    with initialize_analysis_process_group(config):
        warm_start = config.warm_start
        for update in config.updates:
            output = config.output_dir / "spectra" / f"update_{update:06d}"
            run_analysis(config, update=update, output=output, warm_start=warm_start)
            if config.reuse_previous_vector:
                warm_start = output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML file containing measurement settings")
    args = parser.parse_args()
    run_measurements(MeasurementConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
