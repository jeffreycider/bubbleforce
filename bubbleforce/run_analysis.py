"""Measure historical Muon response eigenpairs from saved training snapshots.

Read a measurement YAML file from the command line, load each selected update,
then save eigenvalue reports and eigenvector shards. Launch with torchrun for
distributed measurements, or use run.py to train before measuring snapshots.
"""

import argparse
from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path

import torch
import torch.distributed as dist

from bubbleforce.analysis.eigensolver import EigenpairReadout, solve_eigenpairs
from bubbleforce.analysis.loss_hessian import LossHessian
from bubbleforce.analysis.compute_bubbleforce import compute_bubbleforce
from bubbleforce.analysis.muon_jvp import compute_optimizer_jvp
from configs.measurement_config import LossHessianConfig, MeasurementConfig
from bubbleforce.load_checkpoint import (
    MeasurementCheckpoint, load_checkpoint_for_analysis,
    load_initial_eigenvector, load_validation_token_batch,
)
from harness.schedules import learning_rate_factor
from optimizers.ringed.momentum.config import momentum_scheduler
from optimizers.ringed.momentum.kernel import HISTORY_LENGTH
from bubbleforce.logging import (
    prepare_bubbleforce_output_directory,
    save_eigenvector_shards, save_bubbleforce_eigenpair_readouts,
)


@contextmanager
def initialize_analysis_process_group(config: MeasurementConfig):
    """Configure numerical precision and prepare distributed measurement workers.

    Read worker ranks from the launch environment. Use CUDA when available,
    otherwise use CPU. Reuse an existing process group, or initialize a group
    for multiple workers. Destroy the group on exit if initialized here.

    Args:
        config: Measurement settings for validation token counts, sequence
            length, CPU thread count, and matrix multiplication precision.

    Yields:
        A tuple containing the global rank, worker count, and torch device.

    Raises:
        ValueError: A token count cannot be divided into complete validation
            sequences across all workers.
    """
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
    """Apply the single Muon group's schedules and yield update/prediction pairs.

    Use the available gradient history and actual decay to calculate kappa_star.
    Yield (kappa_star / learning_rate + weight_decay) / training batch tokens.
    Yield None for a nonpositive learning rate or current-gradient weight.
    Raise ValueError if the group count, update, or kernel is invalid.
    """
    muon_groups = [
        group for group in training_config["optimizer_groups"]
        if group["optimizer"] == "muon"
    ]
    if len(muon_groups) != 1:
        raise ValueError("prediction requires exactly one Muon group")
    group = muon_groups[0]
    hyperparams = group["hyperparams"]
    schedule = momentum_scheduler(hyperparams["momentum"])
    weight_decay = hyperparams.get("weight_decay", 0.)
    train_steps = training_config["train_steps"]
    if updates is None:
        stop_step = training_config.get("stop_step")
        updates = range(train_steps if stop_step is None else min(train_steps, stop_step))

    for update in updates:
        if type(update) is not int or not 0 <= update < train_steps:
            raise ValueError("prediction update must be within the training schedule")
        learning_rate = hyperparams["lr"] * learning_rate_factor(
            group["lr_schedule"], update, train_steps,
        )
        kernel = schedule(update)
        history_length = min(update + 1, HISTORY_LENGTH)
        momentum_kernel = kernel.weights[:history_length].cpu()
        if kernel.normalize_available:
            mass = momentum_kernel.sum()
            if mass <= 0:
                raise ValueError("the available kernel mass is not positive")
            momentum_kernel = momentum_kernel / mass

        current_gradient_coefficient = float(momentum_kernel[0])
        predicted_zeta = None
        if learning_rate > 0 and current_gradient_coefficient > 0:
            decay_step = learning_rate * weight_decay
            predicted_bubbleforce = compute_bubbleforce(
                momentum_kernel.numpy(), decay_step=decay_step,
            )
            training_tokens = training_config["batch_tokens"]
            predicted_zeta = (
                predicted_bubbleforce / learning_rate + weight_decay
            ) / training_tokens
        yield dict(update=update, predicted_zeta=predicted_zeta)


def compute_bubbleforce_eigenpairs(
    config: MeasurementConfig,
    checkpoint: MeasurementCheckpoint,
    validation_batch: tuple[torch.Tensor, torch.Tensor],
    token_count: int,
    initial_vector: torch.Tensor | None = None,
) -> tuple[list[EigenpairReadout], torch.Tensor]:
    """Estimate eigenpairs of the historical Muon response.

    Apply J_P(M)[H v] + (weight_decay / training batch tokens) v, using
    validation-mean Hessian products and saved summed-gradient momentum.
    Hold older gradients and Adam parameters fixed.
    Clear cached Hessian programs after solving, including on failure.

    Args:
        config: Measurement settings for microbatch size, Hessian evaluation,
            numerical precision, and the eigensolver.
        checkpoint: Loaded model with momentum tensors, optimizer settings,
            and parameter assignments for each worker.
        validation_batch: This worker's input and target token tensors, each
            with dimensions (local_sequences, sequence_length).
        token_count: Total validation token count across all workers, used to
            normalize the loss Hessian.
        initial_vector: Optional local starting vector with length
            checkpoint.vector_config.local_vector_size. If omitted, the
            solver draws a starting vector using its configured seed.

    Returns:
        Return the solver readouts and this worker's eigenvector shards.
        Return the eigenvectors as rows, with dimensions
        (returned_eigenpairs, local_vector_size).
    """
    vector_config = checkpoint.vector_config
    hessian = LossHessian(
        checkpoint.model, vector_config.parameter_names, validation_batch, config.microbatch_sequences,
        global_tokens=token_count,
        config=LossHessianConfig(mode=config.hvp_mode, gemm_precision=config.gemm_precision),
    )
    try:
        apply_optimizer_jacobian = partial(
            compute_optimizer_jvp, checkpoint, hessian,
        )
        readouts, eigenvectors = solve_eigenpairs(
            apply_optimizer_jacobian, vector_config, config.eigensolver,
            initial_basis_vector=initial_vector,
        )
    finally:
        hessian.clear_program_cache()
    return readouts, eigenvectors


def run_analysis(
    config: MeasurementConfig,
    *,
    update: int,
    output: Path,
    warm_start: Path | None = None,
) -> list[dict]:
    """Load a saved update and measure each configured validation prefix.

    Load the model and optimizer state, then estimate eigenpairs for each
    token count. Save reports on rank zero and eigenvector shards on every
    worker. If vector reuse is enabled, start each later solve with the first
    eigenvector from the preceding solve, when available.

    Args:
        config: Measurement settings for the capture directory, validation
            data, token counts, and eigensolver.
        update: Zero-based training update to load from the capture directory.
        output: Directory to create for this update's reports and vectors.
        warm_start: Optional directory containing current_vector_rank{rank}.pt
            files. Use this worker's saved vector for the first token count.

    Returns:
        Measurement dictionaries in configured token-count order. Include
        the token count and eigensolver readouts in each dictionary.

    Raises:
        FileExistsError: The output directory already exists or cannot be
            created.
        FileNotFoundError: A required snapshot, validation shard, or supplied
            starting-vector file is missing.
        ValueError: Saved state, starting-vector dimensions, or validation
            data fail the measurement checks.
    """
    with initialize_analysis_process_group(config) as (rank, world_size, device):

        checkpoint = load_checkpoint_for_analysis(config.capture_dir, update, rank, world_size, device)
        initial_vector = load_initial_eigenvector(warm_start, checkpoint.vector_config)
        prepare_bubbleforce_output_directory(output, rank, world_size)
        per_update_bubbleforce_measurements = dict(update=update, measurements=[])

        for token_count in config.token_counts:
            validation_batch = load_validation_token_batch(
                config.validation, token_count, rank=rank, world_size=world_size,
                device=device, sequence_length=config.sequence_length,
            )
            readouts, eigenvectors = compute_bubbleforce_eigenpairs(
                config, checkpoint, validation_batch, token_count, initial_vector,
            )
            per_update_bubbleforce_measurements["measurements"].append(
                dict(tokens=token_count, readouts=readouts),
            )
            save_eigenvector_shards(output, token_count, rank, eigenvectors)
            initial_vector = (
                eigenvectors[0].clone() if config.reuse_previous_vector and len(eigenvectors) else None
            )
            if rank == 0:
                save_bubbleforce_eigenpair_readouts(output, per_update_bubbleforce_measurements)

        if world_size > 1:
            dist.barrier()
        return per_update_bubbleforce_measurements["measurements"]


def main(config: MeasurementConfig | None = None) -> None:
    """Load settings if needed, then measure and save each selected update."""
    if config is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("config", type=Path, help="YAML file containing measurement settings")
        args = parser.parse_args()
        config = MeasurementConfig.from_yaml(args.config)

    with initialize_analysis_process_group(config):
        warm_start = config.warm_start
        for update in config.measured_updates:
            output = config.output_dir / "bubbleforce" / str(update)
            run_analysis(config, update=update, output=output, warm_start=warm_start)
            if config.reuse_previous_vector:
                warm_start = output


if __name__ == "__main__":
    main()
