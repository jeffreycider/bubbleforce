# bubbleforce

Experiments on momentum kernels, directional curvature and stability during
language-model training. This repository contains our training harness,
checkpoint and gradient capture, and reproducible analyses. It builds on
[Keller Jordan's modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

1. [Preliminaries](#preliminaries)
2. [Quick start](#quick-start)
3. [Complete documentation](#complete-documentation)
4. [Code guide](#code-guide)

## Preliminaries

Run commands from the repository root. Use a CUDA-enabled PyTorch installation;
the supplied recipes use eight GPUs. Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the FineWeb token shards once:

```bash
python data/cached_fineweb10B.py 20
```

## Quick start

For all our experiments, we train on FineWeb10B with GPT-2 tokenization and next-token cross-entropy loss. All six experiments are reproducible with yaml configs in `configs/`, which also include our specified model dimensions, batch size, and training duration.

To compute optimizer path springiness, $\zeta_t$ and bubbleforce, the runner saves weights and momentum at specified updates during training (e.g. `0, 250, 500, ..., 3000, 3249`). After training finishes, it loads each saved update and estimates the largest positive real eigenvalue of the Muon Jacobian, using 16,777,216 validation tokens per measurement. This calculation measures how Muon's normalized update responds to a small weight perturbation, holding older gradients in the momentum buffer, and AdamW parameters fixed. Then, we plot this value as optimizer path springiness, $\zeta_t$, where $t$ is the training step.

Let's walk through how we can reproduce our experiment results with various learning rate and momentum configs.

### Original schedule

The script `run.py` trains a GPT language model from either a saved restart checkpoint or from scratch.

```bash
python run.py original_implementation
```

The trainer config (link /config/trainconfig.py) assigns learning rate, scheduling and weight decay hyperparameters. For our experiments we use Muon on the Attention and MLP weight matrices, and AdamW on the token embedding, output projection weights, biases and normalization gains.

Training logs are saved in a folder that can be specified in the CLI args:

```text
runs/spectrum/<experiment name>/
```

Our original Muon schedule uses exponential moving average (EMA) momentum with decay `0.95` and Nesterov mix `0.95`. The trainer holds all learning rates constant for the first 30% of training, then reduces them linearly toward zero over the remaining 70%.

You can view each experiments settings in their respective yaml's `[original_implementation.yaml](configs/original_implementation.yaml)`, or print the settings with:

```bash
python run.py original_implementation --print-settings
```

After the measurements finish, plot optimizer path springiness against training
update with the example below. Run it from the repository root, changing `run_dir`
to plot another experiment. Install the plotting packages with
`python -m pip install pandas matplotlib`.

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

run_dir = Path("runs/spectrum/original_implementation")

# Combine the measurement tables, keeping the largest eigenvalue at each update.
tables = []
for csv_path in run_dir.glob("spectra/update_*/eigenvalues.csv"):
    tables.append(pd.read_csv(csv_path))
measured = pd.concat(tables).sort_values("update")
measured = measured[measured["mode"] == 1]

predicted = pd.read_csv(run_dir / "predicted_equilibrium.csv")

plt.figure(figsize=(12, 6))
plt.plot(predicted["update"], predicted["predicted_zeta"],
         linestyle="--", color="black", label="Predicted equilibrium")
plt.plot(measured["update"], measured["eigenvalue"], marker="D", label="Measured")
plt.yscale("log")
plt.ylim(measured["eigenvalue"].min() * 0.8, measured["eigenvalue"].max() * 1.2)
plt.xlabel("Training update")
plt.ylabel(r"Optimizer path springiness $\zeta_t$")
plt.legend()
plt.tight_layout()
plt.savefig(run_dir / "figure.png", dpi=180)
```

### MaxEntSlop

Here we switch from exponential moving average (EMA) momentum to the MaxEnt kernel at update `1000`, then interpolate its settings through to update `3250`. We retain the original learning-rate cooldown. You can read the full settings in `[maxentslop.yaml](configs/maxentslop.yaml)`.

```bash
python run.py maxentslop
```

### Double learning rate

Here we use a constant learning rates until update `2009`, then multiply every optimizer group's learning rate by `2`. You can read the full settings in `lr_double.yaml](configs/lr_double.yaml)`.

```bash
python run.py lr_double
```

### Half learning rate

Here we use constant learning rates until update `2009`, then multiply every optimizer group's learning rate by `0.5`. You can read the full settings in `lr_half.yaml](configs/lr_half.yaml)`.

```bash
python run.py lr_half
```

### Increased momentum

At update `2009`, we change Muon decay and Nesterov mix from `0.95` to `0.9750961447125188` which was chosen to double the maximum tolerable bubbleforce. We keep learning rates constant through training. You can read the full settings in `increased_momentum.yaml](configs/increased_momentum.yaml)`.

```bash
python run.py increased_momentum
```

### Decreased momentum

At update `2009`, we change Muon decay and Nesterov mix from `0.95` to `0.8990666066208468` which was chosen to halve the maximum tolerable bubbleforce. We also keep learning rates constant through training. Again, you can read the full settings in `decreased_momentum.yaml](configs/decreased_momentum.yaml)`.

```bash
python run.py decreased_momentum
```

## Complete documentation

### Experiment arguments

Run `python run.py EXPERIMENT key=value ...`. The experiment name is required; the selected YAML supplies defaults for the optional arguments below. Quote list overrides, for example `'measurement.updates=[2008,2009,2010]'`.

#### Required argument

- `experiment` [`str`]: Choose `original_implementation`, `maxentslop`, `lr_double`, `lr_half`, `increased_momentum`, or `decreased_momentum`.

#### Optional arguments

- `--print-settings` [`bool` flag, default: `false`]: Print the resolved settings and exit. Add the flag without a value.
- `world_size` [`int`, default: `8`]: Set the number of GPU workers. Match the saved process count when loading snapshots or restarting training.
- `measurement.output_dir` [`str` (path), default: `runs/spectrum/<experiment>`]: Save results in a new directory. By default, training logs go in its `training/` subdirectory and snapshots go in `captures/`.
- `training.train_steps` [`int`, default: `3250`]: Set the training schedule length. The default runs updates `0` through `3249`; changing it also changes the learning-rate cooldown timing.
- `training.seed` [`int`, default: `0`]: Change the model initialization seed. The trainer adds the configured seed offset before initializing parameters.
- `training.microbatch_sequences` [`int`, default: `64`]: Process this many 1024-token sequences per GPU at a time. Reduce it to save memory; it must divide the local sequence count.
- `measurement.measure_every` [`int`, default: `250`]: Measure every this many training updates, including update `0` and the final update.
- `measurement.updates` [`list[int]` or `null`, default: `null`]: Measure specific updates instead of using the interval above. Unless `training.stop_step` is set, training stops after the last selected update.
- `measurement.token_counts` [`list[int]`, default: `[16777216]`]: Calculate eigenvalues using each requested number of validation tokens across all GPUs. Supply positive, distinct, increasing counts divisible by `world_size * measurement.sequence_length`.
- `measurement.microbatch_sequences` [`int`, default: `16`]: Process this many validation sequences per GPU at a time during measurement. Reduce it to save memory.
- `measurement.eigensolver.num_requested_eigenpairs` [`int`, default: `1`]: Estimate this many leading positive real eigenvalues and their eigenvectors.

#### Saved snapshots and restarts

- `training` [`dict` or `null`, default: selected YAML's training settings]: Set `training=null` to measure saved snapshots. Supply `measurement.capture_dir` and explicit `measurement.updates`.
- `measurement.capture_dir` [`str` (path), default: `${measurement.output_dir}/captures`]: Save or load snapshots here. Keep `capture_config.json` with the saved tensors.
- `training.checkpoint.every` [`int`, default: `0`]: Save full restart checkpoints every this many completed updates, plus the final boundary when enabled. Zero disables periodic checkpoints.
- `training.resume_checkpoint` [`str` (path) or `null`, default: `null`]: Resume a full checkpoint using the same training recipe and GPU count. Select measurement updates at or after the checkpoint boundary.

For example, inspect settings for measurements at updates `2008`, `2009`, and `2010`:

```bash
python run.py increased_momentum 'measurement.updates=[2008,2009,2010]' measurement.output_dir=runs/spectrum/increased_momentum_transition --print-settings
```

Remove `--print-settings` to run the experiment. Edit the selected [experiment YAML](configs/) for learning rates, momentum schedules, model dimensions, and advanced solver settings. Read the [spectrum experiment instructions](docs/spectrum-experiments.md) for additional examples.

## Code guide

Use this guide to find the experiment and analysis code.

#### Mathematical code

| To inspect                                               | Open                                                                                     | Start with                                                                               |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Arnoldi basis expansion and orthogonalization            | [bubbleforce/analysis/krylov.py](bubbleforce/analysis/krylov.py)                         | `extend_krylov_basis`; `solve_hessenberg_eigenpairs` solves the smaller projected matrix |
| Eigenpair selection, convergence checks, residual errors | [bubbleforce/analysis/eigensolver.py](bubbleforce/analysis/eigensolver.py)               | `solve_eigenpairs`, `compute_relative_eigenpair_residuals`, `reconstruct_eigenvectors`   |
| Muon response to weight perturbations                    | [bubbleforce/analysis/muon_step.py](bubbleforce/analysis/muon_step.py)                   | `compute_muon_response`, `compute_negative_update_jvp`                                   |
| Newton-Schulz directional derivative                     | [bubbleforce/analysis/muon_jvp.py](bubbleforce/analysis/muon_jvp.py)                     | `linearize_newton_schulz`                                                                |
| Bubbleforce                                              | [bubbleforce/analysis/momentum_stability.py](bubbleforce/analysis/momentum_stability.py) | `compute_bubbleforce`                                                                    |
| Momentum coefficients across updates                     | [optimizers/ringed/momentum/schedulers.py](optimizers/ringed/momentum/schedulers.py)     | `EMA`, `KernelInterpolation`, `ScheduleSwitch`                                           |
| Muon and MuonH weight updates                            | [optimizers/ringed/muon.py](optimizers/ringed/muon.py)                                   | `compute_weight_after_muon_step`, `compute_weight_after_muonh_step`                      |

#### Run code and settings

| To change or follow                                           | Open                                                             | Start with                                                                                  |
| ------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Experiment selection, overrides, worker launch                | [run.py](run.py)                                                 | `prepare_experiment`, `run_experiment`, `run_training`                                      |
| Learning rates, momentum interventions, measurement frequency | [configs/](configs/)                                             | The selected experiment YAML                                                                |
| Training settings and parameter groups                        | [train_config.py](configs/train_config.py)                       | `TrainConfig`, `resolve_optimizer_groups`                                                   |
| Measurement settings and solver tolerances                    | [configs/measurement_config.py](configs/measurement_config.py)   | `MeasurementConfig`, `EigenSolveConfig`                                                     |
| Snapshot measurement and scheduled predictions                | [bubbleforce/run_analysis.py](bubbleforce/run_analysis.py)       | `run_analysis`, `compute_bubbleforce_eigenpairs`, `predict_equilibrium`                     |
| Saved weights, momentum, validation tokens                    | [bubbleforce/load_checkpoint.py](bubbleforce/load_checkpoint.py) | `load_checkpoint_for_analysis`, `load_model_from_checkpoint`, `load_validation_token_batch` |

#### Logging and plots

| To inspect or change                       | Open                                             | Start with                                               |
| ------------------------------------------ | ------------------------------------------------ | -------------------------------------------------------- |
| JSON reports and eigenvalue CSV files      | [bubbleforce/logging.py](bubbleforce/logging.py) | `record_eigensolver_readout`, `write_measurement_report` |
| Eigenvector files and prediction CSV files | [bubbleforce/logging.py](bubbleforce/logging.py) | `save_eigenvector_shards`, `write_predicted_equilibrium` |

#### PyTorch infrastructure

| To inspect or change                               | Open                                                                                     | Start with                                                                                |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Hessian-vector products, tracing, compilation      | [bubbleforce/analysis/loss_hessian.py](bubbleforce/analysis/loss_hessian.py)             | `LossHessian` applies loss curvature to a direction without constructing the full Hessian |
| Hessian products across GPUs                       | [bubbleforce/analysis/sharded_hessian.py](bubbleforce/analysis/sharded_hessian.py)       | `compute_distributed_hvp`                                                                 |
| Vector packing, distributed dot products and norms | [bubbleforce/analysis/sharded_vector.py](bubbleforce/analysis/sharded_vector.py)         | `pack_tensor`, `unpack_vector`, `compute_global_dot`, `compute_global_norm`               |
| Loss derivatives with the output head held fixed   | [bubbleforce/analysis/fixed_lm_head_loss.py](bubbleforce/analysis/fixed_lm_head_loss.py) | `fixed_lm_head_cross_entropy`                                                             |
| BF16 matrix multiplication in traced graphs        | [bubbleforce/analysis/bf16_gemm.py](bubbleforce/analysis/bf16_gemm.py)                   | `rewrite_bf16_gemms`                                                                      |
| Device settings and worker communication           | [bubbleforce/run_analysis.py](bubbleforce/run_analysis.py)                               | `initialize_analysis_process_group`                                                       |

For numerical checks, open [tests/](tests/), starting
with `test_eigensolver_correctness.py` for solver checks or
`test_checkpoint_spectrum.py` for snapshot measurements.
