# Bubbleforce experiments

Source paths below are relative to `records/track_3_optimization/`. Run commands
from the repository root.

Choose an experiment from the six YAML files in `configs/`. Each file contains the training, measurement, and shared execution defaults.
The runner merges that YAML with command-line overrides, then passes
`TrainConfig` and `MeasurementConfig` directly to the worker functions.

## Setup

1. Install `requirements.txt` in a Python environment with CUDA-enabled PyTorch.
2. Place the FineWeb shards in `data/fineweb10B/`. To use other paths, add
   `training.train_data` and `training.val_data` to the selected YAML, then update
   its `measurement.validation` field.
3. Run commands from the repository root. The supplied training settings use eight
   GPUs. The experiment runner starts the worker processes automatically.

## Train and measure

```bash
python -m records.track_3_optimization.run original_implementation
```

| YAML file | Training settings |
|---|---|
| `original_implementation.yaml` | Muon decay and Nesterov mix of `0.95`; original learning-rate cooldown |
| `increased_momentum.yaml` | Constant learning rates; change Muon decay and Nesterov mix to `0.9750961447125188` at update `2009` |
| `decreased_momentum.yaml` | Constant learning rates; change both values to `0.8990666066208468` at update `2009` |
| `lr_half.yaml` | Constant learning rates before update `2009`; multiply every group's rate by `0.5` from that update |
| `lr_double.yaml` | Constant learning rates before update `2009`; multiply every group's rate by `2` from that update |
| `maxentslop.yaml` | Original cooldown; switch from EMA to the specified MaxEnt kernel interpolation at update `1000` |

Use the filename without `.yaml` as the command's argument. The momentum
interventions retain the earlier gradient history. The original implementation
uses cooldown, while the intervention experiments use their constant-rate schedules.

To print the complete settings before execution:

```bash
python -m records.track_3_optimization.run increased_momentum --print-settings
```

Pass changes as `key=value` arguments after the experiment name. Quote list
arguments, for example `'measurement.updates=[2008,2009,2010]'`, so the shell
passes them intact. OmegaConf merges the recipe with these changes, then constructs
separate `TrainConfig` and `MeasurementConfig` objects. Their `__post_init__`
methods check the values before execution.
Use `training.model.num_layers=8` or `measurement.eigensolver.start_seed=42`
to change nested settings. Resolve relative paths from the repository root.

When printing settings, validate field types and relationships among settings.
Before execution, check dataset files, saved snapshots, and checkpoint compatibility.

## Assign optimizers to model parameters

Set `block_type` and `optimizer` for each group under `training.optimizer_groups`:

```yaml
training:
  optimizer_groups:
  - block_type: embed
    optimizer: adamw
    hyperparams:
      lr: 0.7
      weight_decay: 0.001
    lr_schedule:
      kind: cooldown
      fraction: 0.7
```

Include all four groups in each recipe:

| `block_type` | Selected parameters | Supplied optimizer |
|---|---|---|
| `embed` | Token embedding weights | AdamW |
| `head` | Output projection weights | AdamW |
| `block_matrices` | Attention and MLP weight matrices in every transformer block | Muon |
| `biases_and_norms` | Biases and normalization gains, including the output bias | AdamW |

`resolve_optimizer_groups()` translates these names for the existing trainer and
checkpoint format. Edit the `training` section of the selected YAML to change
model dimensions, data paths, seeds, recording settings, or the setup sequence.
The printed settings include the merged values.

## Scheduled stability predictions

Each experiment run writes `predicted_stability.csv` with a prediction for every
training update. For each row, evaluate the learning-rate schedule and momentum
kernel at that update. Truncate the kernel to the available gradient history,
then compute the boundary. Use `--print-settings` with the same command-line overrides to inspect
the training settings used for this curve.

For `training=null`, read the schedules from the capture's `capture_config.json`.
If that file is unavailable, compute predictions only at the saved measurement
snapshots. Each snapshot supplies its recorded learning rate and momentum kernel.

The prediction holds each update's settings fixed while solving for its stability
boundary. It does not model evolution through future schedule changes.

## Choose measurement frequency

Set `measurement.measure_every` in the experiment YAML to select the expensive eigenpair measurements.
The prediction curve still includes every update. The default value, `250`, measures
updates `0, 250, 500, ...` and the final update. Training runs for
`training.train_steps` updates. To save full restart checkpoints, explicitly set
`training.checkpoint.every`, for example `500`. With this setting, save a checkpoint
every 500 updates and at the final boundary. The default recording settings save
weights and momentum at the selected measurement updates.

To select particular measurements and stop training after the last selected update:

```bash
python -m records.track_3_optimization.run lr_half \
  'measurement.updates=[2008,2009,2010]' measurement.output_dir=runs/spectrum/lr_half_transition
```

The runner keeps `training.train_steps` fixed when stopping early, so the
learning-rate schedule keeps its configured timing.

## Measure saved snapshots

```bash
python -m records.track_3_optimization.run increased_momentum \
  training=null measurement.capture_dir=/path/to/captures 'measurement.updates=[2009,2010]' \
  measurement.output_dir=runs/spectrum/increased_momentum_remeasurement
```

The runner loads the saved weights and momentum. It recovers actual learning rates
and optimizer settings from the capture. The experiment name selects spectrum
settings; it does not modify the saved training state.

Use `measurement.validation=/path/to/fineweb_val_000000.bin` to select the measurement token
shard. Use `world_size=N` to match the capture's process count.

The schema number identifies the saved file format:

| Format | Files needed for measurement |
|---|---|
| Schema 2 | `rankN/boundary_XXXXXX.pt`, including saved momentum coefficients |
| Schema 3 | `rankN/metrics_snapshot_XXXXXX.pt`, plus `capture_config.json` for reconstructing momentum coefficients |

Keep `capture_config.json` with schema-3 snapshots when copying them. Record
`weights` and `momentum` under `training.metrics_snapshot.quantities` for Muon
response measurements. The supplied recipes request both.

## Resume a training checkpoint

```bash
python -m records.track_3_optimization.run increased_momentum \
  training.resume_checkpoint=/path/to/captures/checkpoint_002000 \
  'measurement.updates=[2008,2009,2010]' measurement.output_dir=runs/spectrum/increased_momentum_restart
```

1. Supply a schema-3 restart bundle from the same training recipe. Keep its
   capture directory and referenced gradient files available. The existing loader
   checks file hashes, recipe compatibility, and process count.
2. Choose updates at or after the restart boundary. Checkpoint `2000` resumes
   before update `2000`; snapshot `2000` includes that update's gradient in momentum.
3. The runner restores weights, optimizer history, and random-number state, then
   records and measures the selected updates. Use `'measurement.updates=[2000]'` to measure only
   the restart boundary after computing its training gradient.

## Configure the spectrum calculation

Edit `measurement.eigensolver` in the selected YAML. Each file explicitly specifies
the eigenpair count, Krylov basis sizes, comparison window, tolerances, symmetry,
and starting seed. Keep the validation path in `measurement.validation`.
The YAML also specifies 16,777,216 validation tokens and a microbatch of
16 sequences. Set `world_size=N` to override the YAML process count for training
or measurement. Keep the capture path in `measurement.capture_dir`.
The runner measures restoring response: it differentiates the normalized Muon
update with older gradients and Adam parameters held fixed. Set
`measurement.eigensolver.symmetric=true` to average the projected matrix with
its transpose before the symmetric eigensolve. With `false`, solve the projected
matrix directly using the general eigensolver.
The calculation uses validation loss Hessian products inside the Muon derivative.

By default, each solve starts from a random direction with seed `921`. For a
single requested eigenpair, set `reuse_previous_vector: true` to reuse the preceding
result. Use `measurement.warm_start=/path/to/measurement` to supply the first direction.
Compare separate random starts when checking an uncertain leading eigenvalue.

The runner executes the current repository code. Historical measurements used
earlier solver implementations and several restarted training runs. Compare
checkpoints and numerical results before combining a new series with those results.

## Read outputs

The runner saves results under `measurement.output_dir`. Use that setting to choose
another directory. Each run requires a new directory.

With the supplied YAML settings, training logs and captures are saved beneath
the measurement output directory:

```text
<measurement.output_dir>/
|-- training/
|   |-- train.log
|   `-- wandb/                         When offline W&B logging is enabled
|-- predicted_stability.csv
|-- captures/
|   |-- capture_config.json
|   |-- rankN/
|   |   |-- manifest.jsonl
|   |   `-- metrics_snapshot_XXXXXX.pt
|   `-- checkpoint_XXXXXX/             When restart checkpoints are enabled
`-- spectra/
    `-- update_XXXXXX/
        |-- measurement_results.json
        |-- eigenvalues.csv
        |-- eigenvectors_N*_rank*.pt
        `-- current_vector_rank*.pt
```

Replace `rankN` with the worker rank, such as `rank0`. Replace `XXXXXX` with
the six-digit update or checkpoint boundary. Override `training.output_dir`
or `measurement.capture_dir` to save those outputs elsewhere.

| Path | Contents |
|---|---|
| `training/train.log` | Training completion, validation losses, Git revision, PyTorch and GPU environment |
| `training/wandb/` | Offline W&B run data when `training.wandb_offline` is enabled |
| `predicted_stability.csv` | Per-update rates, weight decay, momentum coefficient, and predicted stability boundary |
| `captures/capture_config.json` | Saved training settings used to reconstruct optimizer schedules and check compatibility |
| `captures/rankN/manifest.jsonl` | File paths, sizes, and hashes for that worker's recorded tensors |
| `captures/rankN/metrics_snapshot_XXXXXX.pt` | Selected weights and momentum for measuring a saved update |
| `captures/checkpoint_XXXXXX/` | Completion metadata referencing the per-rank restart state, when checkpoint recording is enabled |
| `spectra/update_XXXXXX/measurement_results.json` | Input hashes, source revision, optimizer settings, solver readouts, stability prediction |
| `spectra/update_XXXXXX/eigenvalues.csv` | Eigenvalues, residuals, stopping reasons |
| `spectra/update_XXXXXX/eigenvectors_N*_rank*.pt` | Final eigenvector shards |
| `spectra/update_XXXXXX/current_vector_rank*.pt` | First eigenvector shard for initializing a later measurement, when an eigenvector was found |

Read `measurement_results.json` during a solve to inspect its latest stage and
readouts. The writer refreshes `eigenvalues.csv` as measurements finish. Training
progress also appears in the terminal. Offline W&B files remain local until uploaded.
For a restart, retain the checkpoint directory together with its referenced
per-rank checkpoint and gradient files under `captures/rankN/`.

The training log records progress and the runtime environment. Measurement reports
record results and the optimizer state used for each update. The launcher passes
Python config objects to workers. Keep `capture_config.json` with snapshots so
the loader can reconstruct optimizer schedules and check checkpoint compatibility.

To measure saved snapshots, set `training=null`, provide `measurement.capture_dir`
and explicit `measurement.updates`, choose a new `measurement.output_dir`.
Each worker measures the updates in order and saves results under
`measurement.output_dir/spectra/update_XXXXXX/`. When
`measurement.reuse_previous_vector` is enabled, it loads the preceding result's
eigenvector before measuring the next update.

## Follow the code

1. Use `run.py` to merge the selected YAML and command-line overrides with
   OmegaConf. Resolve paths and measurement updates, validate the separate configs,
   then pass each config to its worker through PyTorch's Python launcher.
2. `run_training(training, capture_dir, world_size)` prepares the training
   settings and initializes distributed training. For a restart, the setup hook
   calls `harness/checkpoint.py` to restore weights, optimizer state, gradient
   history, and random-number state. With `training=null`, skip this call.
3. `run_measurements(measurement)` in `bubbleforce/run_analysis.py` loads
   snapshots, initializes worker communication, computes predictions and
   restoring-response eigenpairs.
4. `bubbleforce/analysis/` contains the Hessian products, Muon derivatives,
   Krylov basis calculations, and momentum stability calculations used by the runner.
5. `bubbleforce/logging.py` writes input hashes, solver readouts, and eigenvector
   files. Read `measurement_results.json` for the current solve status and results.

Edit the selected YAML to change defaults. Use `run.py` to change experiment
choices or checks involving training and measurement together. Use
`configs/train_config.py` for training checks, `configs/measurement_config.py` for measurement
and eigensolver checks.

Use `bubbleforce/load_checkpoint.py` to read model weights and per-rank
momentum, recover optimizer settings and kernel coefficients, rebuild the model,
load a starting eigenvector, and divide validation tokens among workers.

For training restarts, use `training.resume_checkpoint` with a full checkpoint bundle.
For measurements, use `training=null` with metrics snapshots. A metrics snapshot
contains the weights and combined momentum used in the measurement; resuming
training also requires gradient history and the other saved training state.
