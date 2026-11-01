# bubbleforce

This repository contains source code for the blog post by Jeffrey Cheng and Jack Zheng.

1. [Preliminaries](#preliminaries)
2. [Quick start](#quick-start)
3. [Complete documentation](#complete-documentation)
4. [Code guide](#code-guide)

## Preliminaries

To run commands you must use a CUDA-enabled PyTorch installation and one of the supplied experiment configurations. First, install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Then, download the FineWeb token shards once:

```bash
python data/cached_fineweb10B.py 20
```

## Quick start

For all our experiments, we train on FineWeb10B with GPT-2 tokenization and next-token cross-entropy loss.

All six experiments in the blog are reproducible with the yaml configs in `configs/`, which also include our specified model dimensions, batch size, and training duration.

Let's briefly walk through how we can reproduce our experiment results with various learning rate and momentum configs.

### Original schedule

The script `run.py` trains a GPT language model from either a saved restart checkpoint or from scratch.

```bash
python run.py original_implementation
```

Our original Muon schedule uses exponential moving average (EMA) momentum with decay `0.95` and Nesterov mix `0.95`. The trainer holds all learning rates constant for the first 30% of training, then reduces them linearly toward zero over the remaining 70%.

You can view each experiments settings in their respective yaml's, `[original_implementation.yaml](configs/original_implementation.yaml)`, or print the settings with:

```bash
python run.py original_implementation --print-settings
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

### Restarting from a checkpoint

If you don't want to restart training from scratch, we have provided restart checkpoints that can be loaded from [checkpoint bucket](https://huggingface.co/buckets/jacknzheng/bubbleforce) using:

```bash
hf buckets sync hf://buckets/jacknzheng/bubbleforce/restarts/decreased_momentum/decreased_momentum__checkpoint_002201 ./decreased_momentum__checkpoint_002201
```

Just set `training.resume_checkpoint` to where it is downloaded in your local directory. Warning it is ~95GB.

As an example, resume the `increased_momentum` experiment from checkpoint 2000 and measure at updates 2008, 2009, and 2010:

```bash
python run.py increased_momentum \
  training.resume_checkpoint=/path/to/captures/checkpoint_002000 \
  'measurement.updates=[2008,2009,2010]' \
  measurement.output_dir=runs/spectrum/increased_momentum_restart
```

### Measurement methodology

To compute optimizer path springiness, $\zeta_t$ and bubbleforce, the runner saves weights and momentum at specified updates during training (e.g. `0, 250, 500, ..., 3000, 3249`). After training finishes, it loads each saved update and estimates the largest positive real eigenvalue of the Muon Jacobian, using 16,777,216 validation tokens per measurement.

The training logs are then saved in a folder that can be specified in the CLI args:

```text
runs/spectrum/<experiment name>/
```

### Experiment arguments

To run `python run.py EXPERIMENT key=value ...`, the experiment name is required. The selected experiment YAML supplies defaults for the optional arguments below, which can be overridden by passing in the specified CLI arguments.

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

Remove `--print-settings` to run the experiment.

Edit the selected [experiment YAML](configs/) for learning rates, momentum schedules, model dimensions, and advanced solver settings.
