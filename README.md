# bubbleforce

This repository contains source code for the blog post by Jeffrey Cheng and Jack Zheng.

1. [Preliminaries](#preliminaries)
2. [Start training with an experiment config](#start-training-with-an-experiment-config)
3. [Restart training from a checkpoint](#restart-training-from-a-checkpoint)
4. [Measure path spring at a snapshot or a given frequency](#measure-zeta-at-a-snapshot-or-a-given-frequency)
5. [Experiment arguments](AGENTS.md#experiment-arguments)
6. [Code guide](AGENTS.md#code-guide)

## Preliminaries

To run commands you must use a CUDA-enabled PyTorch installation and one of the supplied experiment configurations. First, install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Then, download the FineWeb token shards once:

```bash
python data/cached_fineweb10B.py 20
```

## Quickstart

Let's briefly walk through how you can reproduce our experiments in the blog.

### Start training with an experiment config

For all our experiments, we train on FineWeb10B with GPT-2 tokenization and next-token cross-entropy loss.

You can select an experiment config from `configs/` to set training, measurement, and model options. We have provided the 6 experiment configs we used in the blog. Our supplied configs use eight GPUs.

Start training from scratch with:

```bash
python run.py nanogpt_muon_baseline
```

Our [nanoGPT Muon baseline config](configs/nanogpt_muon_baseline.yaml) follows [Konstantin's PR #323](https://github.com/KellerJordan/modded-nanogpt/pull/323) and uses Muon with exponential moving average (EMA) momentum, decay `0.95`, and Nesterov mix `0.95`. The trainer holds all learning rates constant for the first 30% of training, then reduces them linearly toward zero over the remaining 70%.

You can inspect the settings with:

```bash
python run.py nanogpt_muon_baseline --print-settings
```

Choose another experiment with its corresponding command:

| Experiment           | Description                                                                                                                                                                                                                                                                       | Command                            |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| MaxEntSlop           | Here we switch from exponential moving average (EMA) momentum to the MaxEntSlop kernel at update `1000`, then interpolate its settings through to update `3250`. We retain the original learning-rate cooldown. You can read the full settings in maxentslop.yaml.                | `python run.py maxentslop`         |
| Double learning rate | Here we use a constant learning rates until update `2009`, then multiply every optimizer group's learning rate by `2`. You can read the full settings in lr_double.yaml.                                                                                                          | `python run.py lr_double`          |
| Half learning rate   | Here we use constant learning rates until update `2009`, then multiply every optimizer group's learning rate by `0.5`. You can read the full settings in lr_half.yaml.                                                                                                            | `python run.py lr_half`            |
| Increased momentum   | At update `2009`, we change Muon decay and Nesterov mix from `0.95` to `0.9750961447125188` which was chosen to double the maximum tolerable bubbleforce. We keep learning rates constant through training. You can read the full settings in increased_momentum.yaml.            | `python run.py increased_momentum` |
| Decreased momentum   | At update `2009`, we change Muon decay and Nesterov mix from `0.95` to `0.8990666066208468` which was chosen to halve the maximum tolerable bubbleforce. We also keep learning rates constant through training. Again, you can read the full settings in decreased_momentum.yaml. | `python run.py decreased_momentum` |

### Log experiment results

The runner captures the model weights, momentum buffer and optimizer settings at selected updates, then measures the bubbleforce at those snapshots after training finishes.

By default, snapshots are saved at:

```text
runs/spectrum/<experiment>/captures/
```

You can choose a different output directory by adding `measurement.output_dir=PATH` to the training command. Each run requires a directory that does not already exist. For example:

```bash
python run.py nanogpt_muon_baseline \
  measurement.output_dir=runs/spectrum/original_trial_1
```

## Restart training from a checkpoint

You may resume training from a later step by [downloading a restart checkpoint](AGENTS.md#restart-training-from-a-checkpoint), set `training.resume_checkpoint` to its local checkpoint directory. This example uses eight GPUs to resume the supplied decreased-momentum run through update `2210`, then measures that update. Match the checkpoint's training settings and choose a new output directory.

```bash
python run.py decreased_momentum \
  training.train_steps=6475 \
  training.microbatch_sequences=8 \
  training.resume_checkpoint=./decreased_momentum__checkpoint_002201/checkpoint_002201 \
  'measurement.measured_updates=[2210]' \
  measurement.output_dir=runs/spectrum/decreased_momentum_restart
```

## Measure $\zeta$ at a snapshot or a given frequency

To compute optimizer path spring, $\zeta_t$, the runner saves weights and momentum at specified updates during training. Here, $t$ is the training update.

After training finishes, it loads each saved update and estimates the largest positive real eigenvalue of the normalized Muon restoring response, using 16,777,216 validation tokens per measurement.

### Specify measurement frequency

If you'd like to measure $\zeta_t$ at a fixed interval, set `measurement.measure_every` to the desired interval. The default is every 50 updates.

```bash
python run.py increased_momentum \
  measurement.measure_every=50 \
  measurement.output_dir=runs/spectrum/increased_momentum_every50
```

![Archived increased-momentum path-spring measurements](figures/blog_six/higher_zeta_full.png)

### Measure a saved snapshot

We have also provided snapshots we recorded during our runs. As an example, you can download the supplied decreased-momentum snapshot at update `2025`:

```bash
hf buckets sync \
  hf://buckets/jacknzheng/bubbleforce/snapshots/decreased_momentum/decreased_momentum__update_002025 \
  ./decreased_momentum__update_002025
```

Set `training=null`, as to not kick off a new run, then simply pass in the downloaded directory and select its update:

```bash
python run.py decreased_momentum \
  training=null \
  measurement.capture_dir=./decreased_momentum__update_002025 \
  'measurement.measured_updates=[2025]' \
  measurement.output_dir=runs/spectrum/decreased_momentum_snapshot2025
```

This loads the snapshot for measurement.

You can use it to rerun the eigenvalue solver with a larger Krylov basis if the residual error is too high. For snapshots from your own run, point `measurement.capture_dir` to that run's `captures/` directory.

## Model config

![Bubbleforce model configuration](docs/images/model-architecture.png)
