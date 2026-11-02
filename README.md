# bubbleforce

This repository contains source code for the blog post by Jeffrey Cheng and Jack Zheng.

1. [Preliminaries](#preliminaries)
2. [Start training with an experiment config](#start-training-with-an-experiment-config)
3. [Restart training from a checkpoint](#restart-training-from-a-checkpoint)
4. [Measure path spring at a snapshot or a given frequency](#measure-zeta-at-a-snapshot-or-a-given-frequency)
5. [Experiment arguments](AGENTS.md#experiment-arguments)
6. [Code guide](AGENTS.md#code-guide)
7. [Codebase architecture](#codebase-architecture)

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

Pass the YAML file path to start training from scratch:

```bash
python3 run.py configs/nanogpt_muon_baseline.yaml
```

Our [nanoGPT Muon baseline config](configs/nanogpt_muon_baseline.yaml) follows [Konstantin's PR #323](https://github.com/KellerJordan/modded-nanogpt/pull/323) and uses Muon with exponential moving average (EMA) momentum, decay `0.95`, and Nesterov mix `0.95`. The trainer holds all learning rates constant for the first 30% of training, then reduces them linearly toward zero over the remaining 70%.

You can inspect the settings with:

```bash
python3 run.py configs/nanogpt_muon_baseline.yaml --print-settings
```

Choose another experiment with its corresponding command:

| Experiment           | Settings                                                                                                | Command                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| MaxEntSlop           | Single EMA momentum → MaxEntSlop kernel at update `1000`, interpolated through `3250`. LR cooldown kept | `python3 run.py configs/maxentslop.yaml`         |
| Double learning rate | Constant LR then all LRs × `2` at update `2009`                                                         | `python3 run.py configs/lr_double.yaml`          |
| Half learning rate   | Constant LR then all LRs × `0.5` at update `2009`                                                       | `python3 run.py configs/lr_half.yaml`            |
| Increased momentum   | Constant LR then Muon decay and Nesterov mix change from `0.95` → `0.9750961447125188` at update `2009` | `python3 run.py configs/increased_momentum.yaml` |
| Decreased momentum   | Constant LR then Muon decay and Nesterov mix change from `0.95` → `0.8990666066208468` at update `2009` | `python3 run.py configs/decreased_momentum.yaml` |

### Log experiment results

The runner captures the model weights, momentum buffer and optimizer settings at selected updates, then measures the bubbleforce at those snapshots after training finishes.

By default, snapshots are saved at:

```text
runs/spectrum/<experiment>/captures/
```

You can choose a different output directory by adding `measurement.output_dir=PATH` to the training command. Each run requires a directory that does not already exist. For example:

```bash
python3 run.py configs/nanogpt_muon_baseline.yaml \
  measurement.output_dir=runs/spectrum/original_trial_1
```

## Restart training from a checkpoint

You may resume training from a later step by [downloading a restart checkpoint](https://huggingface.co/buckets/jacknzheng/bubbleforce/tree/restarts), and setting `training.resume_checkpoint` to its local checkpoint directory.

This example uses eight GPUs to resume the supplied decreased-momentum run through update `2210`, then measures that update. Match the checkpoint's training settings and choose a new output directory.

```bash
python3 run.py configs/decreased_momentum.yaml \
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

If you'd like to measure $\zeta_t$ at a fixed interval, set `measurement.measure_every` to the desired interval. The default measurement frequency is every 50 updates.

```bash
python3 run.py configs/increased_momentum.yaml \
  measurement.measure_every=50 \
  measurement.output_dir=runs/spectrum/increased_momentum_every50
```

![Archived increased-momentum path-spring measurements](figures/blog_six/higher_zeta_full.png)

### Measure a saved snapshot

We have also [provided snapshots](https://huggingface.co/buckets/jacknzheng/bubbleforce/tree/snapshots) we recorded during our runs, which can be used to compute the $\zeta_t$ value at the respective step. But you can also use the command below to rerun calculations of snapshots, if for example the eigenvalue solver's residual error is too high.

As an example, you can download the supplied decreased-momentum snapshot at update `2025`:

```bash
hf buckets sync \
  hf://buckets/jacknzheng/bubbleforce/snapshots/decreased_momentum/decreased_momentum__update_002025 \
  ./decreased_momentum__update_002025
```

Set `training=null`, as to not kick off a new run, then simply pass in the downloaded directory and select its update:

```bash
python3 run.py configs/decreased_momentum.yaml \
  training=null \
  measurement.capture_dir=./decreased_momentum__update_002025 \
  'measurement.measured_updates=[2025]' \
  measurement.output_dir=runs/spectrum/decreased_momentum_snapshot2025
```

This loads the snapshot for measurement. For snapshots from your own run, point `measurement.capture_dir` to that run's `captures/` directory.

## Codebase architecture

![Software architecture diagram](figures/architecture.svg)
