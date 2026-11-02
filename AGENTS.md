# Agent guide

This repository trains nanoGPT with Muon, saves optimizer state at chosen updates, and measures the optimizer's stability at those updates. The measured quantity is ζ ("zeta", called *path spring* in [README.md](README.md)); the theoretical limit it is compared with is the *predicted ζ*, derived from the *bubbleforce* boundary of the momentum kernel.

Run every command from the repository root. Read [README.md](README.md) for the experiment story; this file covers how to drive the code, how the numbers are calculated, and how to read them.

## Ground rules

- Check `git status --short` before editing and keep existing changes. Use `rg --files` and `rg -n 'PATTERN' PATH` to find code.
- Training and measurement need CUDA GPUs. The supplied configs use 8 workers (`world_size: 8`).
- Every run needs a **new** `measurement.output_dir`. The runner refuses an existing directory rather than overwriting it.
- Use `--print-settings` to check any command without data, GPUs or checkpoints. It prints the merged config and exits.

## Code guide

Start at `prepare_experiment` in [run.py](run.py), which merges the YAML with command-line overrides. Then follow `run_experiment`: check inputs → train (optional) → write predicted ζ for every update → measure each saved update.


| Task                                     | Files to read                                                                                                                                                                                                              | Follow these operations                                                                                                 |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Change experiment settings               | [run.py](run.py), [train_config.py](configs/train_config.py), [measurement_config.py](configs/measurement_config.py)                                                                                                       | Trace YAML loading, override parsing, validation, then worker launch.                                                   |
| Inspect the model or training loop       | [model_gpt.py](harness/model_gpt.py), [hooks.py](harness/hooks.py), [loop.py](harness/loop.py)                                                                                                                             | Follow model initialization, token loading, loss evaluation, then optimizer updates.                                    |
| Change optimizer behavior                | [optimizers.py](harness/optimizers.py), [muon.py](optimizers/ringed/muon.py), [momentum/](optimizers/ringed/momentum/)                                                                                                     | Follow parameter grouping, gradient-history accumulation, momentum scheduling, then matrix updates.                     |
| Save or restore training                 | [recording/](harness/recording/), [checkpoint.py](harness/checkpoint.py)                                                                                                                                                   | Trace snapshot selection, checkpoint writes, then restoration of optimizer and random-number state.                     |
| Measure saved updates                    | [load_checkpoint.py](bubbleforce/load_checkpoint.py), [run_analysis.py](bubbleforce/run_analysis.py)                                                                                                                       | Load weights and momentum, prepare validation tokens, then estimate response eigenpairs.                                |
| Inspect the measured response            | [loss_hessian.py](bubbleforce/analysis/loss_hessian.py), [sharded_hessian.py](bubbleforce/analysis/sharded_hessian.py), [muon_step.py](bubbleforce/analysis/muon_step.py), [muon_jvp.py](bubbleforce/analysis/muon_jvp.py) | Apply the loss Hessian to a direction, then differentiate the saved Muon update along that direction.                   |
| Inspect numerical stability calculations | [eigensolver.py](bubbleforce/analysis/eigensolver.py), [krylov.py](bubbleforce/analysis/krylov.py), [momentum_stability.py](bubbleforce/analysis/momentum_stability.py)                                                    | Follow eigenvalue estimation separately from the calculation of a stability boundary using fixed momentum coefficients. |
| Inspect saved results                    | [logging.py](bubbleforce/logging.py), [tests/](tests/)                                                                                                                                                                     | Trace result writing, then read the corresponding checks before changing behavior.                                      |




## Prepare the environment

```bash
source .venv/bin/activate                                  # or: python -m venv .venv first
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python data/cached_fineweb10B.py 20                        # validation shard + 20 training shards
python run.py nanogpt_muon_baseline --print-settings       # works without data or GPUs
```

[requirements.txt](requirements.txt) pins PyTorch 2.11.0. Keep that version: PyTorch 2.10 corrupts one bias gradient under `torch.compile`, and a restart then reaches a non-finite loss at its second update. If you must use 2.10, set `export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` before starting Python.

## Run custom commands

Every run has this form:

```bash
python run.py EXPERIMENT [key=value ...] [--print-settings]
```

- `EXPERIMENT` picks [configs/EXPERIMENT.yaml](configs/). Each `key=value` replaces one YAML value, using dotted paths (OmegaConf dotlist syntax).
- Quote lists so the shell leaves them alone: `'measurement.measured_updates=[2000,2050]'`.
- `training=null` skips training and only measures snapshots that already exist.
- Unknown top-level keys are rejected. Nested keys go through the config dataclasses, so a misspelled nested key fails when the config is built.


| Experiment              | Change applied during training                                                                                  |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| `nanogpt_muon_baseline` | Train with the baseline Muon settings and learning-rate cooldown.                                               |
| `maxentslop`            | Switch to the MaxEntSlop momentum kernel at update `1000`, then interpolate its settings through update `3250`. |
| `lr_double`             | Multiply all optimizer learning rates by `2` at update `2009`.                                                  |
| `lr_half`               | Multiply all optimizer learning rates by `0.5` at update `2009`.                                                |
| `increased_momentum`    | Set Muon decay and Nesterov mix to `0.9750961447125188` at update `2009`.                                       |
| `decreased_momentum`    | Set Muon decay and Nesterov mix to `0.8990666066208468` at update `2009`.                                       |




### Common recipes

```bash
# Full run: snapshot every 50 updates (plus the final one), then measure them all
python run.py nanogpt_muon_baseline measurement.output_dir=runs/spectrum/baseline_trial_1

# Measure only chosen updates; training stops after update 2050
python run.py lr_half 'measurement.measured_updates=[2000,2050]' \
  measurement.output_dir=runs/spectrum/lr_half_pair

# Short pilot that keeps the full 3250-update LR schedule but stops at update 500
python run.py nanogpt_muon_baseline training.stop_step=500 \
  measurement.output_dir=runs/spectrum/pilot_500

# Fewer GPUs or less memory: change worker count and microbatch sizes
python run.py nanogpt_muon_baseline world_size=4 training.microbatch_sequences=32 \
  measurement.microbatch_sequences=8 measurement.output_dir=runs/spectrum/four_gpu

# Save restart checkpoints every 500 updates (and at the final boundary)
python run.py nanogpt_muon_baseline training.checkpoint.every=500 \
  measurement.output_dir=runs/spectrum/with_checkpoints

# Re-measure a saved snapshot with more validation tokens and a bigger Krylov basis
python run.py decreased_momentum training=null \
  measurement.capture_dir=./decreased_momentum__update_002025 \
  'measurement.measured_updates=[2025]' \
  'measurement.token_counts=[16777216,33554432]' \
  'measurement.eigensolver.krylov_basis_vector_counts=[16,32,64,128,192]' \
  measurement.output_dir=runs/spectrum/dm2025_bigger
```

Environment variables:

- `DIST_TIMEOUT_S` sets the distributed timeout. It defaults to 1800 seconds; raise it if slow checkpoint writes time out.
- `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` is needed only on PyTorch 2.10 (see above).



## Experiment arguments

The experiment YAML supplies every default; override any value as `key=value`. The most useful ones:

- `experiment` [`str`, required]: `nanogpt_muon_baseline`, `maxentslop`, `lr_double`, `lr_half`, `increased_momentum` or `decreased_momentum`.
- `--print-settings` [flag]: Print the resolved settings and exit.
- `world_size` [`int`, default `8`]: Number of GPU workers. Must equal the saved worker count when you load snapshots or restart checkpoints.
- `measurement.output_dir` [path, default `runs/spectrum/<experiment>`]: New directory for all outputs.
- `measurement.capture_dir` [path, default `<output_dir>/captures`]: Where snapshots and checkpoints are written during training, and read during `training=null` measurement.
- `training.train_steps` [`int`, default `3250`]: Length of the schedule. Changing it also moves the cooldown and changes the restart fingerprint.
- `training.stop_step` [`int` or `null`]: Stop early without changing the schedule. Training runs updates `0 … stop_step-1`.
- `training.seed` [`int`, default `0`]: Initialization seed. The trainer adds `seed_offset` (1337) before initializing.
- `training.microbatch_sequences` [`int`, default `64`]: 1024-token sequences per GPU per forward pass. It must divide the local sequence count, and it is part of the restart fingerprint.
- `training.resume_checkpoint` [path or `null`]: The `checkpoint_<NNNNNN>` folder to resume from.
- `training.checkpoint.every` / `.steps` / `.dense_intervals`: When to save restart checkpoints: every N boundaries, at a list of boundaries, or at every boundary in `{start, end}` ranges.
- `measurement.measure_every` [`int`, default `50`]: Snapshot and measure every N updates, plus update `0` and the final update.
- `measurement.measured_updates` [`list[int]` or `null`]: Measure exactly these updates instead. Without `training.stop_step`, training stops right after the last one.
- `measurement.token_counts` [`list[int]`, default `[16777216]`]: Validation tokens used for each measurement. Counts must be increasing and divisible by `world_size × 1024`.
- `measurement.microbatch_sequences` [`int`, default `16`]: Validation sequences per GPU per Hessian product. Lower it to save memory.
- `measurement.eigensolver.num_requested_eigenpairs` [`int`, default `1`]: How many leading positive real eigenvalues to estimate.
- `measurement.eigensolver.krylov_basis_vector_counts` [`list[int]`, default `[8,16,24,32,48,64,96,128]`]: Basis sizes at which the solver checks whether it has converged. The last entry is the budget.
- `measurement.eigensolver.eigenvalue_rtol` (`1e-3`), `.subspace_tolerance` (`1e-2`), `.comparison_window` (`2`): The agreement test (see below).
- `measurement.warm_start` [path or `null`] and `measurement.reuse_previous_vector` [`bool`]: Start the solver from a saved `current_vector_rank<R>.pt` instead of a random vector, or from the previous update's result.



## How snapshots and restart checkpoints are saved

A **snapshot** holds just enough to *measure* one update. A **restart checkpoint** holds everything needed to *continue training*. Both are written per rank under `measurement.capture_dir`:

```text
captures/
  capture_config.json                  resolved training recipe (written once by rank 0)
  rank<R>/metrics_snapshot_<UUUUUU>.pt snapshot of update U
  rank<R>/checkpoint_<NNNNNN>.pt       this rank's restart shard
  rank<R>/gradient_*.pt                gradient history that checkpoints reference
  rank<R>/manifest.jsonl               sha256 + size receipt for every file the rank wrote
  checkpoint_<NNNNNN>/COMPLETE.json    written only after every rank saved its shard
```

**Two clocks. Mixing them up is the most common mistake.**

- Snapshot `t` describes update `t`: it holds the weights *before* update `t` and the momentum *including* gradient `t`, before the Newton–Schulz step.
- Checkpoint `N` is the state *before* update `N`. Updates `0 … N-1` are finished, so resuming from `checkpoint_002201` runs update 2201 first. Any measured update must be `≥ N`.

**Contents.**

- Snapshots hold the momentum on every rank, plus `model_before` and buffers on rank 0, plus the optimizer settings in force at that update (learning rate, weight decay, current-gradient coefficient, batch tokens). This is the `metrics_snapshot.quantities: [weights, momentum]` setting.
- Checkpoints (format 3) hold the model, the optimizer state, references to the up-to-256-step gradient history ring, and the random-number generator (RNG) states.
- `COMPLETE.json` records `format: 3`, the `boundary`, `world_size`, `protocol_sha256`, and each shard's sha256.

**The restart fingerprint (**`protocol_sha256`**).** `training_config_fingerprint` in [checkpoint.py](harness/checkpoint.py) hashes the resolved training recipe. It leaves out only execution settings: `run_id`, `output_dir`, `capture_dir`, `checkpoint`, `metrics_snapshot`, `resume_checkpoint`, `start_step` and `stop_step`. Everything else must match exactly to resume, including `train_steps`, `microbatch_sequences`, every learning rate and momentum value, and the data paths. Write-once rules: files are never overwritten, a new trajectory needs a new `capture_dir`, and an existing `capture_config.json` must hold the same recipe.

### Supplied snapshots and restarts (Hugging Face bucket)

[hf://buckets/jacknzheng/bubbleforce](https://huggingface.co/buckets/jacknzheng/bubbleforce) holds 393 folders (about 830 GB), all converted to format 3. `MANIFEST.csv` has one row per folder, and `conversion/` holds the conversion scripts and recipes. List the folders with `hf buckets ls jacknzheng/bubbleforce/snapshots/<experiment>`. Download one folder with `hf buckets sync hf://buckets/jacknzheng/bubbleforce/<path> ./<dir>`.

```text
snapshots/<experiment>/<experiment>__update_<UUUUUU>[__<trajectory>]/   capture_config.json, rank0..7/
restarts/<experiment>/<experiment>__checkpoint_<NNNNNN>/                 checkpoint_<NNNNNN>/COMPLETE.json, rank0..7/,
                                                                          training_config.json, CONVERSION.json
```

The bucket was made with an older codebase called *muoff*. Its README shows `python -m records.track_3_optimization.run` and `measurement.updates`. **In this repository, use** `python run.py` **and** `measurement.measured_updates`**.** Bucket experiment names map to this repository like this:


| Bucket experiment                          | Use with `run.py`                                               | Resume overrides (restarts)                                 |
| ------------------------------------------ | --------------------------------------------------------------- | ----------------------------------------------------------- |
| `decreased_momentum`, `increased_momentum` | same name                                                       | `training.train_steps=6475 training.microbatch_sequences=8` |
| `lr_double`, `lr_half`                     | same name (snapshots only)                                      | —                                                           |
| `original_implementation`                  | `nanogpt_muon_baseline` (restarts at 1951 and 2801)             | none                                                        |
| `baseline` (updates 1980–2000)             | `nanogpt_muon_baseline` (snapshots only; shared pre-2009 trunk) | —                                                           |
| `maxentslop`                               | `maxentslop` (snapshots only; different recipe, see below)      | —                                                           |
| `shared_trunk` (checkpoint 2001)           | **cannot resume here**: it stores `protocol_sha256: null`       | —                                                           |


All restarts and snapshots were saved with 8 workers. With the overrides above, this repository's fingerprints match the bucket's `COMPLETE.json` hashes: `a92bb65b…` for decreased, `246bb583…` for increased, and `57d2edad…` for `nanogpt_muon_baseline`.

Bucket caveats:

- Restarts carry fresh seed-0 Python and NumPy RNG states, because the old harness did not save them. The torch CPU and CUDA states are the originals. So a resumed run is not a bit-exact replay; compare the first resumed loss with the original run before trusting it.
- `decreased_momentum__checkpoint_002201` resumed and ran updates 2201–2210 under PyTorch 2.11. The other restarts have not run a training step since conversion.
- `maxentslop` snapshots come from the original MaxEntSlop script, not [configs/maxentslop.yaml](configs/maxentslop.yaml). Its kernel switches at update 780 and blends until 3120 on a 3060-update clock, and Muon weight decay follows the LR schedule from a 0.1 peak. Measured ζ uses the settings saved inside each snapshot, so it is correct. Predicted ζ is computed from `capture_config.json`, so check its weight decay before comparing the two.
- When two trajectories saved the same update, the second one gets a suffix, such as `__extend2200` or `__replay2009_recapture`. Use the plain name unless you need that specific trajectory.



### Measure a saved snapshot

```bash
hf buckets sync \
  hf://buckets/jacknzheng/bubbleforce/snapshots/decreased_momentum/decreased_momentum__update_002025 \
  ./decreased_momentum__update_002025
python run.py decreased_momentum training=null \
  measurement.capture_dir=./decreased_momentum__update_002025 \
  'measurement.measured_updates=[2025]' \
  measurement.output_dir=runs/spectrum/decreased_momentum_snapshot2025
```

For snapshots from your own run, point `measurement.capture_dir` at that run's `captures/` directory. Keep the validation shard available.

### Restart training from a checkpoint

```bash
hf buckets sync \
  hf://buckets/jacknzheng/bubbleforce/restarts/decreased_momentum/decreased_momentum__checkpoint_002201 \
  ./decreased_momentum__checkpoint_002201                    # about 95 GB
python run.py decreased_momentum \
  training.train_steps=6475 training.microbatch_sequences=8 \
  training.resume_checkpoint=./decreased_momentum__checkpoint_002201/checkpoint_002201 \
  'measurement.measured_updates=[2210]' \
  measurement.output_dir=runs/spectrum/decreased_momentum_restart
```

This resumes at update 2201, trains through 2210, and measures 2210. Training data must be present: the loader skips past the earlier batches. To resume from one of your own checkpoints, use `<output_dir>/captures/checkpoint_<NNNNNN>` with the identical recipe.

### Troubleshooting snapshots and restarts


| Error message (source)                                                                                                       | Cause and fix                                                                                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_dir already exists; choose a new one` (run.py)                                                                       | Pass a fresh `measurement.output_dir`.                                                                                                                                                                                                            |
| `measurement snapshot missing: …/rank<R>/metrics_snapshot_<U>.pt` (run.py)                                                   | `capture_dir` must be the folder that *contains* `rank0/…rank7/`. Also check that the update number and `world_size` match the files.                                                                                                             |
| `restart recipe differs; use a checkpoint created with this experiment's training recipe` (run.py)                           | Fingerprint mismatch. Apply the resume overrides in the table above, use the right experiment name, and don't change any training hyperparameter. Debug by comparing `python run.py … --print-settings` with the bucket's `training_config.json`. |
| `resume requires a schema-3 bundle with the selected process count` (run.py)                                                 | Set `world_size` to the checkpoint's `world_size` (8). Point `resume_checkpoint` at the nested `checkpoint_<NNNNNN>` folder, not its parent.                                                                                                      |
| `requested updates precede the restart; select later measurement.measured_updates` (run.py)                                  | Checkpoint `N` can only produce updates `≥ N`.                                                                                                                                                                                                    |
| `measurement updates must precede stop_step and train_steps` (run.py)                                                        | Every measured update must be `< stop_step` (or `< train_steps`).                                                                                                                                                                                 |
| `token_counts must provide whole validation sequences on every rank`                                                         | Each token count must be a multiple of `world_size × 1024`.                                                                                                                                                                                       |
| `checkpoint shard hash mismatch` / `checkpoint gradient hash mismatch` (checkpoint.py)                                       | The download is incomplete or corrupted. Run `hf buckets sync` again and compare with the sha256 values in `COMPLETE.json` and `manifest.jsonl`.                                                                                                  |
| `capture_dir has later outputs; use a new capture_dir for this fork` / `new training trajectories require a new capture_dir` | Captures are write-once. Give a resumed or forked run its own `measurement.output_dir`, so it gets its own `captures/`.                                                                                                                           |
| `recording root has a different config/policy/format; use a new capture_dir`                                                 | The `capture_config.json` already in that folder came from a different recipe.                                                                                                                                                                    |
| `No capture_config.json: predict at saved snapshots only…`                                                                   | A warning, not an error: `predicted_equilibrium.csv` is skipped and the eigenvalues are still measured.                                                                                                                                           |
| Non-finite loss at the second resumed update                                                                                 | PyTorch 2.10 compiler bug. Install 2.11.0 or set `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0`.                                                                                                                                                           |
| CUDA out of memory                                                                                                           | Lower `training.microbatch_sequences` for a new run (for a restart, this changes the fingerprint). For measurement, lower `measurement.microbatch_sequences`, which is always safe.                                                               |




## How the repository calculates ζ

Two numbers are calculated independently. **Measured ζ** is the top positive real eigenvalue of the frozen Muon step's response to a weight nudge. **Predicted ζ** is the stability limit of the momentum kernel alone. Both have units of 1/η.

Measured ζ, in call order:

| File | Role |
| ---- | ---- |
| [run_analysis.py](bubbleforce/run_analysis.py) | `run_analysis` / `compute_bubbleforce_eigenpairs` drive the measurement of one update. |
| [load_checkpoint.py](bubbleforce/load_checkpoint.py) | Loads weights, momentum, per-matrix η, weight decay and `c₀`, and the validation tokens. |
| [loss_hessian.py](bubbleforce/analysis/loss_hessian.py), [sharded_hessian.py](bubbleforce/analysis/sharded_hessian.py), [fixed_lm_head_loss.py](bubbleforce/analysis/fixed_lm_head_loss.py) | Loss-Hessian-vector product `H v` on validation tokens, Muon matrices only. |
| [muon_step.py](bubbleforce/analysis/muon_step.py) | `compute_muon_response`: turns `H v` into a momentum change and returns the change in the next Muon update. This is the operator whose eigenvalue is ζ. |
| [muon_jvp.py](bubbleforce/analysis/muon_jvp.py) | Derivative of the Newton–Schulz step, used by `muon_step.py`. |
| [eigensolver.py](bubbleforce/analysis/eigensolver.py), [krylov.py](bubbleforce/analysis/krylov.py) | Arnoldi solver: top eigenvalue (measured ζ), stopping reason (`agreement`, `budget`, `span_exhausted`) and residual. |
| [logging.py](bubbleforce/logging.py) | Writes `eigenvalues.csv` and `measurement_results.json`. |

Predicted ζ (no model needed):

| File | Role |
| ---- | ---- |
| [run_analysis.py](bubbleforce/run_analysis.py) | `predict_equilibrium`: for each update and Muon group, reads η, weight decay and the momentum kernel; `predicted_zeta = (η·λ_wd + c₀·κ*) / η`. |
| [momentum_stability.py](bubbleforce/analysis/momentum_stability.py) | `compute_bubbleforce`: finds `κ*`, the largest curvature coupling the frozen kernel tolerates. |
| [logging.py](bubbleforce/logging.py) | `write_predicted_equilibrium` writes `predicted_equilibrium.csv`. |

## Output files and how to read them

```text
<output_dir>/
  training/train.log                          training and validation loss
  captures/                                   snapshots and checkpoints (see above)
  predicted_equilibrium.csv                   predicted ζ for every update and Muon group
  spectra/update_<UUUUUU>/
    measurement_results.json                  full report (rewritten as the solver progresses)
    eigenvalues.csv                           one row per (tokens, mode)
    eigenvectors_N<tokens>_rank<R>.pt         eigenvector shards
    current_vector_rank<R>.pt                 top eigenvector, usable as warm_start
```

- `eigenvalues.csv`: `update, tokens, mode, eigenvalue, residual, stopping_reason`. Mode 1 is measured ζ.
- `measurement_results.json`:
  - `stage`: `loaded`, then `eigensolver`, then `complete`. Anything other than `complete` means the job died mid-solve.
  - `current_readout`: the latest estimate while the solver is still running.
  - `measurements[].readouts[]`: the eigenvalues at every basis size, which show how the estimate converged.
  - `stability`: predicted ζ at this update.
  - `optimizer_step_configs`: the saved η, weight decay and `c₀` per matrix.
  - `source_revision` and `source_dirty`: the git commit, and whether the working tree had uncommitted changes.
  - `inputs`: sha256 of every file read.
- `predicted_equilibrium.csv`: `update, parameter_group, learning_rate, weight_decay, current_gradient_coefficient, decay_step, curvature_coupling_boundary, bubble_pressure, predicted_zeta, reason`. When `predicted_zeta` is empty, `reason` says why (for example, the learning rate is zero at the end of the cooldown).

Checklist before interpreting a measured ζ:

1. **Did it converge?** Trust `stopping_reason = agreement` together with a small residual. In the validation runs, converged residuals were about 0.003–0.007. If you see `budget` or a large residual, re-measure with a bigger `krylov_basis_vector_counts`, optionally passing `measurement.warm_start` with the earlier `current_vector_rank*.pt`.
2. **Is it stable across sample sizes?** Measure two `token_counts` (for example 16.8M and 33.6M). If the eigenvalue moves a lot, validation-sample noise dominates. Small token counts, such as 8,192 tokens, check that the code runs; they do not give real estimates.
3. **Compare with the prediction at the same update.** The ratio `measured ζ / predicted_zeta` near 1 means the run sits at the frozen-kernel stability edge. Above 1, the frozen model predicts the top mode grows. Far below 1 means there is room to spare. The [blog figures](figures/blog_six/) plot both curves across training.
4. **Remember what is frozen.** The operator holds older gradients, Adam parameters and the optimizer state fixed, and it uses validation tokens rather than the training minibatch. It is a one-step, fixed-state linearization; it does not simulate the trajectory. Spikes followed by recovery along a run are a separate observation (see the self-stabilization paper below).



## Check changes

```bash
python -m pytest tests/test_config_chains.py tests/test_spectrum_experiments.py -q   # config / CLI changes
python -m pytest -q                                                                  # full suite
```

- Some tests need CUDA, and the distributed tests need permission to open local sockets. Report skipped tests, failures and anything left unverified.
- For documentation changes, check local links and command names with `--print-settings`, then run `git diff --check`.

## Edge-of-stability literature

Read these papers in order when interpreting stability measurements:

1. **Cohen et al., ICLR 2021.** [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/pdf/2103.00065). Start with their full-batch gradient-descent experiments. They track the largest loss-Hessian eigenvalue near 2/η, where η is the learning rate, while observing short-term loss oscillations and longer-term loss reduction.
2. **Cohen, Damian et al., ICLR 2025.** [Understanding Optimization in Deep Learning with Central Flows](https://arxiv.org/abs/2410.24206), with an [illustrated companion website](https://centralflows.github.io/). Read their derivation of differential equations for time-averaged optimizer trajectories to study longer-term motion during oscillatory training.
3. **Islamov et al., ICML 2026.** [Non-Euclidean Gradient Descent Operates at the Edge of Stability](https://arxiv.org/abs/2603.05002). Read their treatment of directional smoothness and generalized sharpness for spectral gradient descent and other non-Euclidean methods. Check the update normalization when comparing these methods with Muon.
4. **Damian, Nichani, and Lee, ICLR 2023.** [Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability](https://arxiv.org/abs/2209.15594). Follow their analysis of how oscillations move gradient descent toward lower curvature and restore stability. Use this mechanism as a hypothesis when investigating curvature spikes followed by recovery in the training runs.
5. **Andreyev and Beneventano.** [Edge of Stochastic Stability: Revisiting the Edge of Stability for SGD](https://arxiv.org/abs/2412.20553), with an [author's companion explanation](https://poggio-lab.mit.edu/blogsupdates/edge-of-stochastic-stability-part-ii). Read their experiments using batch sharpness, the expected minibatch curvature along the corresponding minibatch gradient. Compare their measurement procedure with this repository's fixed-validation measurements before interpreting minibatch training behavior.
6. **Andreyev et al., 2026 preprint.** [Momentum Further Constrains Sharpness at the Edge of Stochastic Stability](https://arxiv.org/abs/2604.14108). Examine their different momentum effects at small and large batch sizes when interpreting the increased- and decreased-momentum experiments. Establish the corresponding Muon analysis before applying their SGD stability thresholds to these runs.

When applying these papers here, first inspect `compute_muon_response` in [muon_step.py](bubbleforce/analysis/muon_step.py). During measurement, the code differentiates the saved normalized Muon update while holding older gradients and Adam parameters fixed. Use this operator definition when comparing measured eigenvalues with a paper's sharpness definition. Evaluate fixed-state stability predictions separately from changes observed along a training run.
