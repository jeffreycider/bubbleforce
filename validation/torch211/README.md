# PyTorch 2.11 validation

Tested the current source with `torch==2.11.0`, then applied that version pin to `requirements.txt`. Ran the checks on RunPod using Python 3.12.3, PyTorch 2.11.0+cu128, CUDA 12.8, and eight H100 80 GB GPUs.

Kept model compilation enabled. Left `TORCHINDUCTOR_MIX_ORDER_REDUCTION` unset and verified that the compiler enabled mixed-order reductions by default.

## Check results

| Check | Result |
| --- | --- |
| Compiler regression reproduction | Passed; the previously corrupted column sum matches eager execution exactly |
| Repository tests | 245 passed, zero failures, errors, or skips |
| Real HF checkpoint restart | Completed updates 2201 through 2210 with finite gradients and parameters on all eight ranks |
| Full-model snapshot measurement | Completed at update 2210 |
| Fresh training and checkpoint creation | Passed on two GPUs |
| Resume from the new checkpoint | Passed; final model and optimizer tensors match the uninterrupted run exactly on both ranks |
| Measurement-only command | Completed using the saved snapshot |

Restarted from `hf://buckets/jacknzheng/bubbleforce/restarts/decreased_momentum/decreased_momentum__checkpoint_002201`. The existing checkpoint loader verified each shard's SHA-256 hash and the recipe fingerprint. Used the full 12-layer model, 524288 tokens per update, eight sequences per microbatch, and the checkpoint's 6475-step schedule.

Measured the full-model snapshot using 8192 validation tokens, one sequence per microbatch, and Krylov basis sizes `[4, 8]`. The solver reported eigenvalue 47441.7659516429, relative residual 0.002666149628939536, and stopped on its agreement criterion. Verified that the saved eigenvectors are finite and have combined squared norm within 1e-5 of one.

For the fresh-training check, used two layers of width 128 on two GPUs with 4096 tokens per update. Saved a checkpoint after update 2, resumed from boundary 3 through boundary 6, then compared the final model and optimizer tensors with a six-update uninterrupted run. Measured the saved snapshot again through the measurement-only CLI.

These checks cover ten updates of the full-model restart and the smaller save/resume comparison. They provide execution and restart evidence for the tested settings.

## Install the tested CUDA build

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The repository requirement pins the PyTorch version to `2.11.0`. The first command selects the CUDA 12.8 build used here.

## Inspect the evidence

Read `results/summary.json` for the verified outcomes. The directory also retains the pytest report, per-rank gradient and parameter checks, commands, training logs, measurement reports, and package hashes. Reproduction scripts are under `control/`.

`source-manifest.json` records the source snapshot before the requirement change and identifies the validation override. `tested-source.tar.gz` contains that snapshot with the tested `torch==2.11.0` requirement. `tested-source.json` records the bundle's SHA-256 hash. The verifier confirmed every source file against the recorded snapshot, allowing only the requirement change.

The larger `torch211-results.tar.gz` retains the checkpoints, snapshots, and eigenvectors locally. `download-verification.json` records its checksum; `cleanup.json` records pod termination and estimated GPU cost.
