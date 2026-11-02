# Hugging Face restart test at boundary 2201

The default compiled training path fails on the requested checkpoint. It restores the weights exactly and completes update 2201, then reports a non-finite loss during update 2202. A diagnostic replay finds a non-finite gradient and parameter update in `blocks.0.attn.proj.bias`.

With model compilation disabled in a separate diagnostic worker, the same restart completes updates 2201 and 2202. The saved parameters remain finite after both updates. The failure therefore depends on the model execution mode in this test; the low-level compiler cause remains unresolved.

## Inputs and settings

Downloaded `hf://buckets/jacknzheng/bubbleforce/restarts/decreased_momentum/decreased_momentum__checkpoint_002201` without transferring credentials to the GPU host. Verified all eight checkpoint files against the SHA-256 hashes in `checkpoint_002201/COMPLETE.json`. The files total 95,270,259,336 bytes.

Used eight NVIDIA H100 80 GB GPUs, Python 3.12.3, PyTorch 2.10.0+cu128, CUDA 12.8. Kept the full 12-layer, width-768 model and 524,288-token training batch. Set `training.train_steps=6475` and `training.microbatch_sequences=8` to match the saved training recipe. The resulting compatibility hash matches `a92bb65be1fa2dec8429ebd606fe8a24b54207e663b6447f05e8e430a81951a8`.

Downloaded the validation shard and training shards 1 through 12. The loader advances through the earlier batches before resuming at update 2201. Validation uses 10,485,760 tokens. The restored compiled model has validation loss 3.49358; the uncompiled control gives 3.49362.

The checkpoint conversion notes say the original Python and NumPy RNG states were unavailable and were replaced with seed-0 states. This test does not establish exact reproduction of the original training trajectory.

## Evidence

| Check | Observed result |
| --- | --- |
| Checkpoint integrity | All eight hashes match. |
| Weight restoration | First snapshot parameters exactly match the checkpoint. |
| Saved optimizer state | All scanned tensors are finite; Adam second moments are nonnegative. |
| Default compiled restart | First update completes; second loss becomes non-finite. |
| Diagnostic compiled replay | `blocks.0.attn.proj.bias` has a non-finite gradient and executed update. Muon momentum and polar outputs are finite. |
| Uncompiled model control | Both updates complete; parameters remain finite after both. Muon optimizer execution remains unchanged. |
| Snapshot measurement | Compiled measurement completes for snapshot 2201 across eight GPUs. |

The measurement uses 8,192 validation tokens, one sequence per microbatch, BF16 operands with FP32 accumulation, and basis sizes `[4,8]`. It reports eigenvalue 1947.60521965559 with relative residual 0.007144560912097201. The solver stops at its basis budget, so this is an execution check rather than a converged full-data estimate. All eight eigenvector files contain finite values; their combined squared norm is 0.9999999310329355.

## Files and code edits

Applied the earlier fixes directly to the requested checkout: restored `run_measurements`, corrected the snapshot comparison's execution mode, replaced the unsafe overwrite of memory-mapped test data. These changes previously passed all 245 tests. The restart diagnostics do not change production model compilation or the Hugging Face checkpoint.

`control/run_validation.py` records the original command. `control/diagnose_update.py` captures extra fields around the first update and calls the existing training and measurement workers. `control/eager_control.py` disables only `GPT.compile` within its worker. `control/summarize_results.py` checks the recorded outcomes.

The downloaded archive retains the complete diagnostic capture and its measurements, plus logs and summaries for the other runs. The original failed run and uncompiled control captures are omitted from the archive after verification. The source checkpoint remains in the Hugging Face bucket. Extract the archive to inspect `results/summary.json`, `results/state-diagnostics.json`, `results/update-diagnostics.json`, and `results/diagnostic/spectra/update_002201/`.

Workers used `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `OMP_NUM_THREADS=4`, `TORCHINDUCTOR_COMPILE_THREADS=2`; the diagnostic workers also used `OPENBLAS_NUM_THREADS=1`. Dependency installation switched the PyTorch download to its official CUDA wheel server after the initial PyPI transfer stalled.

## Cleanup

Downloaded the diagnostic archive and verified its SHA-256 checksum. Terminated pod `5vtbs5bylbf2w8` and verified its removal through the RunPod API. Estimated GPU cost: $14.21, plus minor container storage charges. The existing pod was preserved.

The results branch stores compact reports, logs, verification records, reproduction scripts, and the small tested-source bundle. The full tensor archive remains locally at `/Users/jackzheng/ML/bubbleforce/bubbleforce/validation/hf-restart-002201/results.tar.gz`. See `download-verification.json` and `cleanup.json`.
