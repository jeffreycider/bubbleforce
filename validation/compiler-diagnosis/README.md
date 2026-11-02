# Checkpoint 2201 compiler diagnosis

Replayed the requested Hugging Face checkpoint on eight H100 GPUs with PyTorch 2.10.0+cu128. Traced the bad bias gradient to a missing dtype conversion in PyTorch Inductor's mixed-order reduction code generator. Disabled that optimization, then completed ten compiled training updates and a snapshot measurement.

## Follow the incorrect gradient

1. Inductor combines reductions over different dimensions. For `blocks.0.attn.proj.bias`, it accumulates partial sums in a float32 workspace.
2. The generated Python wrapper sums that workspace into another float32 tensor. PyTorch 2.10 omits the conversion to the expected bfloat16 dtype.
3. The next generated kernel expects a bfloat16 pointer. It reads the float32 tensor's bytes as bfloat16 values, producing enormous numbers or NaNs.
4. Training passes those values to AdamW. The optimizer corrupts the bias, after which the next forward pass produces a non-finite loss.

Captured the kernel inputs and workspace before the incorrect read. Every input and the correctly computed sum is finite. The correct bias sum has maximum absolute value 5.970238208770752; the captured bad gradient has maximum absolute value 2.020426553593072e38. Reinterpreting the correct sum's bytes as bfloat16 reproduces all 768 captured gradient entries exactly.

Run `python validation/compiler-diagnosis/control/verify_dtype_values.py` with PyTorch installed to verify that byte-level match on CPU. The script reads the captured values from `results/dtype-proof-values.json`.

In the full backward code, the wrapper creates `buf637` by summing float32 `workspace_24`. It then passes `buf637` to `triton_poi_fused__to_copy_view_18`, compiled with `in_ptr0: '*bf16'`. The downloaded tensor archive retains the generated code.

## Reproduce without the checkpoint

`control/minimal_reduction.py` reproduces the failure using random bfloat16 input with dimensions `(8, 1024, 768)`:

```python
def fn(x):
    return x.sum((0, 1)).float(), x.float().square().sum(-1)
```

On the tested PyTorch installation, compiling this function produces NaNs in the column sum. Disabling mixed-order reductions produces finite outputs. `results/minimal-generated.py` shows the float32 workspace sum at line 164 and the downstream bfloat16 pointer signature at line 122.

The simpler RMSNorm probe in `control/minimal.py` stays finite with both settings. Its logs are retained to distinguish that unsuccessful reproduction from the reduced sum example.

## Compare identical model inputs

Restored the same model weights, saved rank zero's batch at update 2201, then replayed its eight microbatches independently. Recorded the gradients before any optimizer update.

| Execution | Result |
| --- | --- |
| Eager model | All parameter gradients finite |
| `torch.compile(backend="eager")` | All parameter gradients finite |
| `torch.compile(backend="aot_eager")` | All parameter gradients finite |
| Default Inductor | Corrupted `blocks.0.attn.proj.bias` gradient |
| Inductor with mixed-order reductions disabled | All parameter gradients finite |

The eight-GPU trace also detects corruption before gradient synchronization. Five ranks produce NaNs after their first microbatch; the remaining ranks produce bias gradients around 1e38. This places the failure before GPU communication and the optimizer.

Compared the corrected compiled gradients with eager gradients over the saved batch. The largest relative L2 difference among parameter tensors is 0.02243. These runs use different compiled and eager bfloat16 arithmetic, so the execution checks do not establish identical training trajectories.

## Verify the workaround

Set this variable before starting Python:

```bash
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
```

Kept model compilation and optimizer compilation enabled. Used the original checkpoint, eight GPUs, the full 12-layer model, 524288 tokens per update, eight sequences per microbatch, and the checkpoint's 6475-step schedule. Completed updates 2201 through 2210. Checked every parameter gradient before each optimizer update and every parameter afterward on every rank; all checks passed.

Computed the measurement from snapshot 2210 using 8192 validation tokens and Krylov basis sizes `[4, 8]`. The measurement completed with a reported eigenvalue of 36042.83098410743 and relative residual 0.0048427125289855705. The solver stopped on its agreement criterion. This verifies execution at the stated measurement settings.

`control/verify_workaround.py` records the settings and checks. The exact settings, per-rank checks, training log, and measurement report are under `results/fixed_restart/`.

## Locate the upstream correction

The missing conversion occurs in `SIMDScheduling._codegen_mix_order_reduction`, in `torch/_inductor/codegen/simd.py`. The [PyTorch 2.11 source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/codegen/simd.py) casts the final workspace sum to the expected buffer dtype. The [related PyTorch issue](https://github.com/pytorch/pytorch/issues/190796) reports corrupted bias gradients with the same optimization in 2.10.

Tested the environment-variable workaround on PyTorch 2.10. A full PyTorch 2.11 upgrade was not tested during this diagnosis.

## Retain the checks

Saved diagnostic scripts, compact reports, the reduced reproduction, captured scalar values, and the tested source bundle in this directory. Saved the successful restart snapshots, eigenvectors, captured bias-reduction tensors, batches, and generated compiler code in `compiler-diagnosis-results.tar.gz`; the large archive remains local rather than in Git. `download-verification.json` records its checksum. `cleanup.json` records pod termination and the estimated GPU charge.

Production source files were not edited during this diagnosis. The results are saved on the local branch `codex/hf-restart-002201-check-results`.
