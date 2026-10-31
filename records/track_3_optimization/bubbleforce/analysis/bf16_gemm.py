import torch

from records.track_3_optimization.configs.measurement_config import AddmmConfig

def _require_fp32(*operands):
    if any(value.dtype != torch.float32 for value in operands):
        raise TypeError("BF16 graph substitution requires FP32 inputs")


def _fp32_mm_via_bf16(a, b):
    _require_fp32(a, b)
    a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    if a.is_cuda:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a.float(), b.float())


def _fp32_bmm_via_bf16(a, b):
    _require_fp32(a, b)
    a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    if a.is_cuda:
        return torch.bmm(a, b, out_dtype=torch.float32)
    return torch.bmm(a.float(), b.float())


def _fp32_addmm_via_bf16(bias, a, b, *, config: AddmmConfig = AddmmConfig()):
    _require_fp32(bias, a, b)
    # Keep the bias/epilogue FP32; in particular beta=0 must not read NaN bias.
    product = _fp32_mm_via_bf16(a, b) * config.alpha
    return product if config.beta == 0 else product + bias * config.beta


def rewrite_bf16_gemms(traced: torch.fx.GraphModule) -> dict[str, int]:
    """Replace FP32 dense GEMMs in place and recompile the FX program.

    Matrix operands are rounded to BF16; accumulation, output and the addmm
    bias/epilogue remain FP32. Other operations and input dtypes are unchanged.

    Args:
        traced: Already differentiated FX graph with operand dtype metadata.
            Eligible operations are mm, bmm and addmm with FP32 inputs.

    Returns:
        Replacement counts keyed by original ATen operation name.
    """
    replacements = {
        torch.ops.aten.mm.default: _fp32_mm_via_bf16,
        torch.ops.aten.bmm.default: _fp32_bmm_via_bf16,
        torch.ops.aten.addmm.default: _fp32_addmm_via_bf16,
    }
    counts = {}
    for node in traced.graph.nodes:
        if node.op != "call_function" or node.target not in replacements:
            continue
        # addmm's first operand is the bias; all relevant inputs must be FP32.
        operands = (
            node.args[:3]
            if node.target == torch.ops.aten.addmm.default
            else node.args[:2]
        )
        if not all(
            getattr(
                value.meta.get("val") if isinstance(value, torch.fx.Node) else value,
                "dtype",
                None,
            )
            == torch.float32
            for value in operands
        ):
            continue
        original = str(node.target)
        if node.target == torch.ops.aten.addmm.default:
            with traced.graph.inserting_before(node):
                config = traced.graph.call_function(AddmmConfig, kwargs=dict(node.kwargs))
            node.kwargs = {"config": config}
        node.target = replacements[node.target]
        counts[original] = counts.get(original, 0) + 1
    traced.graph.lint()
    traced.recompile()
    return counts
