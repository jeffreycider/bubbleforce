"""Reinterpret the captured float32 sum, then compare it with the bad bias gradient."""
import json
from pathlib import Path

import torch

values = json.loads(
    (Path(__file__).resolve().parents[1] / "results/dtype-proof-values.json").read_text()
)
correct = torch.tensor(values["workspace_sum"], dtype=torch.float32)
actual = torch.tensor(values["actual_bias"], dtype=torch.float32)
misread = correct.view(torch.bfloat16)[: actual.numel()].float()
assert torch.isfinite(correct).all()
assert torch.equal(misread, actual)
print(f"All {actual.numel()} corrupted bias entries match the wrong pointer dtype.")
