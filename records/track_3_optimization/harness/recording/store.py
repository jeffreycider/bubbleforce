import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils._pytree import tree_map


def file_sha(path: str | Path) -> str:
    """Hash stored bytes without loading the tensor payload into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_formatted_path_to_capture_file(
    capture_root: Path,
    rank: int,
    kind: Literal["gradient", "checkpoint_gradient", "metrics_snapshot", "checkpoint"],
    step: int,
) -> Path:
    """Return the path of one rank's recorded tensor file."""
    return capture_root / f"rank{rank}" / f"{kind}_{step:06d}.pt"


class CaptureWriter:
    """Write one rank's immutable tensor files and durable hash receipts."""

    def __init__(self, root: str | Path, rank: int):
        self.rank = rank
        self.root = Path(root) / f"rank{rank}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipts = {}
        manifest = self.root / "manifest.jsonl"
        if manifest.exists():
            for line in manifest.read_text().splitlines():
                row = json.loads(line)
                self.receipts[row["kind"], row["step"]] = row

    def save(
        self,
        kind: Literal["gradient", "checkpoint_gradient", "metrics_snapshot", "checkpoint"],
        step: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Publish a payload and receipt without overwriting an earlier capture."""
        target = get_formatted_path_to_capture_file(self.root.parent, self.rank, kind, step)
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                # Native checkpoint serialization streams one storage at a time,
                # rather than cloning all optimizer state into host RAM first.
                # Readouts need compact CPU copies: a gradient may view a ring.
                if kind != "checkpoint":
                    payload = tree_map(
                        lambda value: value.detach().to(device="cpu", copy=True)
                        if isinstance(value, torch.Tensor) else value,
                        payload,
                    )
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target)
            receipt = {
                "kind": kind,
                "step": step,
                "file": target.name,
                "sha256": file_sha(target),
                "bytes": target.stat().st_size,
            }
            if kind in ("gradient", "checkpoint_gradient"):
                receipt["parameters"] = sorted(
                    name for name, value in payload.items()
                    if isinstance(value, torch.Tensor)
                )
            with (self.root / "manifest.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(receipt, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.receipts[kind, step] = receipt
            return receipt
        finally:
            os.unlink(temporary)
