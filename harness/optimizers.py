import json
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Protocol
import torch
from torch import nn

from optimizers import Muon, MuonH
from optimizers.ringed.momentum.config import momentum_scheduler


class OptimizerBuilder(Protocol):
    """Construct an optimizer from named parameters and YAML hyperparameters."""

    def __call__(self, named_parameters: list[tuple[str, nn.Parameter]],
                 **hyperparameters: Any) -> torch.optim.Optimizer: ...


def scheduler_config_sha256(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _store_scheduler_config_hash(
    optimizer: torch.optim.Optimizer, identity: dict[str, Any],
) -> torch.optim.Optimizer:
    """Bind scheduler configuration so checkpoint loading rejects a changed schedule."""
    for group in optimizer.param_groups:
        group["scheduler_config_sha256"] = scheduler_config_sha256(identity)
    return optimizer


def _build_muon(named_parameters: list[tuple[str, nn.Parameter]], *,
                momentum: dict[str, Any], **hyperparameters: Any) -> torch.optim.Optimizer:
    return _store_scheduler_config_hash(
        Muon([p for _, p in named_parameters], momentum_scheduler(momentum), **hyperparameters),
        {"momentum": momentum})


def _build_muonh(named_parameters: list[tuple[str, nn.Parameter]], *,
                 momentum: dict[str, Any], **hyperparameters: Any) -> torch.optim.Optimizer:
    return _store_scheduler_config_hash(
        MuonH([p for _, p in named_parameters], momentum_scheduler(momentum), **hyperparameters),
        {"momentum": momentum})


def _build_adamw(named_parameters: list[tuple[str, nn.Parameter]], *, lr: float, weight_decay: float,
                 betas: tuple[float, float] = (0.8, 0.95), eps: float = 1e-10,
                 fused: bool = True) -> torch.optim.Optimizer:
    """Construct AdamW with betas=(0.8, 0.95), eps=1e-10 and CUDA-only fused execution."""
    return torch.optim.AdamW([p for _, p in named_parameters], lr=lr, weight_decay=weight_decay,
                             betas=tuple(betas), eps=eps, fused=fused and torch.cuda.is_available())


optimizer_registry: dict[str, OptimizerBuilder] = {
    "muon": _build_muon,
    "muonh": _build_muonh,
    "adamw": _build_adamw,
}


@dataclass(frozen=True)
class GroupSpec:
    pattern: str
    optimizer: str


class GroupedOptimizers:
    """Regex-routed optimizers.

    Each optimizer owns its updates and state. Fullmatch routing rejects
    overlapping groups and uncovered trainable parameters.
    """
    def __init__(self, model: nn.Module, specifications: list[dict[str, Any]]) -> None:
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        used = set()
        self.groups: list[tuple[GroupSpec, torch.optim.Optimizer]] = []
        for row in specifications:
            spec = GroupSpec(row["pattern"], row["optimizer"])
            selected = [(name, p) for name, p in named if re.fullmatch(spec.pattern, name)]
            if not selected or any(name in used for name, _ in selected):
                raise ValueError("optimizer patterns must select disjoint nonempty groups")
            used.update(name for name, _ in selected)
            built = optimizer_registry[spec.optimizer](selected, **row.get("hyperparams", {}))
            self.groups.append((spec, built))
            for group in built.param_groups:
                group.setdefault("initial_lr", group["lr"])
        if used != {name for name, _ in named}:
            raise ValueError("optimizer specifications omit trainable parameters")

    def step(self) -> None:
        for _, built in self.groups:
            built.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _, built in self.groups:
            built.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Standard state dictionaries for each optimizer group."""
        return {"optimizers": [built.state_dict() for _, built in self.groups]}

    def load_state_dict(self, optimizer_state_dict: dict[str, Any]) -> None:
        """Restore through the owning optimizer; PyTorch checks group sizes."""
        for (_, optimizer), entry in zip(self.groups, optimizer_state_dict["optimizers"], strict=True):
            optimizer.load_state_dict(entry)
