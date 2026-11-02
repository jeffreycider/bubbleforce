import hashlib

import torch

from harness.optimizers import GroupedOptimizers
from harness.recording.snapshots import Recording
from harness.run_state import NanoGPTRunState
from optimizers.ringed.momentum.kernel import HISTORY_LENGTH


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.Linear(3, 2, bias=False)
        self.head = torch.nn.Linear(2, 1, bias=False)


def configuration(root, beta=0.95):
    quantities = [
        "weights",
        "gradients",
        "momentum",
        "muon_update",
        "executed_update",
    ]
    return dict(
        batch_tokens=6,
        train_steps=4,
        capture_dir=str(root),
        metrics_snapshot=dict(dense_intervals=[dict(start=2, end=3)], quantities=quantities),
        checkpoint=dict(steps=[4]),
        optimizer_groups=[
            dict(
                pattern=r"blocks\.weight",
                optimizer="muon",
                hyperparams=dict(momentum=dict(kind="ema", decay=beta), lr=0.02),
            ),
            dict(
                pattern=r"head\.weight",
                optimizer="adamw",
                hyperparams=dict(lr=0.01, weight_decay=0.1),
            ),
        ],
    )


def state_for(config, model=None):
    torch.manual_seed(91)
    model = model or Model()
    state = NanoGPTRunState(rank=0, world_size=1)
    state.model = model
    state.optimizer = GroupedOptimizers(model, config["optimizer_groups"])
    state.step = -1
    state.print_log = lambda *a, **k: None
    return state


def gradients(state, step):
    for index, p in enumerate(state.model.parameters()):
        p.grad = (
            torch.arange(p.numel(), dtype=p.dtype).reshape_as(p) * 0.001 + (step + index + 1) * 0.01
        )


def muon_ring_digests(state) -> dict[str, str]:
    """Compare the exact rings on this rank after a checkpoint round trip."""
    names = {id(p): name for name, p in state.model.named_parameters()}
    return {
        names[id(parameter)]: ring_digest(
            entry["gradient_history"], (entry["gradient_newest"] + 1) % HISTORY_LENGTH
        )
        for _, optimizer in state.optimizer.groups
        for parameter, entry in optimizer.state.items()
        if "gradient_newest" in entry
    }


def ring_digest(ring: torch.Tensor, ring_next: int) -> str:
    """Test-only equality fingerprint for a ring's contents and next-write slot."""
    digest = hashlib.sha256(f"{tuple(ring.shape)}|{ring_next}|float32".encode())
    for row in ring:
        digest.update(row.detach().float().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_recorded_capture(root, updates=8):
    """Produce real schema-3 snapshots through the shared recording interface."""
    config = configuration(root)
    config['train_steps'] = updates
    config['metrics_snapshot']['dense_intervals'] = [dict(start=0, end=updates-1)]
    config['checkpoint'] = dict(steps=[updates])
    state = state_for(config)
    recording = Recording(config)
    recording.setup(state)
    state.train_batches = iter(range(updates))
    for update in range(updates):
        state.step = update
        next(state.train_batches)
        gradients(state, update)
        recording.before_optimizer(state)
        state.optimizer.step()
        state.optimizer.zero_grad()
        recording.after_optimizer(state)
        recording.after_step(state)
    return config
