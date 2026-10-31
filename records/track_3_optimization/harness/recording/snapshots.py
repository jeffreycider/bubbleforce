from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from records.track_3_optimization.harness.constants import SNAPSHOT_QUANTITIES
from records.track_3_optimization.harness.run_state import Config, NanoGPTRunState
from records.track_3_optimization.harness.recording.schedule import get_recording_steps
from records.track_3_optimization.harness.recording.store import CaptureWriter
from records.track_3_optimization.harness.checkpoint import (
    save_checkpoint, training_config_fingerprint, use_capture_directory,
)
from records.track_3_optimization.optimizers import Muon
from records.track_3_optimization.optimizers.ringed.momentum.kernel import (
    HISTORY_LENGTH,
)
from records.track_3_optimization.optimizers.ringed.muon import (
    PolarObservation,
)


@dataclass
class _PendingSnapshot:
    """One update's readout, alive only around optimizer.step()."""
    update: int
    weights_before: dict[str, torch.Tensor]
    payload: dict[str, Any]


class Recording:
    """One run's recording policy, writer and optimizer observer.

    Snapshots describe an update; checkpoints resume before the next update.
    Observe the optimizer immediately, but save checkpoints after the other
    post-update hooks.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.pending: _PendingSnapshot | None = None
        stop = min(config["train_steps"], int(config.get("stop_step", config["train_steps"])))
        snapshot_policy = config.get("metrics_snapshot")
        checkpoint_policy = config.get("checkpoint")
        for policy in (snapshot_policy, checkpoint_policy):
            if policy is not None and not isinstance(policy, dict):
                raise ValueError("recording policies must be mappings")
        snapshot_policy = dict(snapshot_policy or {})
        quantities = snapshot_policy.pop("quantities", SNAPSHOT_QUANTITIES)
        self.quantities = set(quantities)
        if not self.quantities or self.quantities - SNAPSHOT_QUANTITIES:
            raise ValueError("unknown or empty metrics_snapshot quantities")
        self.snapshot_steps = get_recording_steps(range(stop), **snapshot_policy)

        # Snapshot t describes update t; checkpoint N is after N updates.
        checkpoint_policy = checkpoint_policy or {}
        self.checkpoint_steps = get_recording_steps(range(1, stop + 1), **checkpoint_policy)
        if stop > 0 and any(checkpoint_policy.values()):
            self.checkpoint_steps.add(stop)
        self.enabled = bool(self.checkpoint_steps or self.snapshot_steps)
        self.gradient_steps = {
            step for update in self.snapshot_steps
            for step in range(max(0, update - HISTORY_LENGTH), update + 1)
        } if "gradients" in self.quantities else set()

    def setup(self, run_state: NanoGPTRunState) -> None:
        if not self.enabled:
            return
        config = self.config
        root = os.environ.get("CAPTURE_ROOT") or config.get("capture_dir")
        self.writer = CaptureWriter(root or Path(config["output_dir"]) / "recording", run_state.rank)
        use_capture_directory(run_state, self.writer)
        if run_state.rank == 0:
            # Save the resolved recipe once; analysis derives kernel t from it.
            metadata = dict(
                config,
                format=3,
                ring_length=HISTORY_LENGTH,
                clock="snapshot t describes update t; checkpoint N is before update N",
            )
            target = self.writer.root.parent / "capture_config.json"
            if target.exists():
                saved = json.loads(target.read_text())
                if (
                    training_config_fingerprint(saved) != training_config_fingerprint(metadata)
                    or any(saved.get(key) != metadata.get(key)
                           for key in ("metrics_snapshot", "checkpoint"))
                ):
                    raise ValueError(
                        "recording root has a different config/policy/format; use a new capture_dir"
                    )
            else:
                target.write_text(json.dumps(metadata, indent=2))
        if not self.snapshot_steps:
            return
        self.params = dict(run_state.model.named_parameters())
        self.names = {id(p): name for name, p in self.params.items()}
        muon_groups = [
            optimizer for _, optimizer in run_state.optimizer.groups
            if isinstance(optimizer, Muon)
        ]
        if len(muon_groups) != 1 or len(muon_groups[0].param_groups) != 1:
            raise ValueError("metrics snapshots require one single-group Muon/MuonH")
        optimizer = self.optimizer = muon_groups[0]
        self.muon_parameters = {
            self.names[id(p)]: p for p in optimizer.param_groups[0]["params"]
        }
        if optimizer.polar_observer is not None:
            raise ValueError("recording would overwrite an existing polar observer")

    def before_optimizer(self, run_state: NanoGPTRunState) -> None:
        """Archive needed gradients and retain the selected snapshot's inputs."""
        if not self.snapshot_steps:
            return
        update = run_state.step
        parameters = self.params
        store = self.writer
        optimizer = self.optimizer
        if optimizer.param_groups[0]["iteration"] != update:
            raise ValueError("snapshot update and optimizer clock differ")
        if (
            update in self.gradient_steps
            and run_state.rank == 0
        ):
            receipt = store.save(
                "gradient",
                update,
                {name: p.grad.detach() for name, p in self.muon_parameters.items()},
            )
            run_state.gradient_references[update] = dict(
                receipt, file=str((store.root / receipt["file"]).resolve()),
            )
        if update not in self.snapshot_steps:
            return
        weights_before = {
            name: p.detach().clone() for name, p in parameters.items()
        } if run_state.rank == 0 and self.quantities & {"weights", "executed_update"} else {}
        payload: dict[str, Any] = {
            field: {} for quantity, field in (
                ("momentum", "post_insertion_prepolar_momentum"),
                ("muon_update", "polar_output"),
            ) if quantity in self.quantities
        }
        payload["settings"] = [
            dict(
                pattern=spec.pattern,
                optimizer=spec.optimizer,
                groups=[
                    {
                        key: value
                        for key, value in group.items()
                        if key
                        in {"lr", "weight_decay", "initial_lr", "mu", "betas", "eps"}
                    }
                    for group in built.param_groups
                ],
            )
            for spec, built in run_state.optimizer.groups
        ]
        if run_state.rank == 0 and "gradients" in self.quantities:
            payload["gradient_file"] = f"gradient_{update:06d}.pt"
            payload["aux_gradients"] = {
                name: p.grad.detach().clone() for name, p in parameters.items()
                if name not in self.muon_parameters
            }
        self.pending = _PendingSnapshot(update, weights_before, payload)
        optimizer.polar_observer = self.observe_polar

    def observe_polar(self, observation: PolarObservation) -> None:
        """Copy only selected fields from the optimizer's actual computation."""
        snapshot = self.pending
        assert snapshot is not None
        if observation.iteration != snapshot.update:
            raise ValueError("polar observation and snapshot clocks differ")
        name = self.names[id(observation.parameter)]
        if "momentum" in self.quantities:
            snapshot.payload["post_insertion_prepolar_momentum"][name] = observation.momentum.detach().clone()
        if "muon_update" in self.quantities:
            snapshot.payload["polar_output"][name] = (
                observation.polar.detach()
                * self.optimizer.shape_scale(observation.parameter)
            )

    def after_optimizer(self, run_state: NanoGPTRunState) -> None:
        """Detach the observer and write the snapshot before post-update hooks."""
        snapshot = self.pending
        if snapshot is None:
            return
        update = snapshot.update
        optimizer = self.optimizer
        self.pending = None
        payload = snapshot.payload
        optimizer.polar_observer = None
        payload.update(
            format=3,
            update=update,
            rank=run_state.rank,
            training_batch_tokens=self.config["batch_tokens"],
        )
        if run_state.rank == 0:
            if "weights" in self.quantities:
                payload["model_before"] = snapshot.weights_before
                payload["buffers"] = dict(run_state.model.named_buffers())
            if "executed_update" in self.quantities:
                payload["executed_update"] = {
                    name: snapshot.weights_before[name] - parameter.detach()
                    for name, parameter in self.params.items()
                }
        self.writer.save("metrics_snapshot", update, payload)

    def after_step(self, run_state: NanoGPTRunState) -> None:
        """Save a restart checkpoint after all configured post-update hooks."""
        if run_state.step + 1 in self.checkpoint_steps:
            save_checkpoint(self.config, run_state, self.writer)
