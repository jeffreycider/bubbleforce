import torch
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel


class EagerForward(torch.nn.Module):
    """Bypass only torch.compile; keep the model's original forward arithmetic."""

    def __init__(self, model):
        super().__init__()
        self.inner = model

    def forward(self, *args):
        return self.inner.forward(*args)


class FrozenMuonHessian:
    """Exact reverse-over-reverse HVP over ALL selected parameters together.

    Recomputing the graph per microbatch bounds memory. Retaining selected
    first-gradient graphs is optional; cache lifetime is exactly this probe.
    Loss must return a SUM, and normalization is the GLOBAL token count when
    ranks process disjoint subsets of that loss.
    """

    def __init__(
        self,
        model,
        params,
        buffers,
        names,
        batch,
        microbatch,
        global_tokens=None,
        cache_batches=0,
        backend="reverse",
    ):
        if len(set(names)) != len(names) or not names:
            raise ValueError("nonempty unique Muon names required")
        self.names = tuple(names)
        self.params = {
            n: (
                p.detach().float()
                if p.dtype in (torch.bfloat16, torch.float16)
                else p.detach()
            ).requires_grad_(n in names)
            for n, p in params.items()
        }
        self.p = tuple(self.params[n] for n in names)
        self.buffers = {n: b.detach().clone() for n, b in buffers.items()}
        self.model = EagerForward(model)
        self.batch, self.microbatch = batch, microbatch
        self.tokens = global_tokens or batch[1].numel()
        if microbatch < 1 or self.tokens < batch[1].numel():
            raise ValueError("invalid microbatch/global normalization")
        self.cache_batches = cache_batches
        if backend not in ("reverse", "forward"):
            raise ValueError("invalid exact HVP backend")
        self.backend = backend
        self.cache = {}
        self.cpu_rng = torch.get_rng_state().clone()
        device = self.p[0].device
        self.cuda_rng = (
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        )

    def _gradient(self, start):
        x, y = (b[start : start + self.microbatch] for b in self.batch)
        device = self.p[0].device
        rng_devices = [device.index] if device.type == "cuda" else []
        # Same RNG state per call, without advancing caller RNG.
        with torch.random.fork_rng(devices=rng_devices), sdpa_kernel(SDPBackend.MATH):
            torch.set_rng_state(self.cpu_rng)
            if self.cuda_rng is not None:
                torch.cuda.set_rng_state(self.cuda_rng, device)
            loss = functional_call(
                self.model,
                (
                    {"inner." + n: p for n, p in self.params.items()},
                    {"inner." + n: b for n, b in self.buffers.items()},
                ),
                (x, y),
            )
            return torch.autograd.grad(loss, self.p, create_graph=True)

    def __call__(self, direction):
        if len(direction) != len(self.p):
            raise ValueError("one direction per selected matrix required")
        total = [torch.zeros_like(p) for p in self.p]
        for ordinal, start in enumerate(range(0, len(self.batch[0]), self.microbatch)):
            if self.backend == "forward":
                x, y = (b[start : start + self.microbatch] for b in self.batch)

                def loss(current):
                    params = dict(self.params)
                    params.update(zip(self.names, current))
                    return functional_call(
                        self.model,
                        (
                            {"inner." + n: p for n, p in params.items()},
                            {"inner." + n: b for n, b in self.buffers.items()},
                        ),
                        (x, y),
                    )

                device = self.p[0].device
                with (
                    torch.random.fork_rng(
                        devices=[device.index] if device.type == "cuda" else []
                    ),
                    sdpa_kernel(SDPBackend.MATH),
                ):
                    torch.set_rng_state(self.cpu_rng)
                    if self.cuda_rng is not None:
                        torch.cuda.set_rng_state(self.cuda_rng, device)
                    _, h = torch.func.jvp(
                        torch.func.grad(loss),
                        (self.p,),
                        (tuple(v.to(p.dtype) for v, p in zip(direction, self.p)),),
                    )
                for acc, value in zip(total, h):
                    acc.add_(value.detach() / self.tokens)
                del h
                continue
            keep = ordinal < self.cache_batches
            gradient = self.cache.get(start)
            if gradient is None:
                gradient = self._gradient(start)
                if keep:
                    self.cache[start] = gradient
            active = [
                (g, v.to(p.dtype))
                for g, v, p in zip(gradient, direction, self.p)
                if g.requires_grad
            ]
            if active:
                h = torch.autograd.grad(
                    tuple(g for g, _ in active),
                    self.p,
                    grad_outputs=tuple(v for _, v in active),
                    retain_graph=keep,
                    allow_unused=True,
                )
                for acc, value in zip(total, h):
                    if value is not None:
                        acc.add_(value.detach() / self.tokens)
            del gradient
        return tuple(total)

    def close(self):
        self.cache.clear()
