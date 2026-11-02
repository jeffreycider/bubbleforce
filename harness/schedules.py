import math


def learning_rate_factor(spec, step, train_steps):
    kind = spec["kind"]
    if kind == "piecewise_constant":
        boundaries, factors = spec["boundaries"], spec["factors"]
        if (any(type(value) is not int or value < 0 for value in boundaries)
                or boundaries != sorted(set(boundaries))
                or len(factors) != len(boundaries) + 1
                or any(not math.isfinite(value) or value < 0 for value in factors)):
            raise ValueError("require increasing boundaries and one nonnegative factor per interval")
        return factors[sum(step >= boundary for boundary in boundaries)]
    if kind == "cooldown":
        progress = step / train_steps
        fraction = spec["fraction"]
        if not 0 < fraction <= 1:
            raise ValueError("cooldown fraction must lie in (0,1]")
        return 1.0 if progress < 1 - fraction else (1 - progress) / fraction
    if kind == "fast_slow":
        warm, plateau, fast = spec["warmup_end"], spec["plateau_end"], spec["fast_decay_end"]
        peak, floor, minimum = spec["peak_lr"], spec["floor_lr"], spec["min_lr"]
        if not 0 < warm <= plateau < fast < train_steps:
            raise ValueError("invalid fast-slow phase boundaries")
        if step < warm:
            return peak * (step + 1) / warm
        if step < plateau:
            return peak
        if step < fast:
            progress = (step - plateau) / (fast - plateau)
            exponent = spec["fast_decay_exponent"]
            fraction = progress if exponent == 1 else (0 if progress <= 0 else progress ** exponent)
            return peak + (floor - peak) * fraction
        progress = min(max((step - fast) / (train_steps - fast), 0.0), 1.0)
        shape = spec["slow_decay_schedule"]
        if shape == "linear":
            coefficient = 1 - progress
        elif shape == "cosine":
            coefficient = 0.5 * (1 + math.cos(math.pi * progress))
        elif shape == "minus_sqrt":
            coefficient = 1 - math.sqrt(progress)
        else:
            raise ValueError("unknown slow-decay schedule")
        return minimum + coefficient * (floor - minimum)
    if kind == "piecewise_linear":
        knots = spec["knots"]
        knot_steps = [knot_step for knot_step, _ in knots]
        if (not knots or knot_steps != sorted(set(knot_steps))
                or any(not math.isfinite(factor) or factor < 0 for _, factor in knots)):
            raise ValueError("require increasing knot steps and nonnegative factors")
        if step <= knot_steps[0]:
            return knots[0][1]
        for (start_step, start_factor), (end_step, end_factor) in zip(knots, knots[1:]):
            if step <= end_step:
                return start_factor + (end_factor - start_factor) * (step - start_step) / (end_step - start_step)
        return knots[-1][1]
    raise ValueError(f"unknown learning-rate policy {kind!r}")
