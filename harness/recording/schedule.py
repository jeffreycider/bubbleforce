from collections.abc import Sequence


def get_recording_steps(
    step_range: range, *, every: int = 0, steps: Sequence[int] = (),
    dense_intervals: Sequence[dict[str, int]] = (),
) -> set[int]:
    """Union cadence, explicit steps and inclusive intervals within step_range.

    Every N selects multiples of N; zero disables cadence. Selections beyond
    the run's end are clipped, so the same policy works for a short pilot.
    """
    if type(every) is not int or every < 0:
        raise ValueError("recording every must be a nonnegative integer")
    selected = {step for step in step_range if every and step % every == 0}
    for step in steps:
        if type(step) is not int or step < step_range.start:
            raise ValueError("recording steps must be valid integer indices")
        if step in step_range:
            selected.add(step)
    for interval in dense_intervals:
        start, end = interval["start"], interval["end"]
        if (type(start) is not int or type(end) is not int
                or start < step_range.start or end < start):
            raise ValueError("dense interval must have nonnegative start <= end")
        selected.update(range(start, min(end + 1, step_range.stop)))
    return selected
