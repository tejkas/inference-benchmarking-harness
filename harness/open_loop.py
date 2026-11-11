import math
import random


def arrival_offsets(rate: float, n: int, burstiness: float = 1.0, seed: int = 0) -> list[float]:
    """Send times in seconds from the start of the run, for n requests averaging `rate` per second.

    Gaps between arrivals are gamma-distributed with mean 1/rate. burstiness=1 gives
    exponential gaps (a Poisson process); below 1 is burstier, above 1 more regular.
    rate=inf sends everything at once.
    """
    if math.isinf(rate):
        return [0.0] * n
    rng = random.Random(seed)
    shape = burstiness
    scale = 1.0 / (rate * burstiness)  # keeps the mean gap at shape * scale = 1/rate
    offsets = []
    t = 0.0
    for _ in range(n):
        offsets.append(t)
        t += rng.gammavariate(shape, scale)
    return offsets
