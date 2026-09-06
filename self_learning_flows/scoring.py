"""Small statistical helpers used by promotion and evaluation."""

from __future__ import annotations

import math


def beta_posterior_mean(successes: int, failures: int, alpha: float = 1, beta: float = 1) -> float:
    return (successes + alpha) / (successes + failures + alpha + beta)


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Lower bound of a Wilson score interval for a Bernoulli rate."""
    if trials <= 0:
        return 0.0
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = proportion + z * z / (2 * trials)
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * trials)) / trials)
    return max(0.0, (centre - margin) / denominator)


def running_average(previous: float, count_before: int, value: float) -> float:
    if count_before <= 0:
        return value
    return ((previous * count_before) + value) / (count_before + 1)
