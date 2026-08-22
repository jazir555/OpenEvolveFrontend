"""
Maker/MDAP Scaling-Law Analytics.

Faithful implementation of the scaling laws from the MAKER paper
("Solving a Million-Step LLM Task with Zero Errors", docs/Papers/MDAP_MAKER.txt),
section 3.2 (First-to-ahead-by-k Voting and Scaling Laws) and 3.3 (Red-Flagging).

Key formulas (Eq. 9 and the gambler's-ruin analysis):

  * Per-step success after first-to-ahead-by-k voting (worst case: one correct
    candidate racing one most-likely alternative), for p > 0.5:

        P(step correct) = 1 / (1 + ((1 - p) / p) ** k)            (Eq. 9)

  * Full-task success (s independent steps):

        P(full) = P(step) ** s_steps

  * Required votes threshold for a target reliability t grows logarithmically
    with the number of steps:

        k_min = Theta(ln s)

  * Expected cost for maximal decomposition (m = 1) scales log-linearly:

        E[cost] = Theta( p^{-1} * c * s * ln s )

    where the ln s factor is exactly the parallelizable vote threshold k_min.

These functions are PURE and import nothing from the rest of the project, so
they can be unit-tested offline (no API keys, no LLM).
"""

from __future__ import annotations

import math
from typing import Optional

# Upper bound used when binary-searching for the required vote threshold k.
_K_SEARCH_MAX = 1_000_000


def step_success_probability(p: float, k: int) -> float:
    """Per-step success probability after first-to-ahead-by-k voting (Eq. 9).

    Exact closed form for the worst-case race between a correct candidate
    (probability p) and a single most-likely alternative (probability 1 - p):

        P = 1 / (1 + ((1 - p) / p) ** k)

    Behavior:
      * p > 0.5  -> P increases monotonically toward 1 as k grows.
      * p == 0.5 -> P == 0.5 for every k (voting cannot help a fair coin).
      * p < 0.5  -> P < 0.5 (voting amplifies the wrong candidate).

    Args:
        p: intrinsic per-sample probability that a single vote is correct (0..1).
        k: first-to-ahead-by-k vote margin (>= 1).

    Returns:
        Probability that the correct candidate wins the k-ahead vote.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # Trivial / degenerate bases that would otherwise break the pow().
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    ratio = (1.0 - p) / p
    # For p > 0.5 ratio < 1 so ratio**k -> 0 as k -> inf (converges to 1).
    # For p < 0.5 ratio > 1 so ratio**k -> inf (converges to 0). The same
    # formula handles both; we clamp numerically to stay within [0, 1].
    try:
        denom = 1.0 + (ratio ** k)
    except OverflowError:
        # ratio**k overflowed: means p < 0.5 and k huge -> P -> 0.
        return 0.0
    prob = 1.0 / denom
    return max(0.0, min(1.0, prob))


def full_task_success_probability(
    p: float,
    k: int,
    s_steps: int,
    redflag_correlation: float = 0.0,
) -> float:
    """Probability the full s-step task completes with zero errors.

    Independent steps multiply per-step success:

        P(full) = P(step) ** s_steps

    ``redflag_correlation`` models the paper's correlated-error reduction
    (section 3.3): red-flagging discards malformed/structurally-inconsistent
    outputs, which are correlated with deeper reasoning errors. Discarding them
    raises the *effective* per-sample correctness p. We model this as a simple,
    conservative lift: a fraction ``redflag_correlation`` of the (1 - p)
    failures are correlated-and-removed, so

        p_eff = p + (1 - p) * redflag_correlation,

    clamped to <= 1. This is a heuristic proxy, not a derived bound, but it
    captures the paper's qualitative claim that red-flagging decorrelates and
    raises effective reliability.

    Args:
        p: intrinsic per-sample correctness.
        k: vote margin.
        s_steps: total number of dependent steps.
        redflag_correlation: fraction of failures that are correlated and removed
            by red-flagging (0..1). 0 = no red-flag benefit.

    Returns:
        Full-task zero-error probability in [0, 1].
    """
    if not (0.0 <= redflag_correlation <= 1.0):
        raise ValueError(f"redflag_correlation must be in [0, 1], got {redflag_correlation}")
    if s_steps < 0:
        raise ValueError(f"s_steps must be >= 0, got {s_steps}")

    effective_p = p + (1.0 - p) * redflag_correlation
    effective_p = min(1.0, max(0.0, effective_p))
    per_step = step_success_probability(effective_p, k)
    return per_step ** s_steps


def required_k_for_reliability(
    p: float,
    s_steps: int,
    target: float = 0.95,
    redflag_correlation: float = 0.0,
    k_max: int = _K_SEARCH_MAX,
) -> int:
    """Minimal first-to-ahead-by-k margin so full-task success >= target.

    Because ``full_task_success_probability`` is monotonically increasing in k
    (for p > 0.5), we binary-search the smallest integer k in [1, k_max] that
    meets the target. This is the "exceed" feature: a user supplies a target
    reliability (e.g. 0.95) and the system auto-tunes k. With p <= 0.5 the target
    is unreachable, so we return ``k_max`` (caller can detect "infeasible").

    Args:
        p: intrinsic per-sample correctness.
        s_steps: total number of dependent steps.
        target: desired full-task zero-error probability (0..1).
        redflag_correlation: correlated-error removal fraction (see
            ``full_task_success_probability``).
        k_max: search ceiling; also the value returned when infeasible.

    Returns:
        Minimal integer k achieving the target (or k_max if unreachable).
    """
    if not (0.0 <= target <= 1.0):
        raise ValueError(f"target must be in [0, 1], got {target}")

    # Fast path: even infinite k cannot help p <= 0.5 reach a target > 0.5.
    if p <= 0.5 and target > 0.5:
        return k_max

    def meets(k: int) -> bool:
        return full_task_success_probability(p, k, s_steps, redflag_correlation) >= target

    low, high = 1, k_max
    result = k_max
    # The monotonic sequence means binary search is exact.
    while low <= high:
        mid = (low + high) // 2
        if meets(mid):
            result = mid
            high = mid - 1
        else:
            low = mid + 1
    return result


def expected_votes_per_step(p: float, k: int, m: int = 1) -> float:
    """Expected number of samples drawn per subtask under k-ahead voting.

    Modeled as the expected duration of a gambler's-ruin race between the
    correct subtask outcome (probability ``pvote``) and a single most-likely
    alternative (probability ``palt``), with absorbing barriers at +/-k.

    For maximal decomposition (m = 1): pvote = p, palt = 1 - p, and the expected
    duration starting from a tied race is:

        E[T] = k/(q - p) - (2k)/(q - p) * (1 - (q/p)**k) / (1 - (q/p)**(2k))

    with q = 1 - p. At p = 0.5 this reduces to E[T] = k**2. The result is always
    >= k (you must sample at least k times for a winner to lead by k).

    Args:
        p: intrinsic per-sample correctness.
        k: vote margin.
        m: steps per subtask (m = 1 is maximal decomposition).

    Returns:
        Expected number of samples (votes) per subtask.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    pvote = (p ** m)
    palt = ((1.0 - p) * (p ** (m - 1))) if m >= 1 else (1.0 - p)
    # Total probability mass that actually races (the rest is the "other"
    # alternatives that, by the worst-case assumption, never accumulate).
    total = pvote + palt
    if total <= 0.0:
        return float(k)

    q = palt / total
    prob = pvote / total  # P(correct) in the two-way race

    if abs(prob - q) < 1e-12:
        # Fair race: classic gambler's-ruin expected duration.
        return float(k * k)

    ratio = q / prob
    if ratio <= 0.0:
        return float(k)

    try:
        num = 1.0 - (ratio ** k)
        den = 1.0 - (ratio ** (2 * k))
    except OverflowError:
        # ratio > 1 and k huge -> num ~ -inf, den ~ -inf; ratio**k dominates,
        # the race is essentially one-sided and converges in ~k steps.
        return float(k)

    if abs(den) < 1e-15:
        return float(k)

    frac = num / den
    duration = (k / (q - prob)) - ((2.0 * k) / (q - prob)) * frac
    # Each "round" of the two-way race consumes 1/(pvote+palt) raw samples when
    # m > 1 (a subtask sample yields one of possibly many alternatives). Scale.
    return max(float(k), abs(duration) / total)


def expected_cost(
    p: float,
    s_steps: int,
    m: int = 1,
    c: float = 1.0,
    k: Optional[int] = None,
    target: float = 0.95,
    redflag_correlation: float = 0.0,
) -> float:
    """Expected total cost (in LLM calls) to solve the full task.

    Cost = (number of subtasks) * (expected samples per subtask) * per-sample cost c.

        E[cost] = c * (s_steps / m) * expected_votes_per_step(p, k, m)

    When ``k`` is not supplied it is auto-tuned via ``required_k_for_reliability``
    to meet ``target``. For m = 1 this matches the paper's log-linear law
    E[cost] = Theta(p^{-1} c s ln s) because k_min = Theta(ln s).

    Args:
        p: intrinsic per-sample correctness.
        s_steps: total number of dependent steps.
        m: steps per subtask (m = 1 maximal).
        c: cost per single sample (default 1.0 -> cost in "calls").
        k: explicit vote margin; if None, auto-tuned from target.
        target: reliability target used when auto-tuning k.
        redflag_correlation: correlated-error removal fraction.

    Returns:
        Expected total cost.
    """
    if k is None:
        k = required_k_for_reliability(p, s_steps, target, redflag_correlation)

    per_subtask = expected_votes_per_step(p, k, m)
    num_subtasks = max(1, s_steps / m)
    return c * num_subtasks * per_subtask


def parallelization_factor(
    s_steps: int,
    target: float = 0.95,
    p: float = 0.9,
    redflag_correlation: float = 0.0,
) -> int:
    """Parallelization factor: the number of concurrent votes per step.

    The paper notes the Theta(ln s) vote threshold can be parallelized across
    Theta(ln s) processes, so wall-clock time scales only linearly with s. This
    factor is exactly the auto-tuned k_min for the requested reliability.

    Args:
        s_steps: total number of dependent steps.
        target: desired full-task reliability.
        p: intrinsic per-sample correctness.
        redflag_correlation: correlated-error removal fraction.

    Returns:
        Required vote margin k_min (also the parallel-process count).
    """
    return required_k_for_reliability(p, s_steps, target, redflag_correlation)
