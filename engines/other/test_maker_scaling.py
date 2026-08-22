"""
Offline unit tests for the MAKER/MDAP scaling-law analytics (maker_scaling.py).

No API keys, no LLM. Pure math, validated against the paper's Eq. 9 and the
gambler's-ruin expected-duration analysis.
"""

import os
import sys

# Flat-package layout: engines/other has no __init__.py, so add it to sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math

import maker_scaling as ms


def test_step_success_probability_eq9():
    # Eq. 9: P = 1 / (1 + ((1-p)/p)^k) for p > 0.5
    p, k = 0.9, 5
    expected = 1.0 / (1.0 + ((1.0 - p) / p) ** k)
    got = ms.step_success_probability(p, k)
    assert abs(got - expected) < 1e-12
    # p=0.9, k=5 -> ~0.999983 (paper figure, p=0.9,k=5 ~ 0.99995+)
    assert abs(got - 0.9999833) < 1e-4


def test_step_success_monotonic_in_k():
    p = 0.85
    prev = -1.0
    for k in range(1, 12):
        val = ms.step_success_probability(p, k)
        assert val > prev, f"not monotonic at k={k}"
        prev = val
    assert ms.step_success_probability(p, 11) > 0.999


def test_step_success_degenerate_p():
    # Fair coin: voting cannot help -> exactly 0.5 for every k.
    for k in (1, 3, 10):
        assert ms.step_success_probability(0.5, k) == 0.5
    # Worse-than-fair: voting amplifies the wrong candidate (< 0.5).
    assert ms.step_success_probability(0.4, 5) < 0.5


def test_full_task_success_is_product():
    p, k, s = 0.9, 5, 1000
    per = ms.step_success_probability(p, k)
    assert abs(ms.full_task_success_probability(p, k, s) - per ** s) < 1e-12


def test_required_k_meets_target():
    p, s, target = 0.9, 10000, 0.95
    k = ms.required_k_for_reliability(p, s, target)
    assert 5 <= k <= 8, f"unexpected k={k}"
    assert ms.full_task_success_probability(p, k, s) >= target
    if k > 1:
        assert ms.full_task_success_probability(p, k - 1, s) < target


def test_required_k_grows_logarithmically():
    # k_min = Theta(ln s): increasing in s but sublinear (a constant offset).
    ks = [ms.required_k_for_reliability(0.9, 10 ** e, 0.95) for e in (3, 4, 5, 6)]
    assert ks == sorted(ks), "required k must increase with s"
    assert ks[-1] > ks[0]
    # The spread across three orders of magnitude is small (logarithmic).
    assert (ks[-1] - ks[0]) < 15


def test_expected_votes_per_step():
    # p = 0.5 -> classic gambler's ruin expected duration = k^2.
    assert abs(ms.expected_votes_per_step(0.5, 3) - 9.0) < 1e-9
    # Always at least k samples (a winner must lead by k).
    for k in (1, 3, 5):
        assert ms.expected_votes_per_step(0.9, k) >= k
    # Monotonic in k.
    assert ms.expected_votes_per_step(0.9, 5) > ms.expected_votes_per_step(0.9, 1)
    # k = 1 -> first sample decides -> exactly 1 vote expected.
    assert abs(ms.expected_votes_per_step(0.9, 1) - 1.0) < 1e-9


def test_parallelization_factor_equals_required_k():
    assert ms.parallelization_factor(1_000_000, 0.95, 0.9) == ms.required_k_for_reliability(0.9, 1_000_000, 0.95)


def test_expected_cost_scales_with_steps():
    # E[cost] for m=1 is ~ c * s * k_min -> strictly increasing in s.
    c1 = ms.expected_cost(0.9, 1_000, m=1, c=1.0, target=0.95)
    c2 = ms.expected_cost(0.9, 100_000, m=1, c=1.0, target=0.95)
    assert c2 > c1
    # Auto-tuned k (k=None) yields a finite, positive cost.
    assert ms.expected_cost(0.9, 10_000, m=1, c=1.0) > 0


def test_redflag_correlation_raises_effective_p():
    # Decorrelating correlated failures (paper 3.3) raises full-task success.
    base = ms.full_task_success_probability(0.7, 5, 1000, redflag_correlation=0.0)
    boosted = ms.full_task_success_probability(0.7, 5, 1000, redflag_correlation=0.5)
    assert boosted > base
