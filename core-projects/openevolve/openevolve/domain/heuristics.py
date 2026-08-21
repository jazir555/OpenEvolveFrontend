"""
Deterministic Scoring Heuristics for Domain Optimizers
=====================================================

Domain optimizers need a real fitness signal without calling out to backtesting
engines, analytics platforms, FEA solvers or docking software. Everything in this
module is therefore:

- deterministic: the same candidate always yields the same metrics;
- local: only the Python standard library is used, no network or services;
- signal-driven: scores are derived from measurable properties of the candidate.

Three groups of helpers are provided:

1. Text/structure signals (:func:`signal_coverage`, :func:`saturating`,
   :func:`code_structure_score`) used by every domain to turn source text into
   normalized scores.
2. A reproducible synthetic backtest (:func:`synthetic_returns` plus
   :func:`return_statistics`) used by the finance and trading optimizers. The
   return series is generated from a stable hash of the candidate and shaped by
   the candidate's own risk/return characteristics; the reported statistics
   (Sharpe, Sortino, drawdown, VaR, ...) are then computed with the standard
   formulas on that series.
3. Small numeric utilities (:func:`clamp`, :func:`stable_unit`).

Author: OpenEvolve Domain Team
"""

import hashlib
import math
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple
# Trading days per year, used to annualize the synthetic series
PERIODS_PER_YEAR = 252

# Risk-free rate used by the Sharpe calculation
RISK_FREE_RATE = 0.02


# ============================================================================
# NUMERIC UTILITIES
# ============================================================================


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value into ``[low, high]``"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if math.isnan(number):
        return low
    return max(low, min(high, number))


def stable_unit(*parts: Any) -> float:
    """
    Deterministic pseudo-random value in ``[0.0, 1.0)`` derived from the inputs.

    Used to give named entities (assets, materials, molecules) stable synthetic
    properties without any external data source.
    """
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(1 << 48)


def saturating(count: float, target: float) -> float:
    """
    Normalize a count against a target, saturating at 1.0.

    Args:
        count: Observed count
        target: Count that should score 1.0

    Returns:
        Score in ``[0.0, 1.0]``
    """
    if target <= 0:
        return 0.0
    return clamp(float(count) / float(target))


def mean(values: Sequence[float], default: float = 0.0) -> float:
    """Arithmetic mean with a default for empty input"""
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else default


def stdev(values: Sequence[float]) -> float:
    """Population standard deviation"""
    if len(values) < 2:
        return 0.0
    average = mean(values)
    variance = sum((v - average) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile of a sample"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = clamp(fraction, 0.0, 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# ============================================================================
# TEXT / STRUCTURE SIGNALS
# ============================================================================


def signal_coverage(text: str, patterns: Iterable[str]) -> float:
    """
    Fraction of the given signals present in the text (case-insensitive).

    Substrings are matched literally; entries containing regex metacharacters
    are matched as regular expressions.

    Args:
        text: Candidate source/spec
        patterns: Signals to look for

    Returns:
        Coverage ratio in ``[0.0, 1.0]``
    """
    patterns = list(patterns)
    if not patterns:
        return 0.0

    lowered = (text or "").lower()
    hits = 0
    for pattern in patterns:
        needle = pattern.lower()
        if re.search(r"[\\\[\](){}|+*?^$]", needle):
            if re.search(needle, lowered):
                hits += 1
        elif needle in lowered:
            hits += 1
    return hits / len(patterns)


def count_signals(text: str, patterns: Iterable[str]) -> int:
    """Number of signals from ``patterns`` present in the text"""
    patterns = list(patterns)
    return int(round(signal_coverage(text, patterns) * len(patterns)))


def code_structure_score(source: str) -> float:
    """
    Score how well-structured a candidate implementation is.

    Rewards documented, decomposed, commented code and penalizes one-liners and
    unbounded scripts. Purely structural, so it works for any language.

    Args:
        source: Candidate source

    Returns:
        Structure score in ``[0.0, 1.0]``
    """
    text = source or ""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0

    definitions = len(re.findall(r"\b(def|class|function|const|let)\b", text))
    comments = len([line for line in lines if line.strip().startswith(("#", "//", "/*", "<!--"))])
    docstrings = len(re.findall(r'"""|\'\'\'', text)) // 2

    components = [
        saturating(len(lines), 40),               # enough substance
        saturating(definitions, 4),               # decomposition
        saturating(comments + docstrings, 6),     # explained intent
        1.0 - saturating(max(0, len(lines) - 400), 400),  # not a monolith
    ]
    return clamp(mean(components))


# ============================================================================
# SYNTHETIC BACKTEST
# ============================================================================


def synthetic_returns(
    seed: str,
    periods: int = PERIODS_PER_YEAR,
    drift: float = 0.0004,
    volatility: float = 0.012,
) -> List[float]:
    """
    Generate a reproducible pseudo-random return series.

    A linear congruential generator seeded from ``seed`` feeds a Box-Muller
    transform, so the series is normally distributed, deterministic for a given
    seed, and shaped by the supplied drift/volatility. This replaces market data
    for offline scoring: candidates that manage risk better receive a lower
    ``volatility`` and therefore genuinely better risk-adjusted statistics.

    Args:
        seed: Seed text (usually the candidate source or its parsed structure)
        periods: Number of periods to generate
        drift: Per-period expected return
        volatility: Per-period standard deviation

    Returns:
        List of per-period returns
    """
    digest = hashlib.sha256((seed or "").encode("utf-8")).hexdigest()
    state = int(digest[:16], 16) | 1

    def next_uniform() -> float:
        nonlocal state
        # Numerical Recipes LCG constants
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return ((state >> 11) + 0.5) / float(1 << 53)

    returns: List[float] = []
    while len(returns) < max(1, periods):
        u1 = max(next_uniform(), 1e-12)
        u2 = next_uniform()
        radius = math.sqrt(-2.0 * math.log(u1))
        returns.append(radius * math.cos(2.0 * math.pi * u2))
        if len(returns) < periods:
            returns.append(radius * math.sin(2.0 * math.pi * u2))
    returns = returns[:periods]

    # Standardize the draw, then impose the requested drift/volatility exactly.
    # The path shape (ordering, drawdowns, tails) stays a deterministic property
    # of the seed while the moments reflect the candidate being scored, so the
    # statistics are not dominated by sampling noise.
    sample_mean = mean(returns)
    sample_std = stdev(returns) or 1.0
    return [drift + volatility * (value - sample_mean) / sample_std for value in returns]


def return_statistics(
    returns: Sequence[float],
    periods_per_year: int = PERIODS_PER_YEAR,
    risk_free: float = RISK_FREE_RATE,
) -> Dict[str, float]:
    """
    Compute standard performance statistics for a return series.

    Args:
        returns: Per-period returns
        periods_per_year: Periods used for annualization
        risk_free: Annual risk-free rate

    Returns:
        Dict with total_return, annual_return, volatility, sharpe_ratio,
        sortino_ratio, max_drawdown, win_rate, profit_factor, avg_win, avg_loss,
        expectancy, var_95 and cvar_95
    """
    series = [float(r) for r in returns]
    if not series:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 1.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
        }

    # Equity curve and drawdown
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in series:
        equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)

    total_return = equity - 1.0
    period_mean = mean(series)
    period_std = stdev(series)

    annual_return = (1.0 + period_mean) ** periods_per_year - 1.0
    volatility = period_std * math.sqrt(periods_per_year)

    sharpe_ratio = (annual_return - risk_free) / volatility if volatility > 0 else 0.0

    downside = [value for value in series if value < 0]
    downside_std = stdev(downside) * math.sqrt(periods_per_year) if len(downside) > 1 else 0.0
    sortino_ratio = (
        (annual_return - risk_free) / downside_std if downside_std > 0 else sharpe_ratio
    )

    wins = [value for value in series if value > 0]
    losses = [abs(value) for value in series if value < 0]

    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float(len(wins) > 0) * 10.0

    win_rate = len(wins) / len(series)
    avg_win = mean(wins)
    avg_loss = mean(losses)

    var_95 = max(0.0, -percentile(series, 0.05))
    tail = [value for value in series if value <= -var_95] or [-var_95]
    cvar_95 = max(0.0, -mean(tail))

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": win_rate * avg_win - (1.0 - win_rate) * avg_loss,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


# ============================================================================
# SYNTHETIC ASSET MODEL (FINANCE)
# ============================================================================


def asset_profile(name: str) -> Tuple[float, float]:
    """
    Deterministic annual (expected return, volatility) for a named asset.

    Args:
        name: Asset identifier

    Returns:
        Tuple of (expected annual return, annual volatility)
    """
    expected = 0.02 + 0.14 * stable_unit(name, "mu")
    volatility = 0.08 + 0.32 * stable_unit(name, "sigma")
    return expected, volatility


def asset_correlation(left: str, right: str) -> float:
    """Deterministic correlation between two named assets"""
    if left == right:
        return 1.0
    first, second = sorted((str(left), str(right)))
    return 0.05 + 0.65 * stable_unit(first, second, "rho")


def portfolio_moments(weights: Dict[str, float]) -> Tuple[float, float]:
    """
    Expected annual return and volatility of a weighted portfolio.

    Uses the synthetic asset model, so diversification genuinely reduces
    volatility through the correlation matrix.

    Args:
        weights: Asset -> weight (normalized internally)

    Returns:
        Tuple of (expected annual return, annual volatility)
    """
    if not weights:
        return 0.0, 0.0

    total = sum(abs(w) for w in weights.values())
    if total <= 0:
        return 0.0, 0.0

    normalized = {asset: abs(weight) / total for asset, weight in weights.items()}
    profiles = {asset: asset_profile(asset) for asset in normalized}

    expected = sum(normalized[a] * profiles[a][0] for a in normalized)

    variance = 0.0
    for left, left_weight in normalized.items():
        for right, right_weight in normalized.items():
            variance += (
                left_weight
                * right_weight
                * profiles[left][1]
                * profiles[right][1]
                * asset_correlation(left, right)
            )

    return expected, math.sqrt(max(0.0, variance))
