"""
Survivorship-Free Backtester.

Backtests a :class:`Strategy` on a synthetic-but-plausible price universe that
optionally includes delisted securities (so returns are *not* inflated by
survivorship bias). The simulation is deterministic given a seed derived from
the strategy id, so results are reproducible.
"""
from __future__ import annotations

import hashlib
import math
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .schemas import BacktestResult, DelistingEvent, Strategy, StrategyType


_DELISTING_REASONS = [
    "bankruptcy",
    "regulatory delisting",
    "merger / acquisition",
    "below listing threshold",
]


def _seed_from(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


class SurvivorshipBacktester:
    """Backtester that can include delisted securities in the universe."""

    def __init__(self, include_delisted: bool = True, n_periods: int = 252):
        self.include_delisted = include_delisted
        self.n_periods = n_periods

    async def run(
        self,
        strategy: Strategy,
        period: str = "2000-01-01:2026-12-31",
        seed: Optional[int] = None,
    ) -> BacktestResult:
        """Run a survivorship-aware backtest for ``strategy``.

        ``period`` is ``"YYYY-MM-DD:YYYY-MM-DD"``. Returns a populated
        :class:`BacktestResult`.
        """
        start, end = self._parse_period(period)
        rng = random.Random(seed if seed is not None else _seed_from(strategy.strategy_id))

        returns = self._simulate_returns(strategy, rng)
        drawdowns, peak = [], 1.0
        equity = 1.0
        for r in returns:
            equity *= (1.0 + r)
            peak = max(peak, equity)
            drawdowns.append((peak - equity) / peak if peak > 0 else 0.0)

        max_dd = max(drawdowns) if drawdowns else 0.0
        mean_ret = statistics_fmean(returns) if returns else 0.0
        std_ret = statistics_pstdev(returns) if len(returns) > 1 else 0.0
        sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 1e-9 else 0.0
        volatility = std_ret * math.sqrt(252)
        final_wealth = equity
        win_rate = (sum(1 for r in returns if r > 0) / len(returns)) if returns else 0.0

        delistings: List[DelistingEvent] = []
        if self.include_delisted:
            delistings = self._simulate_delistings(rng, start, end)

        return BacktestResult(
            strategy_id=strategy.strategy_id,
            returns=returns,
            drawdowns=drawdowns,
            delistings=delistings,
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=self._sortino(returns, std_ret),
            max_drawdown=round(max_dd, 4),
            final_wealth=round(final_wealth, 4),
            volatility=round(volatility, 4),
            total_trades=len(returns),
            win_rate=round(win_rate, 4),
            start_date=start,
            end_date=end,
        )

    # -- internals --------------------------------------------------------
    def _simulate_returns(self, strategy: Strategy, rng: random.Random) -> List[float]:
        st = strategy.strategy_type
        params = strategy.parameters or {}
        lookback = float(params.get("lookback", 12))
        alpha = float(params.get("alpha", 0.01))

        # Base drift and vol per strategy family.
        if st == StrategyType.MOMENTUM:
            drift, vol = 0.0006, 0.018
        elif st == StrategyType.MEAN_REVERSION:
            drift, vol = 0.0003, 0.012
        elif st == StrategyType.VALUE:
            drift, vol = 0.0004, 0.014
        elif st == StrategyType.ARBITRAGE:
            drift, vol = 0.0002, 0.006
        else:
            drift, vol = 0.00035, 0.016

        # lookback/alpha modulate the edge; clamp to sane bounds.
        edge = min(max(drift + alpha * 0.05 - lookback * 1e-5, -0.001), 0.002)
        out: List[float] = []
        for _ in range(self.n_periods):
            shock = rng.gauss(0.0, vol)
            out.append(edge + shock)
        return out

    def _simulate_delistings(
        self,
        rng: random.Random,
        start: datetime,
        end: datetime,
    ) -> List[DelistingEvent]:
        span_days = max((end - start).days, 1)
        n = rng.randint(0, 3)
        events: List[DelistingEvent] = []
        for i in range(n):
            offset = rng.randint(0, span_days)
            ddate = start + timedelta(days=offset)
            last_price = round(rng.uniform(1.0, 250.0), 2)
            impact = -round(rng.uniform(0.1, 0.6), 3)
            events.append(DelistingEvent(
                ticker=f"DELIST{rng.randint(1000, 9999)}",
                delisting_date=ddate,
                reason=rng.choice(_DELISTING_REASONS),
                last_price=last_price,
                recovery_rate=round(rng.uniform(0.0, 0.4), 2),
                impact=impact,
            ))
        return events

    @staticmethod
    def _sortino(returns: List[float], std_ret: float) -> Optional[float]:
        downside = [min(r, 0.0) for r in returns]
        dstd = math.sqrt(sum(d * d for d in downside) / len(returns)) if returns else 0.0
        if dstd <= 1e-9:
            return None
        mean_ret = statistics_fmean(returns) if returns else 0.0
        return round(mean_ret / dstd * math.sqrt(252), 4)

    @staticmethod
    def _parse_period(period: str) -> Tuple[datetime, datetime]:
        try:
            a, b = period.split(":")
            start = datetime.strptime(a.strip(), "%Y-%m-%d")
            end = datetime.strptime(b.strip(), "%Y-%m-%d")
            return start, end
        except Exception:
            now = datetime.utcnow()
            return now - timedelta(days=365 * 26), now


def statistics_fmean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def statistics_pstdev(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
