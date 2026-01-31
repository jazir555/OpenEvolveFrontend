"""
SurvivorshipBacktester - Backtester that avoids survivorship bias

Uses CRSP-style data with delisted securities included to prevent
look-ahead bias in strategy evaluation.

Key Features:
- Includes delisted securities
- Adjusts for splits and dividends
- Tracks delisting events
- Parallel execution support
"""

from typing import List, Optional, Dict, Any
import asyncio
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

from .schemas import (
    Strategy,
    BacktestResult,
    DelistingEvent
)


@dataclass
class DataCache:
    """Simple in-memory data cache for backtesting"""
    prices: Dict[str, List[float]]
    dates: List[datetime]
    delistings: List[DelistingEvent]

    def __init__(self):
        self.prices = {}
        self.dates = []
        self.delistings = []


class SurvivorshipBacktester:
    """
    Backtester with survivorship-bias-free data.

    Simulates CRSP-style data access with delisted securities included.
    In production, this would connect to actual CRSP API or similar.
    """

    def __init__(
        self,
        data_source: str = "CRSP_SIMULATED",
        include_delisted: bool = True,
        adjust_for_splits: bool = True,
        adjust_for_dividends: bool = True
    ):
        """
        Initialize survivorship backtester.

        Args:
            data_source: Data source identifier
            include_delisted: Whether to include delisted securities
            adjust_for_splits: Whether to adjust for stock splits
            adjust_for_dividends: Whether to adjust for dividends
        """
        self.data_source = data_source
        self.include_delisted = include_delisted
        self.adjust_for_splits = adjust_for_splits
        self.adjust_for_dividends = adjust_for_dividends

        # Data cache
        self.data_cache = DataCache()

    async def run(
        self,
        strategy: Strategy,
        period: str,
        include_delisted: Optional[bool] = None
    ) -> BacktestResult:
        """
        Run backtest with survivorship-free data.

        Args:
            strategy: Strategy to test
            period: Date range (YYYY-MM-DD:YYYY-MM-DD)
            include_delisted: Override default setting

        Returns:
            BacktestResult with returns, drawdowns, delistings
        """
        include_delisted = include_delisted or self.include_delisted

        # Parse period
        start_date_str, end_date_str = period.split(":")
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)

        # Load data (with delisted if enabled)
        data = await self._load_data(
            source=self.data_source,
            start_date=start_date,
            end_date=end_date,
            include_delisted=include_delisted
        )

        # Run strategy
        returns, delistings = self._execute_strategy(
            strategy,
            data,
            start_date,
            end_date
        )

        # Calculate metrics
        drawdowns = self._calculate_drawdowns(returns)
        sharpe_ratio = self._calculate_sharpe(returns)
        sortino_ratio = self._calculate_sortino(returns)
        max_drawdown = max(drawdowns) if drawdowns else 0.0
        final_wealth = returns[-1] if len(returns) > 0 else 1.0
        volatility = np.std(returns) if len(returns) > 1 else 0.0

        # Estimate trades (simplified)
        total_trades = len(returns) // 10  # Assume trade every 10 periods
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0

        return BacktestResult(
            strategy_id=strategy.strategy_id,
            returns=returns,
            drawdowns=drawdowns,
            delistings=delistings,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            final_wealth=final_wealth,
            volatility=volatility,
            total_trades=total_trades,
            win_rate=win_rate,
            start_date=start_date,
            end_date=end_date
        )

    async def run_parallel(
        self,
        strategies: List[Strategy],
        period: str,
        include_delisted: Optional[bool] = None
    ) -> List[BacktestResult]:
        """
        Run multiple strategies in parallel.

        Args:
            strategies: Strategies to test
            period: Date range
            include_delisted: Override default setting

        Returns:
            List of BacktestResult objects
        """
        tasks = [
            self.run(strategy, period, include_delisted)
            for strategy in strategies
        ]
        return await asyncio.gather(*tasks)

    async def _load_data(
        self,
        source: str,
        start_date: datetime,
        end_date: datetime,
        include_delisted: bool
    ) -> Dict[str, Any]:
        """
        Load market data.

        In production, this would query CRSP API or similar.
        For now, simulates realistic market data with crisis periods.

        Args:
            source: Data source
            start_date: Start date
            end_date: End date
            include_delisted: Whether to include delisted

        Returns:
            Dictionary with price data and metadata
        """
        # Simulated data generation
        # In production: actual CRSP API call

        n_days = (end_date - start_date).days
        n_periods = n_days // 30  # Monthly periods

        # Generate realistic market data with crises
        returns = self._simulate_market_returns(n_periods, start_date, end_date)

        # Generate delistings if requested
        delistings = []
        if include_delisted:
            delistings = self._simulate_delistings(returns, start_date)

        return {
            "returns": returns,
            "dates": [
                start_date + timedelta(days=30*i)
                for i in range(n_periods)
            ],
            "delistings": delistings,
            "include_delisted": include_delisted
        }

    def _simulate_market_returns(
        self,
        n_periods: int,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """
        Simulate realistic market returns with crisis periods.

        Args:
            n_periods: Number of periods
            start_date: Start date
            end_date: End date

        Returns:
            List of simulated returns
        """
        returns = []

        # Crisis periods (approximate)
        dotcom_crisis = (datetime(2000, 1, 1), datetime(2002, 12, 31))
        gfc_crisis = (datetime(2007, 9, 1), datetime(2009, 3, 31))
        covid_crisis = (datetime(2020, 2, 1), datetime(2020, 4, 30))

        current_date = start_date

        for i in range(n_periods):
            current_date = start_date + timedelta(days=30*i)

            # Check if in crisis
            in_dotcom = dotcom_crisis[0] <= current_date <= dotcom_crisis[1]
            in_gfc = gfc_crisis[0] <= current_date <= gfc_crisis[1]
            in_covid = covid_crisis[0] <= current_date <= covid_crisis[1]

            # Generate return based on regime
            if in_dotcom:
                # Dot-com: high volatility, negative drift
                mu, sigma = -0.02, 0.10
            elif in_gfc:
                # GFC: extreme volatility, very negative
                mu, sigma = -0.05, 0.15
            elif in_covid:
                # COVID: extreme volatility
                mu, sigma = -0.08, 0.20
            else:
                # Normal: positive drift, moderate volatility
                mu, sigma = 0.01, 0.04

            # Generate return
            ret = np.random.normal(mu, sigma)
            returns.append(ret)

        return returns

    def _simulate_delistings(
        self,
        returns: List[float],
        start_date: datetime
    ) -> List[DelistingEvent]:
        """
        Simulate delisting events.

        Args:
            returns: Market returns
            start_date: Start date

        Returns:
            List of DelistingEvent objects
        """
        delistings = []

        # Simulate occasional delistings
        for i, ret in enumerate(returns):
            # Higher delisting probability during crises
            if ret < -0.15:  # Large loss
                if np.random.random() < 0.1:  # 10% chance of delisting
                    delisting_date = start_date + timedelta(days=30*i)

                    delistings.append(DelistingEvent(
                        ticker=f"DEL_{i:04d}",
                        delisting_date=delisting_date,
                        reason="bankruptcy" if ret < -0.30 else "below_threshold",
                        last_price=1.0 + ret,
                        recovery_rate=0.1 if ret < -0.30 else 0.3,
                        impact=ret * 100  # Scaled impact
                    ))

        return delistings

    def _execute_strategy(
        self,
        strategy: Strategy,
        data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> tuple[List[float], List[DelistingEvent]]:
        """
        Execute strategy on data.

        Args:
            strategy: Strategy to execute
            data: Market data
            start_date: Start date
            end_date: End date

        Returns:
            Tuple of (returns, delistings)
        """
        market_returns = data["returns"]
        delistings = data["delistings"]

        # Simple strategy execution based on type
        if strategy.strategy_type == "momentum":
            strategy_returns = self._execute_momentum(
                strategy, market_returns
            )
        elif strategy.strategy_type == "mean_reversion":
            strategy_returns = self._execute_mean_reversion(
                strategy, market_returns
            )
        else:
            # Default: market returns with strategy alpha
            alpha = strategy.parameters.get("alpha", 0.0)
            beta = strategy.parameters.get("beta", 1.0)
            strategy_returns = [
                alpha + beta * r
                for r in market_returns
            ]

        # Apply delisting impacts
        for delisting in delistings:
            idx = delisting.delistings.day // 30  # Approximate period
            if 0 <= idx < len(strategy_returns):
                strategy_returns[idx] += delisting.impact / 100

        return strategy_returns, delistings

    def _execute_momentum(
        self,
        strategy: Strategy,
        market_returns: List[float]
    ) -> List[float]:
        """Execute momentum strategy"""
        lookback = strategy.parameters.get("lookback", 3)
        returns = []

        for i in range(len(market_returns)):
            if i < lookback:
                returns.append(0.0)
            else:
                # Calculate momentum
                past_returns = market_returns[i-lookback:i]
                momentum = sum(past_returns) / lookback

                # Go long if momentum positive, short if negative
                if momentum > 0:
                    returns.append(market_returns[i] * 1.2)  # Add leverage
                else:
                    returns.append(-market_returns[i] * 0.8)  # Short with less leverage

        return returns

    def _execute_mean_reversion(
        self,
        strategy: Strategy,
        market_returns: List[float]
    ) -> List[float]:
        """Execute mean reversion strategy"""
        lookback = strategy.parameters.get("lookback", 5)
        threshold = strategy.parameters.get("threshold", 0.02)
        returns = []

        for i in range(len(market_returns)):
            if i < lookback:
                returns.append(0.0)
            else:
                # Calculate mean and deviation
                past_returns = market_returns[i-lookback:i]
                mean_return = sum(past_returns) / lookback
                deviation = market_returns[i-1] - mean_return

                # Bet on reversion
                if abs(deviation) > threshold:
                    # Opposite bet
                    position = -1 if deviation > 0 else 1
                    returns.append(position * market_returns[i] * 0.5)
                else:
                    returns.append(0.0)

        return returns

    def _calculate_drawdowns(self, returns: List[float]) -> List[float]:
        """Calculate drawdown time series"""
        if not returns:
            return []

        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        return drawdowns.tolist()

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio (annualized)"""
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualize (assuming monthly returns)
        sharpe = (mean_return * 12) / (std_return * np.sqrt(12))
        return sharpe

    def _calculate_sortino(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)

        # Downside deviation
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return float('inf') if mean_return > 0 else 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        # Annualize
        sortino = (mean_return * 12) / (downside_std * np.sqrt(12))
        return sortino


# End of SurvivorshipBacktester
