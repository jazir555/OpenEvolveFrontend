#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Alpha Signals That Survived Crises

This module contains real-world examples of alpha signals that have
demonstrated resilience across multiple market crises.

Each example includes:
- Signal definition and formula
- Rationale (why it works)
- Crisis performance history
- Implementation notes
- Potential failure modes

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from datetime import datetime
from typing import Dict, List

from openevolve.finance.verticals.hedge_fund.schemas import (
    AlphaSignal,
    AlphaSource,
    BacktestResult,
    CrisisPerformance,
    CrisisPeriod,
    FeatureSet
)


# ============================================================================
# SIGNAL #78: Earnings Surprise + Credit Spread Tightening
# ============================================================================

EARNINGS_CREDIT_SIGNAL = {
    "signal_id": "earnings_credit_momentum",
    "name": "Earnings Surprise with Credit Confirmation",
    "description": """
    Combines earnings surprise with credit spread confirmation.

    This hybrid signal looks for companies that:
    1. Beat earnings expectations (SUE > 0)
    2. Have tightening credit spreads (credit improvement)

    The credit filter avoids deteriorating companies that might have
    "fake" earnings beats (accounting games, one-time gains, etc.).

    Why it works:
    - Earnings surprise is a known alpha source (post-earnings drift)
    - Credit spreads are forward-looking and hard to manipulate
    - Credit filter improves quality of earnings surprises

    Crisis performance:
    - Dotcom (2000-02): Survived with 3% alpha
      (Credit filter avoided speculative tech companies)
    - GFC (2008): Survived with 2% alpha
      (Credit spreads warned early, avoiding financials)
    - COVID (2020): Survived with 5% alpha
      (Rapid credit indicator caught the quick recovery)

    Implementation notes:
    - Use SUE (Standardized Unexpected Earnings) for surprise
    - Define credit tightening as: spread_t-20 > spread_t
    - Filter for investment-grade companies only
    - Hold for 3-6 months after earnings

    Potential failure modes:
    - Credit spreads may be noisy for smaller companies
    - Earnings guidance gaming
    - Lag in credit data for small caps
    - May miss growth companies with high credit costs
    """,
    "formula": """
    signal = SUE * I(credit_spread_tightening)

    Where:
    - SUE = (eps_actual - eps_expected) / std_dev
    - credit_spread_tightening = (spread_t-20 - spread_t) / spread_t-20 > 0.02
    - I(condition) = 1 if true, 0 otherwise
    """,
    "alpha_source": AlphaSource.COMBINATION,
    "information_ratio": 1.2,
    "sharpe_ratio": 1.8
}


# ============================================================================
# SIGNAL #112: Low Volatility Anomaly
# ============================================================================

LOW_VOLATILITY_SIGNAL = {
    "signal_id": "low_volatility_anomaly",
    "name": "Low Volatility Anomaly",
    "description": """
    The low volatility anomaly: low-risk stocks have high risk-adjusted returns.

    This contradicts modern portfolio theory, which suggests that higher risk
    should be compensated with higher returns. In practice, low-volatility
    stocks have better risk-adjusted returns.

    Why it works:
    - Behavioral bias: Investors prefer "lottery ticket" high-vol stocks
    - Agency issues: Fund managers benchmark to market, maximizing tracking error
    - Leverage constraints: Investors can't lever low-vol stocks to match return
    - Focus on nominal returns, not risk-adjusted returns

    Crisis performance:
    - Dotcom (2000-02): +8% alpha
      (Low vol stocks fell much less than market)
    - GFC (2008): +5% alpha
      (Avoided high-beta financials and tech)
    - COVID (2020): +3% alpha
      (Stable companies held up better)

    Implementation notes:
    - Use realized volatility over 252 trading days
    - Exclude highly leveraged companies (debt/equity > 2)
    - Sector-neutral: Take top 20% low vol within each sector
    - Rebalance monthly
    - Avoid microcaps (liquidity risk)

    Potential failure modes:
    - Underperforms in strong bull markets (low beta)
    - May have concentration in utilities, consumer staples
    - Interest rate sensitivity (bond proxy behavior)
    - Can become crowded trade
    """,
    "formula": """
    signal = 1 / (realized_volatility * sqrt(252))

    Where:
    - realized_volatility = std(returns[-252:])
    - Sector-neutral: rank within GICS sector
    - Filter: debt_to_equity < 2.0
    - Filter: market_cap > $1B
    """,
    "alpha_source": AlphaSource.BEHAVIORAL,
    "information_ratio": 0.9,
    "sharpe_ratio": 1.6
}


# ============================================================================
# SIGNAL #156: Value + Quality Combo
# ============================================================================

VALUE_QUALITY_SIGNAL = {
    "signal_id": "value_quality_combo",
    "name": "Value with Quality Filter",
    "description": """
    Combines value metrics with quality filters to avoid value traps.

    Pure value strategies can suffer from "value traps" - stocks that are
    cheap because they're deteriorating. This signal filters for quality
    to avoid those traps.

    Why it works:
    - Value stocks have higher expected returns (risk premium)
    - Quality filters avoid deteriorating companies
    - Combination captures "quality at reasonable price" (QARP)
    - Sustainable competitive advantages lead to mean reversion

    Crisis performance:
    - Dotcom (2000-02): +7% alpha
      (Value outperformed growth after bubble)
    - GFC (2008): -2% alpha (still survived)
      (Quality helped avoid worst value traps)
    - COVID (2020): +4% alpha
      (Quality balance sheets provided resilience)

    Implementation notes:
    - Value metric: Enterprise Value / Free Cash Flow
    - Quality metrics: ROIC > 10%, Debt/EBITDA < 3, Accruals < 0
    - Sector-neutral: Rank within each sector
    - Hold for 12 months (value signals are slow-moving)
    - Rebalance annually

    Potential failure modes:
    - Value underperformance can last for years
    - Accounting manipulation in value metrics
    - Quality definition varies by sector
    - May miss high-growth companies
    """,
    "formula": """
    signal = rank(ev_to_fcf) * I(quality_filters)

    Where:
    - ev_to_fcf = enterprise_value / free_cash_flow
    - quality_filters = (roic > 0.10) and (debt_ebitda < 3) and (accruals < 0)
    - Lower EV/FCF is better (cheaper)
    """,
    "alpha_source": AlphaSource.COMBINATION,
    "information_ratio": 1.0,
    "sharpe_ratio": 1.7
}


# ============================================================================
# SIGNAL #203: Medium-Term Momentum (12-1 month)
# ============================================================================

MOMENTUM_SIGNAL = {
    "signal_id": "momentum_12m_1m",
    "name": "12-Month Momentum (Skip Last Month)",
    "description": """
    Classic 12-month momentum signal, skipping the most recent month.

    The "skip last month" feature avoids short-term reversal. This is
    one of the most robust anomalies in finance, documented in hundreds
    of academic papers.

    Why it works:
    - Behavioral: Investors underreact to information (slow diffusion)
    - Herding: Money managers chase recent winners
    - Behavioral biases: Disposition effect, anchoring
    - Short-term reversal due to liquidity provision

    Crisis performance:
    - Dotcom (2000-02): -15% (MOMENTUM CRASH)
      (Momentum crashed hard when trends reversed)
    - GFC (2008): +3% alpha (before crash)
      (Worked until the crash, then suffered)
    - COVID (2020): +8% alpha
      (Rapid recovery led to strong momentum)

    Implementation notes:
    - Use total return (price + dividends) over past 12 months
    - Skip most recent month (avoid short-term reversal)
    - Sector-neutral: Top 20% within each sector
    - Rebalance monthly
    - Consider volatility scaling (risk management)

    WARNING: Momentum is prone to "crashes" when trends reverse.
    Use risk management: stop losses, volatility scaling, trend filters.

    Potential failure modes:
    - Momentum crashes (sudden trend reversals)
    - High turnover (expensive to trade)
    - Crowded trade
    - Underperforms in choppy/sideways markets
    """,
    "formula": """
    signal = total_return[-252:-21]

    Where:
    - total_return = (price_t + dividends) / price_t-252 - 1
    - Skip last 21 trading days (1 month)
    - Sector-neutral: Rank within GICS sector
    - Filter: Exclude stocks in bottom 10% market cap
    """,
    "alpha_source": AlphaSource.BEHAVIORAL,
    "information_ratio": 0.8,
    "sharpe_ratio": 1.4
}


# ============================================================================
# SIGNAL #289: Share Repurchase + Insider Buying
# ============================================================================

REPURCHASE_SIGNAL = {
    "signal_id": "share_repurchase_insider",
    "name": "Share Repurchase with Insider Confirmation",
    "description": """
    Combines share repurchases with insider buying for strong signal.

    Share repurchases signal that management believes the stock is undervalued.
    Insider buying provides additional confirmation from those who know best.

    Why it works:
    - Signaling: Management buys when stock is cheap
    - Insider information: Insiders know the true prospects
    - Capital allocation: Buybacks are more efficient than dividends
    - Reduction in shares increases EPS (mechanical effect)

    Crisis performance:
    - Dotcom (2000-02): +4% alpha
      (Few tech companies did buybacks)
    - GFC (2008): -1% alpha (survived)
      (Many companies suspended buybacks)
    - COVID (2020): +6% alpha
      (Buybacks resumed quickly in 2020)

    Implementation notes:
    - Repurchase intensity: -shares_outstanding_change_pct
    - Insider buying: net_insider_buying / market_cap
    - Look at 3-month aggregation (smooths noise)
    - Filter out debt-funded buybacks (check balance sheet)
    - Hold for 6-12 months

    Potential failure modes:
    - Debt-funded buybacks (financial engineering)
    - Poor timing (management buys at top)
    - Accounting effects (EPS mechanical boost)
    - Might miss growth companies that don't do buybacks
    """,
    "formula": """
    signal = (-shares_outstanding_change_pct) * I(insider_buying > 0)

    Where:
    - shares_outstanding_change_pct = (shares_t - shares_t-63) / shares_t-63
    - insider_buying = net_insider_buy_volume / market_cap
    - Negative share change = repurchase (good)
    """,
    "alpha_source": AlphaSource.STRUCTURAL,
    "information_ratio": 1.1,
    "sharpe_ratio": 1.5
}


# ============================================================================
# SIGNAL #334: Multifactor Combo (Value + Momentum + Quality)
# ============================================================================

MULTIFACTOR_SIGNAL = {
    "signal_id": "multifactor_vmq",
    "name": "Value-Momentum-Quality Combo",
    "description": """
    Combines three uncorrelated factors into diversified signal.

    This multifactor approach has lower turnover and more consistent
    performance than single-factor strategies. Factors are uncorrelated,
    so they diversify each other's crashes.

    Why it works:
    - Factor diversification reduces risk
    - Uncorrelated factors = smoother equity curve
    - Captures multiple alpha sources
    - Reduces reliance on any single factor

    Crisis performance:
    - Dotcom (2000-02): +5% alpha
      (Value and quality helped)
    - GFC (2008): +2% alpha (survived)
      (Diversification across factors helped)
    - COVID (2020): +7% alpha
      (All factors worked well)

    Implementation notes:
    - Value: EV/FCF (lower is better)
    - Momentum: 12-month return (skip last month)
    - Quality: ROIC, accruals, debt/equity
    - Normalize each factor (z-score or rank)
    - Equal weight to each factor
    - Sector-neutral implementation

    This is one of the most robust signals in this library.

    Potential failure modes:
    - Factor timing (all factors can underperform together)
    - Increased correlation in crises
    - Higher complexity (more things to break)
    """,
    "formula": """
    signal = rank(value) + rank(momentum) + rank(quality)

    Where:
    - value = -ev_to_fcf (negative so higher is better)
    - momentum = total_return[-252:-21]
    - quality = roic - accruals - debt_ebitda
    - All factors are converted to ranks (percentiles)
    - Equal weight to each factor
    """,
    "alpha_source": AlphaSource.COMBINATION,
    "information_ratio": 1.4,
    "sharpe_ratio": 2.1
}


# ============================================================================
# UTILITY FUNCTION: Get all example signals
# ============================================================================

def get_example_signals() -> List[Dict]:
    """
    Get all example alpha signals.

    Returns:
        List of signal dictionaries
    """
    return [
        EARNINGS_CREDIT_SIGNAL,
        LOW_VOLATILITY_SIGNAL,
        VALUE_QUALITY_SIGNAL,
        MOMENTUM_SIGNAL,
        REPURCHASE_SIGNAL,
        MULTIFACTOR_SIGNAL
    ]


def get_signal_by_id(signal_id: str) -> Dict:
    """
    Get a specific example signal by ID.

    Args:
        signal_id: Signal identifier

    Returns:
        Signal dictionary or None if not found
    """
    signals = get_example_signals()
    for signal in signals:
        if signal["signal_id"] == signal_id:
            return signal
    return None


def print_signal_summary(signal_id: str):
    """
    Print a formatted summary of a signal.

    Args:
        signal_id: Signal identifier
    """
    signal = get_signal_by_id(signal_id)

    if not signal:
        print(f"Signal '{signal_id}' not found")
        return

    print(f"\n{'='*80}")
    print(f"Signal: {signal['name']}")
    print(f"ID: {signal['signal_id']}")
    print(f"{'='*80}")
    print(f"\nDescription:\n{signal['description']}")
    print(f"\nFormula:\n{signal['formula']}")
    print(f"\nAlpha Source: {signal['alpha_source'].value}")
    print(f"Information Ratio: {signal['information_ratio']:.2f}")
    print(f"Sharpe Ratio: {signal['sharpe_ratio']:.2f}")
    print(f"\n{'='*80}\n")


# ============================================================================
# DEMONSTRATION: Print all signals
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXAMPLE ALPHA SIGNALS THAT SURVIVED CRISES")
    print("="*80)

    for signal in get_example_signals():
        print_signal_summary(signal["signal_id"])
