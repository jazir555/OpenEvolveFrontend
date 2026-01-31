#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investment Committee Agent Demo

Demonstrates the autonomous investment committee agent running multiple
weekly cycles with learning from outcomes.
"""

import asyncio
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openevolve.agents.investment_committee import (
    InvestmentCommitteeAgent,
    PortfolioState
)


class MockMarketDataProvider:
    """Mock market data provider for demonstration."""

    def __init__(self):
        self.current_week = 0

    async def get_current_state(self, tickers):
        """Get current market state."""
        # Simulate changing market conditions
        self.current_week += 1

        # Cycle through different market regimes
        regime = self.current_week % 4

        if regime == 0:  # Bull market
            return {
                "fundamentals": {
                    ticker: {
                        "pe_ratio": 25.0 + random.uniform(-2, 2),
                        "earnings_growth": 0.15 + random.uniform(-0.03, 0.05)
                    }
                    for ticker in tickers
                },
                "technical": {
                    "market_momentum": 0.05 + random.uniform(0.0, 0.05),
                    "volatility_regime": "low"
                },
                "macro": {
                    "interest_rate": 0.025 + random.uniform(-0.005, 0.005),
                    "inflation": 0.025 + random.uniform(-0.005, 0.005),
                    "gdp_growth": 0.03 + random.uniform(-0.005, 0.01)
                },
                "sentiment": {
                    "market_sentiment": "positive"
                }
            }

        elif regime == 1:  # Normal market
            return {
                "fundamentals": {
                    ticker: {
                        "pe_ratio": 20.0 + random.uniform(-2, 2),
                        "earnings_growth": 0.10 + random.uniform(-0.02, 0.02)
                    }
                    for ticker in tickers
                },
                "technical": {
                    "market_momentum": 0.0 + random.uniform(-0.02, 0.02),
                    "volatility_regime": "normal"
                },
                "macro": {
                    "interest_rate": 0.035 + random.uniform(-0.005, 0.005),
                    "inflation": 0.030 + random.uniform(-0.005, 0.005),
                    "gdp_growth": 0.025 + random.uniform(-0.005, 0.005)
                },
                "sentiment": {
                    "market_sentiment": "neutral"
                }
            }

        elif regime == 2:  # Bear market
            return {
                "fundamentals": {
                    ticker: {
                        "pe_ratio": 15.0 + random.uniform(-2, 2),
                        "earnings_growth": -0.05 + random.uniform(-0.05, 0.03)
                    }
                    for ticker in tickers
                },
                "technical": {
                    "market_momentum": -0.10 + random.uniform(-0.05, 0.0),
                    "volatility_regime": "high"
                },
                "macro": {
                    "interest_rate": 0.045 + random.uniform(-0.005, 0.005),
                    "inflation": 0.040 + random.uniform(-0.005, 0.005),
                    "gdp_growth": 0.01 + random.uniform(-0.01, 0.005)
                },
                "sentiment": {
                    "market_sentiment": "negative"
                }
            }

        else:  # High volatility
            return {
                "fundamentals": {
                    ticker: {
                        "pe_ratio": 20.0 + random.uniform(-5, 5),
                        "earnings_growth": 0.05 + random.uniform(-0.10, 0.10)
                    }
                    for ticker in tickers
                },
                "technical": {
                    "market_momentum": 0.0 + random.uniform(-0.10, 0.10),
                    "volatility_regime": "very high"
                },
                "macro": {
                    "interest_rate": 0.035 + random.uniform(-0.01, 0.01),
                    "inflation": 0.035 + random.uniform(-0.01, 0.01),
                    "gdp_growth": 0.02 + random.uniform(-0.01, 0.01)
                },
                "sentiment": {
                    "market_sentiment": "uncertain"
                }
            }

    async def get_historical_data(self, tickers, period="1y"):
        """Get historical market data."""
        # Simulate historical returns
        num_days = 252  # One trading year

        # Generate returns with some drift and volatility
        returns = []
        for _ in range(num_days):
            daily_return = random.gauss(0.0003, 0.012)  # ~7.5% annual return, ~19% vol
            returns.append(daily_return)

        return {
            "period": period,
            "returns": returns,
            "num_observations": num_days
        }


async def run_investment_committee_demo():
    """Run the investment committee demo."""
    print("=" * 80)
    print("AUTONOMOUS INVESTMENT COMMITTEE AGENT DEMO")
    print("=" * 80)
    print()

    # Create portfolio
    portfolio = PortfolioState(
        holdings={"AAPL": 100, "MSFT": 50, "GOOGL": 30, "AMZN": 20, "TSLA": 10},
        cash=15000.0,
        total_value=75000.0,
        last_rebalance=datetime.utcnow() - timedelta(days=10)
    )

    print(f"Initial Portfolio:")
    print(f"  Holdings: {portfolio.holdings}")
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Total Value: ${portfolio.total_value:,.2f}")
    print()

    # Create market data provider
    market_data = MockMarketDataProvider()

    # Create database path
    db_path = Path("./demo_investment_db")

    # Initialize agent
    print("Initializing Investment Committee Agent...")
    agent = InvestmentCommitteeAgent(
        portfolio_state=portfolio,
        market_data_provider=market_data,
        database_path=db_path,
        risk_tolerance=0.15,
        max_position_size=0.25,
        rebalance_threshold=0.05,
        review_frequency_days=7,
        enable_loongflow=False  # Disabled for demo speed
    )
    print("Agent initialized.")
    print()

    # Run multiple weekly cycles
    num_weeks = 8
    previous_decision_id = None

    for week in range(num_weeks):
        print("=" * 80)
        print(f"WEEK {week + 1} - {datetime.utcnow().strftime('%Y-%m-%d')}")
        print("=" * 80)
        print()

        # Check if review is needed
        if not agent.should_review():
            print("Review not yet due. Skipping.")
            print()
            continue

        # Run weekly review cycle
        print("Running weekly review cycle...")
        decision = await agent.weekly_review_cycle()
        print()

        # Display decision
        print(f"DECISION TYPE: {decision.decision_type.upper()}")
        print(f"CONFIDENCE: {decision.confidence:.2%}")
        print(f"REASONING: {decision.reasoning}")
        print()

        if decision.actions:
            print("RECOMMENDED ACTIONS:")
            for i, action in enumerate(decision.get("actions", []) if isinstance(decision, dict) else decision.actions, 1):
                if isinstance(action, dict):
                    act = action.get("action", "N/A")
                    ticker = action.get("ticker", "N/A")
                    rationale = action.get("rationale", "")
                    print(f"  {i}. {ticker}: {act}")
                    if rationale:
                        print(f"     Rationale: {rationale}")
        else:
            print("No actions recommended - maintain current positions.")

        print()

        # Record outcome from previous week
        if previous_decision_id and week > 0:
            # Simulate outcome
            outcome_type = random.choice(["positive", "positive", "positive", "negative"])
            return_amount = random.uniform(-0.03, 0.06)

            outcome = f"{outcome_type} return of {abs(return_amount):.2%}"
            performance = {
                "return": return_amount,
                "volatility": random.uniform(0.10, 0.20),
                "sharpe": (return_amount - 0.02) / random.uniform(0.10, 0.20)
            }

            print(f"Recording outcome for previous decision:")
            print(f"  Outcome: {outcome}")
            print(f"  Performance: {performance}")
            print()

            await agent.record_outcome(
                previous_decision_id,
                actual_outcome=outcome,
                performance_metrics=performance
            )

        previous_decision_id = decision.decision_id

        # Display performance summary
        summary = agent.get_performance_summary()
        print(f"PERFORMANCE SUMMARY:")
        print(f"  Total Decisions: {summary['total_decisions']}")
        print(f"  Decisions with Outcomes: {summary['decisions_with_outcomes']}")
        print(f"  Average Confidence: {summary['average_confidence']:.2%}")

        if summary['accuracy']:
            print(f"  Accuracy: {summary['accuracy']:.2%}")

        print()

        # Display learned knowledge
        knowledge = agent.knowledge_integrator.get_knowledge_summary()
        print(f"LEARNING PROGRESS:")
        print(f"  Causal Factors: {knowledge['total_causal_factors']}")
        print(f"  Heuristics: {knowledge['total_heuristics']}")
        print(f"  Lessons Learned: {knowledge['total_lessons']}")
        print(f"  Scenarios Stored: {knowledge['total_scenarios']}")

        if knowledge.get('top_predictive_factors'):
            print()
            print("  TOP PREDICTIVE FACTORS:")
            for factor in knowledge['top_predictive_factors'][:3]:
                print(f"    - {factor['name']}: {factor['predictive_power']:.2%} predictive power")

        print()

        # Small delay for readability
        await asyncio.sleep(0.5)

    # Final summary
    print("=" * 80)
    print("DEMO COMPLETE - FINAL SUMMARY")
    print("=" * 80)
    print()

    summary = agent.get_performance_summary()
    print(f"Final Performance:")
    print(f"  Total Decisions: {summary['total_decisions']}")
    print(f"  Accuracy: {summary.get('accuracy', 'N/A')}")
    print()

    knowledge = agent.knowledge_integrator.get_knowledge_summary()
    print(f"Knowledge Acquired:")
    print(f"  Total Causal Factors: {knowledge['total_causal_factors']}")
    print(f"  Total Heuristics: {knowledge['total_heuristics']}")
    print(f"  Total Lessons: {knowledge['total_lessons']}")
    print()

    # Get recent lessons
    recent_lessons = agent.knowledge_integrator.get_recent_lessons(days=30)
    if recent_lessons:
        print(f"Recent Lessons Learned (Last 30 days):")
        for lesson in recent_lessons[:3]:
            print(f"  - {lesson.lesson}")

    print()

    print(f"Database saved to: {db_path}")
    print()
    print("Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_investment_committee_demo())
