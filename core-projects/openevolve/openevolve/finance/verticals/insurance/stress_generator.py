"""
Stress Scenario Generator for Insurance Portfolios

Generates regulatory stress test scenarios for insurance reserve portfolios.
Scenarios are based on historical crises and regulatory requirements.

Scenarios include:
- Historical: 2008 GFC, 2020 COVID, 2000 Dot-com, 1994 Bond massacre
- Rate Shocks: +/- 300bps parallel shifts
- Credit Events: Downgrade cascades, default surges
- Insurance-Specific: Mortality surges, natural catastrophes

Each scenario specifies:
- Shock type and magnitude
- Duration of shock
- Expected impact on asset classes
- Correlation assumptions

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .models import StressScenario


class CrisisType(Enum):
    """Types of crisis scenarios"""
    FINANCIAL_CRISIS = "financial_crisis"
    PANDEMIC = "pandemic"
    RATE_SHOCK = "rate_shock"
    CREDIT_EVENT = "credit_event"
    INSURANCE_CATASTROPHE = "insurance_catastrophe"
    COMPOUNDED_CRISIS = "compounded_crisis"


@dataclass
class HistoricalCrises:
    """Historical crisis parameters"""
    gfc_2008: Dict[str, Any]
    covid_2020: Dict[str, Any]
    dotcom_2000: Dict[str, Any]
    bond_massacre_1994: Dict[str, Any]


@dataclass
class RateShockScenario:
    """Interest rate shock parameters"""
    magnitude_bps: int
    duration_months: int
    curve_flattening: bool


class StressScenarioGenerator:
    """
    Generate stress test scenarios for insurance portfolios.

    This class creates realistic stress scenarios based on historical events
    and regulatory requirements. Scenarios are designed to test portfolio
    resilience under extreme conditions.

    Example:
        >>> generator = StressScenarioGenerator()
        >>> scenarios = generator.generate_all_scenarios()
        >>> for scenario in scenarios:
        ...     print(f"{scenario.name}: {scenario.description}")
        ...     print(f"  Shocks: {scenario.shocks}")
    """

    def __init__(self):
        """Initialize stress scenario generator"""
        # Historical crisis data
        self.historical_crises = HistoricalCrises(
            gfc_2008={
                "equities": -0.52,  # S&P 500 dropped 52%
                "corporate_bonds_oas": +650,  # Spreads widened 650bps
                "treasury_yields": -180,  # Flight to safety
                "defaults": {
                    "aaa": 0.002,
                    "aa": 0.008,
                    "a": 0.025,
                    "bbb": 0.095,
                    "bb": 0.255,
                    "b": 0.455
                },
                "duration_months": 18
            },
            covid_2020={
                "equities": -0.34,  # S&P dropped 34%
                "corporate_bonds_oas": +450,
                "treasury_yields": -120,
                "mortality_rate": +0.15,  # 15% excess mortality
                "duration_months": 12
            },
            dotcom_2000={
                "equities": -0.49,  # Nasdaq dropped 49%
                "corporate_bonds_oas": +250,
                "treasury_yields": -80,
                "duration_months": 30
            },
            bond_massacre_1994={
                "treasury_yields": +250,  # Rapid rate hike
                "corporate_bonds_oas": +100,
                "duration_months": 9
            }
        )

    def gfc_plus_covid(self) -> StressScenario:
        """
        Generate compounded crisis: 2008 GFC + COVID.

        Combines the worst aspects of both crises:
        - GFC: Corporate bond defaults, equity crash, rate cuts
        - COVID: Sudden stop, credit spreads spike, mortality surge

        Returns:
            StressScenario representing compounded crisis

        Example:
            >>> scenario = generator.gfc_plus_covid()
            >>> print(f"Equity shock: {scenario.shocks['equities']}")
            >>> print(f"Spread shock: {scenario.shocks['corporate_bonds_oas']}bps")
        """
        # Combine GFC and COVID shocks
        gfc = self.historical_crises.gfc_2008
        covid = self.historical_crises.covid_2020

        return StressScenario(
            name="gfc_plus_covid",
            description="2008 GFC combined with COVID-19 sudden stop - compounded crisis scenario",
            duration_months=24,
            shocks={
                # Equity: Take worst of both
                "equities": -0.55,  # 55% drop (worse than either alone)

                # Credit: Widespread downgrade cascade
                "corporate_bonds_oas": +700,  # Spreads +700bps

                # Rates: Flight to safety
                "treasury_yields": -200,  # -200bps

                # Insurance: Mortality surge from COVID
                "mortality_rate": +0.20,  # 20% excess deaths

                # Defaults: Elevated across all ratings
                "defaults": {
                    "aaa": 0.003,  # Even AAA see defaults
                    "aa": 0.010,
                    "a": 0.035,
                    "bbb": 0.120,  # 12% BBB default rate
                    "bb": 0.300,   # 30% BB default rate
                    "b": 0.500     # 50% B default rate
                }
            },
            correlations={
                # High correlations in crisis
                "equities_corporate_bonds": 0.95,  # Near-perfect correlation
                "treasuries_corporate_bonds": -0.85,  # Strong flight to quality
                "equities_mortality": 0.60,  # Economic stress -> mortality
            },
            probability=0.005  # 0.5% annual probability (very rare)
        )

    def rate_shock_up(self) -> StressScenario:
        """
        Generate upward interest rate shock scenario.

        Tests portfolio sensitivity to rapid rate increases.
        Uses 300bps parallel shift over 3 months.

        Returns:
            StressScenario with upward rate shock

        Example:
            >>> scenario = generator.rate_shock_up()
            >>> print(f"Rate shock: +{scenario.shocks['treasury_yield_curve']}bps")
        """
        return StressScenario(
            name="rate_shock_up_300bps",
            description="+300bps parallel shift over 3 months (rapid tightening)",
            duration_months=3,
            shocks={
                "treasury_yield_curve": +300,  # +300bps
                "corporate_spreads": +75,  # Spreads widen slightly
                "mortgage_rates": +300,
                "duration_impact": "convexity_adjustment",
                # Rate shocks increase default risk for highly leveraged companies
                "defaults": {
                    "aaa": 0.001,
                    "aa": 0.003,
                    "a": 0.010,
                    "bbb": 0.035,
                    "bb": 0.080,
                    "b": 0.150
                }
            },
            correlations={
                "treasuries_corporate_bonds": 0.7,  # High correlation
                "duration_impact": "nonlinear"  # Convexity effects
            },
            probability=0.02  # 2% annual probability
        )

    def rate_shock_down(self) -> StressScenario:
        """
        Generate downward interest rate shock scenario.

        Tests portfolio sensitivity to rapid rate decreases.
        Uses -300bps parallel shift over 3 months.

        Returns:
            StressScenario with downward rate shock

        Example:
            >>> scenario = generator.rate_shock_down()
            >>> print(f"Rate shock: {scenario.shocks['treasury_yield_curve']}bps")
        """
        return StressScenario(
            name="rate_shock_down_300bps",
            description="-300bps parallel shift over 3 months (rapid easing)",
            duration_months=3,
            shocks={
                "treasury_yield_curve": -300,  # -300bps
                "corporate_spreads": +50,  # Spreads still widen on volatility
                "mortgage_rates": -300,
                "duration_impact": "positive_convexity",
                # Falling rates can signal economic weakness
                "defaults": {
                    "aaa": 0.001,
                    "aa": 0.004,
                    "a": 0.012,
                    "bbb": 0.040,
                    "bb": 0.090,
                    "b": 0.160
                }
            },
            correlations={
                "treasuries_corporate_bonds": 0.6,
            },
            probability=0.02
        )

    def credit_downgrade_cascade(self) -> StressScenario:
        """
        Generate credit downgrade cascade scenario.

        Simulates a mass downgrade event where large numbers of bonds
        are downgraded simultaneously, causing selling pressure and
        spread widening.

        Returns:
            StressScenario with credit cascade

        Example:
            >>> scenario = generator.credit_downgrade_cascade()
            >>> print(f"Spread impact: {scenario.shocks['corporate_bonds_oas']}bps")
        """
        return StressScenario(
            name="credit_downgrade_cascade",
            description="Mass downgrade cascade with 50% of BBB downgraded to BB",
            duration_months=12,
            shocks={
                "corporate_bonds_oas": +500,  # Massive spread widening
                "downgrade_rate": {
                    "aaa_to_aa": 0.05,  # 5% of AAA downgraded
                    "aa_to_a": 0.10,
                    "a_to_bbb": 0.20,
                    "bbb_to_bb": 0.50,  # 50% of BBB downgraded!
                    "bb_to_b": 0.40,
                    "b_to_ccc": 0.30
                },
                "liquidity_premium": +200,  # Liquidity dries up
                "defaults": {
                    "aaa": 0.002,
                    "aa": 0.006,
                    "a": 0.018,
                    "bbb": 0.070,
                    "bb": 0.220,
                    "b": 0.420
                }
            },
            correlations={
                "spread_duration": 0.8,  # Longer duration = more spread pain
                "liquidity_spreads": 0.9  # Liquidity and spreads tightly linked
            },
            probability=0.01  # 1% annual probability
        )

    def mortality_surge(self) -> StressScenario:
        """
        Generate mortality surge scenario.

        Insurance-specific scenario testing excess mortality from:
        - Pandemic
        - Natural disasters
        - Other mass casualty events

        Returns:
            StressScenario with mortality surge

        Example:
            >>> scenario = generator.mortality_surge()
            >>> print(f"Mortality shock: +{scenario.shocks['mortality_rate']*100}%")
        """
        return StressScenario(
            name="mortality_surge_20pct",
            description="20% excess mortality from pandemic or catastrophe",
            duration_months=18,
            shocks={
                "mortality_rate": +0.20,  # 20% excess deaths
                # Mortality surges increase claims immediately
                "liability_surge": 1.20,  # Liabilities increase 20%
                # Market impact from pandemic
                "equities": -0.25,
                "corporate_bonds_oas": +200,
                "treasury_yields": -100,
            },
            correlations={
                "mortality_markets": 0.7,  # Mortality events hurt markets
            },
            probability=0.01
        )

    def natural_catastrophe(self) -> StressScenario:
        """
        Generate natural catastrophe scenario.

        Tests impact of major natural disasters:
        - Hurricanes
        - Earthquakes
        - Wildfires
        - Floods

        Returns:
            StressScenario with natural catastrophe

        Example:
            >>> scenario = generator.natural_catastrophe()
            >>> print(f"Claims surge: {scenario.shocks['claims_surge']*100}%")
        """
        return StressScenario(
            name="natural_catastrophe",
            description="Major natural catastrophe causing $100B+ in insured losses",
            duration_months=6,
            shocks={
                "claims_surge": 1.30,  # Liabilities increase 30%
                "regional_impact": {
                    "hurricane": 0.50,  # 50% of losses from hurricanes
                    "earthquake": 0.25,
                    "wildfire": 0.15,
                    "flood": 0.10
                },
                # Catastrophes don't usually impact markets much
                "equities": -0.05,
                "corporate_bonds_oas": +50,
            },
            correlations={
                "regional_concentration": 0.9,  # Losses concentrated by region
            },
            probability=0.03  # 3% annual probability
        )

    def generate_all_scenarios(self) -> List[StressScenario]:
        """
        Generate all standard stress test scenarios.

        Returns:
            List of all stress scenarios

        Example:
            >>> scenarios = generator.generate_all_scenarios()
            >>> print(f"Generated {len(scenarios)} scenarios")
        """
        return [
            self.gfc_plus_covid(),
            self.rate_shock_up(),
            self.rate_shock_down(),
            self.credit_downgrade_cascade(),
            self.mortality_surge(),
            self.natural_catastrophe()
        ]

    def generate_custom_scenario(
        self,
        name: str,
        description: str,
        equity_shock: Optional[float] = None,
        spread_shock_bps: Optional[int] = None,
        rate_shock_bps: Optional[int] = None,
        mortality_shock: Optional[float] = None,
        custom_shocks: Optional[Dict[str, Any]] = None,
        duration_months: int = 12
    ) -> StressScenario:
        """
        Generate custom stress scenario.

        Args:
            name: Scenario name
            description: Scenario description
            equity_shock: Equity market shock (decimal, e.g., -0.30 for -30%)
            spread_shock_bps: Credit spread shock in basis points
            rate_shock_bps: Interest rate shock in basis points
            mortality_shock: Mortality rate increase (decimal)
            custom_shocks: Dictionary of custom shock parameters
            duration_months: Duration of scenario in months

        Returns:
            Custom StressScenario

        Example:
            >>> scenario = generator.generate_custom_scenario(
            ...     name="my_custom_scenario",
            ...     description="Custom stress test",
            ...     equity_shock=-0.25,
            ...     spread_shock_bps=300
            ... )
        """
        shocks = {}

        if equity_shock is not None:
            shocks["equities"] = equity_shock

        if spread_shock_bps is not None:
            shocks["corporate_bonds_oas"] = spread_shock_bps

        if rate_shock_bps is not None:
            shocks["treasury_yields"] = rate_shock_bps

        if mortality_shock is not None:
            shocks["mortality_rate"] = mortality_shock

        if custom_shocks:
            shocks.update(custom_shocks)

        return StressScenario(
            name=name,
            description=description,
            duration_months=duration_months,
            shocks=shocks,
            probability=0.01  # Default probability
        )
