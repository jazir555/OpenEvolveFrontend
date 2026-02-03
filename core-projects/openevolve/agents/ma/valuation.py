"""
M&A Valuation Engine

Performs comprehensive valuation analysis using multiple methods,
scenario analysis, and synergy valuation.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from openevolve.agents.ma.schemas import (
    Deal,
    ValuationResult,
    ValuationMethod,
    Scenario,
    Synergy,
)


logger = logging.getLogger(__name__)


class ValuationEngine:
    """
    Valuation Engine for M&A deals

    Performs comprehensive valuation analysis:
    - Multiple valuation methods (DCF, comps, precedent transactions)
    - Synergy valuation
    - Scenario and sensitivity analysis
    - Risk assessment
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Valuation Engine"""
        self.config = config or {}
        self.market_data = self._load_market_data()

    def _load_market_data(self) -> Dict[str, Any]:
        """Load market data for valuation"""
        return self.config.get("market_data", {})

    async def valuate_deal(
        self,
        deal: Deal,
        similar_deals: Optional[List[Any]] = None,
    ) -> ValuationResult:
        """
        Perform comprehensive valuation analysis

        Args:
            deal: Deal to value
            similar_deals: Similar completed deals for reference

        Returns:
            ValuationResult: Comprehensive valuation analysis
        """
        logger.info(f"Valuing deal {deal.deal_id}")

        methods = []

        # DCF Valuation
        dcf = await self._dcf_valuation(deal)
        methods.append(dcf)

        # Comparable Companies
        comps = await self._comps_valuation(deal)
        methods.append(comps)

        # Precedent Transactions
        precedent = await self._precedent_valuation(deal, similar_deals)
        methods.append(precedent)

        # Asset-based Valuation
        asset = await self._asset_valuation(deal)
        methods.append(asset)

        # Calculate implied value (weighted average)
        weights = {"dcf": 0.4, "comps": 0.3, "precedent": 0.2, "asset": 0.1}
        implied_value = sum(
            m.value * weights.get(m.method, 0.25)
            for m in methods
        )

        # Generate scenarios
        scenarios = await self._generate_scenarios(deal, implied_value)

        # Value synergies
        synergies = await self._value_synergies(deal)
        synergy_value = sum(s.estimated_value for s in synergies)

        # Calculate ranges
        base_case = implied_value
        best_case = implied_value * 1.2
        worst_case = implied_value * 0.8

        result = ValuationResult(
            deal_id=deal.deal_id,
            target_company=deal.target_company.name,
            methods=methods,
            implied_value=implied_value,
            scenarios=scenarios,
            base_case=base_case,
            best_case=best_case,
            worst_case=worst_case,
            identified_synergies=synergies,
            synergy_value=synergy_value,
            valuation_range=(worst_case, best_case),
            confidence=0.75,
        )

        logger.info(
            f"Valuation complete: ${implied_value:.1f}M "
            f"(range: ${worst_case:.1f}M - ${best_case:.1f}M)"
        )

        return result

    async def _dcf_valuation(self, deal: Deal) -> ValuationMethod:
        """Perform DCF valuation"""
        # Simplified DCF calculation
        revenue = deal.target_company.revenue or 100
        growth_rate = deal.target_company.growth_rate or 0.15
        margin = 0.2  # Assumed EBITDA margin

        # 5-year projection
        fcf = revenue * margin
        terminal_growth = 0.03
        discount_rate = 0.12
        terminal_value = fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)

        value = fcf * 3.5 + terminal_value * 0.5  # Simplified PV calculation

        return ValuationMethod(
            method="dcf",
            value=value,
            assumptions=[
                f"Revenue: ${revenue:.1f}M",
                f"Growth rate: {growth_rate:.1%}",
                f"EBITDA margin: {margin:.1%}",
                f"WACC: {discount_rate:.1%}",
                f"Terminal growth: {terminal_growth:.1%}",
            ],
            confidence=0.7,
        )

    async def _comps_valuation(self, deal: Deal) -> ValuationMethod:
        """Perform comparable company analysis"""
        # In production, use real market data
        revenue = deal.target_company.revenue or 100
        ebitda = deal.target_company.ebitda or (revenue * 0.2)

        # Industry multiples (would come from market data)
        ev_revenue_multiple = 3.0
        ev_ebitda_multiple = 12.0

        value_revenue = revenue * ev_revenue_multiple
        value_ebitda = ebitda * ev_ebitda_multiple
        value = (value_revenue + value_ebitda) / 2

        return ValuationMethod(
            method="comps",
            value=value,
            assumptions=[
                f"EV/Revenue: {ev_revenue_multiple:.1f}x",
                f"EV/EBITDA: {ev_ebitda_multiple:.1f}x",
            ],
            confidence=0.75,
        )

    async def _precedent_valuation(
        self,
        deal: Deal,
        similar_deals: Optional[List[Any]],
    ) -> ValuationMethod:
        """Perform precedent transaction analysis"""
        revenue = deal.target_company.revenue or 100

        # Precedent multiples (would come from transaction database)
        premium_range = (0.2, 0.4)  # 20-40% control premium
        premium = sum(premium_range) / 2

        value = revenue * 3.0 * (1 + premium)

        return ValuationMethod(
            method="precedent",
            value=value,
            assumptions=[
                f"Control premium: {premium:.1%}",
                f"Based on similar transactions in {deal.target_company.industry}",
            ],
            confidence=0.65,
        )

    async def _asset_valuation(self, deal: Deal) -> ValuationMethod:
        """Perform asset-based valuation"""
        # Simplified asset valuation
        value = (deal.target_company.ebitda or 20) * 8

        return ValuationMethod(
            method="asset",
            value=value,
            assumptions=[
                "Fair market value of operating assets",
                "Adjusted for liabilities",
            ],
            confidence=0.6,
        )

    async def _generate_scenarios(
        self,
        deal: Deal,
        base_value: float,
    ) -> List[Scenario]:
        """Generate valuation scenarios"""
        return [
            Scenario(
                name="Base Case",
                description="Most likely outcome with current assumptions",
                probability=0.50,
                valuation=base_value,
                key_assumptions={"growth": deal.target_company.growth_rate or 0.15},
            ),
            Scenario(
                name="Upside Case",
                description="Optimistic growth and synergy realization",
                probability=0.25,
                valuation=base_value * 1.3,
                key_assumptions={"growth": (deal.target_company.growth_rate or 0.15) + 0.10},
            ),
            Scenario(
                name="Downside Case",
                description="Conservative growth and integration challenges",
                probability=0.25,
                valuation=base_value * 0.75,
                key_assumptions={"growth": max((deal.target_company.growth_rate or 0.15) - 0.05, 0)},
            ),
        ]

    async def _value_synergies(self, deal: Deal) -> List[Synergy]:
        """Value synergies"""
        synergies = []

        if deal.target_company.strategic_fit:
            return deal.target_company.strategic_fit.synergies

        # Default synergy estimates
        revenue = deal.target_company.revenue or 100
        synergies = [
            Synergy(
                synergy_type="revenue",
                description="Cross-selling and market expansion",
                estimated_value=revenue * 0.10,
                confidence=0.6,
                time_to_realize=18,
            ),
            Synergy(
                synergy_type="cost",
                description="Operational cost reductions",
                estimated_value=revenue * 0.05,
                confidence=0.7,
                time_to_realize=12,
            ),
        ]

        return synergies
