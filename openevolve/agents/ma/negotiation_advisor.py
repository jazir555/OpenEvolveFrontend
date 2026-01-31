"""
M&A Negotiation Advisor

Provides negotiation strategy, BATNA analysis,
and tactical recommendations.
"""

import logging
from typing import Optional, List, Dict, Any

from openevolve.agents.ma.schemas import (
    Deal,
    NegotiationStrategy,
    BATNA,
    NegotiationTactic,
    ValuationResult,
    DealStructure,
)


logger = logging.getLogger(__name__)


class NegotiationAdvisor:
    """
    Negotiation Advisor

    Provides comprehensive negotiation support:
    - BATNA analysis
    - Strategy development
    - Tactical recommendations
    - Game theory insights
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Negotiation Advisor"""
        self.config = config or {}

    async def create_strategy(
        self,
        deal: Deal,
        valuation: Optional[ValuationResult] = None,
        structure: Optional[DealStructure] = None,
    ) -> NegotiationStrategy:
        """
        Create negotiation strategy

        Args:
            deal: Deal context
            valuation: Valuation analysis
            structure: Deal structure

        Returns:
            NegotiationStrategy: Comprehensive negotiation strategy
        """
        logger.info(f"Creating negotiation strategy for deal {deal.deal_id}")

        # Analyze BATNAs
        batna = await self._analyze_our_batna(deal, valuation)
        their_batna = await self._analyze_their_batna(deal)

        # Determine approach
        approach = "collaborative"  # Default to collaborative

        # Define positions
        if valuation:
            opening_position = {"value": valuation.best_case * 0.9}
            target_position = {"value": valuation.implied_value}
            fallback_position = {"value": valuation.base_case * 1.1}
        else:
            opening_position = None
            target_position = None
            fallback_position = None

        # Generate tactics
        tactics = await self._generate_tactics(deal, batna, their_batna)

        # Define key terms
        must_haves = [
            "Key executive retention",
            "Technology IP assignment",
            "Non-compete agreements",
        ]

        nice_to_haves = [
            "Transition services agreement",
            "Earnout based on performance",
        ]

        tradeables = [
            "Employment agreements",
            "Consulting arrangements",
            "Payment timing",
        ]

        # Assess leverage
        leverage_assessment = self._assess_leverage(deal, batna, their_batna)

        strategy = NegotiationStrategy(
            deal_id=deal.deal_id,
            target_company=deal.target_company.name,
            batna=batna,
            their_batna=their_batna,
            approach=approach,
            opening_position=opening_position,
            target_position=target_position,
            fallback_position=fallback_position,
            recommended_tactics=tactics,
            must_haves=must_haves,
            nice_to_haves=nice_to_haves,
            tradeables=tradeables,
            leverage_assessment=leverage_assessment,
        )

        logger.info(f"Negotiation strategy created: {approach} approach")

        return strategy

    async def _analyze_our_batna(
        self,
        deal: Deal,
        valuation: Optional[ValuationResult],
    ) -> BATNA:
        """Analyze our BATNA"""
        value = valuation.implied_value * 0.7 if valuation else 100

        return BATNA(
            description="Pursue alternative acquisition targets in same sector",
            value=value,
            probability=0.6,
            timeline="6-9 months",
            risks=["Market may heat up", "May pay premium later"],
        )

    async def _analyze_their_batna(self, deal: Deal) -> BATNA:
        """Analyze their BATNA"""
        return BATNA(
            description="Remain independent or seek other buyers",
            value=deal.target_company.revenue * 3 if deal.target_company.revenue else 100,
            probability=0.5,
            timeline="12-18 months",
            risks=["Market conditions may deteriorate", "May not find better offer"],
        )

    async def _generate_tactics(
        self,
        deal: Deal,
        batna: BATNA,
        their_batna: BATNA,
    ) -> List[NegotiationTactic]:
        """Generate negotiation tactics"""
        return [
            NegotiationTactic(
                tactic="Build relationship first",
                rationale="Establish trust before discussing numbers",
                timing="Early meetings",
                expected_outcome="More collaborative atmosphere",
                risks=["May delay process"],
            ),
            NegotiationTactic(
                tactic="Anchor with justified range",
                rationale="Set reasonable anchor based on valuation",
                timing="First formal offer",
                expected_outcome="Negotiation within valuation range",
                risks=["May seem too high or too low"],
            ),
        ]

    def _assess_leverage(
        self,
        deal: Deal,
        batna: BATNA,
        their_batna: BATNA,
    ) -> str:
        """Assess negotiation leverage"""
        if batna.value > their_batna.value:
            return "Strong - we have good alternatives"
        elif batna.value < their_batna.value:
            return "Weak - they have more options"
        else:
            return "Balanced - both sides have similar alternatives"
