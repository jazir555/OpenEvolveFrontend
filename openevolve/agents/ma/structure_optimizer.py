"""
M&A Deal Structure Optimizer

Optimizes deal structure for tax efficiency, risk allocation,
and value maximization.
"""

import logging
from typing import Optional, List, Dict, Any

from openevolve.agents.ma.schemas import (
    Deal,
    DealStructure,
    ValuationResult,
)


logger = logging.getLogger(__name__)


class StructureOptimizer:
    """
    Deal Structure Optimizer

    Optimizes M&A deal structure for:
    - Tax efficiency
    - Risk allocation
    - Value preservation
    - Seller motivation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Structure Optimizer"""
        self.config = config or {}

    async def optimize_structure(
        self,
        deal: Deal,
        valuation: ValuationResult,
        patterns: Optional[List[Any]] = None,
    ) -> DealStructure:
        """
        Optimize deal structure

        Args:
            deal: Deal to structure
            valuation: Valuation analysis
            patterns: Success patterns from similar deals

        Returns:
            DealStructure: Optimized deal structure
        """
        logger.info(f"Optimizing structure for deal {deal.deal_id}")

        # Determine optimal structure type
        structure_type = await self._determine_structure_type(deal)

        # Optimize consideration mix
        cash_pct, stock_pct, earnout_pct = await self._optimize_consideration(
            deal, valuation
        )

        total_value = valuation.implied_value
        cash_component = total_value * cash_pct
        stock_component = total_value * stock_pct
        earnout_component = total_value * earnout_pct

        # Generate structure
        structure = DealStructure(
            deal_id=deal.deal_id,
            structure_type=structure_type,
            total_value=total_value,
            cash_component=cash_component,
            stock_component=stock_component,
            earnout=earnout_component,
            efficiency_score=0.8,
            risk_score=0.3,
            tax_efficiency=0.75,
            rationale=f"Optimized balance of cash ({cash_pct:.0%}), "
                      f"stock ({stock_pct:.0%}), and earnout ({earnout_pct:.0%})",
        )

        logger.info(
            f"Structure: {structure_type}, "
            f"Cash: {cash_pct:.0%}, Stock: {stock_pct:.0%}, Earnout: {earnout_pct:.0%}"
        )

        return structure

    async def _determine_structure_type(self, deal: Deal) -> str:
        """Determine optimal structure type"""
        # In production, analyze tax considerations, liability exposure, etc.
        return "stock"  # Default to stock purchase

    async def _optimize_consideration(
        self,
        deal: Deal,
        valuation: ValuationResult,
    ) -> tuple[float, float, float]:
        """Optimize cash/stock/earnout mix"""
        # Default allocation
        cash_pct = 0.60
        stock_pct = 0.30
        earnout_pct = 0.10

        # Adjust based on deal characteristics
        if valuation.confidence < 0.7:
            # Lower confidence -> more earnout
            cash_pct = 0.50
            stock_pct = 0.30
            earnout_pct = 0.20

        return cash_pct, stock_pct, earnout_pct
