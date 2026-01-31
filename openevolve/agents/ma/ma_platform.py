"""
M&A Deal Intelligence Platform - Main Orchestrator

Orchestrates the entire M&A deal lifecycle from sourcing through integration.
Coordinates multiple agents and manages deal pipeline.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

# Optional LoongFlow integration
try:
    from loongflow.utils import run_loongflow
    LOONGFLOW_AVAILABLE = True
except ImportError:
    LOONGFLOW_AVAILABLE = False

from openevolve.agents.ma.schemas import (
    Deal,
    DealStage,
    DealPriority,
    Company,
    TargetCompany,
    DiligenceReport,
    ValuationResult,
    DealStructure,
    NegotiationStrategy,
    IntegrationPlan,
    DealOutcome,
)
from openevolve.agents.ma.deal_sourcer import DealSourcer
from openevolve.agents.ma.diligence_assistant import DiligenceAssistant
from openevolve.agents.ma.valuation import ValuationEngine
from openevolve.agents.ma.structure_optimizer import StructureOptimizer
from openevolve.agents.ma.negotiation_advisor import NegotiationAdvisor
from openevolve.agents.ma.integration_planner import IntegrationPlanner
from openevolve.agents.ma.knowledge_manager import DealKnowledgeManager


logger = logging.getLogger(__name__)


class MADealPlatform:
    """
    Main M&A Deal Intelligence Platform Orchestrator

    Manages the complete M&A deal lifecycle:
    - Continuous deal sourcing and screening
    - Due diligence planning and execution
    - Valuation and structure optimization
    - Negotiation strategy
    - Integration planning
    - Continuous learning from outcomes
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_loongflow: bool = True,
        workspace_dir: Optional[str] = None,
    ):
        """
        Initialize the M&A Deal Platform

        Args:
            config: Platform configuration
            use_loongflow: Whether to use LoongFlow for complex planning
            workspace_dir: Working directory for deal data
        """
        self.config = config or {}
        self.use_loongflow = use_loongflow and LOONGFLOW_AVAILABLE
        self.workspace_dir = Path(workspace_dir or "./ma_workspace")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Deal pipeline
        self.deals: Dict[str, Deal] = {}
        self.deal_pipeline: Dict[DealStage, List[str]] = {
            stage: [] for stage in DealStage
        }

        # Initialize agents
        self.sourcer = DealSourcer(config=self.config.get("sourcer"))
        self.diligence = DiligenceAssistant(config=self.config.get("diligence"))
        self.valuation = ValuationEngine(config=self.config.get("valuation"))
        self.structure_optimizer = StructureOptimizer(
            config=self.config.get("structure")
        )
        self.negotiation = NegotiationAdvisor(config=self.config.get("negotiation"))
        self.integration = IntegrationPlanner(config=self.config.get("integration"))
        self.knowledge = DealKnowledgeManager(config=self.config.get("knowledge"))

        logger.info(f"Initialized M&A Deal Platform (LoongFlow: {self.use_loongflow})")

    async def start_continuous_sourcing(self) -> None:
        """
        Start continuous market scanning for potential targets
        """
        logger.info("Starting continuous deal sourcing")

        while True:
            try:
                # Scan market for targets
                targets = await self.sourcer.scan_market()

                # Screen and prioritize targets
                for target in targets:
                    if target.company_id not in self.deals:
                        # Analyze strategic fit
                        target.strategic_fit = await self.sourcer.analyze_strategic_fit(
                            target
                        )

                        # Create initial deal record
                        deal = Deal(
                            deal_id=f"deal_{target.company_id}",
                            target_company=target,
                            stage=DealStage.SOURCING,
                            priority=target.priority,
                        )

                        self.deals[deal.deal_id] = deal
                        self.deal_pipeline[DealStage.SOURCING].append(deal.deal_id)

                        logger.info(
                            f"New target identified: {target.name} "
                            f"(Priority: {target.priority})"
                        )

                # Prioritize pipeline
                await self._prioritize_pipeline()

                # Sleep before next scan
                await asyncio.sleep(3600)  # Scan every hour

            except Exception as e:
                logger.error(f"Error in continuous sourcing: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def initiate_diligence(
        self,
        deal_id: str,
        diligence_depth: str = "comprehensive",
    ) -> DiligenceReport:
        """
        Initiate due diligence phase for a deal

        Args:
            deal_id: Deal identifier
            diligence_depth: Depth of diligence (quick, standard, comprehensive)

        Returns:
            DiligenceReport: Generated diligence report
        """
        if deal_id not in self.deals:
            raise ValueError(f"Deal {deal_id} not found")

        deal = self.deals[deal_id]
        deal.stage = DealStage.DILIGENCE
        deal.updated_at = datetime.utcnow()
        self._update_pipeline(deal_id, DealStage.DILIGENCE)

        logger.info(f"Initiating diligence for deal {deal_id}")

        if self.use_loongflow:
            # Use LoongFlow to plan diligence approach
            return await self._plan_diligence_with_loongflow(
                deal, diligence_depth
            )
        else:
            # Use standard diligence planning
            return await self._plan_diligence_standard(deal, diligence_depth)

    async def _plan_diligence_with_loongflow(
        self,
        deal: Deal,
        diligence_depth: str,
    ) -> DiligenceReport:
        """Plan diligence using LoongFlow evolutionary agent"""
        logger.info(f"Planning diligence with LoongFlow for deal {deal.deal_id}")

        # Define diligence planning task
        task_prompt = f"""
        Plan comprehensive due diligence for acquisition of {deal.target_company.name}.

        Target Company:
        - Industry: {deal.target_company.industry}
        - Sector: {deal.target_company.sector}
        - Revenue: ${deal.target_company.revenue:, if deal.target_company.revenue else 'N/A'}
        - Employees: {deal.target_company.employees or 'N/A'}

        Deal Context:
        - Priority: {deal.priority}
        - Deal Size: ${deal.deal_size:, if deal.deal_size else 'TBD'}
        - Depth: {diligence_depth}

        Create a comprehensive due diligence plan covering:
        1. Financial analysis
        2. Legal review
        3. Commercial assessment
        4. Operational evaluation
        5. Technology audit
        6. Human resources review
        7. Tax analysis

        Generate detailed checklist items for each category with:
        - Specific items to investigate
        - Key documents to request
        - Risks to assess
        - Red flags to watch for
        """

        try:
            # Use LoongFlow to evolve diligence plan
            plan_result = await run_loongflow(
                task_description=task_prompt,
                workspace=str(self.workspace_dir / deal.deal_id / "diligence_plan"),
                max_iterations=10,
            )

            # Execute diligence with evolved plan
            report = await self.diligence.execute_diligence(
                deal=deal,
                plan=plan_result,
                depth=diligence_depth,
            )

            return report

        except Exception as e:
            logger.error(f"LoongFlow diligence planning failed: {e}")
            # Fallback to standard planning
            return await self._plan_diligence_standard(deal, diligence_depth)

    async def _plan_diligence_standard(
        self,
        deal: Deal,
        diligence_depth: str,
    ) -> DiligenceReport:
        """Plan diligence using standard approach"""
        return await self.diligence.execute_diligence(
            deal=deal,
            plan=None,
            depth=diligence_depth,
        )

    async def analyze_deal(
        self,
        deal_id: str,
    ) -> tuple[ValuationResult, DealStructure, IntegrationPlan]:
        """
        Perform comprehensive deal analysis

        Args:
            deal_id: Deal identifier

        Returns:
            Tuple of (valuation, structure, integration_plan)
        """
        if deal_id not in self.deals:
            raise ValueError(f"Deal {deal_id} not found")

        deal = self.deals[deal_id]
        deal.stage = DealStage.ANALYSIS
        deal.updated_at = datetime.utcnow()
        self._update_pipeline(deal_id, DealStage.ANALYSIS)

        logger.info(f"Analyzing deal {deal_id}")

        # Get knowledge from past deals
        similar_deals = await self.knowledge.find_similar_deals(deal)
        success_patterns = await self.knowledge.get_success_patterns(
            deal.target_company.industry
        )

        # Run valuations
        valuation = await self.valuation.valuate_deal(
            deal=deal,
            similar_deals=similar_deals,
        )

        # Optimize deal structure
        structure = await self.structure_optimizer.optimize_structure(
            deal=deal,
            valuation=valuation,
            patterns=success_patterns,
        )

        # Create integration plan
        integration_plan = await self.integration.create_integration_plan(
            deal=deal,
            structure=structure,
        )

        return valuation, structure, integration_plan

    async def generate_recommendation(
        self,
        deal_id: str,
    ) -> Dict[str, Any]:
        """
        Generate go/no-go recommendation

        Args:
            deal_id: Deal identifier

        Returns:
            Recommendation dictionary with analysis and rationale
        """
        if deal_id not in self.deals:
            raise ValueError(f"Deal {deal_id} not found")

        deal = self.deals[deal_id]
        deal.stage = DealStage.DECISION
        deal.updated_at = datetime.utcnow()
        self._update_pipeline(deal_id, DealStage.DECISION)

        logger.info(f"Generating recommendation for deal {deal_id}")

        # Collect all analysis
        diligence_report = deal.metadata.get("diligence_report")
        valuation = deal.metadata.get("valuation")
        structure = deal.metadata.get("structure")
        integration_plan = deal.metadata.get("integration_plan")

        # Generate recommendation using knowledge from past deals
        recommendation = await self.knowledge.generate_recommendation(
            deal=deal,
            diligence_report=diligence_report,
            valuation=valuation,
            structure=structure,
            integration_plan=integration_plan,
        )

        return recommendation

    async def prepare_negotiation(
        self,
        deal_id: str,
    ) -> NegotiationStrategy:
        """
        Prepare negotiation strategy

        Args:
            deal_id: Deal identifier

        Returns:
            NegotiationStrategy: Comprehensive negotiation strategy
        """
        if deal_id not in self.deals:
            raise ValueError(f"Deal {deal_id} not found")

        deal = self.deals[deal_id]
        deal.stage = DealStage.NEGOTIATION
        deal.updated_at = datetime.utcnow()
        self._update_pipeline(deal_id, DealStage.NEGOTIATION)

        logger.info(f"Preparing negotiation for deal {deal_id}")

        # Generate negotiation strategy
        strategy = await self.negotiation.create_strategy(
            deal=deal,
            valuation=deal.metadata.get("valuation"),
            structure=deal.metadata.get("structure"),
        )

        return strategy

    async def close_deal(
        self,
        deal_id: str,
        final_value: float,
    ) -> None:
        """
        Mark deal as closed and start integration

        Args:
            deal_id: Deal identifier
            final_value: Final deal value
        """
        if deal_id not in self.deals:
            raise ValueError(f"Deal {deal_id} not found")

        deal = self.deals[deal_id]
        deal.stage = DealStage.CLOSING
        deal.updated_at = datetime.utcnow()
        deal.deal_size = final_value
        self._update_pipeline(deal_id, DealStage.CLOSING)

        logger.info(f"Closing deal {deal_id} at ${final_value:,}")

    async def start_integration(
        self,
        deal_id: str,
    ) -> IntegrationPlan:
        """
        Start post-merger integration

        Args:
            deal_id: Deal identifier

        Returns:
            IntegrationPlan: Detailed integration plan
        """
        if deal_id not in self.deals:
            raise ValueError(f"Deal {deal_id} not found")

        deal = self.deals[deal_id]
        deal.stage = DealStage.INTEGRATION
        deal.updated_at = datetime.utcnow()
        self._update_pipeline(deal_id, DealStage.INTEGRATION)

        logger.info(f"Starting integration for deal {deal_id}")

        # Get or create integration plan
        integration_plan = deal.metadata.get("integration_plan")
        if not integration_plan:
            integration_plan = await self.integration.create_integration_plan(deal=deal)

        # Activate integration tracking
        await self.integration.track_integration(
            deal_id=deal_id,
            plan=integration_plan,
        )

        return integration_plan

    async def record_outcome(
        self,
        deal_id: str,
        outcome: DealOutcome,
    ) -> None:
        """
        Record deal outcome and extract learnings

        Args:
            deal_id: Deal identifier
            outcome: Deal outcome data
        """
        if deal_id not in self.deals:
            raise ValueError(f"Deal {deal_id} not found")

        deal = self.deals[deal_id]
        deal.outcome = outcome
        deal.stage = DealStage.COMPLETED
        deal.updated_at = datetime.utcnow()
        self._update_pipeline(deal_id, DealStage.COMPLETED)

        logger.info(f"Recording outcome for deal {deal_id}")

        # Learn from this deal
        await self.knowledge.learn_from_deal(
            deal=deal,
            outcome=outcome,
        )

        # Update success patterns and heuristics
        await self.knowledge.update_patterns(deal, outcome)

    async def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get summary of deal pipeline

        Returns:
            Pipeline summary with metrics and status
        """
        summary = {
            "total_deals": len(self.deals),
            "by_stage": {
                stage.value: len(deal_ids)
                for stage, deal_ids in self.deal_pipeline.items()
            },
            "by_priority": {},
        }

        # Count by priority
        for deal in self.deals.values():
            priority = deal.priority.value
            summary["by_priority"][priority] = summary["by_priority"].get(priority, 0) + 1

        # Calculate metrics
        summary["conversion_rate"] = self._calculate_conversion_rate()
        summary["average_deal_size"] = self._calculate_average_deal_size()

        return summary

    def _update_pipeline(self, deal_id: str, new_stage: DealStage) -> None:
        """Update deal pipeline stage"""
        # Remove from old stage
        for stage, deal_ids in self.deal_pipeline.items():
            if deal_id in deal_ids:
                deal_ids.remove(deal_id)

        # Add to new stage
        self.deal_pipeline[new_stage].append(deal_id)

    async def _prioritize_pipeline(self) -> None:
        """Prioritize deals in sourcing stage"""
        sourcing_deals = [
            self.deals[deal_id]
            for deal_id in self.deal_pipeline[DealStage.SOURCING]
        ]

        # Sort by priority and strategic fit
        priority_order = {
            DealPriority.CRITICAL: 0,
            DealPriority.HIGH: 1,
            DealPriority.MEDIUM: 2,
            DealPriority.LOW: 3,
        }

        sourcing_deals.sort(
            key=lambda d: (
                priority_order[d.priority],
                -(d.target_company.strategic_fit.overall_score
                  if d.target_company.strategic_fit else 0)
            )
        )

        self.deal_pipeline[DealStage.SOURCING] = [d.deal_id for d in sourcing_deals]

    def _calculate_conversion_rate(self) -> float:
        """Calculate deal conversion rate"""
        completed = len(self.deal_pipeline[DealStage.COMPLETED])
        total = len(self.deals)
        return (completed / total * 100) if total > 0 else 0.0

    def _calculate_average_deal_size(self) -> float:
        """Calculate average deal size for completed deals"""
        completed_deals = [
            d for d in self.deals.values()
            if d.stage == DealStage.COMPLETED and d.deal_size
        ]

        if not completed_deals:
            return 0.0

        return sum(d.deal_size for d in completed_deals) / len(completed_deals)

    async def get_deal(self, deal_id: str) -> Optional[Deal]:
        """Get deal by ID"""
        return self.deals.get(deal_id)

    async def list_deals(
        self,
        stage: Optional[DealStage] = None,
        priority: Optional[DealPriority] = None,
    ) -> List[Deal]:
        """List deals with optional filters"""
        deals = list(self.deals.values())

        if stage:
            deals = [d for d in deals if d.stage == stage]

        if priority:
            deals = [d for d in deals if d.priority == priority]

        return deals
