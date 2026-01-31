"""
M&A Deal Intelligence Platform

A comprehensive system for end-to-end M&A deal workflow automation including:
- Deal sourcing and screening
- Due diligence support
- Deal structuring
- Negotiation assistance
- Integration planning
- Continuous learning from deals
"""

# Relative imports for package compatibility
try:
    from openevolve.agents.ma.ma_platform import MADealPlatform
    from openevolve.agents.ma.deal_sourcer import DealSourcer
    from openevolve.agents.ma.diligence_assistant import DiligenceAssistant
    from openevolve.agents.ma.valuation import ValuationEngine
    from openevolve.agents.ma.structure_optimizer import StructureOptimizer
    from openevolve.agents.ma.negotiation_advisor import NegotiationAdvisor
    from openevolve.agents.ma.integration_planner import IntegrationPlanner
    from openevolve.agents.ma.knowledge_manager import DealKnowledgeManager
    from openevolve.agents.ma.schemas import (
        Deal,
        DealStage,
        Company,
        DiligenceReport,
        ValuationResult,
        DealStructure,
        NegotiationStrategy,
        IntegrationPlan,
        DealOutcome,
    )
except ImportError:
    # Fallback to relative imports
    from .ma_platform import MADealPlatform
    from .deal_sourcer import DealSourcer
    from .diligence_assistant import DiligenceAssistant
    from .valuation import ValuationEngine
    from .structure_optimizer import StructureOptimizer
    from .negotiation_advisor import NegotiationAdvisor
    from .integration_planner import IntegrationPlanner
    from .knowledge_manager import DealKnowledgeManager
    from .schemas import (
        Deal,
        DealStage,
        Company,
        DiligenceReport,
        ValuationResult,
        DealStructure,
        NegotiationStrategy,
        IntegrationPlan,
        DealOutcome,
    )

__all__ = [
    "MADealPlatform",
    "DealSourcer",
    "DiligenceAssistant",
    "ValuationEngine",
    "StructureOptimizer",
    "NegotiationAdvisor",
    "IntegrationPlanner",
    "DealKnowledgeManager",
    "Deal",
    "DealStage",
    "Company",
    "DiligenceReport",
    "ValuationResult",
    "DealStructure",
    "NegotiationStrategy",
    "IntegrationPlan",
    "DealOutcome",
]

__version__ = "0.1.0"
