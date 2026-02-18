"""
OpenEvolve Integration for Adaptive MDAP/MAKER Adapter

This module provides comprehensive integration between the Adaptive MDAP/MAKER adapter
and the OpenEvolve evolution system, enabling:
- Complexity-based resource allocation for OpenEvolve workflows
- MAKER voting for workflow decision points
- Adaptive gauntlet selection based on problem complexity
- ICR pattern learning from workflow executions

Federation Constitution Compliance:
- Law 1: Air Gap - No imports from core-projects, uses adapter with canonical schema
- Law 2: Runtime Truth - Validates against actual OpenEvolve API behavior
- Law 3: Untouchable DB - Read-only access to workflow state
- Law 4: Idempotency - All operations safe to retry
- Law 5: Configuration Explicitness - All config via environment variables
- Law 6: UTC - All timestamps in UTC ISO-8601
"""

import os
import sys
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import canonical schemas from adapter
from adaptive_mdap_adapter import (
    get_adapter,
    CanonicalSubProblem,
    CanonicalComplexityScore,
    CanonicalStrategy,
    CanonicalRequest,
    CanonicalResponse,
    TaskStatus,
    AdaptiveMDAPAdapterConfig
)

from maker_adapter import (
    get_maker_adapter,
    CanonicalMakerConfig,
    CanonicalMakerStep,
    CanonicalMakerResult
)

logger = logging.getLogger(__name__)


class OpenEvolveWorkflowType(Enum):
    """OpenEvolve workflow types that can use MDAP/MAKER integration."""
    EVOLUTION = "evolution"
    ADVERSARIAL = "adversarial"
    SOVEREIGN = "sovereign"
    WEB3 = "web3"
    RAG = "rag"


class OpenEvolveStage(Enum):
    """OpenEvolve workflow stages where MDAP/MAKER can assist."""
    DECOMPOSITION = "decomposition"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    ASSEMBLY = "assembly"


@dataclass
class OpenEvolveIntegrationConfig:
    """Configuration for OpenEvolve integration."""
    # MDAP Adapter settings
    mdap_timeout_ms: int = 5000
    mdap_max_retries: int = 3
    mdap_enable_complexity_analysis: bool = True
    mdap_enable_resource_adaptation: bool = True

    # MAKER Adapter settings
    maker_enable_voting: bool = True
    maker_k_ahead: int = 3
    maker_max_agents: int = 5
    maker_enable_red_flagging: bool = True

    # ICR Integration settings
    icr_enable_learning: bool = True
    icr_store_patterns: bool = True
    icr_min_confidence: float = 0.7

    # Gauntlet Integration settings
    gauntlet_enable_adaptation: bool = True
    gauntlet_min_complexity_threshold: float = 0.3
    gauntlet_max_complexity_threshold: float = 0.8


@dataclass
class WorkflowComplexityAnalysis:
    """Result of analyzing workflow complexity."""
    workflow_id: str
    workflow_type: str
    overall_complexity: float
    sub_problems: List[Dict[str, Any]]
    recommended_strategy: str
    recommended_resources: Dict[str, Any]
    estimated_duration_ms: float
    timestamp: str


@dataclass
class MAKERWorkflowDecision:
    """Result of MAKER voting on workflow decision."""
    workflow_id: str
    stage: str
    decision_point: str
    votes_collected: int
    consensus_reached: bool
    consensus_score: float
    recommended_action: str
    red_flags: List[Dict[str, Any]]
    timestamp: str


class OpenEvolveMDAPIntegration:
    """
    Integration between Adaptive MDAP/MAKER adapter and OpenEvolve workflows.

    Features:
    1. Complexity analysis for workflow decomposition
    2. Resource allocation based on problem complexity
    3. MAKER voting for workflow decision points
    4. Adaptive gauntlet selection
    5. ICR pattern learning from executions
    """

    def __init__(self, config: Optional[OpenEvolveIntegrationConfig] = None):
        """
        Initialize OpenEvolve integration.

        Args:
            config: Integration configuration (uses env vars if None)
        """
        # Load configuration from environment if not provided
        if config is None:
            config = self._load_config_from_env()

        self.config = config

        # Initialize adapters
        self.mdap_adapter = get_adapter()
        self.maker_adapter = get_maker_adapter()

        # ICR integration (optional)
        self.icr_integration = None
        if config.icr_enable_learning:
            try:
                from icr_integration import get_icr_integration
                self.icr_integration = get_icr_integration()
                logger.info("ICR integration enabled")
            except ImportError as e:
                logger.warning(f"ICR integration not available: {e}")

        # Gauntlet integration (optional)
        self.gauntlet_integration = None
        if config.gauntlet_enable_adaptation:
            try:
                from mdap_maker_gauntlet_integration import create_mdap_maker_integration
                self.gauntlet_integration = create_mdap_maker_integration()
                logger.info("Gauntlet integration enabled")
            except ImportError as e:
                logger.warning(f"Gauntlet integration not available: {e}")

        # Active workflow tracking
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_lock = threading.Lock()

        logger.info("OpenEvolve MDAP Integration initialized")

    def _load_config_from_env(self) -> OpenEvolveIntegrationConfig:
        """Load configuration from environment variables."""
        return OpenEvolveIntegrationConfig(
            mdap_timeout_ms=int(os.getenv("MDAP_TIMEOUT_MS", "5000")),
            mdap_max_retries=int(os.getenv("MDAP_MAX_RETRIES", "3")),
            mdap_enable_complexity_analysis=os.getenv("MDAP_ENABLE_COMPLEXITY", "true").lower() == "true",
            mdap_enable_resource_adaptation=os.getenv("MDAP_ENABLE_ADAPTATION", "true").lower() == "true",
            maker_enable_voting=os.getenv("MAKER_ENABLE_VOTING", "true").lower() == "true",
            maker_k_ahead=int(os.getenv("MAKER_K_AHEAD", "3")),
            maker_max_agents=int(os.getenv("MAKER_MAX_AGENTS", "5")),
            maker_enable_red_flagging=os.getenv("MAKER_ENABLE_REDFLAGGING", "true").lower() == "true",
            icr_enable_learning=os.getenv("ICR_ENABLE_LEARNING", "true").lower() == "true",
            icr_store_patterns=os.getenv("ICR_STORE_PATTERNS", "true").lower() == "true",
            icr_min_confidence=float(os.getenv("ICR_MIN_CONFIDENCE", "0.7")),
            gauntlet_enable_adaptation=os.getenv("GAUNTLET_ENABLE_ADAPTATION", "true").lower() == "true",
            gauntlet_min_complexity_threshold=float(os.getenv("GAUNTLET_MIN_COMPLEXITY", "0.3")),
            gauntlet_max_complexity_threshold=float(os.getenv("GAUNTLET_MAX_COMPLEXITY", "0.8"))
        )

    def analyze_workflow_complexity(
        self,
        workflow_id: str,
        workflow_type: str,
        problem_statement: str,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowComplexityAnalysis:
        """
        Analyze complexity of an OpenEvolve workflow.

        Args:
            workflow_id: OpenEvolve workflow ID
            workflow_type: Type of workflow
            problem_statement: Problem to solve
            context: Additional context

        Returns:
            WorkflowComplexityAnalysis with recommendations
        """
        start_time = time.time()
        context = context or {}

        logger.info(f"Analyzing complexity for workflow {workflow_id}")

        # Create canonical subproblem for analysis
        subproblem = CanonicalSubProblem(
            id=f"workflow_{workflow_id}",
            description=problem_statement[:2000],  # Limit length
            domain=context.get("domain", workflow_type),
            depth=context.get("depth", 1),
            dependencies=context.get("dependencies", []),
            metadata={
                "workflow_type": workflow_type,
                "workflow_id": workflow_id,
                **context
            }
        )

        # Analyze complexity using MDAP adapter
        response = self.mdap_adapter.analyze_complexity(subproblem)

        if response.status != TaskStatus.COMPLETED:
            logger.error(f"Complexity analysis failed: {response.error}")
            # Return default analysis
            return WorkflowComplexityAnalysis(
                workflow_id=workflow_id,
                workflow_type=workflow_type,
                overall_complexity=0.5,
                sub_problems=[],
                recommended_strategy="DIRECT",
                recommended_resources={},
                estimated_duration_ms=30000,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        complexity_score = response.complexity_score

        # Get recommended strategy
        strategy_response = self.mdap_adapter.allocate_resources(complexity_score)
        recommended_strategy = "DIRECT"
        recommended_resources = {}

        if strategy_response.status == TaskStatus.COMPLETED:
            strategy = strategy_response.strategy
            recommended_strategy = strategy.strategy.value if hasattr(strategy.strategy, 'value') else str(strategy.strategy)
            recommended_resources = {
                "n_agents": strategy.n_agents,
                "k_ahead": strategy.k_ahead,
                "timeout_ms": strategy.timeout_ms,
                "max_retries": strategy.max_retries
            }

        # Estimate duration based on complexity
        estimated_duration_ms = 30000 * (1 + complexity_score.overall_score)

        analysis = WorkflowComplexityAnalysis(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            overall_complexity=complexity_score.overall_score,
            sub_problems=[{
                "id": subproblem.id,
                "description": subproblem.description,
                "domain": subproblem.domain,
                "depth": subproblem.depth,
                "complexity": complexity_score.overall_score
            }],
            recommended_strategy=recommended_strategy,
            recommended_resources=recommended_resources,
            estimated_duration_ms=estimated_duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Store in active workflows
        with self.workflow_lock:
            self.active_workflows[workflow_id] = {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "complexity_analysis": analysis,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

        # Store ICR pattern if enabled
        if self.icr_integration and self.config.icr_store_patterns:
            try:
                from icr_integration import ICRPatternType
                self.icr_integration.store_pattern(
                    pattern_type=ICRPatternType.WORKFLOW_EXECUTION,
                    passed=True,
                    context={
                        "workflow_type": workflow_type,
                        "complexity_score": complexity_score.overall_score,
                        "strategy": recommended_strategy
                    },
                    metrics={
                        "overall_complexity": complexity_score.overall_score,
                        "text_length": complexity_score.text_length_score,
                        "dependencies": complexity_score.dependency_score,
                        "depth": complexity_score.depth_score
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to store ICR pattern: {e}")

        logger.info(
            f"Complexity analysis complete: {complexity_score.overall_score:.3f}, "
            f"strategy: {recommended_strategy}"
        )

        return analysis

    def make_workflow_decision(
        self,
        workflow_id: str,
        stage: str,
        decision_point: str,
        options: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> MAKERWorkflowDecision:
        """
        Use MAKER voting to make a workflow decision.

        Args:
            workflow_id: OpenEvolve workflow ID
            stage: Current workflow stage
            decision_point: Description of decision to make
            options: List of possible options
            context: Additional context

        Returns:
            MAKERWorkflowDecision with voting results
        """
        if not self.config.maker_enable_voting:
            logger.info("MAKER voting disabled, returning first option")
            return MAKERWorkflowDecision(
                workflow_id=workflow_id,
                stage=stage,
                decision_point=decision_point,
                votes_collected=1,
                consensus_reached=True,
                consensus_score=1.0,
                recommended_action=options[0].get("action", "proceed") if options else "proceed",
                red_flags=[],
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        logger.info(f"MAKER voting for workflow {workflow_id}, stage {stage}")

        context = context or {}

        # Create MAKER step for voting
        step = CanonicalMakerStep(
            step_id=f"decision_{workflow_id}_{stage}",
            prompt_template=f"""
Decision Point: {decision_point}

Context:
{context.get('context_description', 'No additional context')}

Options:
{self._format_options(options)}

Previous History:
{{history}}

Provide your recommendation as JSON:
{{
    "selected_option": <index or description>,
    "confidence": <0.0-1.0>,
    "reasoning": "<explanation>",
    "red_flags": ["<any concerns>"]
}}
            """.strip(),
            task_type="decision",
            priority=1,
            metadata={
                "workflow_id": workflow_id,
                "stage": stage,
                "decision_point": decision_point
            }
        )

        # Execute MAKER voting
        try:
            result = self.maker_adapter.execute_maker_step(
                step=step,
                current_state={"options": options, **context},
                history=[],
                team=None,  # Use default team
                correlation_id=f"workflow_{workflow_id}"
            )

            if result.status == TaskStatus.COMPLETED:
                # Extract decision from result
                selected_option = self._extract_selected_option(result, options)
                confidence = result.metadata.get("confidence", 0.5)
                reasoning = result.metadata.get("reasoning", "")

                decision = MAKERWorkflowDecision(
                    workflow_id=workflow_id,
                    stage=stage,
                    decision_point=decision_point,
                    votes_collected=result.votes_cast,
                    consensus_reached=result.consensus_reached,
                    consensus_score=result.consensus_score,
                    recommended_action=selected_option,
                    red_flags=result.red_flags_detected,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )

                # Store ICR pattern if enabled
                if self.icr_integration and self.config.icr_store_patterns:
                    try:
                        from icr_integration import ICRPatternType
                        self.icr_integration.store_pattern(
                            pattern_type=ICRPatternType.RETRY_PATTERN,
                            passed=decision.consensus_reached,
                            context={
                                "workflow_type": context.get("workflow_type", "unknown"),
                                "stage": stage,
                                "decision_point": decision_point
                            },
                            metrics={
                                "votes_collected": decision.votes_collected,
                                "consensus_score": decision.consensus_score,
                                "red_flags": len(decision.red_flags)
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store ICR pattern: {e}")

                logger.info(
                    f"MAKER decision: {selected_option}, "
                    f"consensus: {decision.consensus_reached} ({decision.consensus_score:.3f})"
                )

                return decision
            else:
                logger.error(f"MAKER execution failed: {result.error}")
                # Fallback to first option
                return MAKERWorkflowDecision(
                    workflow_id=workflow_id,
                    stage=stage,
                    decision_point=decision_point,
                    votes_collected=0,
                    consensus_reached=False,
                    consensus_score=0.0,
                    recommended_action=options[0].get("action", "proceed") if options else "proceed",
                    red_flags=[],
                    timestamp=datetime.now(timezone.utc).isoformat()
                )

        except Exception as e:
            logger.error(f"MAKER voting failed: {e}")
            # Fallback to first option
            return MAKERWorkflowDecision(
                workflow_id=workflow_id,
                stage=stage,
                decision_point=decision_point,
                votes_collected=0,
                consensus_reached=False,
                consensus_score=0.0,
                recommended_action=options[0].get("action", "proceed") if options else "proceed",
                red_flags=[],
                timestamp=datetime.now(timezone.utc).isoformat()
            )

    def select_adaptive_gauntlet(
        self,
        workflow_id: str,
        complexity_score: float,
        base_gauntlet_type: str = "adversarial"
    ) -> Dict[str, Any]:
        """
        Select appropriate gauntlet based on complexity analysis.

        Args:
            workflow_id: OpenEvolve workflow ID
            complexity_score: Overall complexity score (0.0-1.0)
            base_gauntlet_type: Base gauntlet type to adapt

        Returns:
            Gauntlet configuration
        """
        if not self.gauntlet_integration:
            logger.info("Gauntlet integration not enabled, using default")
            return {
                "gauntlet_type": base_gauntlet_type,
                "adapted": False,
                "complexity_score": complexity_score
            }

        logger.info(f"Selecting adaptive gauntlet for workflow {workflow_id}")

        try:
            from gauntlet_types import GauntletType

            # Map string to GauntletType
            gauntlet_type_map = {
                "adversarial": GauntletType.ADVERSARIAL,
                "formal_verification": GauntletType.FORMAL_VERIFICATION,
                "statistical": GauntletType.STATISTICAL,
                "multi_objective": GauntletType.MULTI_OBJECTIVE,
                "evolutionary": GauntletType.EVOLUTIONARY
            }

            base_type = gauntlet_type_map.get(base_gauntlet_type, GauntletType.ADVERSARIAL)

            # Use gauntlet integration to select appropriate type
            selected_type = base_type

            if complexity_score > self.config.gauntlet_max_complexity_threshold:
                # High complexity: use formal verification or multi-objective
                selected_type = GauntletType.FORMAL_VERIFICATION
            elif complexity_score < self.config.gauntlet_min_complexity_threshold:
                # Low complexity: use statistical
                selected_type = GauntletType.STATISTICAL

            config = {
                "gauntlet_type": selected_type.value,
                "adapted": True,
                "complexity_score": complexity_score,
                "complexity_threshold_min": self.config.gauntlet_min_complexity_threshold,
                "complexity_threshold_max": self.config.gauntlet_max_complexity_threshold,
                "adaptation_reason": self._get_gauntlet_adaptation_reason(complexity_score)
            }

            logger.info(f"Selected gauntlet: {selected_type.value} for complexity {complexity_score:.3f}")

            return config

        except Exception as e:
            logger.error(f"Gauntlet selection failed: {e}")
            return {
                "gauntlet_type": base_gauntlet_type,
                "adapted": False,
                "complexity_score": complexity_score,
                "error": str(e)
            }

    def _format_options(self, options: List[Dict[str, Any]]) -> str:
        """Format options for MAKER prompt."""
        formatted = []
        for i, option in enumerate(options):
            formatted.append(f"{i + 1}. {option.get('description', option.get('action', 'Unknown'))}")
        return "\n".join(formatted)

    def _extract_selected_option(self, result: CanonicalMakerResult, options: List[Dict[str, Any]]) -> str:
        """Extract selected option from MAKER result."""
        # Try to get selected option from metadata
        if result.metadata and "selected_option" in result.metadata:
            selected = result.metadata["selected_option"]

            # If it's an index, map to option
            if isinstance(selected, int) and 0 <= selected < len(options):
                return options[selected].get("action", "proceed")

            # If it's a string, return as-is
            if isinstance(selected, str):
                return selected

        # Fallback to first option
        return options[0].get("action", "proceed") if options else "proceed"

    def _get_gauntlet_adaptation_reason(self, complexity_score: float) -> str:
        """Get explanation for gauntlet adaptation."""
        if complexity_score > self.config.gauntlet_max_complexity_threshold:
            return "High complexity problem requires formal verification"
        elif complexity_score < self.config.gauntlet_min_complexity_threshold:
            return "Low complexity problem can use lighter statistical validation"
        else:
            return "Medium complexity problem uses standard gauntlet"

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active workflow."""
        with self.workflow_lock:
            return self.active_workflows.get(workflow_id)

    def cleanup_workflow(self, workflow_id: str) -> bool:
        """Clean up completed workflow."""
        with self.workflow_lock:
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
                return True
        return False


# Global instance
_openevolve_integration: Optional[OpenEvolveMDAPIntegration] = None


def get_openevolve_integration() -> OpenEvolveMDAPIntegration:
    """Get or create global OpenEvolve integration instance."""
    global _openevolve_integration
    if _openevolve_integration is None:
        _openevolve_integration = OpenEvolveMDAPIntegration()
    return _openevolve_integration


__all__ = [
    "OpenEvolveWorkflowType",
    "OpenEvolveStage",
    "OpenEvolveIntegrationConfig",
    "WorkflowComplexityAnalysis",
    "MAKERWorkflowDecision",
    "OpenEvolveMDAPIntegration",
    "get_openevolve_integration"
]
