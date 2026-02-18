"""
Comprehensive Integration Manager for Adaptive MDAP/MAKER Adapter

This module provides a unified integration manager that coordinates between:
- OpenEvolve workflows
- BubbleLab UI
- Gauntlet system
- ICR pattern learning

The integration manager ensures all components work together seamlessly
while maintaining Federation Constitution compliance.

Usage:
    from integration_manager import get_integration_manager

    manager = get_integration_manager()

    # Analyze workflow complexity
    analysis = manager.analyze_workflow(workflow_id, problem)

    # Make workflow decision
    decision = manager.make_decision(workflow_id, options)

    # Get UI data
    ui_data = manager.get_ui_data()
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

try:
    from .openevolve_integration import (
        get_openevolve_integration,
        OpenEvolveMDAPIntegration,
        WorkflowComplexityAnalysis,
        MAKERWorkflowDecision,
        OpenEvolveIntegrationConfig
    )
    from .bubblelab_ui_integration import (
        get_bubblelab_ui_integration,
        BubbleLabUIIntegration,
        ComplexityAnalysisResult
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from openevolve_integration import (
        get_openevolve_integration,
        OpenEvolveMDAPIntegration,
        WorkflowComplexityAnalysis,
        MAKERWorkflowDecision,
        OpenEvolveIntegrationConfig
    )
    from bubblelab_ui_integration import (
        get_bubblelab_ui_integration,
        BubbleLabUIIntegration,
        ComplexityAnalysisResult
    )

logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Status of integration components."""
    INITIALIZED = "initialized"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass
class IntegrationHealth:
    """Health status of all integration components."""
    overall_status: IntegrationStatus
    mdap_adapter_status: str
    maker_adapter_status: str
    openevolve_integration_status: str
    bubblelab_ui_status: str
    icr_integration_status: str
    gauntlet_integration_status: str
    timestamp: str
    errors: List[str] = field(default_factory=list)


class ComprehensiveIntegrationManager:
    """
    Unified integration manager for Adaptive MDAP/MAKER adapter.

    Coordinates all integration components and provides a single interface
    for OpenEvolve, BubbleLab, Gauntlet, and ICR systems.
    """

    def __init__(self):
        """Initialize comprehensive integration manager."""
        logger.info("Initializing Comprehensive Integration Manager...")

        # Initialize all integration components
        self.openevolve_integration = get_openevolve_integration()
        self.bubblelab_ui_integration = get_bubblelab_ui_integration()

        # Track integration health
        self._health_status = IntegrationStatus.INITIALIZED
        self._last_health_check = None
        self._health_lock = threading.Lock()

        # Auto-cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()

        # Start auto-cleanup
        self._start_cleanup_thread()

        logger.info("Comprehensive Integration Manager ready")

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_loop():
            while not self._stop_cleanup.wait(300):  # Run every 5 minutes
                try:
                    self.bubblelab_ui_integration.clear_old_data(max_age_seconds=3600)
                    logger.debug("Auto-cleanup completed")
                except Exception as e:
                    logger.error(f"Auto-cleanup failed: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("Auto-cleanup thread started")

    def analyze_workflow(
        self,
        workflow_id: str,
        problem_statement: str,
        workflow_type: str = "evolution",
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowComplexityAnalysis:
        """
        Analyze workflow complexity using OpenEvolve integration.

        Args:
            workflow_id: OpenEvolve workflow ID
            problem_statement: Problem to solve
            workflow_type: Type of workflow
            context: Additional context

        Returns:
            WorkflowComplexityAnalysis with recommendations
        """
        return self.openevolve_integration.analyze_workflow_complexity(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            problem_statement=problem_statement,
            context=context
        )

    def make_decision(
        self,
        workflow_id: str,
        stage: str,
        decision_point: str,
        options: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> MAKERWorkflowDecision:
        """
        Make workflow decision using MAKER voting.

        Args:
            workflow_id: OpenEvolve workflow ID
            stage: Current workflow stage
            decision_point: Description of decision
            options: List of possible options
            context: Additional context

        Returns:
            MAKERWorkflowDecision with voting results
        """
        return self.openevolve_integration.make_workflow_decision(
            workflow_id=workflow_id,
            stage=stage,
            decision_point=decision_point,
            options=options,
            context=context
        )

    def select_gauntlet(
        self,
        workflow_id: str,
        complexity_score: float,
        base_gauntlet_type: str = "adversarial"
    ) -> Dict[str, Any]:
        """
        Select appropriate gauntlet based on complexity.

        Args:
            workflow_id: OpenEvolve workflow ID
            complexity_score: Overall complexity score
            base_gauntlet_type: Base gauntlet type

        Returns:
            Gauntlet configuration
        """
        return self.openevolve_integration.select_adaptive_gauntlet(
            workflow_id=workflow_id,
            complexity_score=complexity_score,
            base_gauntlet_type=base_gauntlet_type
        )

    def analyze_for_ui(
        self,
        problem_description: str,
        domain: str = "general",
        depth: int = 1,
        dependencies: Optional[List[str]] = None
    ) -> ComplexityAnalysisResult:
        """
        Analyze complexity for BubbleLab UI display.

        Args:
            problem_description: Problem to analyze
            domain: Problem domain
            depth: Problem depth
            dependencies: List of dependencies

        Returns:
            ComplexityAnalysisResult for UI
        """
        return self.bubblelab_ui_integration.analyze_complexity_for_ui(
            problem_description=problem_description,
            domain=domain,
            depth=depth,
            dependencies=dependencies
        )

    def get_ui_data(self) -> Dict[str, Any]:
        """
        Get all UI data for BubbleLab display.

        Returns:
            UI data including analyses, health, workflows, etc.
        """
        return self.bubblelab_ui_integration.export_ui_data(format="dict")

    def get_health_status(self) -> IntegrationHealth:
        """
        Get comprehensive health status of all integrations.

        Returns:
            IntegrationHealth with status of all components
        """
        with self._health_lock:
            # Check if we need to refresh health status
            if self._last_health_check:
                time_since_check = (datetime.now(timezone.utc) -
                                   self._last_health_check).total_seconds()
                if time_since_check < 30:  # Cache for 30 seconds
                    return self._health_status

            # Gather health status from all components
            errors = []
            overall_status = IntegrationStatus.READY

            try:
                # Get adapter health from BubbleLab UI integration
                adapter_health = self.bubblelab_ui_integration.get_adapter_health_status()

                mdap_status = adapter_health.get("mdap_adapter", {}).get("status", "unknown")
                maker_status = adapter_health.get("maker_adapter", {}).get("status", "unknown")

                # Determine overall status
                if mdap_status == "healthy" and maker_status == "healthy":
                    overall_status = IntegrationStatus.READY
                elif mdap_status == "degraded" or maker_status == "degraded":
                    overall_status = IntegrationStatus.DEGRADED
                else:
                    overall_status = IntegrationStatus.ERROR
                    errors.append("Adapter health check failed")

            except Exception as e:
                logger.error(f"Health check failed: {e}")
                overall_status = IntegrationStatus.ERROR
                errors.append(str(e))
                mdap_status = "error"
                maker_status = "error"

            # Create health status
            self._health_status = IntegrationHealth(
                overall_status=overall_status,
                mdap_adapter_status=mdap_status,
                maker_adapter_status=maker_status,
                openevolve_integration_status="ready",  # Always ready if initialized
                bubblelab_ui_status="ready",
                icr_integration_status="ready" if self.openevolve_integration.icr_integration else "disabled",
                gauntlet_integration_status="ready" if self.openevolve_integration.gauntlet_integration else "disabled",
                timestamp=datetime.now(timezone.utc).isoformat(),
                errors=errors
            )

            self._last_health_check = datetime.now(timezone.utc)

            return self._health_status

    def execute_full_workflow(
        self,
        workflow_id: str,
        problem_statement: str,
        workflow_type: str = "evolution",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute full workflow with complexity analysis, decision making, and gauntlet selection.

        Args:
            workflow_id: OpenEvolve workflow ID
            problem_statement: Problem to solve
            workflow_type: Type of workflow
            context: Additional context

        Returns:
            Complete workflow execution results
        """
        logger.info(f"Executing full workflow {workflow_id} of type {workflow_type}")

        start_time = time.time()
        context = context or {}

        results = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "steps": []
        }

        try:
            # Step 1: Analyze complexity
            logger.info("Step 1: Analyzing workflow complexity...")
            complexity_analysis = self.analyze_workflow(
                workflow_id=workflow_id,
                problem_statement=problem_statement,
                workflow_type=workflow_type,
                context=context
            )

            results["steps"].append({
                "step": "complexity_analysis",
                "status": "completed",
                "complexity": complexity_analysis.overall_complexity,
                "strategy": complexity_analysis.recommended_strategy,
                "resources": complexity_analysis.recommended_resources
            })

            # Step 2: Select appropriate gauntlet
            logger.info("Step 2: Selecting adaptive gauntlet...")
            gauntlet_config = self.select_gauntlet(
                workflow_id=workflow_id,
                complexity_score=complexity_analysis.overall_complexity,
                base_gauntlet_type=context.get("base_gauntlet_type", "adversarial")
            )

            results["steps"].append({
                "step": "gauntlet_selection",
                "status": "completed",
                "gauntlet_type": gauntlet_config.get("gauntlet_type"),
                "adapted": gauntlet_config.get("adapted", False)
            })

            # Step 3: Make initial workflow decision
            logger.info("Step 3: Making initial workflow decision...")
            decision = self.make_decision(
                workflow_id=workflow_id,
                stage="planning",
                decision_point="Select execution strategy",
                options=[
                    {"action": "proceed", "description": "Proceed with recommended strategy"},
                    {"action": "fallback", "description": "Use fallback strategy"}
                ],
                context={
                    **context,
                    "complexity_analysis": complexity_analysis,
                    "gauntlet_config": gauntlet_config
                }
            )

            results["steps"].append({
                "step": "initial_decision",
                "status": "completed",
                "action": decision.recommended_action,
                "consensus_reached": decision.consensus_reached,
                "consensus_score": decision.consensus_score
            })

            # Overall results
            results["overall_status"] = "completed"
            results["execution_time_ms"] = (time.time() - start_time) * 1000
            results["timestamp"] = datetime.now(timezone.utc).isoformat()

            logger.info(f"Full workflow execution completed in {results['execution_time_ms']:.2f}ms")

            return results

        except Exception as e:
            logger.error(f"Full workflow execution failed: {e}")
            results["overall_status"] = "failed"
            results["error"] = str(e)
            results["execution_time_ms"] = (time.time() - start_time) * 1000
            results["timestamp"] = datetime.now(timezone.utc).isoformat()
            return results

    def cleanup(self):
        """Cleanup resources and stop background threads."""
        logger.info("Cleaning up Comprehensive Integration Manager...")

        # Stop cleanup thread
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5)
            logger.info("Cleanup thread stopped")

        # Cleanup old data
        try:
            self.bubblelab_ui_integration.clear_old_data(max_age_seconds=0)
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

        logger.info("Cleanup complete")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass


# Global instance
_integration_manager: Optional[ComprehensiveIntegrationManager] = None


def get_integration_manager() -> ComprehensiveIntegrationManager:
    """Get or create global integration manager instance."""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = ComprehensiveIntegrationManager()
    return _integration_manager


__all__ = [
    "IntegrationStatus",
    "IntegrationHealth",
    "ComprehensiveIntegrationManager",
    "get_integration_manager"
]
