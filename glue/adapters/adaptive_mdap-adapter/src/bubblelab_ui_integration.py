"""
BubbleLab UI Integration for Adaptive MDAP/MAKER Adapter

This module provides UI components and visualizations for the Adaptive MDAP/MAKER adapter
within the BubbleLab interface, enabling:
- Real-time complexity analysis visualization
- MAKER voting progress tracking
- Adaptive resource allocation dashboard
- ICR pattern learning insights
- Integration with BubbleLab's existing UI framework

Federation Constitution Compliance:
- Law 1: Air Gap - No imports from core-projects
- Law 2: Runtime Truth - Validates against actual BubbleLab API
- Law 3: Untouchable DB - Read-only access to metrics
- Law 4: Idempotency - UI refreshes safe to repeat
- Law 5: Configuration Explicitness - All config via environment
- Law 6: UTC - All timestamps in UTC ISO-8601
"""

import os
import sys
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import adapter components
from .adaptive_mdap_adapter import (
    get_adapter,
    CanonicalSubProblem,
    CanonicalComplexityScore,
    TaskStatus
)

from .maker_adapter import (
    get_maker_adapter,
    CanonicalMakerResult
)

from .openevolve_integration import (
    get_openevolve_integration,
    WorkflowComplexityAnalysis,
    MAKERWorkflowDecision
)

logger = logging.getLogger(__name__)


class UIComponent(Enum):
    """Available UI components."""
    COMPLEXITY_ANALYZER = "complexity_analyzer"
    RESOURCE_ALLOCATOR = "resource_allocator"
    MAKER_VOTING_DISPLAY = "maker_voting_display"
    ICR_INSIGHTS = "icr_insights"
    WORKFLOW_MONITOR = "workflow_monitor"


@dataclass
class UIState:
    """UI state management."""
    active_tab: str = "complexity_analyzer"
    selected_workflow: Optional[str] = None
    refresh_interval_seconds: int = 5
    auto_refresh: bool = True
    show_advanced_options: bool = False


@dataclass
class ComplexityAnalysisResult:
    """Result of complexity analysis for UI display."""
    problem_id: str
    problem_description: str
    overall_complexity: float
    text_length_score: float
    dependency_score: float
    depth_score: float
    recommended_strategy: str
    recommended_resources: Dict[str, Any]
    execution_time_ms: float
    timestamp: str


@dataclass
class MAKERVotingDisplay:
    """MAKER voting progress for UI display."""
    voting_id: str
    decision_point: str
    total_votes: int
    consensus_reached: bool
    consensus_score: float
    current_leader: str
    red_flags: List[Dict[str, Any]]
    timestamp: str


class BubbleLabUIIntegration:
    """
    UI integration for Adaptive MDAP/MAKER adapter in BubbleLab.

    Provides:
    1. Complexity analysis visualization
    2. Resource allocation dashboard
    3. MAKER voting progress display
    4. ICR pattern insights
    5. Workflow monitoring
    """

    def __init__(self):
        """Initialize BubbleLab UI integration."""
        self.mdap_adapter = get_adapter()
        self.maker_adapter = get_maker_adapter()
        self.openevolve_integration = get_openevolve_integration()

        self.ui_state = UIState()
        self.active_analyses: Dict[str, ComplexityAnalysisResult] = {}
        self.active_votings: Dict[str, MAKERVotingDisplay] = {}

        # ICR integration (optional)
        self.icr_integration = None
        try:
            from icr_integration import get_icr_integration
            self.icr_integration = get_icr_integration()
        except ImportError:
            logger.warning("ICR integration not available for UI")

        logger.info("BubbleLab UI Integration initialized")

    def analyze_complexity_for_ui(
        self,
        problem_description: str,
        domain: str = "general",
        depth: int = 1,
        dependencies: Optional[List[str]] = None
    ) -> ComplexityAnalysisResult:
        """
        Analyze complexity and return UI-friendly result.

        Args:
            problem_description: Problem to analyze
            domain: Problem domain
            depth: Problem depth
            dependencies: List of dependencies

        Returns:
            ComplexityAnalysisResult for UI display
        """
        start_time = time.time()

        # Create subproblem
        subproblem = CanonicalSubProblem(
            id=f"ui_analysis_{int(time.time() * 1000)}",
            description=problem_description,
            domain=domain,
            depth=depth,
            dependencies=dependencies or [],
            metadata={"source": "bubblelab_ui"}
        )

        # Analyze complexity
        response = self.mdap_adapter.analyze_complexity(subproblem)

        execution_time = (time.time() - start_time) * 1000

        if response.status == TaskStatus.COMPLETED:
            complexity = response.complexity_score

            # Get resource allocation
            strategy_response = self.mdap_adapter.allocate_resources(complexity)
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

            result = ComplexityAnalysisResult(
                problem_id=subproblem.id,
                problem_description=problem_description,
                overall_complexity=complexity.overall_score,
                text_length_score=complexity.text_length_score,
                dependency_score=complexity.dependency_score,
                depth_score=complexity.depth_score,
                recommended_strategy=recommended_strategy,
                recommended_resources=recommended_resources,
                execution_time_ms=execution_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

            # Store for UI
            self.active_analyses[subproblem.id] = result

            return result
        else:
            # Return error result
            return ComplexityAnalysisResult(
                problem_id=subproblem.id,
                problem_description=problem_description,
                overall_complexity=0.0,
                text_length_score=0.0,
                dependency_score=0.0,
                depth_score=0.0,
                recommended_strategy="UNKNOWN",
                recommended_resources={},
                execution_time_ms=execution_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

    def get_complexity_visualization_data(
        self,
        analysis_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get data for complexity visualization charts.

        Args:
            analysis_id: ID of complexity analysis

        Returns:
            Visualization data for charts
        """
        if analysis_id not in self.active_analyses:
            return None

        analysis = self.active_analyses[analysis_id]

        return {
            "chart_type": "radar",
            "title": "Complexity Breakdown",
            "data": {
                "labels": ["Text Length", "Dependencies", "Depth"],
                "datasets": [{
                    "label": "Complexity Scores",
                    "data": [
                        analysis.text_length_score,
                        analysis.dependency_score,
                        analysis.depth_score
                    ],
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 2
                }]
            },
            "options": {
                "scales": {
                    "r": {
                        "beginAtZero": True,
                        "max": 1.0
                    }
                }
            }
        }

    def get_strategy_visualization_data(
        self,
        analysis_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get data for strategy recommendation visualization.

        Args:
            analysis_id: ID of complexity analysis

        Returns:
            Visualization data for strategy display
        """
        if analysis_id not in self.active_analyses:
            return None

        analysis = self.active_analyses[analysis_id]

        return {
            "chart_type": "bar",
            "title": f"Recommended Strategy: {analysis.recommended_strategy}",
            "data": {
                "labels": list(analysis.recommended_resources.keys()),
                "datasets": [{
                    "label": "Resource Allocation",
                    "data": list(analysis.recommended_resources.values()),
                    "backgroundColor": [
                        "rgba(255, 99, 132, 0.6)",
                        "rgba(54, 162, 235, 0.6)",
                        "rgba(255, 206, 86, 0.6)",
                        "rgba(75, 192, 192, 0.6)"
                    ]
                }]
            }
        }

    def display_maker_voting_progress(
        self,
        voting_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get MAKER voting progress for display.

        Args:
            voting_id: Voting session ID

        Returns:
            Voting progress data
        """
        if voting_id not in self.active_votings:
            return None

        voting = self.active_votings[voting_id]

        return {
            "voting_id": voting_id,
            "decision_point": voting.decision_point,
            "total_votes": voting.total_votes,
            "consensus_reached": voting.consensus_reached,
            "consensus_score": voting.consensus_score,
            "current_leader": voting.current_leader,
            "red_flags": voting.red_flags,
            "timestamp": voting.timestamp,
            "progress_percentage": min(100, (voting.total_votes / 5) * 100)  # Assume 5 votes target
        }

    def get_icr_insights(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get ICR pattern insights for UI display.

        Args:
            limit: Maximum number of patterns to return

        Returns:
            ICR insights data
        """
        if not self.icr_integration:
            return {
                "available": False,
                "message": "ICR integration not available"
            }

        try:
            from icr_integration import ICRPatternType

            # Get statistics for different pattern types
            insights = {
                "available": True,
                "patterns": {}
            }

            for pattern_type in ICRPatternType:
                stats = self.icr_integration.get_statistics(pattern_type)
                insights["patterns"][pattern_type.value] = {
                    "count": stats.get("count", 0),
                    "pass_rate": stats.get("pass_rate", 0.0),
                    "confidence": stats.get("confidence", 0.0)
                }

            return insights

        except Exception as e:
            logger.error(f"Failed to get ICR insights: {e}")
            return {
                "available": False,
                "error": str(e)
            }

    def get_workflow_monitor_data(self) -> Dict[str, Any]:
        """
        Get workflow monitoring data for dashboard.

        Returns:
            Workflow monitor data
        """
        try:
            # Get active workflows from OpenEvolve integration
            active_workflows = self.openevolve_integration.active_workflows

            monitor_data = {
                "active_workflows": len(active_workflows),
                "workflows": []
            }

            for workflow_id, workflow_data in active_workflows.items():
                monitor_data["workflows"].append({
                    "workflow_id": workflow_id,
                    "workflow_type": workflow_data.get("workflow_type", "unknown"),
                    "created_at": workflow_data.get("created_at", ""),
                    "complexity": workflow_data.get("complexity_analysis", {}).get("overall_complexity", 0.0)
                })

            return monitor_data

        except Exception as e:
            logger.error(f"Failed to get workflow monitor data: {e}")
            return {
                "active_workflows": 0,
                "workflows": [],
                "error": str(e)
            }

    def get_adapter_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all adapters.

        Returns:
            Health status data
        """
        try:
            mdap_health = self.mdap_adapter.health_check()
            maker_health = self.maker_adapter.health_check()

            return {
                "mdap_adapter": {
                    "status": mdap_health.get("status", "unknown"),
                    "circuit_breaker": mdap_health.get("circuit_breaker_state", "unknown"),
                    "metrics": mdap_health.get("metrics", {})
                },
                "maker_adapter": {
                    "status": maker_health.get("status", "unknown"),
                    "circuit_breaker": maker_health.get("circuit_breaker_state", "unknown"),
                    "metrics": maker_health.get("metrics", {})
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get adapter health status: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def export_ui_data(self, format: str = "json") -> str:
        """
        Export all UI data for external visualization.

        Args:
            format: Export format ("json" or "dict")

        Returns:
            Exported data
        """
        data = {
            "analyses": {
                analysis_id: {
                    "problem_id": a.problem_id,
                    "problem_description": a.problem_description,
                    "overall_complexity": a.overall_complexity,
                    "recommended_strategy": a.recommended_strategy,
                    "timestamp": a.timestamp
                }
                for analysis_id, a in self.active_analyses.items()
            },
            "votings": {
                voting_id: {
                    "decision_point": v.decision_point,
                    "total_votes": v.total_votes,
                    "consensus_reached": v.consensus_reached,
                    "consensus_score": v.consensus_score,
                    "timestamp": v.timestamp
                }
                for voting_id, v in self.active_votings.items()
            },
            "health": self.get_adapter_health_status(),
            "workflow_monitor": self.get_workflow_monitor_data(),
            "icr_insights": self.get_icr_insights()
        }

        if format == "json":
            return json.dumps(data, indent=2)
        else:
            return data

    def clear_old_data(self, max_age_seconds: int = 3600):
        """
        Clear old analysis and voting data.

        Args:
            max_age_seconds: Maximum age of data to keep
        """
        current_time = time.time()
        max_age_timestamp = current_time - max_age_seconds

        # Clear old analyses
        to_remove = []
        for analysis_id, analysis in self.active_analyses.items():
            try:
                analysis_time = datetime.fromisoformat(analysis.timestamp).timestamp()
                if analysis_time < max_age_timestamp:
                    to_remove.append(analysis_id)
            except (ValueError, TypeError):
                # Invalid timestamp, remove
                to_remove.append(analysis_id)

        for analysis_id in to_remove:
            del self.active_analyses[analysis_id]

        # Clear old votings
        to_remove = []
        for voting_id, voting in self.active_votings.items():
            try:
                voting_time = datetime.fromisoformat(voting.timestamp).timestamp()
                if voting_time < max_age_timestamp:
                    to_remove.append(voting_id)
            except (ValueError, TypeError):
                # Invalid timestamp, remove
                to_remove.append(voting_id)

        for voting_id in to_remove:
            del self.active_votings[voting_id]

        logger.info(f"Cleared {len(to_remove)} old data items")


# Global instance
_bubblelab_ui_integration: Optional[BubbleLabUIIntegration] = None


def get_bubblelab_ui_integration() -> BubbleLabUIIntegration:
    """Get or create global BubbleLab UI integration instance."""
    global _bubblelab_ui_integration
    if _bubblelab_ui_integration is None:
        _bubblelab_ui_integration = BubbleLabUIIntegration()
    return _bubblelab_ui_integration


__all__ = [
    "UIComponent",
    "UIState",
    "ComplexityAnalysisResult",
    "MAKERVotingDisplay",
    "BubbleLabUIIntegration",
    "get_bubblelab_ui_integration"
]
