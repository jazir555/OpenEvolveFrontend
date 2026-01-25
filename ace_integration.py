"""
ACE (Agent, Critique, Enhancement) Integration

Provides self-improving agents with automated critique mechanisms,
enhancement suggestions, and performance improvement tracking.
ACE-enhanced agents should improve 20-35% over time according to specification.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from sovereign_data_models import (
    DecompositionPlan,
    SubProblem,
    ProblemDefinition,
    EnhancedAgent,
    ACECritiqueReport,
    EnhancementSuggestion,
    ImprovementMetrics,
    generate_id
)

logger = logging.getLogger(__name__)


class ACEIntegration:
    """
    Integration with ACE (Agent, Critique, Enhancement) system.

    ACE provides:
    - Self-improving agents
    - Automated critique mechanisms
    - Enhancement suggestions
    - Performance improvement
    """

    def __init__(self, ace_endpoint: str = None):
        """
        Initialize with ACE endpoint.

        Args:
            ace_endpoint: Optional endpoint for ACE service
        """
        self.ace_enabled = ace_endpoint is not None
        self.ace_endpoint = ace_endpoint

        # Track enhanced agents
        self.enhanced_agents: Dict[str, EnhancedAgent] = {}

        # Track performance over time
        self.performance_history: Dict[str, List[float]] = {}

        # Track enhancement count
        self.enhancement_count: int = 0

        logger.info(f"ACE Integration initialized (enabled={self.ace_enabled})")

    def enhance_agent_with_ace(
        self,
        agent_type: str,
        agent_config: Dict[str, Any],
        domain: str
    ) -> EnhancedAgent:
        """
        Create ACE-enhanced agent.

        ACE-enhanced agents:
        - Learn from experience
        - Self-critique their work
        - Auto-improve over time
        - Adapt to domain patterns

        Args:
            agent_type: Type of agent ("solver", "patcher", "red_team", "gold_team")
            agent_config: Base model configuration
            domain: Domain for adaptation

        Returns:
            EnhancedAgent configuration
        """
        agent_id = f"ace_{agent_type}_{domain}_{generate_id()}"

        # Create enhanced agent
        enhanced_agent = EnhancedAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            base_config=agent_config,
            ace_enabled=self.ace_enabled,
            learning_rate=0.1,  # Moderate learning rate
            critique_threshold=0.7,  # Critique when confidence < 70%
            initial_performance=0.5,
            current_performance=0.5,
            improvement_percentage=0.0,
            self_critique=True,
            auto_enhancement=True,
            domain_adaptation=True,
            enhanced_at=datetime.now(),
            enhancement_count=0
        )

        # Store agent
        self.enhanced_agents[agent_id] = enhanced_agent
        self.performance_history[agent_id] = [0.5]

        logger.info(f"Created ACE-enhanced agent: {agent_id} (type={agent_type}, domain={domain})")

        return enhanced_agent

    def generate_critique(
        self,
        work_product: Any,
        critique_type: str,
        ace_agent_id: str = None
    ) -> ACECritiqueReport:
        """
        Generate automated critique using ACE.

        Critique types:
        - "solution": Critique solution quality
        - "decomposition": Critique decomposition quality
        - "strategy": Critique strategy choice
        - "team": Critique team assignment

        Args:
            work_product: The work to critique (plan, solution, etc.)
            critique_type: Type of critique
            ace_agent_id: Optional ACE agent to use

        Returns:
            ACECritiqueReport with findings
        """
        work_product_id = getattr(work_product, 'id', 'unknown')

        critique_id = generate_id("critique")

        # In production, this would use ACE system or LLM
        # For now, provide a basic implementation
        critique = ACECritiqueReport(
            critique_id=critique_id,
            work_product_id=work_product_id,
            critique_type=critique_type,
            strengths=[],
            weaknesses=[],
            suggestions=[],
            overall_score=0.7,
            dimension_scores={},
            critical_issues=[],
            high_priority_issues=[],
            medium_priority_issues=[],
            generated_at=datetime.now(),
            confidence=0.7
        )

        # Analyze based on critique type
        if critique_type == "decomposition" and isinstance(work_product, DecompositionPlan):
            critique = self._critique_decomposition(work_product, critique_id)
        elif critique_type == "solution" and hasattr(work_product, 'solution_content'):
            critique = self._critique_solution(work_product, critique_id)
        else:
            # Generic critique
            critique = self._generate_generic_critique(work_product, critique_type, critique_id)

        logger.info(f"Generated ACE critique: {critique_id} (type={critique_type}, score={critique.overall_score:.2f})")

        return critique

    def _critique_decomposition(
        self,
        plan: DecompositionPlan,
        critique_id: str
    ) -> ACECritiqueReport:
        """Critique a decomposition plan."""
        strengths = []
        weaknesses = []
        suggestions = []
        critical_issues = []
        high_priority_issues = []
        medium_priority_issues = []

        # Check completeness
        if len(plan.sub_problems) == 0:
            critical_issues.append("No sub-problems defined")
        elif len(plan.sub_problems) < 3:
            weaknesses.append("Decomposition may be too coarse")
        else:
            strengths.append(f"Good granularity with {len(plan.sub_problems)} sub-problems")

        # Check quality scores
        if plan.enhanced_quality_scores:
            eqs = plan.enhanced_quality_scores

            if eqs.completeness_score < 0.7:
                high_priority_issues.append("Completeness score below threshold")
                weaknesses.append("Some aspects may be missing")

            if eqs.consistency_score < 0.7:
                high_priority_issues.append("Consistency score below threshold")
                weaknesses.append("Inconsistencies detected")

            if eqs.feasibility_score < 0.7:
                medium_priority_issues.append("Feasibility concerns")
                weaknesses.append("Some sub-problems may be difficult")

            if eqs.dependency_score < 0.7:
                medium_priority_issues.append("Dependency issues")
                suggestions.append("Review and refine dependencies")

        # Check complexity balance
        complexities = [sp.complexity_score.overall_complexity for sp in plan.sub_problems]
        if complexities:
            avg_complexity = sum(complexities) / len(complexities)
            max_complexity = max(complexities)
            min_complexity = min(complexities)

            if max_complexity - min_complexity > 5:
                suggestions.append("Consider balancing complexity across sub-problems")
                weaknesses.append("High complexity variance")

            if avg_complexity > 7:
                suggestions.append("Consider further decomposition of complex sub-problems")
                weaknesses.append("Overall complexity high")

        # Generate suggestions
        if not plan.dependency_graph:
            suggestions.append("Consider creating a dependency graph")

        if not plan.validation_checkpoints:
            suggestions.append("Add validation checkpoints for quality assurance")

        # Calculate overall score
        dimension_scores = {
            "completeness": plan.enhanced_quality_scores.completeness_score if plan.enhanced_quality_scores else 0.7,
            "consistency": plan.enhanced_quality_scores.consistency_score if plan.enhanced_quality_scores else 0.7,
            "feasibility": plan.enhanced_quality_scores.feasibility_score if plan.enhanced_quality_scores else 0.7,
            "balance": 0.8 if complexities and max_complexity - min_complexity < 5 else 0.6
        }

        overall_score = sum(dimension_scores.values()) / len(dimension_scores)

        return ACECritiqueReport(
            critique_id=critique_id,
            work_product_id=plan.id,
            critique_type="decomposition",
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            critical_issues=critical_issues,
            high_priority_issues=high_priority_issues,
            medium_priority_issues=medium_priority_issues,
            generated_at=datetime.now(),
            confidence=0.8
        )

    def _critique_solution(self, solution: Any, critique_id: str) -> ACECritiqueReport:
        """Critique a solution attempt."""
        strengths = ["Solution provided"]
        weaknesses = []
        suggestions = []
        critical_issues = []
        high_priority_issues = []
        medium_priority_issues = []

        solution_content = getattr(solution, 'solution_content', '')

        if not solution_content:
            critical_issues.append("No solution content provided")
        elif len(solution_content) < 100:
            weaknesses.append("Solution appears too brief")
            suggestions.append("Expand solution with more details")
        else:
            strengths.append("Comprehensive solution provided")

        return ACECritiqueReport(
            critique_id=critique_id,
            work_product_id=getattr(solution, 'id', 'unknown'),
            critique_type="solution",
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            overall_score=0.7,
            dimension_scores={"completeness": 0.7, "clarity": 0.7},
            critical_issues=critical_issues,
            high_priority_issues=high_priority_issues,
            medium_priority_issues=medium_priority_issues,
            generated_at=datetime.now(),
            confidence=0.7
        )

    def _generate_generic_critique(
        self,
        work_product: Any,
        critique_type: str,
        critique_id: str
    ) -> ACECritiqueReport:
        """Generate a generic critique."""
        return ACECritiqueReport(
            critique_id=critique_id,
            work_product_id=getattr(work_product, 'id', 'unknown'),
            critique_type=critique_type,
            strengths=["Work product provided"],
            weaknesses=[],
            suggestions=["Review for improvements"],
            overall_score=0.7,
            dimension_scores={},
            generated_at=datetime.now(),
            confidence=0.6
        )

    def suggest_enhancements(
        self,
        work_product: Any,
        performance_context: Dict[str, float]
    ) -> List[EnhancementSuggestion]:
        """
        Suggest enhancements using ACE.

        Analyzes:
        - Performance metrics
        - Quality issues
        - Optimization opportunities
        - Best practice violations

        Args:
            work_product: The work to enhance
            performance_context: Performance metrics

        Returns:
            List of EnhancementSuggestion objects
        """
        suggestions = []

        work_product_id = getattr(work_product, 'id', 'unknown')

        # Analyze performance context
        for metric_name, metric_value in performance_context.items():
            if metric_value < 0.7:
                suggestion = EnhancementSuggestion(
                    suggestion_id=generate_id("suggestion"),
                    work_product_id=work_product_id,
                    category=metric_name,
                    description=f"Improve {metric_name} performance",
                    rationale=f"Current {metric_name} score of {metric_value:.2f} is below threshold",
                    implementation_difficulty="medium",
                    estimated_effort=5.0,
                    expected_improvement=0.15,
                    priority="high" if metric_value < 0.5 else "medium"
                )
                suggestions.append(suggestion)

        # Add specific suggestions based on work product type
        if isinstance(work_product, DecompositionPlan):
            # Decomposition-specific enhancements
            if len(work_product.sub_problems) < 3:
                suggestion = EnhancementSuggestion(
                    suggestion_id=generate_id("suggestion"),
                    work_product_id=work_product_id,
                    category="quality",
                    description="Consider finer-grained decomposition",
                    rationale="More sub-problems can improve parallel execution and reduce individual complexity",
                    implementation_difficulty="medium",
                    estimated_effort=6.0,
                    expected_improvement=0.2,
                    priority="medium"
                )
                suggestions.append(suggestion)

        logger.info(f"Generated {len(suggestions)} enhancement suggestions for {work_product_id}")

        return suggestions

    def apply_auto_enhancement(
        self,
        decomposition_plan: DecompositionPlan,
        enhancement_areas: List[str] = None
    ) -> DecompositionPlan:
        """
        Automatically enhance decomposition plan.

        Enhancements:
        - Improve sub-problem boundaries
        - Add missing dependencies
        - Balance complexity
        - Improve quality scores

        Args:
            decomposition_plan: Original plan
            enhancement_areas: Specific areas to enhance (None = all)

        Returns:
            Enhanced decomposition plan
        """
        if not self.ace_enabled:
            logger.warning("ACE not enabled, returning original plan")
            return decomposition_plan

        logger.info(f"Applying ACE auto-enhancement to plan {decomposition_plan.id}")

        # Determine enhancement areas
        if enhancement_areas is None:
            enhancement_areas = ["complexity", "dependencies", "quality"]

        enhanced_sub_problems = []

        for sub_problem in decomposition_plan.sub_problems:
            enhanced_sub = sub_problem

            # Apply complexity balancing
            if "complexity" in enhancement_areas:
                enhanced_sub = self._enhance_complexity(enhanced_sub)

            # Apply dependency improvements
            if "dependencies" in enhancement_areas:
                enhanced_sub = self._enhance_dependencies(enhanced_sub)

            # Apply quality improvements
            if "quality" in enhancement_areas:
                enhanced_sub = self._enhance_quality(enhanced_sub)

            enhanced_sub_problems.append(enhanced_sub)

        # Create enhanced plan
        enhanced_plan = DecompositionPlan(
            id=decomposition_plan.id,
            problem_id=decomposition_plan.problem_id,
            strategy=decomposition_plan.strategy,
            sub_problems=enhanced_sub_problems,
            dependency_graph=decomposition_plan.dependency_graph,
            validation_checkpoints=decomposition_plan.validation_checkpoints,
            quality_scores=decomposition_plan.quality_scores,
            enhanced_quality_scores=decomposition_plan.enhanced_quality_scores,
            confidence_level=min(decomposition_plan.confidence_level + 0.05, 1.0),
            created_by=decomposition_plan.created_by,
            approved_by=decomposition_plan.approved_by,
            status=decomposition_plan.status,
            created_at=decomposition_plan.created_at,
            updated_at=datetime.now(),
            metadata={
                **decomposition_plan.metadata,
                'ace_enhanced': True,
                'enhancement_areas': enhancement_areas,
                'enhancement_count': self.enhancement_count + 1
            }
        )

        self.enhancement_count += 1

        logger.info(f"ACE auto-enhancement complete for plan {decomposition_plan.id}")

        return enhanced_plan

    def _enhance_complexity(self, sub_problem: SubProblem) -> SubProblem:
        """Enhance complexity assessment of sub-problem."""
        # This would analyze and adjust complexity
        # For now, return as-is
        return sub_problem

    def _enhance_dependencies(self, sub_problem: SubProblem) -> SubProblem:
        """Enhance dependencies of sub-problem."""
        # This would analyze and improve dependencies
        # For now, return as-is
        return sub_problem

    def _enhance_quality(self, sub_problem: SubProblem) -> SubProblem:
        """Enhance quality aspects of sub-problem."""
        # This would improve quality metrics
        # For now, return as-is
        return sub_problem

    def track_agent_improvement(
        self,
        agent_id: str,
        before_performance: float,
        after_performance: float
    ):
        """
        Track agent improvement over time.

        ACE agents should improve 20-35% over time according to specification.

        Args:
            agent_id: Agent identifier
            before_performance: Performance before enhancement
            after_performance: Performance after enhancement
        """
        if agent_id not in self.enhanced_agents:
            logger.warning(f"Unknown agent {agent_id}, cannot track improvement")
            return

        agent = self.enhanced_agents[agent_id]

        # Update performance
        agent.current_performance = after_performance

        # Calculate improvement percentage
        if agent.initial_performance > 0:
            improvement = (after_performance - agent.initial_performance) / agent.initial_performance
            agent.improvement_percentage = improvement

        # Update enhancement count
        agent.enhancement_count += 1

        # Track history
        if agent_id not in self.performance_history:
            self.performance_history[agent_id] = []
        self.performance_history[agent_id].append(after_performance)

        logger.info(
            f"Tracked improvement for {agent_id}: "
            f"{before_performance:.3f} -> {after_performance:.3f} "
            f"({agent.improvement_percentage*100:.1f}% improvement)"
        )

    def get_improvement_metrics(
        self,
        agent_id: str,
        time_period: str = "all"
    ) -> ImprovementMetrics:
        """
        Get improvement metrics for ACE-enhanced agent.

        Args:
            agent_id: Agent identifier
            time_period: Time period for metrics ("all", "week", "month", "year")

        Returns:
            ImprovementMetrics with performance data
        """
        if agent_id not in self.enhanced_agents:
            raise ValueError(f"Unknown agent: {agent_id}")

        agent = self.enhanced_agents[agent_id]

        # Get performance history
        history = self.performance_history.get(agent_id, [])

        # Filter by time period
        if time_period == "week":
            cutoff = datetime.now() - timedelta(weeks=1)
        elif time_period == "month":
            cutoff = datetime.now() - timedelta(days=30)
        elif time_period == "year":
            cutoff = datetime.now() - timedelta(days=365)
        else:  # "all"
            cutoff = None

        # Calculate metrics
        initial_performance = agent.initial_performance
        final_performance = agent.current_performance
        improvement_percentage = agent.improvement_percentage

        # Determine trend
        if len(history) >= 3:
            recent = history[-3:]
            if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
                trend = "improving"
            elif all(recent[i] >= recent[i+1] for i in range(len(recent)-1)):
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Calculate rate of improvement
        if len(history) >= 2 and time_period == "all":
            rate_of_improvement = (history[-1] - history[0]) / len(history)
        else:
            rate_of_improvement = 0.0

        metrics = ImprovementMetrics(
            agent_id=agent_id,
            time_period=time_period,
            initial_performance=initial_performance,
            final_performance=final_performance,
            improvement_percentage=improvement_percentage,
            target_improvement=0.275,  # 20-35% target, midpoint
            trend=trend,
            rate_of_improvement=rate_of_improvement,
            total_enhancements=agent.enhancement_count,
            successful_enhancements=sum(1 for p in history if p > agent.initial_performance),
            failed_enhancements=sum(1 for p in history if p <= agent.initial_performance),
            measured_at=datetime.now()
        )

        logger.info(
            f"Improvement metrics for {agent_id}: "
            f"{improvement_percentage*100:.1f}% improvement, "
            f"trend={trend}, "
            f"enhancements={agent.enhancement_count}"
        )

        return metrics

    def get_enhanced_agent(self, agent_id: str) -> Optional[EnhancedAgent]:
        """
        Get enhanced agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            EnhancedAgent or None if not found
        """
        return self.enhanced_agents.get(agent_id)

    def list_enhanced_agents(self) -> List[str]:
        """Get list of all enhanced agent IDs."""
        return list(self.enhanced_agents.keys())

    def is_agent_improving(self, agent_id: str) -> bool:
        """
        Check if agent is showing improvement.

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent is improving (20-35% target range)
        """
        if agent_id not in self.enhanced_agents:
            return False

        agent = self.enhanced_agents[agent_id]

        # Check if improvement is in target range (20-35%)
        target_min = 0.20
        target_max = 0.35

        return target_min <= agent.improvement_percentage <= target_max
