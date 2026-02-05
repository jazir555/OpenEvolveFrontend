"""
Result Summarizer for OpenEvolve + LoongFlow PES Integration.

Summarizes evolution results and extracts actionable insights
and reusable patterns for the knowledge base.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of reusable patterns."""
    PARAMETER_SET = "parameter_set"
    STRATEGY = "strategy"
    MUTATION_OPERATOR = "mutation_operator"
    CONVERGENCE_BEHAVIOR = "convergence_behavior"


@dataclass
class Pattern:
    """A reusable pattern discovered during evolution."""
    pattern_type: PatternType
    description: str
    context: Dict[str, Any]
    applicability_score: float  # 0.0 - 1.0
    success_rate: float  # 0.0 - 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Strategy:
    """A reusable strategy discovered during evolution."""
    name: str
    description: str
    applicable_problems: List[str]
    effectiveness: float  # 0.0 - 1.0
    avg_cost: float
    avg_iterations: int
    success_rate: float


@dataclass
class EvolutionSummary:
    """Summary of evolution execution."""
    success: bool
    final_fitness: float
    iterations_completed: int
    budget_used: Dict[str, float]
    
    # Insights
    key_insights: List[str]
    failure_modes: List[str]
    success_factors: List[str]
    
    # Recommendations
    recommended_next_actions: List[str]
    suggested_parameter_adjustments: Dict[str, Any]
    
    # Knowledge extracted
    patterns_discovered: List[Pattern]
    reusable_strategies: List[Strategy]
    
    # Execution metadata
    execution_trace: Dict[str, Any] = field(default_factory=dict)


class ResultAnalyzer:
    """Analyzes evolution execution results."""
    
    def analyze(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze execution results."""
        analysis = {
            "success": execution.get("success", False),
            "fitness_achieved": execution.get("final_fitness", 0),
            "iterations_used": execution.get("iterations", 0),
            "convergence_analysis": self._analyze_convergence(execution),
            "cost_efficiency": self._analyze_cost_efficiency(execution),
            "strategy_effectiveness": self._analyze_strategy(execution)
        }
        return analysis
    
    def _analyze_convergence(self, execution: Dict) -> Dict:
        """Analyze convergence behavior."""
        state = execution.get("state")
        if not state:
            return {}
        
        phase_results = getattr(state, "phase_results", [])
        
        if not phase_results:
            return {"status": "unknown"}
        
        fitness_progression = [p.final_fitness for p in phase_results]
        
        return {
            "status": "converged" if execution.get("success") else "stagnated",
            "fitness_progression": fitness_progression,
            "improvement_rate": (
                (fitness_progression[-1] - fitness_progression[0]) / len(fitness_progression)
                if len(fitness_progression) > 1 else 0
            )
        }
    
    def _analyze_cost_efficiency(self, execution: Dict) -> Dict:
        """Analyze cost efficiency."""
        cost = execution.get("cost_summary", {})
        
        consumed = cost.get("consumed", {})
        allocated = cost.get("allocated", {})
        
        if not allocated:
            return {"efficiency": "unknown"}
        
        cost_efficiency = consumed.get("cost_usd", 0) / allocated.get("cost_usd", 1)
        token_efficiency = consumed.get("tokens", 0) / allocated.get("tokens", 1)
        
        return {
            "cost_efficiency": cost_efficiency,
            "token_efficiency": token_efficiency,
            "budget_utilization": (cost_efficiency + token_efficiency) / 2,
            "cost_per_fitness": consumed.get("cost_usd", 0) / max(execution.get("final_fitness", 0.01), 0.01)
        }
    
    def _analyze_strategy(self, execution: Dict) -> Dict:
        """Analyze strategy effectiveness."""
        plan = execution.get("plan", {})
        adaptations = plan.get("adaptations", [])
        
        return {
            "phases_executed": plan.get("phases_executed", 0),
            "phases_planned": plan.get("phases_planned", 0),
            "adaptations_made": len(adaptations),
            "adaptation_effectiveness": self._calculate_adaptation_effectiveness(adaptations)
        }
    
    def _calculate_adaptation_effectiveness(self, adaptations: List) -> float:
        """Calculate effectiveness of adaptations."""
        if not adaptations:
            return 1.0  # No adaptations needed
        
        # Count successful adaptations
        # (In practice, would track before/after metrics)
        return 0.8  # Placeholder


class PatternExtractor:
    """Extracts reusable patterns from evolution results."""
    
    def extract_patterns(self, execution: Dict) -> List[Pattern]:
        """Extract patterns from execution."""
        patterns = []
        
        # Extract parameter patterns
        param_pattern = self._extract_parameter_pattern(execution)
        if param_pattern:
            patterns.append(param_pattern)
        
        # Extract convergence patterns
        conv_pattern = self._extract_convergence_pattern(execution)
        if conv_pattern:
            patterns.append(conv_pattern)
        
        # Extract strategy patterns
        strat_pattern = self._extract_strategy_pattern(execution)
        if strat_pattern:
            patterns.append(strat_pattern)
        
        return patterns
    
    def _extract_parameter_pattern(self, execution: Dict) -> Optional[Pattern]:
        """Extract parameter setting pattern."""
        plan = execution.get("plan", {})
        
        if not plan:
            return None
        
        return Pattern(
            pattern_type=PatternType.PARAMETER_SET,
            description="Effective parameter configuration for this problem type",
            context={
                "parameters": plan.get("suggested_parameters", {}),
                "problem_characteristics": self._extract_problem_characteristics(execution)
            },
            applicability_score=execution.get("final_fitness", 0),
            success_rate=1.0 if execution.get("success") else 0.0
        )
    
    def _extract_convergence_pattern(self, execution: Dict) -> Optional[Pattern]:
        """Extract convergence behavior pattern."""
        state = execution.get("state")
        
        if not state:
            return None
        
        phase_results = getattr(state, "phase_results", [])
        
        return Pattern(
            pattern_type=PatternType.CONVERGENCE_BEHAVIOR,
            description=f"Converged in {len(phase_results)} phases",
            context={
                "iterations_per_phase": [p.iterations for p in phase_results],
                "fitness_per_phase": [p.final_fitness for p in phase_results]
            },
            applicability_score=0.8,
            success_rate=1.0 if execution.get("success") else 0.0
        )
    
    def _extract_strategy_pattern(self, execution: Dict) -> Optional[Pattern]:
        """Extract strategy pattern."""
        plan = execution.get("plan", {})
        
        if not plan:
            return None
        
        return Pattern(
            pattern_type=PatternType.STRATEGY,
            description=f"Strategy: {plan.get('recommended_mode', 'unknown')}",
            context={
                "mode": plan.get("recommended_mode"),
                "phases": [p.name for p in plan.get("phases", [])]
            },
            applicability_score=0.75,
            success_rate=1.0 if execution.get("success") else 0.0
        )
    
    def _extract_problem_characteristics(self, execution: Dict) -> Dict:
        """Extract problem characteristics."""
        # Would extract from problem definition
        return {}


class InsightGenerator:
    """Generates insights from evolution results."""
    
    def generate_insights(self, analysis: Dict, execution: Dict) -> List[str]:
        """Generate key insights."""
        insights = []
        
        # Convergence insights
        conv = analysis.get("convergence_analysis", {})
        if conv.get("status") == "converged":
            insights.append("Evolution converged successfully")
            
            improvement = conv.get("improvement_rate", 0)
            if improvement > 0.1:
                insights.append(f"Rapid improvement rate: {improvement:.3f} per phase")
        
        # Cost insights
        cost = analysis.get("cost_efficiency", {})
        utilization = cost.get("budget_utilization", 1.0)
        if utilization < 0.5:
            insights.append(f"Efficient budget usage: {utilization:.1%} consumed")
        elif utilization > 0.9:
            insights.append("Budget fully utilized - consider increasing for better results")
        
        # Strategy insights
        strategy = analysis.get("strategy_effectiveness", {})
        if strategy.get("adaptations_made", 0) > 0:
            insights.append(f"{strategy['adaptations_made']} parameter adaptations improved convergence")
        
        return insights
    
    def identify_failure_modes(self, analysis: Dict, execution: Dict) -> List[str]:
        """Identify failure modes."""
        failures = []
        
        if not execution.get("success"):
            conv = analysis.get("convergence_analysis", {})
            if conv.get("status") == "stagnated":
                failures.append("Fitness stagnated before reaching target")
            
            cost = analysis.get("cost_efficiency", {})
            if cost.get("budget_utilization", 0) > 0.95:
                failures.append("Budget exhausted before convergence")
        
        return failures
    
    def identify_success_factors(self, analysis: Dict, execution: Dict) -> List[str]:
        """Identify factors contributing to success."""
        factors = []
        
        if execution.get("success"):
            strategy = analysis.get("strategy_effectiveness", {})
            if strategy.get("adaptations_made", 0) > 0:
                factors.append("Adaptive parameter adjustment")
            
            conv = analysis.get("convergence_analysis", {})
            if conv.get("improvement_rate", 0) > 0.05:
                factors.append("Strong fitness progression")
        
        return factors


class RecommendationEngine:
    """Generates recommendations based on results."""
    
    def recommend_actions(self, analysis: Dict, execution: Dict) -> List[str]:
        """Recommend next actions."""
        actions = []
        
        conv = analysis.get("convergence_analysis", {})
        cost = analysis.get("cost_efficiency", {})
        
        if conv.get("status") == "stagnated":
            actions.append("Increase mutation rate for more exploration")
            actions.append("Consider restarting with different initial population")
        
        if cost.get("budget_utilization", 0) > 0.9:
            actions.append("Increase budget allocation for more iterations")
        
        if not execution.get("success"):
            actions.append("Review problem constraints for feasibility")
        
        return actions
    
    def suggest_parameter_adjustments(self, analysis: Dict) -> Dict[str, Any]:
        """Suggest parameter adjustments for future runs."""
        adjustments = {}
        
        conv = analysis.get("convergence_analysis", {})
        if conv.get("improvement_rate", 0) < 0.01:
            adjustments["mutation_rate"] = "increase by 20%"
            adjustments["population_size"] = "increase by 25%"
        
        cost = analysis.get("cost_efficiency", {})
        if cost.get("cost_per_fitness", 0) > 1.0:
            adjustments["evaluation_budget"] = "reduce by 20%"
        
        return adjustments


class ResultSummarizer:
    """
    Summarizes evolution results for knowledge extraction.
    
    Responsibilities:
    - Analyze execution trace
    - Extract patterns and insights
    - Generate recommendations
    - Update knowledge base
    """
    
    def __init__(self, knowledge_engine=None):
        self.knowledge_engine = knowledge_engine
        self.result_analyzer = ResultAnalyzer()
        self.pattern_extractor = PatternExtractor()
        self.insight_generator = InsightGenerator()
        self.recommendation_engine = RecommendationEngine()
    
    def summarize(self, execution: Dict[str, Any]) -> EvolutionSummary:
        """
        Summarize evolution execution.
        
        Steps:
        1. Analyze execution results
        2. Extract patterns
        3. Generate insights
        4. Create recommendations
        5. Update knowledge base (if available)
        """
        logger.info("Summarizing evolution results")
        
        # 1. Analyze results
        analysis = self.result_analyzer.analyze(execution)
        
        # 2. Extract patterns
        patterns = self.pattern_extractor.extract_patterns(execution)
        
        # 3. Generate insights
        insights = self.insight_generator.generate_insights(analysis, execution)
        failures = self.insight_generator.identify_failure_modes(analysis, execution)
        successes = self.insight_generator.identify_success_factors(analysis, execution)
        
        # 4. Generate recommendations
        actions = self.recommendation_engine.recommend_actions(analysis, execution)
        adjustments = self.recommendation_engine.suggest_parameter_adjustments(analysis)
        
        # 5. Build strategies list
        strategies = self._build_strategies(patterns, execution)
        
        # 6. Update knowledge base
        if self.knowledge_engine:
            self._update_knowledge(patterns, execution)
        
        # Build summary
        cost_summary = execution.get("cost_summary", {})
        consumed = cost_summary.get("consumed", {})
        
        return EvolutionSummary(
            success=execution.get("success", False),
            final_fitness=execution.get("final_fitness", 0.0),
            iterations_completed=execution.get("iterations", 0),
            budget_used=consumed,
            key_insights=insights,
            failure_modes=failures,
            success_factors=successes,
            recommended_next_actions=actions,
            suggested_parameter_adjustments=adjustments,
            patterns_discovered=patterns,
            reusable_strategies=strategies,
            execution_trace=execution.get("state", {}).__dict__ if execution.get("state") else {}
        )
    
    def _build_strategies(
        self,
        patterns: List[Pattern],
        execution: Dict
    ) -> List[Strategy]:
        """Build reusable strategies from patterns."""
        strategies = []
        
        for pattern in patterns:
            if pattern.pattern_type == PatternType.STRATEGY:
                strategy = Strategy(
                    name=pattern.description,
                    description=f"Strategy for {pattern.context.get('mode', 'unknown')} evolution",
                    applicable_problems=["similar_complexity"],
                    effectiveness=pattern.applicability_score,
                    avg_cost=execution.get("cost_summary", {}).get("consumed", {}).get("cost_usd", 0),
                    avg_iterations=execution.get("iterations", 0),
                    success_rate=pattern.success_rate
                )
                strategies.append(strategy)
        
        return strategies
    
    def _update_knowledge(self, patterns: List[Pattern], execution: Dict):
        """Update knowledge base with extracted patterns."""
        if not self.knowledge_engine:
            return
        
        for pattern in patterns:
            try:
                # Store pattern in knowledge base
                self.knowledge_engine.store_pattern(
                    pattern_type=pattern.pattern_type.value,
                    pattern_data=pattern.context,
                    metadata={
                        "success_rate": pattern.success_rate,
                        "applicability": pattern.applicability_score
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to store pattern: {e}")
