"""Summarization and learning extraction - from LoongFlow PES.

Extracts insights, patterns, and learning from evolution runs.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Discovered pattern during evolution."""
    pattern_type: str  # success, failure, optimization
    description: str
    frequency: int
    examples: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class SuccessFactor:
    """Factor contributing to success."""
    factor: str
    impact_score: float  # -1.0 to 1.0
    evidence: str
    applicable_strategies: List[str] = field(default_factory=list)


@dataclass
class FailureMode:
    """Identified failure mode."""
    mode: str
    frequency: int
    root_cause: str
    prevention_strategy: str
    examples: List[str] = field(default_factory=list)


@dataclass
class EvolutionSummary:
    """Complete summary of evolution run."""
    
    # Basic stats
    total_iterations: int
    total_evaluations: int
    duration_ms: int
    
    # Performance
    initial_fitness: float
    final_fitness: float
    best_fitness: float
    improvement_rate: float
    
    # Convergence
    converged: bool
    convergence_iteration: Optional[int]
    convergence_score: float
    
    # Cost
    total_cost_usd: float
    cost_per_improvement: float
    efficiency_gain: float
    
    # Insights
    patterns: List[Pattern]
    success_factors: List[SuccessFactor]
    failure_modes: List[FailureMode]
    
    # Recommendations
    recommendations: List[str]
    suggested_parameters_for_similar: Dict[str, Any]
    
    # Raw data
    convergence_curve: List[float]
    diversity_curve: List[float]
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    strategy_used: str = ""
    problem_type: str = ""


class InsightExtractor:
    """Extracts insights from evolution history - from LoongFlow."""
    
    def __init__(self):
        self.patterns: List[Pattern] = []
    
    def extract_patterns(
        self,
        fitness_history: List[float],
        diversity_history: List[float],
        code_changes: Optional[List[str]] = None
    ) -> List[Pattern]:
        """Extract patterns from evolution history."""
        patterns = []
        
        # Pattern 1: Rapid early improvement
        if len(fitness_history) >= 5:
            early_improvement = fitness_history[4] - fitness_history[0]
            if early_improvement > 0.2:
                patterns.append(Pattern(
                    pattern_type="success",
                    description="Rapid early improvement detected",
                    frequency=1,
                    confidence=early_improvement,
                    examples=[f"Generation 0->4: {fitness_history[0]:.3f}->{fitness_history[4]:.3f}"]
                ))
        
        # Pattern 2: Plateau detection
        if len(fitness_history) >= 10:
            recent = fitness_history[-10:]
            if statistics.stdev(recent) < 0.01:
                patterns.append(Pattern(
                    pattern_type="optimization",
                    description="Fitness plateau reached - consider increasing mutation",
                    frequency=1,
                    confidence=0.8,
                    examples=[f"Generations {len(fitness_history)-10}-{len(fitness_history)}: std={statistics.stdev(recent):.4f}"]
                ))
        
        # Pattern 3: Diversity collapse
        if diversity_history:
            if diversity_history[-1] < 0.1 and len(diversity_history) > 5:
                patterns.append(Pattern(
                    pattern_type="failure",
                    description="Premature convergence - diversity lost",
                    frequency=1,
                    confidence=0.9,
                    examples=[f"Diversity: {diversity_history[0]:.3f}->{diversity_history[-1]:.3f}"]
                ))
        
        # Pattern 4: Oscillating fitness
        if len(fitness_history) >= 6:
            oscillations = sum(
                1 for i in range(2, len(fitness_history))
                if (fitness_history[i] > fitness_history[i-1]) != (fitness_history[i-1] > fitness_history[i-2])
            )
            if oscillations > len(fitness_history) * 0.4:
                patterns.append(Pattern(
                    pattern_type="optimization",
                    description="Unstable optimization - mutation rate may be too high",
                    frequency=oscillations,
                    confidence=0.7,
                    examples=[f"{oscillations} direction changes in {len(fitness_history)} iterations"]
                ))
        
        self.patterns = patterns
        return patterns
    
    def identify_success_factors(
        self,
        fitness_history: List[float],
        parameter_history: Optional[List[Dict]] = None
    ) -> List[SuccessFactor]:
        """Identify factors that contributed to success."""
        factors = []
        
        if len(fitness_history) < 2:
            return factors
        
        total_improvement = fitness_history[-1] - fitness_history[0]
        
        if total_improvement > 0.5:
            factors.append(SuccessFactor(
                factor="Strong initial population",
                impact_score=0.8,
                evidence=f"Total improvement: {total_improvement:.3f}",
                applicable_strategies=["standard", "pes_enhanced"]
            ))
        
        # Check for consistent improvement
        improvements = [fitness_history[i] - fitness_history[i-1] 
                       for i in range(1, len(fitness_history))]
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        
        if positive_improvements > len(improvements) * 0.6:
            factors.append(SuccessFactor(
                factor="Consistent improvement pattern",
                impact_score=0.7,
                evidence=f"{positive_improvements}/{len(improvements)} iterations improved",
                applicable_strategies=["all"]
            ))
        
        # Check parameter effectiveness
        if parameter_history and len(parameter_history) > 5:
            mutation_rates = [p.get("mutation_rate", 0.1) for p in parameter_history]
            if statistics.stdev(mutation_rates) > 0.05:
                factors.append(SuccessFactor(
                    factor="Adaptive parameter tuning",
                    impact_score=0.6,
                    evidence=f"Mutation rate varied {min(mutation_rates):.3f}-{max(mutation_rates):.3f}",
                    applicable_strategies=["pes_enhanced"]
                ))
        
        return factors
    
    def identify_failure_modes(
        self,
        fitness_history: List[float],
        diversity_history: Optional[List[float]] = None,
        error_history: Optional[List[str]] = None
    ) -> List[FailureMode]:
        """Identify failure modes that occurred."""
        modes = []
        
        # Failure 1: Stagnation
        if len(fitness_history) >= 10:
            recent_improvement = fitness_history[-1] - fitness_history[-10]
            if recent_improvement < 0.01:
                modes.append(FailureMode(
                    mode="Late stagnation",
                    frequency=1,
                    root_cause="Population converged to local optimum",
                    prevention_strategy="Increase mutation rate or restart with diversity injection",
                    examples=[f"Improvement in last 10 iterations: {recent_improvement:.4f}"]
                ))
        
        # Failure 2: Premature convergence
        if diversity_history and len(diversity_history) > 5:
            early_diversity = statistics.mean(diversity_history[:5])
            late_diversity = statistics.mean(diversity_history[-5:])
            if late_diversity < early_diversity * 0.3:
                modes.append(FailureMode(
                    mode="Premature convergence",
                    frequency=1,
                    root_cause="Selection pressure too high or population too small",
                    prevention_strategy="Increase population size or use island model",
                    examples=[f"Diversity: {early_diversity:.3f}->{late_diversity:.3f}"]
                ))
        
        # Failure 3: No improvement
        if fitness_history[-1] <= fitness_history[0]:
            modes.append(FailureMode(
                mode="No net improvement",
                frequency=1,
                root_cause="Evolution parameters unsuitable for problem",
                prevention_strategy="Try different strategy or increase mutation/crossover rates",
                examples=[f"Start: {fitness_history[0]:.3f}, End: {fitness_history[-1]:.3f}"]
            ))
        
        return modes


class LearningCapture:
    """Captures learning for future runs - from LoongFlow."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.learned_parameters: Dict[str, Any] = {}
        self.strategy_effectiveness: Dict[str, List[float]] = {}
    
    def capture_run(
        self,
        problem_type: str,
        strategy: str,
        parameters: Dict[str, Any],
        outcome: Dict[str, Any]
    ):
        """Capture learning from a run."""
        
        # Track strategy effectiveness
        if strategy not in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy] = []
        
        fitness = outcome.get("final_fitness", 0.0)
        self.strategy_effectiveness[strategy].append(fitness)
        
        # Learn good parameters for problem type
        if fitness > 0.8:  # Successful run
            if problem_type not in self.learned_parameters:
                self.learned_parameters[problem_type] = []
            
            self.learned_parameters[problem_type].append({
                "parameters": parameters,
                "fitness": fitness,
                "evaluations": outcome.get("total_evaluations", 0),
            })
        
        logger.info(f"Captured learning: {problem_type}/{strategy} -> fitness={fitness:.3f}")
    
    def get_recommended_parameters(self, problem_type: str) -> Optional[Dict[str, Any]]:
        """Get recommended parameters based on past learning."""
        if problem_type not in self.learned_parameters:
            return None
        
        runs = self.learned_parameters[problem_type]
        if not runs:
            return None
        
        # Find best run
        best_run = max(runs, key=lambda r: r["fitness"])
        
        return {
            "recommended": best_run["parameters"],
            "expected_fitness": best_run["fitness"],
            "based_on_runs": len(runs),
        }
    
    def get_strategy_ranking(self) -> List[tuple]:
        """Get ranking of strategies by effectiveness."""
        rankings = []
        
        for strategy, fitnesses in self.strategy_effectiveness.items():
            if fitnesses:
                avg_fitness = statistics.mean(fitnesses)
                rankings.append((strategy, avg_fitness, len(fitnesses)))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


class SummarizationEngine:
    """Main summarization engine - wraps around existing evolution."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.insight_extractor = InsightExtractor()
        self.learning_capture = LearningCapture()
    
    def summarize(
        self,
        execution_history: List[Dict[str, Any]],
        cost_data: Optional[Dict[str, float]] = None,
        strategy: str = "",
        problem_type: str = ""
    ) -> EvolutionSummary:
        """Generate comprehensive summary of evolution run.
        
        Args:
            execution_history: List of iteration snapshots
            cost_data: Optional cost tracking data
            strategy: Strategy used
            problem_type: Type of problem
            
        Returns:
            EvolutionSummary with insights and recommendations
        """
        if not execution_history:
            raise ValueError("Empty execution history")
        
        # Extract curves
        fitness_curve = [s.get("best_fitness", 0.0) for s in execution_history]
        diversity_curve = [s.get("diversity", 0.0) for s in execution_history]
        
        # Basic stats
        total_iterations = len(execution_history)
        total_evaluations = sum(s.get("evaluations", 0) for s in execution_history)
        
        # Performance metrics
        initial_fitness = fitness_curve[0] if fitness_curve else 0.0
        final_fitness = fitness_curve[-1] if fitness_curve else 0.0
        best_fitness = max(fitness_curve) if fitness_curve else 0.0
        improvement_rate = (final_fitness - initial_fitness) / total_iterations if total_iterations > 0 else 0.0
        
        # Convergence detection
        converged = False
        convergence_iteration = None
        for i, fit in enumerate(fitness_curve):
            if fit >= 0.95:  # Threshold
                converged = True
                convergence_iteration = i
                break
        
        # Calculate convergence score
        convergence_score = best_fitness / 0.95 if best_fitness < 0.95 else 1.0
        
        # Cost metrics
        total_cost = cost_data.get("total_cost_usd", 0.0) if cost_data else 0.0
        cost_per_improvement = total_cost / (final_fitness - initial_fitness) if (final_fitness - initial_fitness) > 0 else float('inf')
        
        # Efficiency gain (vs baseline 2.5x evaluations)
        baseline_evals = total_evaluations * 2.5
        efficiency_gain = (baseline_evals - total_evaluations) / baseline_evals if baseline_evals > 0 else 0.0
        
        # Extract patterns
        patterns = self.insight_extractor.extract_patterns(fitness_curve, diversity_curve)
        
        # Identify success factors
        success_factors = self.insight_extractor.identify_success_factors(fitness_curve)
        
        # Identify failure modes
        failure_modes = self.insight_extractor.identify_failure_modes(fitness_curve, diversity_curve)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            patterns, success_factors, failure_modes, fitness_curve
        )
        
        # Suggest parameters for similar problems
        suggested_params = self._suggest_parameters(success_factors, failure_modes)
        
        # Duration
        duration_ms = execution_history[-1].get("timestamp_ms", 0) - execution_history[0].get("timestamp_ms", 0) if len(execution_history) > 1 else 0
        
        return EvolutionSummary(
            total_iterations=total_iterations,
            total_evaluations=total_evaluations,
            duration_ms=duration_ms,
            initial_fitness=initial_fitness,
            final_fitness=final_fitness,
            best_fitness=best_fitness,
            improvement_rate=improvement_rate,
            converged=converged,
            convergence_iteration=convergence_iteration,
            convergence_score=convergence_score,
            total_cost_usd=total_cost,
            cost_per_improvement=cost_per_improvement,
            efficiency_gain=efficiency_gain,
            patterns=patterns,
            success_factors=success_factors,
            failure_modes=failure_modes,
            recommendations=recommendations,
            suggested_parameters_for_similar=suggested_params,
            convergence_curve=fitness_curve,
            diversity_curve=diversity_curve,
            strategy_used=strategy,
            problem_type=problem_type,
        )
    
    def _generate_recommendations(
        self,
        patterns: List[Pattern],
        success_factors: List[SuccessFactor],
        failure_modes: List[FailureMode],
        fitness_curve: List[float]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Based on patterns
        for pattern in patterns:
            if pattern.pattern_type == "optimization" and "mutation" in pattern.description:
                recommendations.append("Consider adjusting mutation rate based on plateau detection")
            elif pattern.pattern_type == "failure" and "diversity" in pattern.description:
                recommendations.append("Increase population size or enable island model to maintain diversity")
        
        # Based on success factors
        if success_factors:
            top_factor = max(success_factors, key=lambda f: f.impact_score)
            recommendations.append(f"Continue using: {top_factor.factor}")
        
        # Based on failure modes
        for mode in failure_modes:
            recommendations.append(f"Address {mode.mode}: {mode.prevention_strategy}")
        
        # Based on fitness curve
        if len(fitness_curve) >= 10:
            recent_avg = statistics.mean(fitness_curve[-5:])
            early_avg = statistics.mean(fitness_curve[:5])
            if recent_avg - early_avg < 0.1:
                recommendations.append("Consider increasing iterations or using PES-enhanced strategy for better convergence")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _suggest_parameters(
        self,
        success_factors: List[SuccessFactor],
        failure_modes: List[FailureMode]
    ) -> Dict[str, Any]:
        """Suggest parameters for similar problems."""
        params = {
            "iterations": 50,
            "population_size": 30,
            "mutation_rate": 0.1,
            "early_stopping": True,
        }
        
        # Adjust based on failure modes
        if any(m.mode == "Premature convergence" for m in failure_modes):
            params["population_size"] = 50
            params["mutation_rate"] = 0.15
        
        if any(m.mode == "Late stagnation" for m in failure_modes):
            params["adaptive_mutation"] = True
            params["restart_on_stagnation"] = True
        
        # Adjust based on success factors
        if any("adaptive" in f.factor.lower() for f in success_factors):
            params["adaptive_mutation"] = True
            params["adaptive_population"] = True
        
        return params
    
    def capture_learning(
        self,
        problem_type: str,
        strategy: str,
        parameters: Dict[str, Any],
        summary: EvolutionSummary
    ):
        """Capture learning from this run."""
        self.learning_capture.capture_run(
            problem_type=problem_type,
            strategy=strategy,
            parameters=parameters,
            outcome={
                "final_fitness": summary.final_fitness,
                "total_evaluations": summary.total_evaluations,
                "converged": summary.converged,
            }
        )
