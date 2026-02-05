"""Strategy enhancement layer - adds cost-aware strategy selection.

Wraps around existing strategy selection without modifying it.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available evolution strategies."""
    STANDARD = "standard"
    PES_ENHANCED = "pes_enhanced"
    QUALITY_DIVERSITY = "quality_diversity"
    MULTI_OBJECTIVE = "multi_objective"
    ADVERSARIAL = "adversarial"
    LANGUAGE_AGNOSTIC = "language_agnostic"
    LEAN_PROOF = "lean_proof"


@dataclass
class StrategyDecision:
    """Decision about which strategy to use."""
    strategy: StrategyType
    reasoning: str
    estimated_cost_usd: float
    estimated_evaluations: int
    confidence: float
    recommended_parameters: Dict[str, Any]


class AdaptiveParameterTuner:
    """Dynamically tunes parameters based on progress - from LoongFlow."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.iteration_count = 0
        self.improvement_history = []
    
    def tune_parameters(
        self,
        current_params: Dict[str, Any],
        current_fitness: float,
        previous_fitness: float,
        diversity: float
    ) -> Dict[str, Any]:
        """Adapt parameters based on progress.
        
        This is the LoongFlow-style dynamic adaptation that OpenEvolve lacked.
        """
        self.iteration_count += 1
        improvement = current_fitness - previous_fitness
        self.improvement_history.append(improvement)
        
        tuned = current_params.copy()
        
        # Adapt mutation rate based on progress
        if len(self.improvement_history) >= 3:
            recent_improvements = self.improvement_history[-3:]
            avg_improvement = sum(recent_improvements) / 3
            
            if avg_improvement < 0.001:  # Plateau
                # Increase mutation to escape local optima
                tuned["mutation_rate"] = min(
                    0.5,
                    current_params.get("mutation_rate", 0.1) * 1.5
                )
                logger.debug("Increasing mutation rate to escape plateau")
            elif avg_improvement > 0.05:  # Rapid improvement
                # Decrease mutation to fine-tune
                tuned["mutation_rate"] = max(
                    0.01,
                    current_params.get("mutation_rate", 0.1) * 0.8
                )
                logger.debug("Decreasing mutation rate for fine-tuning")
        
        # Adapt population size based on diversity
        if diversity < 0.1:  # Low diversity
            tuned["population_size"] = max(
                10,
                int(current_params.get("population_size", 50) * 1.2)
            )
            logger.debug("Increasing population size to boost diversity")
        elif diversity > 0.8:  # High diversity
            # Can afford smaller population
            tuned["population_size"] = max(
                10,
                int(current_params.get("population_size", 50) * 0.9)
            )
        
        # Simulated annealing-like temperature adjustment
        if "temperature" in current_params:
            # Decrease temperature over time
            tuned["temperature"] = current_params["temperature"] * 0.99
        
        return tuned
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptations made."""
        return {
            "iterations_monitored": self.iteration_count,
            "avg_improvement": sum(self.improvement_history) / len(self.improvement_history) if self.improvement_history else 0.0,
            "total_improvement": sum(self.improvement_history),
        }


class CostAwareStrategySelector:
    """Selects strategy based on cost constraints - from LoongFlow."""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def select_strategy(
        self,
        problem_description: str,
        code: Optional[str],
        language: Optional[str],
        max_cost_usd: float = 10.0,
        max_time_seconds: int = 1800,
        complexity_hint: Optional[str] = None
    ) -> StrategyDecision:
        """Select best strategy given constraints.
        
        This addresses the gap where OpenEvolve had no cost-aware
        strategy selection.
        """
        
        # Analyze complexity
        complexity = complexity_hint or self._estimate_complexity(problem_description, code)
        
        # Check for special cases
        if language == "lean" or "lean" in problem_description.lower():
            return StrategyDecision(
                strategy=StrategyType.LEAN_PROOF,
                reasoning="Lean 4 theorem proving detected",
                estimated_cost_usd=min(5.0, max_cost_usd * 0.5),
                estimated_evaluations=500,
                confidence=0.85,
                recommended_parameters={
                    "iterations": 50,
                    "population_size": 10,
                    "use_proof_strategies": True,
                }
            )
        
        if language in ["auto", "universal"] or self._is_multi_language(problem_description):
            return StrategyDecision(
                strategy=StrategyType.LANGUAGE_AGNOSTIC,
                reasoning="Multi-language support required",
                estimated_cost_usd=max_cost_usd * 0.8,
                estimated_evaluations=1000,
                confidence=0.80,
                recommended_parameters={
                    "iterations": 50,
                    "population_size": 20,
                    "language_detection": True,
                }
            )
        
        # Budget-based selection
        cost_per_eval = 0.001  # Rough estimate
        
        if max_cost_usd < 1.0:
            # Very tight budget - use PES for efficiency
            return StrategyDecision(
                strategy=StrategyType.PES_ENHANCED,
                reasoning="Tight budget - PES reduces evaluations by ~60%",
                estimated_cost_usd=max_cost_usd * 0.9,
                estimated_evaluations=int(max_cost_usd / cost_per_eval * 0.4),
                confidence=0.75,
                recommended_parameters={
                    "iterations": int(max_cost_usd * 10),
                    "population_size": 10,
                    "early_stopping": True,
                    "adaptive_mutation": True,
                }
            )
        
        if complexity == "high" or complexity == "very_high":
            if max_cost_usd > 5.0:
                # Complex problem with good budget - use PES
                return StrategyDecision(
                    strategy=StrategyType.PES_ENHANCED,
                    reasoning=f"High complexity ({complexity}) with adequate budget",
                    estimated_cost_usd=min(max_cost_usd * 0.8, 20.0),
                    estimated_evaluations=5000,
                    confidence=0.80,
                    recommended_parameters={
                        "iterations": 100,
                        "population_size": 50,
                        "early_stopping": True,
                        "directed_mutation": True,
                    }
                )
            else:
                # Complex but tight budget - standard with early stopping
                return StrategyDecision(
                    strategy=StrategyType.STANDARD,
                    reasoning="High complexity but limited budget - standard approach",
                    estimated_cost_usd=max_cost_usd * 0.9,
                    estimated_evaluations=int(max_cost_usd / cost_per_eval),
                    confidence=0.65,
                    recommended_parameters={
                        "iterations": int(max_cost_usd * 20),
                        "population_size": 20,
                        "early_stopping": True,
                    }
                )
        
        # Default: Standard evolution
        return StrategyDecision(
            strategy=StrategyType.STANDARD,
            reasoning="Standard approach for moderate complexity",
            estimated_cost_usd=max_cost_usd * 0.7,
            estimated_evaluations=int(max_cost_usd / cost_per_eval * 0.8),
            confidence=0.75,
            recommended_parameters={
                "iterations": int(max_cost_usd * 15),
                "population_size": 30,
                "early_stopping": True,
            }
        )
    
    def _estimate_complexity(self, description: str, code: Optional[str]) -> str:
        """Estimate problem complexity."""
        score = 0
        
        # Description length
        words = len(description.split())
        if words > 100:
            score += 2
        elif words > 50:
            score += 1
        
        # Code complexity
        if code:
            lines = len(code.split('\n'))
            if lines > 200:
                score += 2
            elif lines > 50:
                score += 1
            
            if any(kw in code for kw in ['class ', 'async ', 'decorator']):
                score += 1
        
        # Keywords indicating complexity
        complex_keywords = ['optimization', 'constraint', 'theorem', 'prove', 'verify']
        for kw in complex_keywords:
            if kw in description.lower():
                score += 1
                break
        
        if score >= 5:
            return "very_high"
        elif score >= 3:
            return "high"
        elif score >= 1:
            return "medium"
        return "low"
    
    def _is_multi_language(self, description: str) -> bool:
        """Check if description suggests multi-language needs."""
        language_keywords = ['python', 'javascript', 'php', 'java', 'multi-language', 'language-agnostic']
        found = sum(1 for kw in language_keywords if kw in description.lower())
        return found >= 2


class StrategyEnhancer:
    """Main strategy enhancement class - wraps existing strategy selection."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.selector = CostAwareStrategySelector(config)
        self.tuner = AdaptiveParameterTuner(config)
    
    def enhance_strategy_selection(
        self,
        original_strategy_func,
        problem_description: str,
        code: Optional[str] = None,
        language: Optional[str] = None,
        budget: Optional[Dict[str, float]] = None
    ) -> StrategyDecision:
        """Enhance strategy selection with cost awareness.
        
        Args:
            original_strategy_func: The existing strategy selection function
            problem_description: Problem description
            code: Optional code
            language: Programming language
            budget: Budget constraints
            
        Returns:
            Enhanced StrategyDecision
        """
        max_cost = budget.get("max_cost_usd", 10.0) if budget else 10.0
        max_time = budget.get("max_time_seconds", 1800) if budget else 1800
        
        # Use cost-aware selector
        decision = self.selector.select_strategy(
            problem_description,
            code,
            language,
            max_cost_usd=max_cost,
            max_time_seconds=max_time
        )
        
        logger.info(f"Strategy selected: {decision.strategy.value} "
                   f"(cost=${decision.estimated_cost_usd:.2f}, "
                   f"confidence={decision.confidence:.0%})")
        
        return decision
