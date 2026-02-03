"""
Complete Strategy Recommender Implementation

Provides ensemble-based strategy selection that combines multiple recommendation
approaches to select the best decomposition or evolution strategy.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecommendation:
    """A strategy recommendation with confidence and reasoning."""
    strategy_name: str
    confidence: float  # 0-1
    reasoning: str
    factors: Dict[str, float] = field(default_factory=dict)
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "factors": self.factors,
            "alternatives": self.alternatives
        }


class BaseStrategyRecommender(ABC):
    """Base class for strategy recommenders."""
    
    @abstractmethod
    def recommend_strategy(
        self,
        problem_description: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        available_strategies: Optional[List[str]] = None
    ) -> StrategyRecommendation:
        """Recommend a strategy for the given problem."""
        pass
    
    @abstractmethod
    def get_confidence(
        self,
        problem_description: str,
        strategy: str
    ) -> float:
        """Get confidence score for a strategy given a problem."""
        pass


class KeywordBasedRecommender(BaseStrategyRecommender):
    """Recommends strategies based on keyword matching."""
    
    # Strategy keywords
    STRATEGY_KEYWORDS = {
        "semantic": [
            "meaning", "semantic", "understand", "context", "intent",
            "concept", "relationship", "ontology", "taxonomy"
        ],
        "dependency": [
            "depend", "require", "prerequisite", "before", "after",
            "order", "sequence", "flow", "pipeline"
        ],
        "divide_and_conquer": [
            "split", "divide", "subproblem", "recursive", "parallel",
            "concurrent", "independent", "modular"
        ],
        "evolutionary": [
            "evolve", "generation", "population", "mutation", "crossover",
            "fitness", "optimize", "selection"
        ],
        "adversarial": [
            "attack", "defend", "robust", "vulnerability", "exploit",
            "adversary", "red team", "blue team"
        ],
        "hybrid": [
            "combine", "multiple", "ensemble", "blend", "mix",
            "approaches", "strategies", "together"
        ]
    }
    
    def recommend_strategy(
        self,
        problem_description: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        available_strategies: Optional[List[str]] = None
    ) -> StrategyRecommendation:
        """Recommend based on keyword matches."""
        text = problem_description.lower()
        scores = {}
        
        available = available_strategies or list(self.STRATEGY_KEYWORDS.keys())
        
        for strategy in available:
            keywords = self.STRATEGY_KEYWORDS.get(strategy, [])
            score = sum(1 for kw in keywords if kw in text)
            scores[strategy] = score / max(len(keywords), 1) if keywords else 0
        
        # Get best strategy
        if not scores or max(scores.values()) == 0:
            best = "semantic" if "semantic" in available else available[0]
            confidence = 0.3
        else:
            best = max(scores, key=scores.get)
            confidence = min(0.9, 0.4 + scores[best] * 0.5)
        
        # Get alternatives
        alternatives = sorted(
            [(s, scores.get(s, 0)) for s in available if s != best],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return StrategyRecommendation(
            strategy_name=best,
            confidence=confidence,
            reasoning=f"Matched keywords for {best} strategy",
            factors=scores,
            alternatives=alternatives
        )
    
    def get_confidence(
        self,
        problem_description: str,
        strategy: str
    ) -> float:
        """Get confidence based on keyword match."""
        text = problem_description.lower()
        keywords = self.STRATEGY_KEYWORDS.get(strategy, [])
        
        if not keywords:
            return 0.5
        
        matches = sum(1 for kw in keywords if kw in text)
        return min(0.9, 0.3 + (matches / len(keywords)) * 0.6)


class DomainBasedRecommender(BaseStrategyRecommender):
    """Recommends strategies based on domain."""
    
    DOMAIN_PREFERENCES = {
        "finance": {
            "preferred": ["dependency", "semantic", "hybrid"],
            "avoid": ["adversarial"]
        },
        "science": {
            "preferred": ["semantic", "divide_and_conquer", "hybrid"],
            "avoid": []
        },
        "engineering": {
            "preferred": ["divide_and_conquer", "dependency", "hybrid"],
            "avoid": []
        },
        "security": {
            "preferred": ["adversarial", "evolutionary", "hybrid"],
            "avoid": []
        },
        "optimization": {
            "preferred": ["evolutionary", "hybrid", "divide_and_conquer"],
            "avoid": []
        },
        "general": {
            "preferred": ["semantic", "hybrid", "divide_and_conquer"],
            "avoid": []
        }
    }
    
    def recommend_strategy(
        self,
        problem_description: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        available_strategies: Optional[List[str]] = None
    ) -> StrategyRecommendation:
        """Recommend based on domain."""
        domain_prefs = self.DOMAIN_PREFERENCES.get(domain, self.DOMAIN_PREFERENCES["general"])
        available = available_strategies or []
        
        # Find preferred strategy that's available
        best = None
        for pref in domain_prefs["preferred"]:
            if not available or pref in available:
                best = pref
                break
        
        if not best:
            best = available[0] if available else "semantic"
        
        confidence = 0.75 if domain in self.DOMAIN_PREFERENCES else 0.5
        
        # Get all scores
        scores = {}
        for strategy in (available or domain_prefs["preferred"]):
            if strategy in domain_prefs["preferred"]:
                scores[strategy] = 0.8 - domain_prefs["preferred"].index(strategy) * 0.1
            elif strategy in domain_prefs.get("avoid", []):
                scores[strategy] = 0.2
            else:
                scores[strategy] = 0.5
        
        alternatives = sorted(
            [(s, scores.get(s, 0.5)) for s in (available or []) if s != best],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return StrategyRecommendation(
            strategy_name=best,
            confidence=confidence,
            reasoning=f"Domain '{domain}' prefers {best} strategy",
            factors=scores,
            alternatives=alternatives
        )
    
    def get_confidence(
        self,
        problem_description: str,
        strategy: str
    ) -> float:
        """Get confidence based on domain match."""
        # Default medium confidence
        return 0.6


class ComplexityBasedRecommender(BaseStrategyRecommender):
    """Recommends strategies based on problem complexity."""
    
    def recommend_strategy(
        self,
        problem_description: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        available_strategies: Optional[List[str]] = None
    ) -> StrategyRecommendation:
        """Recommend based on estimated complexity."""
        complexity = self._estimate_complexity(problem_description)
        available = available_strategies or ["semantic", "divide_and_conquer", "hybrid"]
        
        # Strategy selection based on complexity
        if complexity > 0.8:
            # Very complex - use hybrid
            best = "hybrid" if "hybrid" in available else "divide_and_conquer"
            reasoning = "High complexity problem requires combined approach"
        elif complexity > 0.5:
            # Medium complexity - use divide and conquer
            best = "divide_and_conquer" if "divide_and_conquer" in available else "semantic"
            reasoning = "Medium complexity benefits from modular decomposition"
        else:
            # Simple - use semantic
            best = "semantic" if "semantic" in available else available[0]
            reasoning = "Lower complexity suited for semantic analysis"
        
        # Confidence based on complexity clarity
        confidence = 0.9 - abs(complexity - 0.5)
        
        scores = {
            "complexity_estimate": complexity,
            "semantic": 1.0 - complexity,
            "divide_and_conquer": complexity if complexity > 0.3 else 0.3,
            "hybrid": complexity if complexity > 0.7 else 0.4
        }
        
        alternatives = sorted(
            [(s, scores.get(s, 0.5)) for s in available if s != best],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return StrategyRecommendation(
            strategy_name=best,
            confidence=confidence,
            reasoning=reasoning,
            factors=scores,
            alternatives=alternatives
        )
    
    def _estimate_complexity(self, text: str) -> float:
        """Estimate problem complexity from text."""
        # Factors: length, vocabulary, structure
        words = text.split()
        
        # Length factor
        length_score = min(1.0, len(words) / 100)
        
        # Vocabulary diversity
        unique_words = len(set(w.lower() for w in words))
        vocab_score = min(1.0, unique_words / max(len(words), 1) * 2)
        
        # Structural indicators
        complex_indicators = [
            "and", "or", "but", "however", "although", "moreover",
            "furthermore", "consequently", "therefore", "nevertheless"
        ]
        structure_score = sum(1 for ind in complex_indicators if ind in text.lower()) / 5
        structure_score = min(1.0, structure_score)
        
        # Combine factors
        complexity = (length_score * 0.3 + vocab_score * 0.3 + structure_score * 0.4)
        
        return complexity
    
    def get_confidence(
        self,
        problem_description: str,
        strategy: str
    ) -> float:
        """Get confidence based on complexity match."""
        complexity = self._estimate_complexity(problem_description)
        
        if strategy == "hybrid" and complexity > 0.7:
            return 0.85
        elif strategy == "divide_and_conquer" and 0.4 < complexity < 0.8:
            return 0.8
        elif strategy == "semantic" and complexity < 0.5:
            return 0.85
        
        return 0.5


class HistoricalPerformanceRecommender(BaseStrategyRecommender):
    """Recommends based on historical performance data."""
    
    def __init__(self):
        self.performance_history: Dict[str, Dict[str, Any]] = {}
        self.load_history()
    
    def load_history(self):
        """Load historical performance data."""
        # In real implementation, load from database
        self.performance_history = {
            "semantic": {"success_rate": 0.82, "avg_time": 1.2, "count": 150},
            "dependency": {"success_rate": 0.78, "avg_time": 1.5, "count": 120},
            "divide_and_conquer": {"success_rate": 0.85, "avg_time": 2.1, "count": 200},
            "evolutionary": {"success_rate": 0.71, "avg_time": 5.5, "count": 80},
            "adversarial": {"success_rate": 0.75, "avg_time": 3.2, "count": 60},
            "hybrid": {"success_rate": 0.88, "avg_time": 3.0, "count": 100}
        }
    
    def recommend_strategy(
        self,
        problem_description: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        available_strategies: Optional[List[str]] = None
    ) -> StrategyRecommendation:
        """Recommend based on historical performance."""
        available = available_strategies or list(self.performance_history.keys())
        
        # Score by success rate weighted by count
        scores = {}
        for strategy in available:
            hist = self.performance_history.get(strategy, {})
            success_rate = hist.get("success_rate", 0.5)
            count = hist.get("count", 0)
            # Weight by count (more data = more reliable)
            weight = min(1.0, count / 100)
            scores[strategy] = success_rate * weight + 0.5 * (1 - weight)
        
        if not scores:
            return StrategyRecommendation(
                strategy_name="semantic",
                confidence=0.5,
                reasoning="No historical data available",
                factors={}
            )
        
        best = max(scores, key=scores.get)
        
        alternatives = sorted(
            [(s, scores[s]) for s in available if s != best],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        hist_info = self.performance_history.get(best, {})
        reasoning = (
            f"{best} has {hist_info.get('success_rate', 0):.0%} success rate "
            f"from {hist_info.get('count', 0)} historical runs"
        )
        
        return StrategyRecommendation(
            strategy_name=best,
            confidence=scores[best],
            reasoning=reasoning,
            factors=scores,
            alternatives=alternatives
        )
    
    def get_confidence(
        self,
        problem_description: str,
        strategy: str
    ) -> float:
        """Get confidence from historical success rate."""
        hist = self.performance_history.get(strategy, {})
        return hist.get("success_rate", 0.5)
    
    def record_result(
        self,
        strategy: str,
        success: bool,
        execution_time: float,
        quality_score: Optional[float] = None
    ):
        """Record execution result for learning."""
        if strategy not in self.performance_history:
            self.performance_history[strategy] = {
                "success_rate": 0.5,
                "avg_time": execution_time,
                "count": 0
            }
        
        hist = self.performance_history[strategy]
        count = hist["count"]
        
        # Update success rate with exponential moving average
        alpha = 0.1  # Learning rate
        hist["success_rate"] = (1 - alpha) * hist["success_rate"] + alpha * (1.0 if success else 0)
        hist["avg_time"] = (1 - alpha) * hist["avg_time"] + alpha * execution_time
        hist["count"] = count + 1


class EnsembleStrategySelector(BaseStrategyRecommender):
    """
    Ensemble selector that combines multiple recommenders.
    Uses weighted voting to select the best strategy.
    """
    
    def __init__(
        self,
        recommenders: Optional[List[BaseStrategyRecommender]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ensemble selector.
        
        Args:
            recommenders: List of recommenders to ensemble
            weights: Weights for each recommender type
        """
        self.recommenders = recommenders or [
            KeywordBasedRecommender(),
            DomainBasedRecommender(),
            ComplexityBasedRecommender(),
            HistoricalPerformanceRecommender()
        ]
        
        self.weights = weights or {
            "KeywordBasedRecommender": 0.25,
            "DomainBasedRecommender": 0.25,
            "ComplexityBasedRecommender": 0.25,
            "HistoricalPerformanceRecommender": 0.25
        }
    
    def recommend_strategy(
        self,
        problem_description: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        available_strategies: Optional[List[str]] = None
    ) -> StrategyRecommendation:
        """
        Recommend strategy using ensemble of recommenders.
        """
        # Collect recommendations from all recommenders
        all_recommendations = []
        
        for recommender in self.recommenders:
            try:
                rec = recommender.recommend_strategy(
                    problem_description,
                    domain,
                    constraints,
                    available_strategies
                )
                all_recommendations.append((recommender, rec))
            except Exception as e:
                logger.warning({
                    "msg": "Recommender failed",
                    "recommender": type(recommender).__name__,
                    "error": str(e)
                })
        
        if not all_recommendations:
            return StrategyRecommendation(
                strategy_name="semantic",
                confidence=0.5,
                reasoning="No recommenders available",
                factors={}
            )
        
        # Aggregate scores
        strategy_scores: Dict[str, float] = {}
        strategy_confidences: Dict[str, List[float]] = {}
        
        for recommender, rec in all_recommendations:
            weight = self.weights.get(type(recommender).__name__, 0.25)
            
            if rec.strategy_name not in strategy_scores:
                strategy_scores[rec.strategy_name] = 0
                strategy_confidences[rec.strategy_name] = []
            
            strategy_scores[rec.strategy_name] += rec.confidence * weight
            strategy_confidences[rec.strategy_name].append(rec.confidence)
        
        # Select best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best_strategy]
        
        # Normalize confidence
        total_weight = sum(self.weights.values())
        normalized_confidence = min(1.0, best_score / total_weight)
        
        # Build reasoning
        reasons = []
        for recommender, rec in all_recommendations:
            if rec.strategy_name == best_strategy:
                reasons.append(type(recommender).__name__.replace("Recommender", ""))
        
        reasoning = f"Selected by: {', '.join(reasons)}"
        
        # Get alternatives
        alternatives = sorted(
            [(s, strategy_scores[s]) for s in strategy_scores if s != best_strategy],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return StrategyRecommendation(
            strategy_name=best_strategy,
            confidence=normalized_confidence,
            reasoning=reasoning,
            factors=strategy_scores,
            alternatives=alternatives
        )
    
    def get_confidence(
        self,
        problem_description: str,
        strategy: str
    ) -> float:
        """Get ensemble confidence for a strategy."""
        confidences = []
        
        for recommender in self.recommenders:
            try:
                conf = recommender.get_confidence(problem_description, strategy)
                weight = self.weights.get(type(recommender).__name__, 0.25)
                confidences.append(conf * weight)
            except:
                pass
        
        if not confidences:
            return 0.5
        
        return sum(confidences) / sum(self.weights.values())


# Convenience function
def recommend_strategy(
    problem_description: str,
    domain: str = "general",
    available_strategies: Optional[List[str]] = None,
    use_ensemble: bool = True
) -> StrategyRecommendation:
    """
    Recommend a strategy for the given problem.
    
    Args:
        problem_description: Description of the problem
        domain: Problem domain
        available_strategies: List of available strategies
        use_ensemble: Whether to use ensemble selector
        
    Returns:
        Strategy recommendation
    """
    if use_ensemble:
        selector = EnsembleStrategySelector()
    else:
        selector = KeywordBasedRecommender()
    
    return selector.recommend_strategy(
        problem_description,
        domain,
        available_strategies=available_strategies
    )


__all__ = [
    'StrategyRecommendation',
    'BaseStrategyRecommender',
    'KeywordBasedRecommender',
    'DomainBasedRecommender',
    'ComplexityBasedRecommender',
    'HistoricalPerformanceRecommender',
    'EnsembleStrategySelector',
    'recommend_strategy'
]
