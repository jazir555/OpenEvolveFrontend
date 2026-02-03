"""
AI-Powered Strategy Recommender for Evolutionary Optimization

Automatically selects optimal evolutionary strategies (OpenEvolve vs LoongFlow,
and which mode) based on problem characteristics and historical performance.

Enhanced with ensemble methods, confidence intervals, and real-time learning.

Author: AI Architecture Team
Date: 2026-01-30
Version: 2.0 - Ensemble Strategy Selector
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
import json
import asyncio
from collections import defaultdict
import random
import math

# Optional imports for advanced features
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from openai import AsyncOpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# **ACTUAL INTEGRATION**: Adaptive MDAP for strategy recommendation v2
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


class EvolutionSystem(str, Enum):
    """Evolutionary systems"""
    OPENEVOLVE = "openevolve"
    LOONGFLOW = "loongflow"
    HYBRID = "hybrid"


class EvolutionMode(str, Enum):
    """Evolutionary modes"""
    PES = "pes"                    # Plan-Execute-Summarize (LoongFlow)
    QD = "qd"                      # Quality-Diversity MAP-Elites (OpenEvolve)
    MO = "mo"                      # Multi-Objective optimization
    ADVERSARIAL = "adversarial"    # Adversarial co-evolution
    STANDARD = "standard"          # Traditional evolutionary algorithm


class DomainType(str, Enum):
    """Problem domains"""
    FINANCE = "finance"
    TRADING = "trading"
    SCIENCE = "science"
    ENGINEERING = "engineering"
    PHARMA = "pharma"
    WEB = "web"
    GENERAL = "general"


class EvaluationCost(str, Enum):
    """Evaluation cost categories"""
    CHEAP = "cheap"                      # < 1 second
    MODERATE = "moderate"                # 1-60 seconds
    EXPENSIVE = "expensive"              # 1-10 minutes
    VERY_EXPENSIVE = "very_expensive"    # > 10 minutes


class ComplexityLevel(str, Enum):
    """Problem complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionMethod(str, Enum):
    """Ensemble prediction methods"""
    RULE_BASED = "rule_based"
    SIMILARITY = "similarity"
    TREND = "trend"
    ML = "ml"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ProblemCharacteristics:
    """Analysis of problem characteristics"""
    domain: str
    complexity: str
    evaluation_cost: str
    has_multiple_objectives: bool
    requires_diversity: bool
    requires_robustness: bool
    constraint_count: int
    estimated_iterations: int
    similar_problems: List[str] = field(default_factory=list)

    # Additional context
    keywords: List[str] = field(default_factory=list)
    domain_specific_factors: Dict[str, Any] = field(default_factory=dict)


    # Vector representation for similarity (optional)
    feature_vector: Optional[List[float]] = None


@dataclass
class HistoricalRun:
    """Historical evolutionary run data"""
    run_id: str
    domain: str
    strategy_used: str  # "pes", "qd", "mo", "adversarial", "standard"
    mode_used: str
    problem_complexity: str
    final_score: float
    convergence_speed: int  # iterations to convergence
    evaluation_count: int
    diversity_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    evaluation_cost: str = "moderate"
    sample_efficiency: float = 1.0  # fitness / evaluations
    time_efficiency: float = 1.0    # fitness / time


@dataclass
class RankedStrategy:
    """Ranked strategy recommendation"""
    system: str
    mode: str
    score: float  # 0.0 to 100.0
    expected_performance: Dict[str, float]
    pros: List[str]
    cons: List[str]
    confidence: float = 0.0


@dataclass
class AlternativeStrategy:
    """Alternative strategy option"""
    system: str
    mode: str
    score: float
    reason: str
    when_to_use: str


@dataclass
class PerformancePrediction:
    """Predicted performance metrics"""
    expected_iterations: int
    expected_time_seconds: float
    expected_score: float
    confidence_interval: Tuple[float, float]
    success_probability: float


@dataclass
class Explanation:
    """Explanation of recommendation"""
    primary_reason: str
    detailed_reasoning: List[str]
    evidence_from_history: List[str]
    domain_considerations: List[str]
    risk_factors: List[str]


@dataclass
class StrategyRecommendation:
    """Complete strategy recommendation"""
    recommended_system: str  # "openevolve", "loongflow", "hybrid"
    recommended_mode: str  # "pes", "qd", "mo", "adversarial", "standard"
    config_overrides: Dict[str, Any]  # Recommended parameter adjustments
    confidence: float  # 0.0 to 1.0
    reasoning: Explanation
    alternatives: List[AlternativeStrategy]
    expected_performance: PerformancePrediction

    # Metadata
    problem_analysis: ProblemCharacteristics = None
    historical_context: List[HistoricalRun] = None
    ranking: List[RankedStrategy] = None


# ============================================================================
# ENSEMBLE PREDICTION STRUCTURES
# ============================================================================

@dataclass
class EnsemblePrediction:
    """Prediction from ensemble methods"""
    strategy: Tuple[str, str]  # (system, mode)
    point_estimate: float  # Expected performance
    confidence_interval: Tuple[float, float]  # (lower, upper)
    confidence_level: float  # 0.90, 0.95, etc.
    prediction_methods: List[str]  # Which methods agreed
    disagreement_ratio: float  # 0.0 = unanimous, 1.0 = split
    reasoning: str
    method_weights: Dict[str, float]  # Weight of each method
    individual_predictions: Dict[str, Tuple[str, str, float]]  # method -> (system, mode, confidence)


@dataclass
class MethodPrediction:
    """Individual prediction from a method"""
    method: str
    system: str
    mode: str
    confidence: float
    reasoning: str
    evidence: Dict[str, Any]


# ============================================================================
# ONLINE LEARNING TRACKER
# ============================================================================

class OnlineLearningTracker:
    """Track recommendation accuracy and adapt in real-time"""

    def __init__(self, window_size: int = 50):
        """
        Initialize learning tracker

        Args:
            window_size: Rolling window size for accuracy tracking
        """
        self.window_size = window_size

        # Tracking data
        self.recommendations_made: List[Dict[str, Any]] = []
        self.actual_performance: List[Dict[str, Any]] = []
        self.accuracy_history: List[float] = []

        # Method-specific tracking
        self.method_accuracies: Dict[str, List[float]] = defaultdict(list)

        # Current ensemble weights
        self.current_weights: Dict[str, float] = {
            'rule_based': 0.25,
            'similarity': 0.35,
            'trend': 0.25,
            'ml': 0.15
        }

        # Performance metrics
        self.total_recommendations = 0
        self.total_accuracy = 0.0

    def record_recommendation(
        self,
        recommendation: EnsemblePrediction,
        problem_chars: ProblemCharacteristics,
        timestamp: datetime = None
    ) -> str:
        """
        Record a recommendation was made

        Args:
            recommendation: The ensemble prediction made
            problem_chars: Analyzed problem characteristics
            timestamp: When recommendation was made

        Returns:
            recommendation_id: Unique ID for this recommendation
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        rec_id = f"rec_{timestamp.isoformat()}_{self.total_recommendations}"

        self.recommendations_made.append({
            'id': rec_id,
            'prediction': recommendation,
            'problem_chars': problem_chars,
            'timestamp': timestamp,
            'predicted_performance': prediction.point_estimate if hasattr(prediction, 'point_estimate') else 0.8
        })

        self.total_recommendations += 1
        return rec_id

    def record_actual_performance(
        self,
        recommendation_id: str,
        actual_performance: float,
        run_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Record the actual performance of a run

        Args:
            recommendation_id: ID of recommendation to match
            actual_performance: Actual performance achieved (0.0 to 1.0)
            run_id: Optional run identifier
            metadata: Optional additional metadata

        Returns:
            accuracy_metrics: Dictionary with accuracy metrics
        """
        # Find the recommendation
        rec = None
        for r in self.recommendations_made:
            if r['id'] == recommendation_id:
                rec = r
                break

        if rec is None:
            print(f"Warning: Recommendation {recommendation_id} not found")
            return {}

        # Calculate accuracy
        predicted = rec['predicted_performance']
        error = abs(predicted - actual_performance)
        accuracy = 1.0 - min(error, 1.0)  # Clamp to [0, 1]

        self.actual_performance.append({
            'recommendation_id': recommendation_id,
            'run_id': run_id,
            'actual_performance': actual_performance,
            'predicted_performance': predicted,
            'accuracy': accuracy,
            'error': error,
            'timestamp': datetime.now(UTC),
            'metadata': metadata or {}
        })

        # Update rolling accuracy history
        self.accuracy_history.append(accuracy)
        if len(self.accuracy_history) > self.window_size:
            self.accuracy_history.pop(0)

        # Update total accuracy
        self.total_accuracy = sum(self.accuracy_history) / len(self.accuracy_history)

        # Track method-specific accuracies
        prediction = rec['prediction']
        if hasattr(prediction, 'individual_predictions'):
            for method, (system, mode, conf) in prediction.individual_predictions.items():
                # Estimate method contribution (simplified)
                method_accuracy = accuracy * conf
                self.method_accuracies[method].append(method_accuracy)

                # Keep window size
                if len(self.method_accuracies[method]) > self.window_size:
                    self.method_accuracies[method].pop(0)

        # Adapt weights if enough data
        if len(self.accuracy_history) >= 20:
            new_weights = self._adapt_ensemble_weights()
            return {
                'accuracy': accuracy,
                'error': error,
                'weights_adapted': True,
                'new_weights': new_weights
            }

        return {
            'accuracy': accuracy,
            'error': error,
            'weights_adapted': False
        }

    def _adapt_ensemble_weights(self) -> Dict[str, float]:
        """Adjust ensemble weights based on recent performance"""
        # Calculate recent accuracy for each method
        method_accuracies = {}

        for method in self.current_weights.keys():
            if method in self.method_accuracies and self.method_accuracies[method]:
                recent_avg = sum(self.method_accuracies[method][-20:]) / min(20, len(self.method_accuracies[method]))
                method_accuracies[method] = recent_avg
            else:
                # Use default if no data
                method_accuracies[method] = 0.7

        # Renormalize to sum to 1.0
        total = sum(method_accuracies.values())

        if total > 0:
            new_weights = {
                method: acc / total
                for method, acc in method_accuracies.items()
            }

            # Smooth transition (don't change too drastically)
            alpha = 0.3  # Learning rate
            smoothed_weights = {}

            for method in self.current_weights:
                old_weight = self.current_weights[method]
                new_weight = new_weights.get(method, old_weight)
                smoothed_weights[method] = (1 - alpha) * old_weight + alpha * new_weight

            self.current_weights = smoothed_weights

        return self.current_weights

    def get_current_weights(self) -> Dict[str, float]:
        """Get current ensemble weights"""
        return self.current_weights.copy()

    def get_accuracy_metrics(self) -> Dict[str, Any]:
        """Get accuracy metrics"""
        if not self.accuracy_history:
            return {
                'average_accuracy': 0.0,
                'total_recommendations': 0,
                'recent_trend': 'unknown'
            }

        recent_avg = sum(self.accuracy_history[-10:]) / min(10, len(self.accuracy_history))
        overall_avg = sum(self.accuracy_history) / len(self.accuracy_history)

        # Determine trend
        if len(self.accuracy_history) >= 20:
            old_avg = sum(self.accuracy_history[-20:-10]) / 10
            if recent_avg > old_avg + 0.05:
                trend = 'improving'
            elif recent_avg < old_avg - 0.05:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'average_accuracy': overall_avg,
            'recent_accuracy': recent_avg,
            'total_recommendations': len(self.accuracy_history),
            'recent_trend': trend,
            'method_weights': self.current_weights
        }

class StrategyRecommender:
    """Base strategy recommender (minimal implementation for reference)"""

    def __init__(
        self,
        knowledge_engine=None,
        llm_client=None,
        use_ai_analysis: bool = True,
        learning_enabled: bool = True
    ):
        self.knowledge_engine = knowledge_engine
        self.llm_client = llm_client
        self.use_ai_analysis = use_ai_analysis and LLM_AVAILABLE
        self.learning_enabled = learning_enabled
        self.historical_runs: Dict[str, HistoricalRun] = {}
        self.recommendation_accuracy: List[float] = []
        self.domain_heuristics = self._init_domain_heuristics()
        
        # Initialize Adaptive MDAP components if available
        self._complexity_classifier = None
        self._mdap_allocator = None
        if ADAPTIVE_MDAP_AVAILABLE:
            try:
                self._complexity_classifier = TaskComplexityClassifier()
                self._mdap_allocator = AdaptiveMDAPAllocator()
            except Exception:
                # Fall back to None if initialization fails
                pass

    def _init_domain_heuristics(self) -> Dict[str, Dict[str, Any]]:
        """Initialize domain-specific strategy heuristics"""
        return {
            "finance": {
                "preferred_modes": ["pes", "mo", "standard"],
                "evaluation_cost": "expensive",
                "requires_diversity": True,
                "requires_robustness": True,
                "typical_iterations": 50,
            },
            "trading": {
                "preferred_modes": ["qd", "pes", "adversarial"],
                "evaluation_cost": "expensive",
                "requires_diversity": True,
                "requires_robustness": True,
                "typical_iterations": 100,
            },
            "science": {
                "preferred_modes": ["pes", "qd", "standard"],
                "evaluation_cost": "very_expensive",
                "requires_diversity": True,
                "requires_robustness": False,
                "typical_iterations": 30,
            },
            "engineering": {
                "preferred_modes": ["pes", "qd", "adversarial"],
                "evaluation_cost": "expensive",
                "requires_diversity": True,
                "requires_robustness": True,
                "typical_iterations": 50,
            },
            "pharma": {
                "preferred_modes": ["qd", "mo", "pes"],
                "evaluation_cost": "very_expensive",
                "requires_diversity": True,
                "requires_robustness": True,
                "typical_iterations": 100,
            },
            "web": {
                "preferred_modes": ["standard", "qd", "pes"],
                "evaluation_cost": "cheap",
                "requires_diversity": False,
                "requires_robustness": False,
                "typical_iterations": 200,
            },
            "general": {
                "preferred_modes": ["pes", "qd", "standard"],
                "evaluation_cost": "moderate",
                "requires_diversity": False,
                "requires_robustness": False,
                "typical_iterations": 100,
            },
        }

    async def recommend_strategy(
        self,
        problem_description: str,
        domain: str,
        constraints: Dict[str, Any]
    ) -> StrategyRecommendation:
        """Basic strategy recommendation (override in subclass)"""
        # This is a minimal implementation
        # EnsembleStrategySelector provides the full implementation
        raise NotImplementedError("Use EnsembleStrategySelector for full functionality")

    async def analyze_problem_characteristics(
        self,
        problem: str,
        domain: str,
        constraints: Dict[str, Any]
    ) -> ProblemCharacteristics:
        """Analyze problem characteristics"""
        complexity = self._assess_complexity(problem, constraints)
        eval_cost = self._assess_evaluation_cost(problem, domain, constraints)
        has_multi_obj = len(constraints.get("objectives", [])) > 1
        needs_diversity = self._check_diversity_need(problem, domain)
        needs_robustness = self._check_robustness_need(domain, constraints)
        constraint_count = len(constraints.get("constraints", []))
        estimated_iters = self._estimate_iterations(domain, complexity, eval_cost)

        return ProblemCharacteristics(
            domain=domain,
            complexity=complexity,
            evaluation_cost=eval_cost,
            has_multiple_objectives=has_multi_obj,
            requires_diversity=needs_diversity,
            requires_robustness=needs_robustness,
            constraint_count=constraint_count,
            estimated_iterations=estimated_iters,
            keywords=self._extract_keywords(problem)
        )

    async def query_historical_performance(
        self,
        domain: str,
        problem_type: str
    ) -> List[HistoricalRun]:
        """Query historical performance data"""
        return [
            r for r in self.historical_runs.values()
            if r.domain == domain
        ][:20]

    async def learn_from_run(self, run_result: Dict[str, Any]) -> None:
        """Learn from completed run"""
        run_id = run_result.get("run_id", f"run_{datetime.now(UTC).isoformat()}")
        domain = run_result.get("domain", "general")

        historical = HistoricalRun(
            run_id=run_id,
            domain=domain,
            strategy_used=run_result.get("strategy_used", "standard"),
            mode_used=run_result.get("mode_used", "standard"),
            problem_complexity=run_result.get("complexity", "medium"),
            final_score=run_result.get("final_score", 0.0),
            convergence_speed=run_result.get("iterations", 0),
            evaluation_count=run_result.get("evaluations", 0),
            diversity_score=run_result.get("diversity_score", 0.5),
            timestamp=datetime.now(UTC),
            metadata=run_result.get("metadata", {}),
            evaluation_cost=run_result.get("evaluation_cost", "moderate"),
            sample_efficiency=run_result.get("final_score", 0.0) / max(run_result.get("evaluations", 1), 1)
        )

        self.historical_runs[run_id] = historical

    def _assess_complexity(self, problem: str, constraints: Dict[str, Any]) -> str:
        """Assess problem complexity using Adaptive MDAP when available, else fall back to keyword-based assessment."""
        # Try Adaptive MDAP classifier first
        if self._complexity_classifier is not None:
            try:
                # Create a subproblem-like structure for the classifier
                subproblem_data = {
                    "description": problem,
                    "constraints": constraints.get("constraints", []),
                    "objectives": constraints.get("objectives", []),
                }
                complexity_level = self._complexity_classifier.classify(subproblem_data)
                # Map classifier output to ComplexityLevel
                complexity_map = {
                    "simple": ComplexityLevel.LOW,
                    "moderate": ComplexityLevel.MEDIUM,
                    "complex": ComplexityLevel.HIGH,
                }
                return complexity_map.get(complexity_level.lower(), ComplexityLevel.MEDIUM)
            except Exception:
                # Fall through to keyword-based assessment on error
                pass
        
        # Fall back to keyword-based assessment
        high_complexity_keywords = [
            "optimize", "maximize", "minimize",
            "multi-objective", "tradeoff", "balance",
            "constraint", "requirement", "specification"
        ]

        low_complexity_keywords = [
            "simple", "basic", "straightforward",
            "single", "linear"
        ]

        problem_lower = problem.lower()

        high_count = sum(1 for kw in high_complexity_keywords if kw in problem_lower)
        low_count = sum(1 for kw in low_complexity_keywords if kw in problem_lower)

        constraint_count = len(constraints.get("constraints", []))
        objective_count = len(constraints.get("objectives", []))

        if high_count > low_count or constraint_count > 3 or objective_count > 1:
            return ComplexityLevel.HIGH
        elif low_count > high_count and constraint_count <= 1:
            return ComplexityLevel.LOW
        else:
            return ComplexityLevel.MEDIUM

    def _assess_evaluation_cost(
        self,
        problem: str,
        domain: str,
        constraints: Dict[str, Any]
    ) -> str:
        """Assess evaluation cost"""
        domain_costs = {
            "science": EvaluationCost.VERY_EXPENSIVE,
            "engineering": EvaluationCost.EXPENSIVE,
            "pharma": EvaluationCost.VERY_EXPENSIVE,
            "finance": EvaluationCost.EXPENSIVE,
            "trading": EvaluationCost.EXPENSIVE,
            "web": EvaluationCost.CHEAP,
        }

        expensive_keywords = [
            "backtest", "simulation", "experiment", "training",
            "monte carlo", "finite element", "docking"
        ]

        problem_lower = problem.lower()
        has_expensive_keyword = any(kw in problem_lower for kw in expensive_keywords)

        base_cost = domain_costs.get(domain, EvaluationCost.MODERATE)

        if has_expensive_keyword and base_cost == EvaluationCost.MODERATE:
            return EvaluationCost.EXPENSIVE

        time_limit = constraints.get("time_limit_seconds", 0)
        if time_limit > 300:
            return EvaluationCost.VERY_EXPENSIVE
        elif time_limit > 60:
            return EvaluationCost.EXPENSIVE
        elif time_limit < 1:
            return EvaluationCost.CHEAP

        return base_cost

    def _check_diversity_need(self, problem: str, domain: str) -> bool:
        """Check if problem needs diverse solutions"""
        diversity_domains = ["finance", "trading", "science", "engineering", "pharma"]

        if domain in diversity_domains:
            return True

        diversity_keywords = [
            "explore", "diverse", "alternative", "novel",
            "different", "variety", "multiple approaches"
        ]

        problem_lower = problem.lower()
        return any(kw in problem_lower for kw in diversity_keywords)

    def _check_robustness_need(self, domain: str, constraints: Dict[str, Any]) -> bool:
        """Check if problem needs robustness testing"""
        robust_domains = ["engineering", "pharma", "finance"]

        if domain in robust_domains:
            return True

        return constraints.get("safety_critical", False)

    def _estimate_iterations(self, domain: str, complexity: str, eval_cost: str) -> int:
        """Estimate required iterations"""
        base = self.domain_heuristics.get(domain, {}).get("typical_iterations", 100)

        if complexity == ComplexityLevel.HIGH:
            base = int(base * 1.5)
        elif complexity == ComplexityLevel.LOW:
            base = int(base * 0.7)

        if eval_cost == EvaluationCost.VERY_EXPENSIVE:
            base = int(base * 0.3)
        elif eval_cost == EvaluationCost.EXPENSIVE:
            base = int(base * 0.5)

        return max(10, min(10000, base))

    def _extract_keywords(self, problem: str) -> List[str]:
        """Extract key terms from problem description"""
        words = problem.lower().split()
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "as", "is"
        }

        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        return keywords[:20]
