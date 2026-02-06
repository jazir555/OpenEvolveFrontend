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
            'predicted_performance': recommendation.point_estimate if hasattr(recommendation, 'point_estimate') else 0.8
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


# ============================================================================
# BASE STRATEGY RECOMMENDER
# ============================================================================

class StrategyRecommender:
    """Base strategy recommender (minimal implementation)"""

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

        # Analyze problem
        problem_chars = await self.analyze_problem_characteristics(
            problem_description, domain, constraints
        )

        # Default recommendation (simple rule-based)
        if problem_chars.requires_diversity:
            system = "openevolve"
            mode = "qd"
            reasoning_text = "Diverse solutions required, use MAP-Elites"
        elif problem_chars.evaluation_cost in [EvaluationCost.EXPENSIVE, EvaluationCost.VERY_EXPENSIVE]:
            system = "loongflow"
            mode = "pes"
            reasoning_text = "Expensive evaluations, use PES for efficiency"
        else:
            system = "openevolve"
            mode = "standard"
            reasoning_text = "Standard optimization problem"

        # Create explanation object
        explanation = Explanation(
            primary_reason=reasoning_text,
            detailed_reasoning=[reasoning_text],
            evidence_from_history=[],
            domain_considerations=[f"Domain: {domain}"],
            risk_factors=["Basic recommender has limited accuracy"]
        )

        # Create performance prediction
        performance = PerformancePrediction(
            expected_iterations=100,
            expected_time_seconds=60.0,
            expected_score=0.75,
            confidence_interval=(0.65, 0.85),
            success_probability=0.7
        )

        return StrategyRecommendation(
            recommended_system=system,
            recommended_mode=mode,
            config_overrides={},
            confidence=0.7,
            reasoning=explanation,
            alternatives=[],
            expected_performance=performance,
            problem_analysis=problem_chars,
            historical_context=[],
            ranking=[]
        )

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
        """Assess problem complexity"""
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


# ============================================================================
# ENSEMBLE STRATEGY SELECTOR
# ============================================================================

class LoongFlowChecker:
    """Check LoongFlow availability and status"""

    _is_available = None
    _check_attempted = False

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if LoongFlow is available

        Returns:
            True if LoongFlow can be imported and used, False otherwise
        """
        if cls._check_attempted:
            return cls._is_available

        cls._check_attempted = True

        try:
            # Try to import LoongFlow
            import sys
            import importlib

            # Check if loongflow package exists
            spec = importlib.util.find_spec("loongflow")
            if spec is None:
                cls._is_available = False
                return False

            # Try to import key modules
            from loongflow.agents.math_agent import MathEvolveAgent
            from loongflow.agents.ml_agent import MLEvolveAgent

            # If we got here, LoongFlow is available
            cls._is_available = True
            return True

        except ImportError:
            cls._is_available = False
            return False
        except Exception as e:
            # Any other error means not available
            cls._is_available = False
            return False

    @classmethod
    def reset(cls):
        """Reset cached availability check (for testing)"""
        cls._is_available = None
        cls._check_attempted = False

    async def rank_strategies(
        self,
        problem_chars: ProblemCharacteristics,
        historical_runs: List[HistoricalRun]
    ) -> List[Tuple[EvolutionSystem, EvolutionMode]]:
        """Rank strategies based on problem and historical data"""
        strategies = [
            (EvolutionSystem.OPENEVOLVE, EvolutionMode.QD),
            (EvolutionSystem.OPENEVOLVE, EvolutionMode.OPTIMIZATION),
            (EvolutionSystem.LOONGFLOW, EvolutionMode.PES),
        ]

        # Score each strategy
        scored = []
        for strategy in strategies:
            score = await self._score_strategy(strategy, problem_chars, historical_runs)
            scored.append((strategy, score))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    async def _score_strategy(
        self,
        strategy: Tuple[EvolutionSystem, EvolutionMode],
        problem_chars: ProblemCharacteristics,
        historical_runs: List[HistoricalRun]
    ) -> float:
        """Score a strategy for the given problem"""
        system, mode = strategy
        score = 0.5  # Base score

        # Diversity bonus
        if problem_chars.requires_diversity and system == EvolutionSystem.OPENEVOLVE:
            if mode == EvolutionMode.QD:
                score += 0.3

        # Efficiency bonus
        if problem_chars.evaluation_cost in [EvaluationCost.EXPENSIVE, EvaluationCost.VERY_EXPENSIVE]:
            if system == EvolutionSystem.LOONGFLOW and mode == EvolutionMode.PES:
                score += 0.3

        # Multi-objective bonus
        if problem_chars.has_multiple_objectives and system == EvolutionSystem.OPENEVOLVE:
            if mode == EvolutionMode.QD:
                score += 0.2

        return min(1.0, score)

    def _parse_historical_run(self, raw_data: Dict[str, Any]) -> HistoricalRun:
        """Parse raw historical run data into HistoricalRun object"""
        return HistoricalRun(
            run_id=raw_data.get("run_id", "unknown"),
            domain=raw_data.get("domain", "general"),
            strategy_used=raw_data.get("strategy_used", "openevolve"),
            mode_used=raw_data.get("mode_used", "qd"),
            problem_complexity=raw_data.get("complexity", "medium"),
            evaluation_cost=raw_data.get("evaluation_cost", "medium"),
            performance_score=raw_data.get("performance", 0.75),
            timestamp=raw_data.get("timestamp", datetime.now(UTC)),
            metadata=raw_data.get("metadata", {})
        )


class EnsembleStrategySelector(StrategyRecommender):
    """
    Enhanced ensemble strategy selector with real-time learning and OpenEvolve-only support

    Uses multiple prediction methods combined with weighted voting:
    1. Rule-Based: Deterministic rules based on problem characteristics
    2. Similarity-Based: Find similar historical problems
    3. Trend-Based: Analyze recent performance trends
    4. ML-Based: Machine learning model (optional)

    Provides confidence intervals and adapts weights based on accuracy.
    Automatically falls back to OpenEvolve-only mode when LoongFlow is unavailable.
    """

    def __init__(
        self,
        knowledge_engine=None,
        llm_client=None,
        use_ai_analysis: bool = True,
        learning_enabled: bool = True,
        enable_ml: bool = False,
        enable_loongflow: bool = True
    ):
        """
        Initialize ensemble strategy selector

        Args:
            knowledge_engine: Knowledge engine for historical data
            llm_client: LLM client for AI-powered analysis
            use_ai_analysis: Enable AI-powered problem analysis
            learning_enabled: Enable learning from new runs
            enable_ml: Enable ML-based prediction (requires scikit-learn)
            enable_loongflow: Enable LoongFlow recommendations (auto-disabled if unavailable)
        """
        # Initialize base class
        super().__init__(
            knowledge_engine=knowledge_engine,
            llm_client=llm_client,
            use_ai_analysis=use_ai_analysis,
            learning_enabled=learning_enabled
        )

        # Ensemble settings
        self.enable_ml = enable_ml
        self.learning_tracker = OnlineLearningTracker()

        # Method weights (will adapt over time)
        self.method_weights = self.learning_tracker.get_current_weights()

        # Minimum samples for each method
        self.min_samples_for_similarity = 5
        self.min_samples_for_trend = 10
        self.min_samples_for_ml = 50

        # LoongFlow availability
        self.enable_loongflow = enable_loongflow
        self.loongflow_available = LoongFlowChecker.is_available() if enable_loongflow else False

    # ========================================================================
    # MAIN ENSEMBLE API
    # ========================================================================

    async def recommend_with_ensemble(
        self,
        problem_description: str,
        domain: str,
        constraints: Dict[str, Any],
        confidence_level: float = 0.95,
        enable_loongflow: Optional[bool] = None
    ) -> EnsemblePrediction:
        """
        Generate recommendation using ensemble methods

        Args:
            problem_description: Text description of the problem
            domain: Problem domain
            constraints: Additional constraints
            confidence_level: Confidence level for intervals (0.90, 0.95, 0.99)
            enable_loongflow: Override to force OpenEvolve-only or LoongFlow-only

        Returns:
            EnsemblePrediction with strategy and confidence interval
        """
        # Determine if LoongFlow should be considered
        can_use_loongflow = self._determine_loongflow_usage(enable_loongflow)

        # Step 1: Analyze problem characteristics
        problem_chars = await self.analyze_problem_characteristics(
            problem_description, domain, constraints
        )

        # Step 2: Query historical performance
        history = await self.query_historical_performance(
            domain, problem_chars.complexity
        )

        # Step 3: Get predictions from all methods (OpenEvolve-only if needed)
        individual_predictions = await self._get_all_predictions(
            problem_chars, history, domain, can_use_loongflow
        )

        # Step 4: Combine predictions with weighted voting
        final_strategy, agreement = self._weighted_voting(
            individual_predictions, self.method_weights
        )

        # Step 5: Calculate confidence interval
        point_estimate, confidence_interval = await self._calculate_confidence_interval(
            final_strategy, problem_chars, history, confidence_level
        )

        # Step 6: Generate reasoning
        reasoning = self._generate_ensemble_reasoning(
            individual_predictions, final_strategy, agreement, can_use_loongflow
        )

        # Step 7: Create ensemble prediction
        prediction = EnsemblePrediction(
            strategy=final_strategy,
            point_estimate=point_estimate,
            confidence_interval=confidence_interval,
            confidence_level=confidence_level,
            prediction_methods=[p.method for p in individual_predictions],
            disagreement_ratio=1.0 - agreement,
            reasoning=reasoning,
            method_weights=self.method_weights.copy(),
            individual_predictions={
                p.method: (p.system, p.mode, p.confidence)
                for p in individual_predictions
            }
        )

        # Step 8: Record recommendation for learning
        if self.learning_enabled:
            self.learning_tracker.record_recommendation(
                prediction, problem_chars
            )

        return prediction

    def _determine_loongflow_usage(self, enable_loongflow: Optional[bool] = None) -> bool:
        """
        Determine if LoongFlow should be used for this recommendation

        Args:
            enable_loongflow: Override to force OpenEvolve-only or LoongFlow-only

        Returns:
            True if LoongFlow should be considered, False for OpenEvolve-only
        """
        if enable_loongflow is not None:
            # Runtime override takes precedence
            return enable_loongflow
        elif not self.enable_loongflow:
            # Config says disabled
            return False
        elif not self.loongflow_available:
            # Not available
            return False
        else:
            # Available and enabled
            return True

    async def _get_all_predictions(
        self,
        problem_chars: ProblemCharacteristics,
        history: List[HistoricalRun],
        domain: str,
        can_use_loongflow: bool = True
    ) -> List[MethodPrediction]:
        """Get predictions from all available methods"""
        predictions = []

        # Method 1: Rule-Based (always available)
        if can_use_loongflow:
            rule_pred = await self._rule_based_prediction(problem_chars, domain)
        else:
            rule_pred = await self._openevolve_rule_based(problem_chars, domain)
        predictions.append(rule_pred)

        # Method 2: Similarity-Based (if enough data)
        if len(history) >= self.min_samples_for_similarity:
            if can_use_loongflow:
                sim_pred = await self._similarity_based_prediction(
                    problem_chars, history
                )
            else:
                sim_pred = await self._similarity_based_openevolve(
                    problem_chars, history
                )
            predictions.append(sim_pred)
        else:
            # Use rule-based as fallback
            predictions.append(rule_pred)

        # Method 3: Trend-Based (if enough data)
        if len(history) >= self.min_samples_for_trend:
            if can_use_loongflow:
                trend_pred = await self._trend_based_prediction(
                    problem_chars, history, domain
                )
            else:
                trend_pred = await self._trend_based_openevolve(
                    problem_chars, history, domain
                )
            predictions.append(trend_pred)
        else:
            # Use rule-based as fallback
            predictions.append(rule_pred)

        # Method 4: ML-Based (if enabled and enough data)
        if self.enable_ml and len(history) >= self.min_samples_for_ml:
            try:
                if can_use_loongflow:
                    ml_pred = await self._ml_based_prediction(
                        problem_chars, history
                    )
                else:
                    ml_pred = await self._ml_based_prediction_openevolve(
                        problem_chars, history
                    )
                predictions.append(ml_pred)
            except Exception as e:
                print(f"ML prediction failed: {e}, using rule-based fallback")
                predictions.append(rule_pred)
        else:
            # Use rule-based as fallback
            predictions.append(rule_pred)

        return predictions

    async def _openevolve_rule_based(
        self,
        problem_chars: ProblemCharacteristics,
        domain: str
    ) -> MethodPrediction:
        """
        OpenEvolve-only rule-based prediction

        Decision tree for OpenEvolve modes only:
        1. Multiple objectives -> MO
        2. Diversity required -> QD
        3. Robustness required -> Adversarial
        4. Default -> Standard

        Args:
            problem_chars: Analyzed problem characteristics
            domain: Problem domain

        Returns:
            MethodPrediction with OpenEvolve system and mode
        """
        reasoning = []
        evidence = {}

        # Rule 1: Multiple objectives
        if problem_chars.has_multiple_objectives:
            system = EvolutionSystem.OPENEVOLVE
            mode = EvolutionMode.MO
            confidence = 0.90
            reasoning.append("Multiple objectives require Pareto optimization")
            evidence['rule_1'] = True

        # Rule 2: Diversity required
        elif problem_chars.requires_diversity:
            system = EvolutionSystem.OPENEVOLVE
            mode = EvolutionMode.QD
            confidence = 0.80
            reasoning.append("Diverse solutions required, use MAP-Elites")
            evidence['rule_2'] = True

        # Rule 3: Robustness required
        elif problem_chars.requires_robustness:
            system = EvolutionSystem.OPENEVOLVE
            mode = EvolutionMode.ADVERSARIAL
            confidence = 0.85
            reasoning.append("Safety-critical, use adversarial testing")
            evidence['rule_3'] = True

        # Default: Standard
        else:
            system = EvolutionSystem.OPENEVOLVE
            mode = EvolutionMode.STANDARD
            confidence = 0.75
            reasoning.append(f"Default to Standard mode for {domain}")
            evidence['default'] = True

        return MethodPrediction(
            method=PredictionMethod.RULE_BASED,
            system=system,
            mode=mode,
            confidence=confidence,
            reasoning="; ".join(reasoning),
            evidence=evidence
        )

    def _get_default_openevolve_mode(self, domain: str) -> MethodPrediction:
        """
        Get default OpenEvolve mode for domain

        Args:
            domain: Problem domain

        Returns:
            MethodPrediction with default mode
        """
        domain_defaults = {
            "finance": "standard",
            "trading": "adversarial",
            "science": "qd",
            "engineering": "standard",
            "pharma": "qd",
            "web": "standard",
            "general": "standard"
        }

        mode = domain_defaults.get(domain, "standard")

        return MethodPrediction(
            system=EvolutionSystem.OPENEVOLVE,
            mode=mode,
            confidence=0.65,
            reasoning=f"Default OpenEvolve mode for {domain} is {mode}"
        )

    async def _similarity_based_openevolve(
        self,
        problem_chars: ProblemCharacteristics,
        history: List[HistoricalRun]
    ) -> MethodPrediction:
        """
        Similarity-based prediction using only OpenEvolve historical data

        Args:
            problem_chars: Analyzed problem characteristics
            history: Historical runs (OpenEvolve only)

        Returns:
            MethodPrediction with best OpenEvolve mode
        """
        # Filter to OpenEvolve runs only
        openevolve_history = [
            r for r in history
            if r.mode_used in ["qd", "mo", "adversarial", "standard"]
        ]

        if len(openevolve_history) < self.min_samples_for_similarity:
            # Not enough OpenEvolve data, use rule-based
            return await self._openevolve_rule_based(problem_chars, problem_chars.domain)

        # Calculate similarity scores for each historical run
        similar_runs = []

        for run in openevolve_history:
            # Keyword overlap
            run_keywords = set(run.metadata.get("keywords", []))
            problem_keywords = set(problem_chars.keywords)

            if not run_keywords:
                similarity = 0.0
            else:
                overlap = len(run_keywords & problem_keywords)
                similarity = overlap / max(len(run_keywords), 1)

            # Domain match bonus
            if run.domain == problem_chars.domain:
                similarity += 0.2

            # Complexity match bonus
            if run.problem_complexity == problem_chars.complexity:
                similarity += 0.1

            similar_runs.append((run, similarity))

        # Sort by similarity and get top k
        k = min(10, len(similar_runs))
        top_runs = sorted(similar_runs, key=lambda x: x[1], reverse=True)[:k]

        # Aggregate performance by strategy (OpenEvolve modes only)
        strategy_performance = defaultdict(list)

        for run, similarity in top_runs:
            key = run.mode_used  # Only OpenEvolve modes
            # Weight by similarity and sample efficiency
            weighted_score = run.final_score * similarity * run.sample_efficiency
            strategy_performance[key].append(weighted_score)

        # Find best strategy
        if not strategy_performance:
            # Fallback to rule-based
            return await self._openevolve_rule_based(problem_chars, problem_chars.domain)

        best_mode = None
        best_score = 0.0

        for mode, scores in strategy_performance.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_mode = mode

        # Confidence based on similarity scores
        avg_similarity = sum(sim for _, sim in top_runs) / len(top_runs)
        confidence = min(0.95, 0.5 + avg_similarity)

        reasoning = (
            f"OpenEvolve-only: Found {len(top_runs)} similar runs (avg similarity: {avg_similarity:.2f}). "
            f"Best mode: {best_mode} with avg score {best_score:.2f}"
        )

        return MethodPrediction(
            method=PredictionMethod.SIMILARITY,
            system=EvolutionSystem.OPENEVOLVE,
            mode=best_mode,
            confidence=confidence,
            reasoning=reasoning,
            evidence={
                'similar_runs': len(top_runs),
                'avg_similarity': avg_similarity,
                'best_score': best_score,
                'openevolve_only': True
            }
        )

    async def _trend_based_openevolve(
        self,
        problem_chars: ProblemCharacteristics,
        history: List[HistoricalRun],
        domain: str
    ) -> MethodPrediction:
        """
        Trend-based prediction using only OpenEvolve data

        Args:
            problem_chars: Analyzed problem characteristics
            history: Historical runs
            domain: Problem domain

        Returns:
            MethodPrediction with best OpenEvolve mode based on trends
        """
        # Filter to OpenEvolve runs only
        openevolve_history = [
            r for r in history
            if r.mode_used in ["qd", "mo", "adversarial", "standard"] and r.domain == domain
        ]

        if len(openevolve_history) < 5:
            # Not enough OpenEvolve data, use rule-based
            return await self._openevolve_rule_based(problem_chars, domain)

        # Sort by timestamp (most recent last)
        openevolve_history.sort(key=lambda r: r.timestamp)

        # Calculate trend for each OpenEvolve mode
        trends = {}

        for mode in ["qd", "mo", "adversarial", "standard"]:
            mode_runs = [r for r in openevolve_history if r.mode_used == mode]

            if len(mode_runs) < 3:
                continue

            # Calculate scores over time
            scores = [r.final_score for r in mode_runs]

            # Simple linear trend (last vs first)
            if len(scores) >= 2:
                # Calculate moving averages
                window = min(5, len(scores))
                recent_avg = sum(scores[-window:]) / window

                if len(scores) > window:
                    old_avg = sum(scores[-(window*2):-window]) / window
                else:
                    old_avg = scores[0]

                # Trend: positive if improving
                trend = recent_avg - old_avg
                trends[mode] = {
                    'trend': trend,
                    'recent_avg': recent_avg,
                    'sample_count': len(mode_runs)
                }

        if not trends:
            # No trend data, use rule-based
            return await self._openevolve_rule_based(problem_chars, domain)

        # Find mode with best improving trend
        best_mode_data = max(trends.items(), key=lambda x: x[1]['trend'])
        mode_name, trend_data = best_mode_data

        # Confidence based on trend strength and sample count
        trend_strength = abs(trend_data['trend'])
        confidence = min(0.90, 0.5 + trend_strength * 2)
        confidence *= min(1.0, trend_data['sample_count'] / 20)

        reasoning = (
            f"OpenEvolve-only: Analyzing {trend_data['sample_count']} recent runs. "
            f"{mode_name.upper()} shows improving trend "
            f"({trend_data['trend']:+.3f}, avg: {trend_data['recent_avg']:.2f})"
        )

        return MethodPrediction(
            method=PredictionMethod.TREND,
            system=EvolutionSystem.OPENEVOLVE,
            mode=mode_name,
            confidence=confidence,
            reasoning=reasoning,
            evidence={
                **trend_data,
                'openevolve_only': True
            }
        )

    async def _ml_based_prediction_openevolve(
        self,
        problem_chars: ProblemCharacteristics,
        history: List[HistoricalRun]
    ) -> MethodPrediction:
        """
        ML-based prediction using only OpenEvolve data

        Args:
            problem_chars: Analyzed problem characteristics
            history: Historical runs (OpenEvolve only)

        Returns:
            MethodPrediction with ML-predicted OpenEvolve mode
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
        except ImportError:
            # ML not available, use rule-based
            return await self._openevolve_rule_based(problem_chars, problem_chars.domain)

        # Filter to OpenEvolve runs only
        openevolve_history = [
            r for r in history
            if r.mode_used in ["qd", "mo", "adversarial", "standard"]
        ]

        if len(openevolve_history) < self.min_samples_for_ml:
            return await self._openevolve_rule_based(problem_chars, problem_chars.domain)

        # Prepare training data
        X = []  # Features
        y = []  # Labels (OpenEvolve mode)

        for run in openevolve_history:
            # Extract features
            features = self._extract_features(run, problem_chars)
            X.append(features)

            # Label: OpenEvolve mode that worked best
            y.append(run.mode_used)

        # Train model
        X = np.array(X)
        y = np.array(y)

        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)

        # Predict for current problem
        current_features = self._extract_features_for_prediction(problem_chars)
        prediction = model.predict([current_features])[0]
        probabilities = model.predict_proba([current_features])[0]

        # Get probability for predicted class
        class_idx = list(model.classes_).index(prediction)
        confidence = probabilities[class_idx]

        reasoning = (
            f"OpenEvolve-only ML: Model trained on {len(X)} historical runs. "
            f"Predicted: {prediction} (confidence: {confidence:.2f})"
        )

        return MethodPrediction(
            method=PredictionMethod.ML,
            system=EvolutionSystem.OPENEVOLVE,
            mode=prediction,
            confidence=confidence,
            reasoning=reasoning,
            evidence={
                'training_samples': len(X),
                'model_classes': list(model.classes_),
                'openevolve_only': True
            }
        )

    async def _rule_based_prediction(
        self,
        problem_chars: ProblemCharacteristics,
        domain: str
    ) -> MethodPrediction:
        """
        Deterministic rules based on problem characteristics

        Decision tree:
        1. Evaluation cost expensive -> PES
        2. Multiple objectives -> MO
        3. Diversity required -> QD
        4. Robustness required -> Adversarial
        5. Default -> PES (best general performance)
        """
        reasoning = []
        evidence = {}

        # Rule 1: Expensive evaluations
        if problem_chars.evaluation_cost in ["expensive", "very_expensive"]:
            system = EvolutionSystem.LOONGFLOW
            mode = EvolutionMode.PES
            confidence = 0.85
            reasoning.append("Expensive evaluations favor PES (60% fewer evaluations)")
            evidence['rule_1'] = True

        # Rule 2: Multiple objectives
        elif problem_chars.has_multiple_objectives:
            system = EvolutionSystem.OPENEVOLVE
            mode = EvolutionMode.MO
            confidence = 0.90
            reasoning.append("Multiple objectives require Pareto optimization")
            evidence['rule_2'] = True

        # Rule 3: Diversity required
        elif problem_chars.requires_diversity:
            system = EvolutionSystem.OPENEVOLVE
            mode = EvolutionMode.QD
            confidence = 0.80
            reasoning.append("Diverse solutions required, use MAP-Elites")
            evidence['rule_3'] = True

        # Rule 4: Robustness required
        elif problem_chars.requires_robustness:
            system = EvolutionSystem.OPENEVOLVE
            mode = EvolutionMode.ADVERSARIAL
            confidence = 0.85
            reasoning.append("Safety-critical, use adversarial testing")
            evidence['rule_4'] = True

        # Default: PES
        else:
            system = EvolutionSystem.LOONGFLOW
            mode = EvolutionMode.PES
            confidence = 0.75
            reasoning.append("Default to PES for best general performance")
            evidence['default'] = True

        return MethodPrediction(
            method=PredictionMethod.RULE_BASED,
            system=system,
            mode=mode,
            confidence=confidence,
            reasoning="; ".join(reasoning),
            evidence=evidence
        )

    async def _similarity_based_prediction(
        self,
        problem_chars: ProblemCharacteristics,
        history: List[HistoricalRun]
    ) -> MethodPrediction:
        """
        Find similar problems and use their best strategies

        Uses keyword overlap and domain matching to find similar runs.
        Aggregates performance by strategy and returns best performing.
        """
        # Calculate similarity scores for each historical run
        similar_runs = []

        for run in history:
            # Keyword overlap
            run_keywords = set(run.metadata.get("keywords", []))
            problem_keywords = set(problem_chars.keywords)

            if not run_keywords:
                similarity = 0.0
            else:
                overlap = len(run_keywords & problem_keywords)
                similarity = overlap / max(len(run_keywords), 1)

            # Domain match bonus
            if run.domain == problem_chars.domain:
                similarity += 0.2

            # Complexity match bonus
            if run.problem_complexity == problem_chars.complexity:
                similarity += 0.1

            similar_runs.append((run, similarity))

        # Sort by similarity and get top k
        k = min(10, len(similar_runs))
        top_runs = sorted(similar_runs, key=lambda x: x[1], reverse=True)[:k]

        # Aggregate performance by strategy
        strategy_performance = defaultdict(list)

        for run, similarity in top_runs:
            key = (run.strategy_used, run.mode_used)
            # Weight by similarity and sample efficiency
            weighted_score = run.final_score * similarity * run.sample_efficiency
            strategy_performance[key].append(weighted_score)

        # Find best strategy
        if not strategy_performance:
            # Fallback to rule-based
            return await self._rule_based_prediction(problem_chars, problem_chars.domain)

        best_strategy = None
        best_score = 0.0

        for strategy, scores in strategy_performance.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_strategy = strategy

        strategy_used, mode_used = best_strategy

        # Determine system from mode
        if mode_used == "pes":
            system = EvolutionSystem.LOONGFLOW
        else:
            system = EvolutionSystem.OPENEVOLVE

        # Confidence based on similarity scores
        avg_similarity = sum(sim for _, sim in top_runs) / len(top_runs)
        confidence = min(0.95, 0.5 + avg_similarity)

        reasoning = (
            f"Found {len(top_runs)} similar runs (avg similarity: {avg_similarity:.2f}). "
            f"Best strategy: {strategy_used} with avg score {best_score:.2f}"
        )

        return MethodPrediction(
            method=PredictionMethod.SIMILARITY,
            system=system,
            mode=mode_used,
            confidence=confidence,
            reasoning=reasoning,
            evidence={
                'similar_runs': len(top_runs),
                'avg_similarity': avg_similarity,
                'best_score': best_score
            }
        )

    async def _trend_based_prediction(
        self,
        problem_chars: ProblemCharacteristics,
        history: List[HistoricalRun],
        domain: str
    ) -> MethodPrediction:
        """
        Analyze recent performance trends

        Gets last N runs for this domain and calculates trend
        for each strategy. Returns strategy with improving trend.
        """
        # Filter by domain and get recent runs
        domain_runs = [r for r in history if r.domain == domain]

        if len(domain_runs) < 5:
            # Not enough data, use rule-based
            return await self._rule_based_prediction(problem_chars, domain)

        # Sort by timestamp (most recent last)
        domain_runs.sort(key=lambda r: r.timestamp)

        # Calculate trend for each strategy
        trends = {}

        for strategy in ["pes", "qd", "mo", "adversarial", "standard"]:
            strategy_runs = [r for r in domain_runs if r.strategy_used == strategy]

            if len(strategy_runs) < 3:
                continue

            # Calculate scores over time
            scores = [r.final_score for r in strategy_runs]

            # Simple linear trend (last vs first)
            if len(scores) >= 2:
                # Calculate moving averages
                window = min(5, len(scores))
                recent_avg = sum(scores[-window:]) / window

                if len(scores) > window:
                    old_avg = sum(scores[-(window*2):-window]) / window
                else:
                    old_avg = scores[0]

                # Trend: positive if improving
                trend = recent_avg - old_avg
                trends[strategy] = {
                    'trend': trend,
                    'recent_avg': recent_avg,
                    'sample_count': len(strategy_runs)
                }

        if not trends:
            # No trend data, use rule-based
            return await self._rule_based_prediction(problem_chars, domain)

        # Find strategy with best improving trend
        best_strategy = max(trends.items(), key=lambda x: x[1]['trend'])
        strategy_name, trend_data = best_strategy

        # Determine system from mode
        if strategy_name == "pes":
            system = EvolutionSystem.LOONGFLOW
        else:
            system = EvolutionSystem.OPENEVOLVE

        # Confidence based on trend strength and sample count
        trend_strength = abs(trend_data['trend'])
        confidence = min(0.90, 0.5 + trend_strength * 2)
        confidence *= min(1.0, trend_data['sample_count'] / 20)

        reasoning = (
            f"Analyzing {trend_data['sample_count']} recent runs. "
            f"{strategy_name.upper()} shows improving trend "
            f"({trend_data['trend']:+.3f}, avg: {trend_data['recent_avg']:.2f})"
        )

        return MethodPrediction(
            method=PredictionMethod.TREND,
            system=system,
            mode=strategy_name,
            confidence=confidence,
            reasoning=reasoning,
            evidence=trend_data
        )

    async def _ml_based_prediction(
        self,
        problem_chars: ProblemCharacteristics,
        history: List[HistoricalRun]
    ) -> MethodPrediction:
        """
        Train simple model on historical data

        Features: problem characteristics
        Labels: best strategy
        Model: Random forest or similar

        Note: Requires scikit-learn
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
        except ImportError:
            # ML not available, use rule-based
            return await self._rule_based_prediction(problem_chars, problem_chars.domain)

        # Prepare training data
        X = []  # Features
        y = []  # Labels (strategy)

        for run in history:
            # Extract features
            features = self._extract_features(run, problem_chars)
            X.append(features)

            # Label: strategy that worked best
            y.append(run.strategy_used)

        if len(X) < self.min_samples_for_ml:
            return await self._rule_based_prediction(problem_chars, problem_chars.domain)

        # Train model
        X = np.array(X)
        y = np.array(y)

        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)

        # Predict for current problem
        current_features = self._extract_features_for_prediction(problem_chars)
        prediction = model.predict([current_features])[0]
        probabilities = model.predict_proba([current_features])[0]

        # Get probability for predicted class
        class_idx = list(model.classes_).index(prediction)
        confidence = probabilities[class_idx]

        # Determine system from mode
        if prediction == "pes":
            system = EvolutionSystem.LOONGFLOW
        else:
            system = EvolutionSystem.OPENEVOLVE

        reasoning = (
            f"ML model trained on {len(X)} historical runs. "
            f"Predicted: {prediction} (confidence: {confidence:.2f})"
        )

        return MethodPrediction(
            method=PredictionMethod.ML,
            system=system,
            mode=prediction,
            confidence=confidence,
            reasoning=reasoning,
            evidence={
                'training_samples': len(X),
                'model_classes': list(model.classes_),
                'feature_importance': dict(zip(
                    ['eval_cost', 'complexity', 'multi_obj', 'diversity', 'robustness'],
                    model.feature_importances_.tolist()
                )) if hasattr(model, 'feature_importances_') else {}
            }
        )

    def _extract_features(
        self,
        run: HistoricalRun,
        problem_chars: ProblemCharacteristics
    ) -> List[float]:
        """Extract feature vector from historical run"""
        features = []

        # Numerical encoding
        cost_map = {"cheap": 0, "moderate": 1, "expensive": 2, "very_expensive": 3}
        complexity_map = {"low": 0, "medium": 1, "high": 2}

        features.append(cost_map.get(run.evaluation_cost, 1))
        features.append(complexity_map.get(run.problem_complexity, 1))

        # Boolean features from problem chars (use defaults for historical)
        features.append(1.0)  # has_multiple_objectives (placeholder)
        features.append(0.0)  # requires_diversity (placeholder)
        features.append(0.0)  # requires_robustness (placeholder)

        return features

    def _extract_features_for_prediction(
        self,
        problem_chars: ProblemCharacteristics
    ) -> List[float]:
        """Extract feature vector for prediction"""
        features = []

        # Numerical encoding
        cost_map = {"cheap": 0, "moderate": 1, "expensive": 2, "very_expensive": 3}
        complexity_map = {"low": 0, "medium": 1, "high": 2}

        features.append(cost_map.get(problem_chars.evaluation_cost, 1))
        features.append(complexity_map.get(problem_chars.complexity, 1))
        features.append(float(problem_chars.has_multiple_objectives))
        features.append(float(problem_chars.requires_diversity))
        features.append(float(problem_chars.requires_robustness))

        return features

    def _weighted_voting(
        self,
        predictions: List[MethodPrediction],
        weights: Dict[str, float]
    ) -> Tuple[Tuple[str, str], float]:
        """
        Combine predictions using weighted voting

        Args:
            predictions: List of individual predictions
            weights: Weight for each method

        Returns:
            ((system, mode), agreement_score)
        """
        # Count weighted votes
        votes = defaultdict(float)

        for pred in predictions:
            weight = weights.get(pred.method, 0.25)
            key = (pred.system, pred.mode)
            votes[key] += weight * pred.confidence

        # Find winner
        winner = max(votes.items(), key=lambda x: x[1])

        # Calculate agreement (entropy-based)
        total_votes = sum(votes.values())
        if total_votes > 0:
            # Normalize
            normalized_votes = {k: v / total_votes for k, v in votes.items()}

            # Calculate agreement (1 - entropy)
            entropy = -sum(p * math.log(p) for p in normalized_votes.values() if p > 0)
            max_entropy = math.log(len(votes))
            agreement = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)
        else:
            agreement = 0.0

        return winner[0], agreement

    async def _calculate_confidence_interval(
        self,
        strategy: Tuple[str, str],
        problem_chars: ProblemCharacteristics,
        history: List[HistoricalRun],
        confidence_level: float = 0.95
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate confidence interval using bootstrap method

        Args:
            strategy: Selected strategy (system, mode)
            problem_chars: Problem characteristics
            history: Historical runs
            confidence_level: Confidence level (0.90, 0.95, 0.99)

        Returns:
            (point_estimate, (lower_bound, upper_bound))
        """
        # Filter relevant history
        system, mode = strategy
        relevant_runs = [
            r for r in history
            if r.strategy_used == mode
        ]

        if len(relevant_runs) < 3:
            # Not enough data, use heuristic interval
            point_estimate = 0.75
            margin = 0.15
            return point_estimate, (point_estimate - margin, point_estimate + margin)

        # Bootstrap sampling
        n_samples = 1000
        bootstrap_scores = []

        for _ in range(n_samples):
            # Sample with replacement
            sample = random.choices(relevant_runs, k=len(relevant_runs))
            # Calculate mean score
            mean_score = sum(r.final_score for r in sample) / len(sample)
            bootstrap_scores.append(mean_score)

        # Calculate percentiles
        alpha = 1.0 - confidence_level
        lower = np.percentile(bootstrap_scores, alpha / 2 * 100) if NUMPY_AVAILABLE else sorted(bootstrap_scores)[int(alpha / 2 * n_samples)]
        upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100) if NUMPY_AVAILABLE else sorted(bootstrap_scores)[int((1 - alpha / 2) * n_samples)]

        # Point estimate is mean of bootstrap distribution
        point_estimate = sum(bootstrap_scores) / n_samples

        return point_estimate, (lower, upper)

    def _generate_ensemble_reasoning(
        self,
        predictions: List[MethodPrediction],
        final_strategy: Tuple[str, str],
        agreement: float,
        can_use_loongflow: bool = True
    ) -> str:
        """Generate explanation for ensemble decision"""
        lines = [
            "## Ensemble Strategy Selection",
            f"",
            f"**Selected Strategy:** {str(final_strategy[0]).upper()} / {str(final_strategy[1]).upper()}",
            f"**Method Agreement:** {agreement:.1%}",
            f"",
        ]

        # Add mode indicator
        if not can_use_loongflow:
            lines.append("**Mode:** OpenEvolve-Only (LoongFlow unavailable)")
            lines.append("")

        lines.append("### Individual Method Predictions:")

        for pred in predictions:
            lines.append(
                f"- **{pred.method.value}**: {pred.system}/{pred.mode} "
                f"(confidence: {pred.confidence:.1%})"
            )
            lines.append(f"  - Reasoning: {pred.reasoning}")

        lines.extend([
            f"",
            "### Ensemble Decision:",
            f"The weighted vote selected {str(final_strategy[0]).upper()}/{str(final_strategy[1]).upper()} "
            f"based on {len(predictions)} prediction methods."
        ])

        if agreement > 0.8:
            lines.append("High agreement among methods indicates strong consensus.")
        elif agreement > 0.5:
            lines.append("Moderate agreement among methods.")
        else:
            lines.append("Low disagreement among methods - recommendation may be uncertain.")

        if not can_use_loongflow:
            lines.append("")
            lines.append("**Note:** Running in OpenEvolve-only mode. LoongFlow is not available or disabled.")

        return "\n".join(lines)

    # ========================================================================
    # LEARNING FROM RESULTS
    # ========================================================================

    async def learn_from_run(self, run_result: Dict[str, Any]) -> None:
        """
        Learn from completed evolutionary run

        Overrides base method to also update learning tracker.

        Args:
            run_result: Result data from evolutionary run
        """
        # Call base class learning
        await super().learn_from_run(run_result)

        # Update learning tracker if this was tracked
        if 'recommendation_id' in run_result:
            actual_performance = run_result.get('final_score', 0.0)
            metrics = self.learning_tracker.record_actual_performance(
                recommendation_id=run_result['recommendation_id'],
                actual_performance=actual_performance,
                run_id=run_result.get('run_id'),
                metadata=run_result.get('metadata', {})
            )

            # Update weights if adapted
            if metrics.get('weights_adapted'):
                self.method_weights = metrics.get('new_weights', self.method_weights)

    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning tracker metrics"""
        return self.learning_tracker.get_accuracy_metrics()

    def explain_ensemble_recommendation(
        self,
        prediction: EnsemblePrediction,
        problem_chars: ProblemCharacteristics
    ) -> str:
        """
        Explain ensemble recommendation in detail

        Args:
            prediction: Ensemble prediction
            problem_chars: Analyzed problem characteristics

        Returns:
            Formatted explanation
        """
        lines = [
            "# Ensemble Strategy Recommendation",
            f"",
            f"## Selected Strategy",
            f"**System:** {prediction.strategy[0].upper()}",
            f"**Mode:** {prediction.strategy[1].upper()}",
            f"",
            f"## Expected Performance",
            f"**Point Estimate:** {prediction.point_estimate:.2%}",
            f"**{prediction.confidence_level*100:.0f}% Confidence Interval:** "
            f"[{prediction.confidence_interval[0]:.2%}, {prediction.confidence_interval[1]:.2%}]",
            f"",
            f"## Method Agreement",
            f"**Agreement Level:** {(1.0 - prediction.disagreement_ratio):.1%}",
            f"**Disagreement Ratio:** {prediction.disagreement_ratio:.1%}",
            f"",
            f"### Prediction Methods Used:"
        ]

        for method in prediction.prediction_methods:
            weight = prediction.method_weights.get(method, 0.0)
            lines.append(f"- **{method}**: {weight:.1%} weight")

        lines.extend([
            f"",
            f"### Individual Predictions:"
        ])

        for method, (system, mode, confidence) in prediction.individual_predictions.items():
            lines.append(f"- **{method}**: {system}/{mode} (confidence: {confidence:.1%})")

        lines.extend([
            f"",
            f"## Detailed Reasoning",
            prediction.reasoning,
            f"",
            f"## Problem Analysis",
            f"- **Domain:** {problem_chars.domain}",
            f"- **Complexity:** {problem_chars.complexity}",
            f"- **Evaluation Cost:** {problem_chars.evaluation_cost}",
            f"- **Multiple Objectives:** {problem_chars.has_multiple_objectives}",
            f"- **Requires Diversity:** {problem_chars.requires_diversity}",
            f"- **Requires Robustness:** {problem_chars.requires_robustness}",
            f"",
            f"## Learning Metrics"
        ])

        # Add learning metrics
        metrics = self.get_learning_metrics()
        lines.append(f"- **Average Accuracy:** {metrics['average_accuracy']:.1%}")
        lines.append(f"- **Total Recommendations:** {metrics['total_recommendations']}")
        lines.append(f"- **Trend:** {metrics['recent_trend']}")

        if 'method_weights' in metrics:
            lines.append(f"- **Current Method Weights:**")
            for method, weight in metrics['method_weights'].items():
                lines.append(f"  - {method}: {weight:.1%}")

        return "\n".join(lines)

    # ========================================================================
    # COLD START HANDLING
    # ========================================================================

    async def handle_cold_start(
        self,
        problem_chars: ProblemCharacteristics,
        domain: str,
        enable_loongflow: Optional[bool] = None
    ) -> EnsemblePrediction:
        """
        Generate good recommendations even without historical data

        Uses rule-based with lower confidence and domain-specific defaults.

        Args:
            problem_chars: Analyzed problem characteristics
            domain: Problem domain
            enable_loongflow: Force OpenEvolve-only or LoongFlow-only

        Returns:
            EnsemblePrediction with cold-start adjustments
        """
        # Determine if LoongFlow should be used
        can_use_loongflow = self._determine_loongflow_usage(enable_loongflow)

        # Get rule-based prediction
        if can_use_loongflow:
            rule_pred = await self._rule_based_prediction(problem_chars, domain)
        else:
            rule_pred = await self._openevolve_rule_based(problem_chars, domain)

        # Lower confidence due to cold start
        rule_pred.confidence *= 0.8

        # Get domain defaults
        defaults = self.domain_heuristics.get(domain, {})

        # Create prediction
        mode_indicator = "OpenEvolve-Only" if not can_use_loongflow else "Full"
        prediction = EnsemblePrediction(
            strategy=(rule_pred.system, rule_pred.mode),
            point_estimate=defaults.get('expected_score', 0.70),
            confidence_interval=(0.60, 0.80),
            confidence_level=0.80,  # Lower due to cold start
            prediction_methods=['rule_based'],
            disagreement_ratio=0.0,  # Only one method
            reasoning=f"Cold start ({mode_indicator}): Using rule-based defaults for {domain} domain. "
                      f"{rule_pred.reasoning} [Cold start: limited historical data]",
            method_weights={'rule_based': 1.0},
            individual_predictions={
                'rule_based': (rule_pred.system, rule_pred.mode, rule_pred.confidence)
            }
        )

        return prediction

    async def recommend_openevolve_only(
        self,
        problem_description: str,
        domain: str,
        constraints: Dict[str, Any],
        confidence_level: float = 0.95
    ) -> EnsemblePrediction:
        """
        Convenience method for OpenEvolve-only recommendation

        This explicitly requests no LoongFlow usage, regardless of availability.

        Args:
            problem_description: Text description of the problem
            domain: Problem domain
            constraints: Additional constraints
            confidence_level: Confidence level for intervals (0.90, 0.95, 0.99)

        Returns:
            EnsemblePrediction with OpenEvolve-only strategy
        """
        return await self.recommend_with_ensemble(
            problem_description=problem_description,
            domain=domain,
            constraints=constraints,
            confidence_level=confidence_level,
            enable_loongflow=False  # Explicitly disable LoongFlow
        )

    def is_loongflow_available(self) -> bool:
        """
        Check if LoongFlow is available for recommendations

        Returns:
            True if LoongFlow is available and enabled
        """
        return self.loongflow_available and self.enable_loongflow

    def get_available_modes(self) -> List[str]:
        """
        Get list of available evolutionary modes

        Returns:
            List of available mode names
        """
        if self.is_loongflow_available():
            return ["pes", "qd", "mo", "adversarial", "standard"]
        else:
            return ["qd", "mo", "adversarial", "standard"]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def recommend_evolutionary_strategy(
    problem_description: str,
    domain: str = "general",
    constraints: Optional[Dict[str, Any]] = None,
    knowledge_engine=None,
    use_ensemble: bool = True
) -> EnsemblePrediction:
    """
    Convenience function for strategy recommendation

    Args:
        problem_description: Text description of the problem
        domain: Problem domain
        constraints: Additional constraints
        knowledge_engine: Optional knowledge engine for historical data
        use_ensemble: Use ensemble methods (default: True)

    Returns:
        EnsemblePrediction
    """
    if use_ensemble:
        selector = EnsembleStrategySelector(knowledge_engine=knowledge_engine)
        return await selector.recommend_with_ensemble(
            problem_description, domain, constraints or {}
        )
    else:
        # Use base class
        from knowledge_engine.core.strategy_recommender import StrategyRecommender
        recommender = StrategyRecommender(knowledge_engine=knowledge_engine)
        return await recommender.recommend_strategy(
            problem_description, domain, constraints or {}
        )


