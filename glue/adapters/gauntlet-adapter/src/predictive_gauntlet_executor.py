"""
Predictive Gauntlet Executor

Integrates success prediction with gauntlet execution for intelligent resource allocation.

Features:
- Pre-execution success assessment
- Dynamic difficulty adjustment
- Resource optimization
- Outcome-based learning
- Integration with success prediction models

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ExecutionDecision(Enum):
    """Decision for gauntlet execution"""
    PROCEED = "proceed"
    SKIP_LOW_PROBABILITY = "skip_low_probability"
    SKIP_HIGH_COST = "skip_high_cost"
    ADJUST_DIFFICULTY = "adjust_difficulty"


@dataclass
class PredictionResult:
    """
    Result from success prediction model.

    Attributes:
        success_probability: Predicted probability of success (0.0-1.0)
        confidence: Confidence in prediction (0.0-1.0)
        risk_factors: Identified risk factors
        recommended_difficulty: Suggested difficulty level
        estimated_time: Estimated execution time (seconds)
        estimated_cost: Estimated computational cost
    """
    success_probability: float
    confidence: float
    risk_factors: List[str] = field(default_factory=list)
    recommended_difficulty: str = "medium"
    estimated_time: float = 30.0
    estimated_cost: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success_probability": self.success_probability,
            "confidence": self.confidence,
            "risk_factors": self.risk_factors,
            "recommended_difficulty": self.recommended_difficulty,
            "estimated_time": self.estimated_time,
            "estimated_cost": self.estimated_cost
        }


@dataclass
class ExecutionPlan:
    """
    Execution plan based on prediction.

    Attributes:
        decision: Execution decision
        adjusted_config: Adjusted gauntlet configuration
        reasoning: Explanation for decision
        expected_outcome: Expected execution outcome
        resource_allocation: Resource allocation strategy
    """
    decision: ExecutionDecision
    adjusted_config: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    expected_outcome: str = ""
    resource_allocation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "decision": self.decision.value,
            "adjusted_config": self.adjusted_config,
            "reasoning": self.reasoning,
            "expected_outcome": self.expected_outcome,
            "resource_allocation": self.resource_allocation
        }


@dataclass
class ExecutionResult:
    """
    Result from predictive gauntlet execution.

    Attributes:
        prediction: Initial prediction
        actual_outcome: Actual execution outcome
        prediction_accuracy: How accurate the prediction was
        execution_time: Actual execution time
        cost_savings: Cost savings from prediction (if any)
        learning_data: Data for improving future predictions
    """
    prediction: PredictionResult
    actual_outcome: Dict[str, Any]
    prediction_accuracy: float
    execution_time: float
    cost_savings: float = 0.0
    learning_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "prediction": self.prediction.to_dict(),
            "actual_outcome": self.actual_outcome,
            "prediction_accuracy": self.prediction_accuracy,
            "execution_time": self.execution_time,
            "cost_savings": self.cost_savings,
            "learning_data": self.learning_data
        }


class PredictiveGauntletExecutor:
    """
    Gauntlet executor with predictive capabilities.

    Uses success prediction to make intelligent decisions about:
    - Whether to execute gauntlet
    - How to adjust difficulty
    - How to allocate resources
    - When to skip expensive evaluations

    Example:
        >>> executor = PredictiveGauntletExecutor()
        >>>
        >>> # Predict success before execution
        >>> prediction = executor.predict_success(
        ...     solution="def solve(): return optimal",
        ...     problem="Optimize portfolio",
        ...     domain="finance"
        ... )
        >>>
        >>> # Get execution plan
        >>> plan = executor.create_execution_plan(prediction)
        >>>
        >>> # Execute if recommended
        >>> if plan.decision == ExecutionDecision.PROCEED:
        ...     result = executor.execute_with_prediction(
        ...         solution="def solve(): return optimal",
        ...         problem="Optimize portfolio",
        ...         domain="finance",
        ...         prediction=prediction
        ...     )
    """

    def __init__(
        self,
        success_threshold: float = 0.3,
        confidence_threshold: float = 0.6,
        cost_threshold: float = 100.0
    ):
        """
        Initialize predictive executor.

        Args:
            success_threshold: Minimum success probability to proceed
            confidence_threshold: Minimum prediction confidence to act on
            cost_threshold: Maximum estimated cost to proceed
        """
        self.success_threshold = success_threshold
        self.confidence_threshold = confidence_threshold
        self.cost_threshold = cost_threshold

        # Historical data for learning
        self.prediction_history: List[Dict[str, Any]] = []

        logger.info(
            f"Predictive Gauntlet Executor initialized: "
            f"success_threshold={success_threshold}, "
            f"confidence_threshold={confidence_threshold}"
        )

    def predict_success(
        self,
        solution: str,
        problem: str,
        domain: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """
        Predict success probability for solution.

        Analyzes solution and problem to estimate likelihood of passing
        the gauntlet evaluation.

        Args:
            solution: Solution code/content
            problem: Problem statement
            domain: Problem domain
            context: Additional context

        Returns:
            PredictionResult with probability and risk factors
        """
        start_time = time.time()

        # Extract features from solution and problem
        features = self._extract_features(solution, problem, domain, context)

        # Calculate base success probability
        success_prob = self._calculate_success_probability(features)

        # Calculate prediction confidence
        confidence = self._calculate_confidence(features)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(features)

        # Recommend difficulty
        difficulty = self._recommend_difficulty(success_prob, features)

        # Estimate execution time and cost
        estimated_time = self._estimate_execution_time(features, domain)
        estimated_cost = self._estimate_cost(features, estimated_time)

        prediction_time = time.time() - start_time
        logger.debug(
            f"Prediction complete: success_prob={success_prob:.2f}, "
            f"confidence={confidence:.2f}, time={prediction_time:.3f}s"
        )

        return PredictionResult(
            success_probability=success_prob,
            confidence=confidence,
            risk_factors=risk_factors,
            recommended_difficulty=difficulty,
            estimated_time=estimated_time,
            estimated_cost=estimated_cost
        )

    def create_execution_plan(
        self,
        prediction: PredictionResult,
        base_config: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create execution plan based on prediction.

        Args:
            prediction: Prediction result
            base_config: Base gauntlet configuration

        Returns:
            ExecutionPlan with decision and adjustments
        """
        config = base_config or {}

        # Check confidence threshold
        if prediction.confidence < self.confidence_threshold:
            return ExecutionPlan(
                decision=ExecutionDecision.SKIP_LOW_PROBABILITY,
                reasoning=f"Prediction confidence ({prediction.confidence:.2f}) below threshold ({self.confidence_threshold})"
            )

        # Check success threshold
        if prediction.success_probability < self.success_threshold:
            return ExecutionPlan(
                decision=ExecutionDecision.SKIP_LOW_PROBABILITY,
                reasoning=f"Success probability ({prediction.success_probability:.2f}) below threshold ({self.success_threshold})",
                expected_outcome="Likely to fail gauntlet evaluation"
            )

        # Check cost threshold
        if prediction.estimated_cost > self.cost_threshold:
            return ExecutionPlan(
                decision=ExecutionDecision.SKIP_HIGH_COST,
                reasoning=f"Estimated cost ({prediction.estimated_cost:.1f}) exceeds threshold ({self.cost_threshold})",
                expected_outcome="Too expensive to evaluate"
            )

        # Success probability is high enough to proceed
        # Consider adjusting difficulty based on prediction
        if prediction.success_probability > 0.8:
            # High success probability - consider increasing difficulty
            adjusted_config = config.copy()
            adjusted_config["round1_threshold"] = adjusted_config.get("round1_threshold", 0.5) + 0.1
            adjusted_config["round2_threshold"] = adjusted_config.get("round2_threshold", 0.6) + 0.1
            adjusted_config["round3_threshold"] = adjusted_config.get("round3_threshold", 0.7) + 0.1

            return ExecutionPlan(
                decision=ExecutionDecision.ADJUST_DIFFICULTY,
                adjusted_config=adjusted_config,
                reasoning=f"High success probability ({prediction.success_probability:.2f}) - increasing difficulty for better quality assessment",
                expected_outcome="Expected to pass with higher thresholds",
                resource_allocation={"strategy": "standard"}
            )
        elif prediction.success_probability < 0.5:
            # Low success probability - consider decreasing difficulty
            adjusted_config = config.copy()
            adjusted_config["round1_threshold"] = max(0.3, adjusted_config.get("round1_threshold", 0.5) - 0.1)
            adjusted_config["round2_threshold"] = max(0.4, adjusted_config.get("round2_threshold", 0.6) - 0.1)
            adjusted_config["round3_threshold"] = max(0.5, adjusted_config.get("round3_threshold", 0.7) - 0.1)

            return ExecutionPlan(
                decision=ExecutionDecision.ADJUST_DIFFICULTY,
                adjusted_config=adjusted_config,
                reasoning=f"Moderate success probability ({prediction.success_probability:.2f}) - lowering thresholds to avoid false negatives",
                expected_outcome="May pass with adjusted thresholds",
                resource_allocation={"strategy": "conservative"}
            )

        # Moderate success probability - proceed with standard config
        return ExecutionPlan(
            decision=ExecutionDecision.PROCEED,
            adjusted_config=config,
            reasoning=f"Success probability ({prediction.success_probability:.2f}) within acceptable range",
            expected_outcome="Expected to pass standard gauntlet",
            resource_allocation={"strategy": "standard"}
        )

    def execute_with_prediction(
        self,
        solution: str,
        problem: str,
        domain: str,
        prediction: Optional[PredictionResult] = None,
        config: Optional[Dict[str, Any]] = None,
        gauntlet_executor: Optional[Any] = None
    ) -> ExecutionResult:
        """
        Execute gauntlet with prediction guidance.

        Args:
            solution: Solution to evaluate
            problem: Problem statement
            domain: Problem domain
            prediction: Prediction result (will generate if None)
            config: Gauntlet configuration
            gauntlet_executor: Actual gauntlet executor (simulated if None)

        Returns:
            ExecutionResult with prediction accuracy and learning data
        """
        start_time = time.time()

        # Generate prediction if not provided
        if prediction is None:
            prediction = self.predict_success(solution, problem, domain)

        # Create execution plan
        plan = self.create_execution_plan(prediction, config)

        logger.info(f"Execution plan: {plan.decision.value} - {plan.reasoning}")

        # Execute based on plan decision
        if plan.decision == ExecutionDecision.SKIP_LOW_PROBABILITY:
            actual_outcome = {
                "passed": False,
                "score": 0.0,
                "skipped": True,
                "reason": plan.reasoning
            }
            execution_time = 0.0
            cost_savings = prediction.estimated_cost

        elif plan.decision == ExecutionDecision.SKIP_HIGH_COST:
            actual_outcome = {
                "passed": False,
                "score": 0.0,
                "skipped": True,
                "reason": plan.reasoning
            }
            execution_time = 0.0
            cost_savings = prediction.estimated_cost - self.cost_threshold

        else:
            # Proceed with execution (standard or adjusted)
            actual_config = plan.adjusted_config
            actual_outcome, execution_time = self._simulate_execution(
                solution, problem, domain, actual_config, gauntlet_executor
            )
            cost_savings = 0.0

        # Calculate prediction accuracy
        actual_passed = actual_outcome.get("passed", False)
        actual_score = actual_outcome.get("score", 0.0)
        predicted_passed = prediction.success_probability > 0.5

        # Binary accuracy for pass/fail
        pass_fail_accuracy = 1.0 if (predicted_passed == actual_passed) else 0.0

        # Score accuracy
        score_error = abs(actual_score - prediction.success_probability)
        score_accuracy = max(0.0, 1.0 - score_error)

        # Overall accuracy
        prediction_accuracy = (pass_fail_accuracy + score_accuracy) / 2

        # Collect learning data
        learning_data = {
            "solution_hash": hash(solution) % 10000,
            "domain": domain,
            "prediction": prediction.to_dict(),
            "plan": plan.to_dict(),
            "actual": actual_outcome,
            "accuracy": prediction_accuracy,
            "timestamp": datetime.now(UTC).isoformat()
        }

        self.prediction_history.append(learning_data)

        execution_time_total = time.time() - start_time

        return ExecutionResult(
            prediction=prediction,
            actual_outcome=actual_outcome,
            prediction_accuracy=prediction_accuracy,
            execution_time=execution_time_total,
            cost_savings=cost_savings,
            learning_data=learning_data
        )

    def _extract_features(
        self,
        solution: str,
        problem: str,
        domain: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract features for prediction"""
        features = {
            "solution_length": len(solution),
            "solution_lines": len(solution.splitlines()),
            "problem_length": len(problem),
            "domain": domain,
            "has_functions": "def " in solution,
            "has_classes": "class " in solution,
            "has_imports": "import " in solution,
            "complexity_score": self._calculate_complexity(solution),
            "domain_risk": self._get_domain_risk(domain)
        }
        return features

    def _calculate_complexity(self, solution: str) -> float:
        """Calculate solution complexity score"""
        complexity = 0.5

        # Length-based complexity
        lines = len(solution.splitlines())
        if lines > 100:
            complexity += 0.2
        elif lines > 50:
            complexity += 0.1
        elif lines < 10:
            complexity -= 0.2

        # Structure-based complexity
        func_count = solution.count("def ")
        class_count = solution.count("class ")
        complexity += min(0.2, func_count * 0.03)
        complexity += min(0.1, class_count * 0.05)

        # Keyword-based complexity
        advanced_keywords = ["async", "await", "yield", "lambda", "decorator"]
        for keyword in advanced_keywords:
            if keyword in solution.lower():
                complexity += 0.05

        return max(0.0, min(1.0, complexity))

    def _get_domain_risk(self, domain: str) -> float:
        """Get inherent difficulty score for domain"""
        domain_risks = {
            "math": 0.7,
            "algorithm": 0.8,
            "ml": 0.8,
            "optimization": 0.75,
            "code": 0.5,
            "general": 0.4
        }
        return domain_risks.get(domain.lower(), 0.5)

    def _calculate_success_probability(self, features: Dict[str, Any]) -> float:
        """Calculate success probability from features"""
        base_prob = 0.7

        # Adjust for complexity
        complexity = features["complexity_score"]
        if complexity > 0.7:
            base_prob -= 0.15
        elif complexity < 0.3:
            base_prob += 0.1

        # Adjust for domain risk
        domain_risk = features["domain_risk"]
        if domain_risk > 0.7:
            base_prob -= 0.1

        # Adjust for solution quality indicators
        if features["has_functions"]:
            base_prob += 0.05
        if features["has_imports"]:
            base_prob += 0.05

        # Adjust for solution length
        if features["solution_lines"] < 5:
            base_prob -= 0.2  # Too short likely incomplete
        elif features["solution_lines"] > 200:
            base_prob -= 0.1  # Very long might have issues

        return max(0.0, min(1.0, base_prob))

    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate prediction confidence"""
        confidence = 0.7

        # Higher confidence for familiar domains
        if features["domain"] in ["code", "general"]:
            confidence += 0.1

        # Lower confidence for very complex solutions
        if features["complexity_score"] > 0.8:
            confidence -= 0.15

        # Higher confidence for moderate-length solutions
        if 10 <= features["solution_lines"] <= 100:
            confidence += 0.1

        return max(0.3, min(0.95, confidence))

    def _identify_risk_factors(self, features: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors"""
        risks = []

        if features["complexity_score"] > 0.7:
            risks.append("High complexity")

        if features["domain_risk"] > 0.7:
            risks.append(f"Challenging domain: {features['domain']}")

        if features["solution_lines"] < 5:
            risks.append("Solution appears incomplete")

        if features["solution_lines"] > 200:
            risks.append("Very long solution may have maintenance issues")

        if not features["has_functions"] and features["solution_lines"] > 20:
            risks.append("Lacks functional structure")

        return risks

    def _recommend_difficulty(self, success_prob: float, features: Dict[str, Any]) -> str:
        """Recommend difficulty level"""
        if success_prob > 0.8:
            return "hard"
        elif success_prob > 0.5:
            return "medium"
        else:
            return "easy"

    def _estimate_execution_time(self, features: Dict[str, Any], domain: str) -> float:
        """Estimate execution time in seconds"""
        base_time = 30.0

        # Adjust for complexity
        complexity_multiplier = 1.0 + features["complexity_score"]
        base_time *= complexity_multiplier

        # Adjust for domain
        domain_multipliers = {
            "math": 1.5,
            "algorithm": 1.3,
            "ml": 1.4,
            "code": 1.0,
            "general": 0.8
        }
        base_time *= domain_multipliers.get(domain.lower(), 1.0)

        return base_time

    def _estimate_cost(self, features: Dict[str, Any], execution_time: float) -> float:
        """Estimate computational cost (arbitrary units)"""
        # Cost based on time and complexity
        return execution_time * (1.0 + features["complexity_score"])

    def _simulate_execution(
        self,
        solution: str,
        problem: str,
        domain: str,
        config: Dict[str, Any],
        executor: Optional[Any]
    ) -> Tuple[Dict[str, Any], float]:
        """Simulate gauntlet execution (or use real executor if provided)"""
        start_time = time.time()

        # If real executor provided, use it
        if executor is not None:
            try:
                result = executor.run_full_gauntlet(
                    solution=solution,
                    problem=problem,
                    domain=domain,
                    **config
                )
                execution_time = time.time() - start_time
                return {
                    "passed": result.passed,
                    "score": result.final_score,
                    "rounds_completed": result.rounds_completed
                }, execution_time
            except Exception as e:
                logger.error(f"Executor failed: {e}")

        # Otherwise, simulate execution
        # Simulate execution time
        simulated_time = np.random.uniform(10, 60)
        time.sleep(0.01)  # Tiny sleep to simulate work

        # Simulate result based on config thresholds
        thresholds = [
            config.get("round1_threshold", 0.5),
            config.get("round2_threshold", 0.6),
            config.get("round3_threshold", 0.7)
        ]

        # Generate random scores
        scores = np.random.uniform(0.4, 0.9, len(thresholds))

        # Check if passes all rounds
        passed = all(s >= t for s, t in zip(scores, thresholds))
        final_score = np.mean(scores) if passed else 0.0

        return {
            "passed": passed,
            "score": final_score,
            "rounds_completed": len(thresholds) if passed else np.sum([s >= t for s, t in zip(scores, thresholds)])
        }, simulated_time

    def get_prediction_accuracy_stats(self) -> Dict[str, float]:
        """Get statistics about prediction accuracy"""
        if not self.prediction_history:
            return {"error": "No predictions made yet"}

        accuracies = [p["accuracy"] for p in self.prediction_history]

        return {
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "min_accuracy": np.min(accuracies),
            "max_accuracy": np.max(accuracies),
            "total_predictions": len(accuracies)
        }
