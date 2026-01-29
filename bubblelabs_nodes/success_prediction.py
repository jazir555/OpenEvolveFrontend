"""
Success Prediction for OpenEvolve Gauntlet System

Predicts the likelihood of success for problems before execution,
enabling better planning and resource allocation.

Key Features:
- Feature extraction from problems
- Success probability prediction
- Go/no-go recommendations
- Model training interface
- Feedback loop for continuous learning
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class PredictionOutcome(Enum):
    """Prediction outcomes"""
    GO = "go"  # Likely to succeed
    NO_GO = "no_go"  # Likely to fail
    UNCERTAIN = "uncertain"  # Not enough data


@dataclass
class ProblemFeatures:
    """Features extracted from a problem for prediction"""
    problem_id: str
    complexity_score: float  # 0-1
    domain_familiarity: float  # 0-1
    team_workload: float  # 0-1
    resource_availability: float  # 0-1
    historical_success_rate: float  # 0-1
    estimated_effort: float  # hours
    requirements_count: int
    dependencies_count: int
    team_size: int
    time_constraint: float  # hours available
    quality_requirements: str  # 'low', 'medium', 'high'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuccessPrediction:
    """Prediction of problem success likelihood"""
    problem_id: str
    success_probability: float  # 0-1
    confidence: float  # 0-1
    outcome: PredictionOutcome
    reasoning: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0  # hours
    risk_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)


class FeatureExtractor:
    """
    Extracts features from problems for prediction.
    """

    def extract_features(
        self,
        problem: Dict[str, Any],
        team_context: Dict[str, Any] = None,
        historical_data: Dict[str, Any] = None
    ) -> ProblemFeatures:
        """
        Extract features from a problem.

        Args:
            problem: Problem definition
            team_context: Team context and capabilities
            historical_data: Historical performance data

        Returns:
            ProblemFeatures
        """
        team_context = team_context or {}
        historical_data = historical_data or {}

        # Calculate complexity score
        complexity_score = self._calculate_complexity(problem)

        # Domain familiarity
        domain_familiarity = self._calculate_domain_familiarity(
            problem,
            team_context
        )

        # Team workload
        team_workload = team_context.get('workload', 0.5)

        # Resource availability
        resource_availability = team_context.get('resource_availability', 0.7)

        # Historical success rate
        historical_success_rate = self._get_historical_success_rate(
            problem,
            historical_data
        )

        # Estimated effort
        estimated_effort = self._estimate_effort(problem)

        # Requirements count
        requirements = problem.get('requirements', [])
        requirements_count = len(requirements) if isinstance(requirements, list) else 1

        # Dependencies count
        dependencies = problem.get('dependencies', [])
        dependencies_count = len(dependencies) if isinstance(dependencies, list) else 0

        # Team size
        team_size = team_context.get('team_size', 3)

        # Time constraint
        time_constraint = problem.get('deadline_hours', 120)  # Default 5 days

        # Quality requirements
        quality_requirements = problem.get('quality', 'medium')

        return ProblemFeatures(
            problem_id=problem.get('id', 'unknown'),
            complexity_score=complexity_score,
            domain_familiarity=domain_familiarity,
            team_workload=team_workload,
            resource_availability=resource_availability,
            historical_success_rate=historical_success_rate,
            estimated_effort=estimated_effort,
            requirements_count=requirements_count,
            dependencies_count=dependencies_count,
            team_size=team_size,
            time_constraint=time_constraint,
            quality_requirements=quality_requirements,
            metadata={
                'statement_length': len(problem.get('statement', '')),
                'has_subproblems': bool(problem.get('subproblems')),
                'subproblem_count': len(problem.get('subproblems', [])),
            }
        )

    def _calculate_complexity(self, problem: Dict[str, Any]) -> float:
        """Calculate complexity score (0-1)"""
        factors = []

        # Statement length (normalized to 0-1)
        statement = problem.get('statement', '')
        statement_factor = min(1.0, len(statement) / 500)
        factors.append(statement_factor)

        # Number of requirements
        requirements = problem.get('requirements', [])
        if isinstance(requirements, list):
            req_factor = min(1.0, len(requirements) / 10)
        else:
            req_factor = 0.3
        factors.append(req_factor)

        # Subproblems
        subproblems = problem.get('subproblems', [])
        if isinstance(subproblems, list):
            sub_factor = min(1.0, len(subproblems) / 5)
        else:
            sub_factor = 0.0
        factors.append(sub_factor)

        # Domain complexity
        statement_lower = statement.lower()
        if any(word in statement_lower for word in ['machine learning', 'ai', 'optimization']):
            factors.append(0.8)
        elif any(word in statement_lower for word in ['simple', 'basic', 'straightforward']):
            factors.append(0.2)
        else:
            factors.append(0.5)

        return statistics.mean(factors)

    def _calculate_domain_familiarity(
        self,
        problem: Dict[str, Any],
        team_context: Dict[str, Any]
    ) -> float:
        """Calculate domain familiarity score (0-1)"""
        # Check if team has experience in this domain
        statement = problem.get('statement', '').lower()
        team_domains = team_context.get('domains', [])
        team_expertise = team_context.get('expertise', {})

        # Check team's domain expertise
        for domain in team_domains:
            if domain.lower() in statement:
                # Team has experience in this domain
                expertise_level = team_expertise.get(domain, 0.5)
                return min(1.0, expertise_level + 0.3)

        # No direct domain match, return baseline
        return team_context.get('base_familiarity', 0.5)

    def _get_historical_success_rate(
        self,
        problem: Dict[str, Any],
        historical_data: Dict[str, Any]
    ) -> float:
        """Get historical success rate for similar problems"""
        # In a real system, this would query ML model or database
        # For now, return baseline
        return historical_data.get('baseline_success_rate', 0.65)

    def _estimate_effort(self, problem: Dict[str, Any]) -> float:
        """Estimate effort in hours"""
        # Base estimation
        base_effort = 8  # 1 day

        # Adjust for complexity
        complexity = self._calculate_complexity(problem)
        effort = base_effort * (1 + complexity * 2)

        # Adjust for requirements
        requirements = problem.get('requirements', [])
        if isinstance(requirements, list):
            effort *= (1 + len(requirements) * 0.1)

        return effort


class SuccessPredictor:
    """
    Predicts success likelihood for problems.

    Uses feature-based scoring with configurable weights.
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        success_threshold: float = 0.6,
        uncertainty_threshold: float = 0.4
    ):
        self.weights = weights or {
            'complexity': -0.3,  # Higher complexity = lower success
            'domain_familiarity': 0.4,  # Higher familiarity = higher success
            'team_workload': -0.2,  # Higher workload = lower success
            'resource_availability': 0.2,  # More resources = higher success
            'historical_success': 0.3,  # Past success predicts future
            'requirements_count': -0.1,  # More requirements = lower success
            'dependencies_count': -0.15,  # More deps = lower success
            'team_size': 0.1,  # Larger team = higher success
            'time_constraint': 0.1,  # More time = higher success
        }
        self.success_threshold = success_threshold
        self.uncertainty_threshold = uncertainty_threshold

    def predict(
        self,
        features: ProblemFeatures
    ) -> SuccessPrediction:
        """
        Predict success likelihood for a problem.

        Args:
            features: Extracted problem features

        Returns:
            SuccessPrediction
        """
        # Calculate weighted score
        score = 0.5  # Base score (50%)

        # Complexity
        score += self.weights.get('complexity', 0) * features.complexity_score

        # Domain familiarity
        score += self.weights.get('domain_familiarity', 0) * features.domain_familiarity

        # Team workload
        score += self.weights.get('team_workload', 0) * features.team_workload

        # Resource availability
        score += self.weights.get('resource_availability', 0) * features.resource_availability

        # Historical success
        score += self.weights.get('historical_success', 0) * features.historical_success_rate

        # Requirements count
        score += self.weights.get('requirements_count', 0) * (features.requirements_count / 10)

        # Dependencies count
        score += self.weights.get('dependencies_count', 0) * (features.dependencies_count / 5)

        # Team size
        score += self.weights.get('team_size', 0) * (features.team_size / 10)

        # Time constraint
        score += self.weights.get('time_constraint', 0) * min(1.0, features.time_constraint / 120)

        # Normalize to 0-1
        success_probability = max(0.0, min(1.0, score))

        # Determine outcome
        if success_probability >= self.success_threshold:
            outcome = PredictionOutcome.GO
        elif success_probability <= self.uncertainty_threshold:
            outcome = PredictionOutcome.NO_GO
        else:
            outcome = PredictionOutcome.UNCERTAIN

        # Calculate confidence
        confidence = self._calculate_confidence(features, success_probability)

        # Generate reasoning
        reasoning = self._generate_reasoning(features, success_probability)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(features)

        # Generate recommendations
        recommendations = self._generate_recommendations(features, success_probability)

        # Estimate duration
        estimated_duration = features.estimated_effort

        prediction = SuccessPrediction(
            problem_id=features.problem_id,
            success_probability=success_probability,
            confidence=confidence,
            outcome=outcome,
            reasoning=reasoning,
            estimated_duration=estimated_duration,
            risk_factors=risk_factors,
            recommendations=recommendations
        )

        logger.info(
            f"Prediction for {features.problem_id}: "
            f"P(success)={success_probability:.1%}, "
            f"outcome={outcome.value}, "
            f"confidence={confidence:.1%}"
        )

        return prediction

    def _calculate_confidence(
        self,
        features: ProblemFeatures,
        success_probability: float
    ) -> float:
        """Calculate confidence in the prediction"""
        confidence = 0.5  # Base confidence

        # Increase confidence if historical data available
        if features.historical_success_rate > 0:
            confidence += 0.2

        # Increase confidence if probability is extreme
        if success_probability > 0.8 or success_probability < 0.2:
            confidence += 0.2

        # Decrease confidence if unfamiliar domain
        if features.domain_familiarity < 0.3:
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def _generate_reasoning(
        self,
        features: ProblemFeatures,
        success_probability: float
    ) -> List[str]:
        """Generate human-readable reasoning"""
        reasons = []

        if success_probability > 0.7:
            reasons.append(f"High success probability ({success_probability:.1%})")
        elif success_probability < 0.4:
            reasons.append(f"Low success probability ({success_probability:.1%})")
        else:
            reasons.append(f"Moderate success probability ({success_probability:.1%})")

        if features.complexity_score > 0.7:
            reasons.append("High complexity increases risk")
        elif features.complexity_score < 0.3:
            reasons.append("Low complexity reduces risk")

        if features.domain_familiarity > 0.7:
            reasons.append("Team has strong domain expertise")
        elif features.domain_familiarity < 0.3:
            reasons.append("Limited domain familiarity")

        if features.team_workload > 0.7:
            reasons.append("High team workload may impact quality")

        if features.requirements_count > 7:
            reasons.append("Many requirements increase complexity")

        return reasons

    def _identify_risk_factors(self, features: ProblemFeatures) -> List[str]:
        """Identify specific risk factors"""
        risks = []

        if features.complexity_score > 0.8:
            risks.append("Very high complexity")

        if features.domain_familiarity < 0.3:
            risks.append("Unfamiliar domain")

        if features.team_workload > 0.8:
            risks.append("Team overcapacity")

        if features.resource_availability < 0.3:
            risks.append("Resource constraints")

        if features.estimated_effort > features.time_constraint:
            risks.append("Insufficient time allocated")

        if features.dependencies_count > 5:
            risks.append("Many external dependencies")

        if features.historical_success_rate < 0.3:
            risks.append("Poor historical performance")

        return risks

    def _generate_recommendations(
        self,
        features: ProblemFeatures,
        success_probability: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if success_probability < 0.5:
            # Low probability, recommend caution
            if features.complexity_score > 0.7:
                recommendations.append("Consider breaking down into smaller problems")

            if features.domain_familiarity < 0.5:
                recommendations.append("Allocate time for domain research")

            if features.estimated_effort > features.time_constraint:
                recommendations.append("Extend timeline or reduce scope")

            recommendations.append("Conduct proof-of-concept before full implementation")

        elif success_probability > 0.8:
            # High probability, recommend proceeding
            recommendations.append("Green light for execution")
            recommendations.append("Monitor progress closely")

        else:
            # Moderate probability, recommend caution
            recommendations.append("Proceed with careful planning")
            recommendations.append("Set up frequent checkpoints")

        return recommendations


class SuccessPredictionSystem:
    """
    Main system for success prediction.

    Integrates feature extraction and prediction.
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        success_threshold: float = 0.6
    ):
        self.extractor = FeatureExtractor()
        self.predictor = SuccessPredictor(
            weights=weights,
            success_threshold=success_threshold
        )

    async def predict_success(
        self,
        problem: Dict[str, Any],
        team_context: Dict[str, Any] = None,
        historical_data: Dict[str, Any] = None
    ) -> SuccessPrediction:
        """
        Predict success likelihood for a problem.

        Args:
            problem: Problem definition
            team_context: Team capabilities and context
            historical_data: Historical performance data

        Returns:
            SuccessPrediction
        """
        # Extract features
        features = self.extractor.extract_features(
            problem,
            team_context,
            historical_data
        )

        # Make prediction
        prediction = self.predictor.predict(features)

        return prediction

    def get_recommendation(self, prediction: SuccessPrediction) -> str:
        """
        Get go/no-go recommendation with explanation.

        Args:
            prediction: Success prediction

        Returns:
            Recommendation string
        """
        lines = [
            f"Problem: {prediction.problem_id}",
            f"Recommendation: {prediction.outcome.value.upper()}",
            f"Success Probability: {prediction.success_probability:.1%}",
            f"Confidence: {prediction.confidence:.1%}",
            f"Estimated Duration: {prediction.estimated_duration:.1f} hours",
            "",
            "Reasoning:"
        ]

        for reason in prediction.reasoning:
            lines.append(f"  - {reason}")

        if prediction.risk_factors:
            lines.append("")
            lines.append("Risk Factors:")
            for risk in prediction.risk_factors:
                lines.append(f"  ⚠️ {risk}")

        if prediction.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in prediction.recommendations:
                lines.append(f"  → {rec}")

        return "\n".join(lines)


# Convenience function
async def predict_problem_success(
    problem: Dict[str, Any],
    team_context: Dict[str, Any] = None
) -> SuccessPrediction:
    """Convenience function to predict problem success"""
    system = SuccessPredictionSystem()
    return await system.predict_success(problem, team_context)


# Example usage
async def demo_success_prediction():
    """Demonstration of success prediction"""

    system = SuccessPredictionSystem()

    # Example problem
    problem = {
        'id': 'problem_1',
        'statement': 'Build a machine learning model for image classification',
        'requirements': [
            'high accuracy',
            'fast inference',
            'handle edge cases',
            'documentation'
        ],
        'deadline_hours': 80,  # ~2 weeks
    }

    # Team context
    team_context = {
        'domains': ['web_development', 'data_processing'],
        'expertise': {
            'web_development': 0.8,
            'data_processing': 0.6,
            'machine_learning': 0.2,
        },
        'workload': 0.4,
        'resource_availability': 0.7,
        'team_size': 3,
        'base_familiarity': 0.5,
    }

    print("\n" + "=" * 60)
    print("Success Prediction Demo")
    print("=" * 60)

    # Make prediction
    prediction = await system.predict_success(problem, team_context)

    # Get recommendation
    recommendation = system.get_recommendation(prediction)
    print("\n" + recommendation)

    print("\n" + "=" * 60)


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_success_prediction())
