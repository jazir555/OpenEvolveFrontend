"""
Resource Estimation Engine for Decomposition Engine

Provides automatic resource estimation for sub-problems based on:
- Base complexity scores
- Domain-specific multipliers
- Risk-based adjustments
- Dependency coordination overhead
- Quality metrics requirements

Author: OpenEvolve
Version: 1.0.0
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from sovereign_data_models import (
    SubProblem,
    ComplexityScore,
    ResourceEstimate,
    ComplexityBreakdown
)

logger = logging.getLogger(__name__)


class DomainMultipliers:
    """
    Domain-specific resource multipliers.

    Different domains have different resource requirements due to:
    - Computational intensity
    - Uncertainty levels
    - Data requirements
    - Infrastructure needs
    """

    MACHINE_LEARNING = 1.5  # More compute resources needed
    SOFTWARE_DEVELOPMENT = 1.2  # Moderate overhead
    RESEARCH = 1.8  # High uncertainty requires more buffer
    DATA_ENGINEERING = 1.3  # Data intensive operations
    DEVOPS = 1.1  # Infrastructure focused
    DEFAULT = 1.0  # Baseline

    @classmethod
    def get_multiplier(cls, domain: str) -> float:
        """
        Get resource multiplier for domain.

        Args:
            domain: Domain string (case-insensitive)

        Returns:
            Resource multiplier (float)
        """
        domain_upper = domain.upper().replace(" ", "_").replace("-", "_")

        multiplier_map = {
            "MACHINE_LEARNING": cls.MACHINE_LEARNING,
            "ML": cls.MACHINE_LEARNING,
            "SOFTWARE_DEVELOPMENT": cls.SOFTWARE_DEVELOPMENT,
            "SOFTWARE": cls.SOFTWARE_DEVELOPMENT,
            "DEV": cls.SOFTWARE_DEVELOPMENT,
            "RESEARCH": cls.RESEARCH,
            "DATA_ENGINEERING": cls.DATA_ENGINEERING,
            "DATA": cls.DATA_ENGINEERING,
            "DEVOPS": cls.DEVOPS,
            "OPS": cls.DEVOPS,
        }

        return multiplier_map.get(domain_upper, cls.DEFAULT)


@dataclass
class BaseResourceRequirements:
    """
    Base resource requirements for different complexity levels.

    These are minimum requirements that get scaled by complexity and other factors.
    """

    # Low complexity (0-3)
    LOW_TIME_HOURS = 2.0
    LOW_API_TOKENS = 1000
    LOW_COMPUTE_UNITS = 1.0
    LOW_REVIEW_MINUTES = 15

    # Medium complexity (3-7)
    MEDIUM_TIME_HOURS = 8.0
    MEDIUM_API_TOKENS = 5000
    MEDIUM_COMPUTE_UNITS = 5.0
    MEDIUM_REVIEW_MINUTES = 60

    # High complexity (7-10)
    HIGH_TIME_HOURS = 24.0
    HIGH_API_TOKENS = 20000
    HIGH_COMPUTE_UNITS = 20.0
    HIGH_REVIEW_MINUTES = 180

    @classmethod
    def get_base_requirements(cls, complexity_score: float) -> Dict[str, Any]:
        """
        Get base requirements based on complexity score (0-10).

        Uses linear interpolation between LOW, MEDIUM, and HIGH benchmarks.

        Args:
            complexity_score: Overall complexity score (0-10)

        Returns:
            Dict with base_time_hours, base_api_tokens, base_compute_units, base_review_minutes
        """
        if complexity_score <= 3.0:
            # Low complexity range
            ratio = complexity_score / 3.0
            return {
                "base_time_hours": cls.LOW_TIME_HOURS * (1 + ratio),
                "base_api_tokens": int(cls.LOW_API_TOKENS * (1 + ratio)),
                "base_compute_units": cls.LOW_COMPUTE_UNITS * (1 + ratio),
                "base_review_minutes": int(cls.LOW_REVIEW_MINUTES * (1 + ratio))
            }
        elif complexity_score <= 7.0:
            # Medium complexity range
            ratio = (complexity_score - 3.0) / 4.0
            return {
                "base_time_hours": cls.MEDIUM_TIME_HOURS * (1 + ratio * 0.5),
                "base_api_tokens": int(cls.MEDIUM_API_TOKENS * (1 + ratio * 0.5)),
                "base_compute_units": cls.MEDIUM_COMPUTE_UNITS * (1 + ratio * 0.5),
                "base_review_minutes": int(cls.MEDIUM_REVIEW_MINUTES * (1 + ratio * 0.5))
            }
        else:
            # High complexity range
            ratio = (complexity_score - 7.0) / 3.0
            return {
                "base_time_hours": cls.HIGH_TIME_HOURS * (1 + ratio * 0.5),
                "base_api_tokens": int(cls.HIGH_API_TOKENS * (1 + ratio * 0.5)),
                "base_compute_units": cls.HIGH_COMPUTE_UNITS * (1 + ratio * 0.5),
                "base_review_minutes": int(cls.HIGH_REVIEW_MINUTES * (1 + ratio * 0.5))
            }


class ResourceEstimationEngine:
    """
    Automatic resource estimation engine for sub-problems.

    Estimates resources based on:
    1. Base complexity (non-linear scaling)
    2. Domain-specific multipliers
    3. Risk-based adjustments
    4. Dependency coordination overhead
    5. Quality metrics requirements

    Usage:
        engine = ResourceEstimationEngine()
        estimate = engine.estimate_resources(
            sub_problem=sub_problem,
            domain="machine_learning",
            base_complexity=7.5
        )
    """

    def __init__(self):
        """Initialize resource estimation engine."""
        self.logger = logging.getLogger(__name__)
        self.domain_multipliers = DomainMultipliers()
        self.base_requirements = BaseResourceRequirements()

    def estimate_resources(
        self,
        sub_problem: SubProblem,
        domain: Optional[str] = None,
        base_complexity: Optional[float] = None
    ) -> ResourceEstimate:
        """
        Estimate resources required for a sub-problem.

        Args:
            sub_problem: The SubProblem to estimate resources for
            domain: Optional domain string for domain-specific multipliers
            base_complexity: Optional complexity score (0-10). If not provided,
                           uses sub_problem.complexity_score.overall_complexity

        Returns:
            ResourceEstimate with time_hours, api_tokens, computational_units, human_review_minutes
        """
        try:
            # Step 1: Determine base complexity
            if base_complexity is None:
                base_complexity = sub_problem.complexity_score.overall_complexity

            # Normalize complexity to 0-1 range for calculations
            complexity_normalized = base_complexity / 10.0

            self.logger.info(
                f"Estimating resources for sub-problem {sub_problem.id} "
                f"(complexity: {base_complexity:.1f}/10, normalized: {complexity_normalized:.2f})"
            )

            # Step 2: Get base requirements from complexity
            base_reqs = self.base_requirements.get_base_requirements(base_complexity)

            # Step 3: Apply non-linear complexity scaling
            # Formula: base * (1 + complexity * multiplier)
            time_hours = base_reqs["base_time_hours"] * (1 + complexity_normalized * 2)
            api_tokens = int(base_reqs["base_api_tokens"] * (1 + complexity_normalized * 1.5))
            compute_units = base_reqs["base_compute_units"] * (1 + complexity_normalized * 2)
            review_minutes = int(base_reqs["base_review_minutes"] * (1 + complexity_normalized))

            self.logger.debug(
                f"Base estimates - Time: {time_hours:.1f}h, Tokens: {api_tokens}, "
                f"Compute: {compute_units:.1f}, Review: {review_minutes}m"
            )

            # Step 4: Apply domain-specific multipliers
            if domain:
                domain_multiplier = self.domain_multipliers.get_multiplier(domain)
                time_hours *= domain_multiplier
                compute_units *= domain_multiplier
                self.logger.debug(f"Applied domain multiplier {domain_multiplier}x for '{domain}'")

            # Step 5: Apply risk-based adjustments
            risk_buffer = self._calculate_risk_buffer(sub_problem)
            time_hours *= (1 + risk_buffer)
            api_tokens = int(api_tokens * (1 + risk_buffer))
            compute_units *= (1 + risk_buffer)
            review_minutes = int(review_minutes * (1 + risk_buffer * 0.5))  # Review time scales less
            self.logger.debug(f"Applied risk buffer: {risk_buffer:.1%}")

            # Step 6: Apply dependency coordination overhead
            dependency_buffer = self._calculate_dependency_buffer(sub_problem)
            time_hours *= (1 + dependency_buffer)
            self.logger.debug(f"Applied dependency buffer: {dependency_buffer:.1%}")

            # Step 7: Apply quality metrics adjustments
            quality_buffer = self._calculate_quality_buffer(sub_problem)
            time_hours *= (1 + quality_buffer)
            review_minutes = int(review_minutes * (1 + quality_buffer))
            self.logger.debug(f"Applied quality buffer: {quality_buffer:.1%}")

            # Step 8: Create and return ResourceEstimate
            estimate = ResourceEstimate(
                time_hours=round(time_hours, 2),
                api_tokens=api_tokens,
                computational_units=round(compute_units, 2),
                human_review_minutes=review_minutes,
                metadata={
                    "complexity_score": base_complexity,
                    "complexity_normalized": round(complexity_normalized, 3),
                    "domain": domain,
                    "domain_multiplier": self.domain_multipliers.get_multiplier(domain) if domain else 1.0,
                    "risk_buffer": round(risk_buffer, 3),
                    "num_risks": len(sub_problem.associated_risks),
                    "dependency_buffer": round(dependency_buffer, 3),
                    "num_dependencies": len(sub_problem.success_dependencies),
                    "quality_buffer": round(quality_buffer, 3),
                    "estimation_method": "automatic"
                }
            )

            self.logger.info(
                f"Final estimates - Time: {estimate.time_hours:.1f}h, "
                f"Tokens: {estimate.api_tokens:,}, "
                f"Compute: {estimate.computational_units:.1f}, "
                f"Review: {estimate.human_review_minutes}m"
            )

            return estimate

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            self.logger.error(f"Error estimating resources for {sub_problem.id}: {e}", exc_info=True)
            # Return conservative default estimate
            return ResourceEstimate(
                time_hours=8.0,
                api_tokens=5000,
                computational_units=5.0,
                human_review_minutes=60,
                metadata={"error": str(e), "estimation_method": "fallback"}
            )

    def _calculate_risk_buffer(self, sub_problem: SubProblem) -> float:
        """
        Calculate risk-based buffer percentage.

        Risk levels add buffer:
        - High risk: 15% per risk
        - Medium risk: 10% per risk
        - Low risk: 5% per risk

        Args:
            sub_problem: SubProblem with associated_risks

        Returns:
            Buffer percentage (0.0 to 1.0)
        """
        if not sub_problem.associated_risks:
            return 0.0

        # Parse risk severity from risk descriptions
        # Expected format: "HIGH: risk description", "MEDIUM: ...", "LOW: ..."
        high_risks = 0
        medium_risks = 0
        low_risks = 0

        for risk in sub_problem.associated_risks:
            risk_upper = risk.upper()
            if "HIGH:" in risk_upper or risk_upper.startswith("HIGH"):
                high_risks += 1
            elif "MEDIUM:" in risk_upper or risk_upper.startswith("MEDIUM"):
                medium_risks += 1
            elif "LOW:" in risk_upper or risk_upper.startswith("LOW"):
                low_risks += 1
            else:
                # Default to medium if severity not specified
                medium_risks += 1

        # Calculate buffer
        buffer = (high_risks * 0.15) + (medium_risks * 0.10) + (low_risks * 0.05)

        # Cap maximum risk buffer at 50%
        return min(buffer, 0.50)

    def _calculate_dependency_buffer(self, sub_problem: SubProblem) -> float:
        """
        Calculate dependency coordination overhead buffer.

        Each dependency adds 5% buffer for coordination overhead.
        Maximum buffer from dependencies: 25%.

        Args:
            sub_problem: SubProblem with success_dependencies

        Returns:
            Buffer percentage (0.0 to 0.25)
        """
        if not sub_problem.success_dependencies:
            return 0.0

        num_dependencies = len(sub_problem.success_dependencies)
        buffer = num_dependencies * 0.05  # 5% per dependency

        # Cap at 25%
        return min(buffer, 0.25)

    def _calculate_quality_buffer(self, sub_problem: SubProblem) -> float:
        """
        Calculate quality metrics requirements buffer.

        Higher quality targets add buffer:
        - High accuracy (>0.95): +20%
        - High security requirements: +15%
        - High compliance requirements: +25%

        Args:
            sub_problem: SubProblem with quality_metrics

        Returns:
            Buffer percentage (0.0 to 1.0)
        """
        if not sub_problem.quality_metrics:
            return 0.0

        buffer = 0.0
        quality = sub_problem.quality_metrics

        # Check accuracy target
        if quality.accuracy_target > 0.95:
            buffer += 0.20
        elif quality.accuracy_target > 0.90:
            buffer += 0.10

        # Check security requirements
        if quality.security_requirements:
            num_security = len(quality.security_requirements)
            if num_security >= 3:
                buffer += 0.15
            elif num_security >= 1:
                buffer += 0.05

        # Check compliance requirements
        if quality.compliance_requirements:
            num_compliance = len(quality.compliance_requirements)
            if num_compliance >= 2:
                buffer += 0.25
            elif num_compliance >= 1:
                buffer += 0.10

        # Cap maximum quality buffer at 50%
        return min(buffer, 0.50)


def estimate_resources_simple(
    complexity_score: float,
    domain: str = "DEFAULT",
    num_risks: int = 0,
    risk_level: str = "medium",
    num_dependencies: int = 0,
    high_accuracy: bool = False,
    security_required: bool = False,
    compliance_required: bool = False
) -> ResourceEstimate:
    """
    Simplified resource estimation function for quick calculations.

    Args:
        complexity_score: Overall complexity (0-10)
        domain: Domain for multiplier
        num_risks: Number of associated risks
        risk_level: Default risk level ("low", "medium", "high")
        num_dependencies: Number of success dependencies
        high_accuracy: Whether accuracy target > 0.95
        security_required: Whether security requirements exist
        compliance_required: Whether compliance requirements exist

    Returns:
        ResourceEstimate
    """
    engine = ResourceEstimationEngine()

    # Create a minimal SubProblem for estimation
    from sovereign_data_models import SubProblemType, SubProblemStatus, QualityMetrics

    # Build risk list
    risks = [f"{risk_level.upper()}: Risk {i+1}" for i in range(num_risks)]

    # Build dependencies list
    dependencies = [f"dep_{i+1}" for i in range(num_dependencies)]

    # Build quality metrics
    quality = None
    if high_accuracy or security_required or compliance_required:
        quality = QualityMetrics(
            accuracy_target=0.98 if high_accuracy else 0.90,
            security_requirements=["security"] if security_required else [],
            compliance_requirements=["compliance"] if compliance_required else []
        )

    # Create minimal SubProblem
    sub_problem = SubProblem(
        id="temp",
        parent_id="temp_parent",
        title="Temporary",
        description="For estimation",
        type=SubProblemType.IMPLEMENTATION,
        complexity_score=ComplexityScore(
            explanation="",
            cognitive_complexity=complexity_score,
            computational_complexity=complexity_score,
            domain_complexity=complexity_score,
            integration_complexity=complexity_score,
            overall_complexity=complexity_score
        ),
        associated_risks=risks,
        success_dependencies=dependencies,
        quality_metrics=quality
    )

    return engine.estimate_resources(sub_problem, domain=domain)
