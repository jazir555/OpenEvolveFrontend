"""
Experimentation Interface - Base abstraction for automated scientific experimentation.

This module defines the abstract interface that experimentation agents (like Curie)
must implement to integrate with OpenEvolve's Knowledge Engine.

Key Concepts:
- Hypothesis formulation and experiment design
- Protocol execution and data collection
- Statistical analysis and validation
- Reflection and iterative refinement

Author: Curie Integration Specialist
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ExperimentDomain(Enum):
    """Supported experimental domains"""
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    MATERIALS_SCIENCE = "materials_science"
    ML_ENGINEERING = "ml_engineering"
    GENERAL = "general"


class ExperimentStatus(Enum):
    """Experiment execution status"""
    DESIGNED = "designed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ANALYZED = "analyzed"
    REFINED = "refined"


@dataclass
class Hypothesis:
    """Experimental hypothesis"""
    statement: str
    domain: ExperimentDomain
    independent_variables: List[str]
    dependent_variables: List[str]
    control_variables: List[str]
    assumptions: List[str]
    confidence: float = 0.5  # Initial confidence before testing


@dataclass
class ExperimentProtocol:
    """Detailed experimental protocol"""
    protocol_id: str
    hypothesis: Hypothesis
    steps: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    equipment: List[str]
    materials: List[str]
    duration_estimate: float  # in seconds
    reproducibility_checks: List[str]


@dataclass
class ExperimentResults:
    """Results from experiment execution"""
    protocol_id: str
    status: ExperimentStatus
    data: Dict[str, Any]
    metrics: Dict[str, float]
    observations: List[str]
    execution_time: float
    reproducibility_score: float
    validation_passed: bool


@dataclass
class StatisticalAnalysis:
    """Statistical analysis of experimental results"""
    significance_tests: Dict[str, Any]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, tuple]
    statistical_power: float
    recommendations: List[str]
    validation_passed: bool


@dataclass
class ReflectionReport:
    """Reflection and refinement recommendations"""
    hypothesis_validated: bool
    confidence_delta: float
    methodological_issues: List[str]
    suggested_improvements: List[str]
    next_experiments: List[str]
    should_continue: bool


@dataclass
class VerificationReport:
    """Verification report for OpenEvolve integration"""
    experiment_valid: bool
    statistical_significance: bool
    reproducibility_confirmed: bool
    methodology_sound: bool
    confidence_level: float
    gaps_identified: List[str]
    recommendations: List[str]
    raw_data: Dict[str, Any]


class ExperimentationInterface(ABC):
    """
    Abstract interface for automated experimentation systems.

    Implementations must support the full scientific workflow:
    1. Design experiments from hypotheses
    2. Execute protocols with validation
    3. Analyze results statistically
    4. Reflect and refine iteratively
    """

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the experimentation system.

        Args:
            config: Configuration dictionary including:
                - api_keys: LLM API credentials
                - workspace_dir: Working directory
                - docker_enabled: Whether to use Docker isolation
                - max_runtime: Maximum experiment duration
                - domains: List of enabled domains
        """
        pass

    @abstractmethod
    async def design_experiment(
        self,
        hypothesis: str,
        domain: ExperimentDomain,
        constraints: Optional[List[str]] = None,
        available_equipment: Optional[List[str]] = None
    ) -> ExperimentProtocol:
        """
        Design an experiment to test a hypothesis.

        Args:
            hypothesis: Hypothesis statement to test
            domain: Scientific domain
            constraints: Experimental constraints
            available_equipment: Available equipment

        Returns:
            Complete experimental protocol
        """
        pass

    @abstractmethod
    async def run_experiment(
        self,
        protocol: ExperimentProtocol,
        iterations: int = 1
    ) -> ExperimentResults:
        """
        Execute an experimental protocol.

        Args:
            protocol: Protocol to execute
            iterations: Number of times to repeat for reproducibility

        Returns:
            Experimental results with validation
        """
        pass

    @abstractmethod
    async def analyze_results(
        self,
        results: ExperimentResults,
        hypothesis: Hypothesis
    ) -> StatisticalAnalysis:
        """
        Perform statistical analysis on experimental results.

        Args:
            results: Experimental results
            hypothesis: Original hypothesis being tested

        Returns:
            Statistical analysis with validation
        """
        pass

    @abstractmethod
    async def reflect_and_refine(
        self,
        protocol: ExperimentProtocol,
        results: ExperimentResults,
        analysis: StatisticalAnalysis
    ) -> ReflectionReport:
        """
        Reflect on results and suggest refinements.

        Args:
            protocol: Protocol that was executed
            results: Results from execution
            analysis: Statistical analysis

        Returns:
            Reflection report with recommendations
        """
        pass

    @abstractmethod
    async def validate(self) -> Dict[str, Any]:
        """
        Validate the experimentation system is properly configured.

        Returns:
            Validation report with:
                - system_available: bool
                - domains_supported: List[str]
                - issues: List[str]
                - capabilities: Dict[str, Any]
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the experimentation system and cleanup resources.
        """
        pass

    async def execute_full_workflow(
        self,
        hypothesis: str,
        domain: ExperimentDomain,
        max_iterations: int = 3
    ) -> VerificationReport:
        """
        Execute complete workflow: design -> run -> analyze -> reflect.

        Args:
            hypothesis: Hypothesis to test
            domain: Scientific domain
            max_iterations: Maximum refinement iterations

        Returns:
            Verification report for OpenEvolve
        """
        iteration = 0
        current_protocol = None
        results = None
        analysis = None

        while iteration < max_iterations:
            # Design experiment
            protocol = await self.design_experiment(hypothesis, domain)

            if iteration == 0:
                current_protocol = protocol

            # Run experiment
            results = await self.run_experiment(protocol)

            if results.status == ExperimentStatus.FAILED:
                return VerificationReport(
                    experiment_valid=False,
                    statistical_significance=False,
                    reproducibility_confirmed=False,
                    methodology_sound=False,
                    confidence_level=0.0,
                    gaps_identified=["Experiment execution failed"],
                    recommendations=["Review protocol and constraints"],
                    raw_data=results.data
                )

            # Analyze results
            analysis = await self.analyze_results(results, protocol.hypothesis)

            # Reflect and refine
            reflection = await self.reflect_and_refine(protocol, results, analysis)

            if not reflection.should_continue:
                break

            iteration += 1

        # Generate verification report
        return VerificationReport(
            experiment_valid=analysis.validation_passed,
            statistical_significance=analysis.validation_passed,
            reproducibility_confirmed=results.reproducibility_score > 0.8,
            methodology_sound=len(reflection.methodological_issues) == 0,
            confidence_level=protocol.hypothesis.confidence,
            gaps_identified=reflection.methodological_issues,
            recommendations=reflection.suggested_improvements,
            raw_data=results.data
        )
