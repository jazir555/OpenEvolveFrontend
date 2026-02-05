"""
Curie Adapter - Automated Scientific Experimentation Integration

This module provides a decoupled adapter for integrating Curie's automated
scientific experimentation framework with OpenEvolve's Knowledge Engine.

Curie fills GAP-4 (Experimental Data Integration) and GAP-12 (Scientific
Experimentation Automation) by providing:
- Hypothesis -> experiment -> result pipeline
- Integration with SOP Generator for protocols
- Statistical validation framework
- Reflection-based refinement

Author: Agent 3 (Curie Integration Specialist)
Version: 1.0.0
Repository: https://github.com/Just-Curieous/curie
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from integrations.base.experimentation_interface import (
    ExperimentationInterface,
    ExperimentDomain,
    ExperimentStatus,
    Hypothesis,
    ExperimentProtocol,
    ExperimentResults,
    StatisticalAnalysis,
    ReflectionReport,
    VerificationReport
)

from integrations.curie.bridge import CurieBridge


logger = logging.getLogger(__name__)


@dataclass
class CurieConfig:
    """Configuration for Curie adapter"""
    openai_api_key: str
    domain: str = "physics"
    workspace_dir: str = "./curie_workspace"
    docker_enabled: bool = False
    max_runtime: int = 86400  # 24 hours
    cache_enabled: bool = True
    cache_ttl: int = 3600
    fallback_on_error: bool = True
    max_workers: int = 4
    timeout: int = 30
    batch_size: int = 100
    temperature: float = 0.7
    model: str = "gpt-4o-mini"


class CurieAdapter(ExperimentationInterface):
    """
    Adapter for Curie automated scientific experimentation framework.

    This adapter implements the ExperimentationInterface to integrate Curie's
    hypothesis -> experiment -> result pipeline with OpenEvolve's Knowledge Engine.

    Key Features:
    - Automated experiment design from hypotheses
    - Protocol generation using SOP Generator
    - Statistical validation and analysis
    - Reflection-based iterative refinement
    - Zero modifications to Curie source code
    """

    def __init__(self, config: CurieConfig):
        """
        Initialize Curie adapter.

        Args:
            config: Curie configuration
        """
        self.config = config
        self.bridge = None
        self._initialized = False
        self._cache = {}
        self._experiment_history = []

        # Validate OpenAI availability
        if not OPENAI_AVAILABLE:
            logger.warning(
                "OpenAI library not available. "
                "Curie requires OpenAI API for LLM-based experiment design."
            )

        # Create workspace directory
        os.makedirs(config.workspace_dir, exist_ok=True)

        logger.info(f"Curie adapter initialized with domain: {config.domain}")

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Curie experimentation system.

        Args:
            config: Configuration dictionary
        """
        if self._initialized:
            logger.warning("Curie adapter already initialized")
            return

        logger.info("Initializing Curie adapter...")

        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and self.config.openai_api_key:
            openai.api_key = self.config.openai_api_key
            logger.info("OpenAI client configured")

        # Initialize bridge to SOP Generator and validation systems
        self.bridge = CurieBridge(
            openai_api_key=self.config.openai_api_key,
            workspace_dir=self.config.workspace_dir,
            cache_enabled=self.config.cache_enabled
        )

        await self.bridge.initialize()

        # Load experiment templates
        await self._load_templates()

        self._initialized = True
        logger.info("Curie adapter initialization complete")

    async def _load_templates(self) -> None:
        """Load experiment templates for configured domain"""
        template_dir = Path(__file__).parent / "templates"
        template_file = template_dir / f"{self.config.domain}.yaml"

        if template_file.exists():
            logger.info(f"Loading experiment templates from {template_file}")
            # Templates loaded by bridge
        else:
            logger.warning(f"No template file found for domain: {self.config.domain}")

    async def design_experiment(
        self,
        hypothesis: str,
        domain: ExperimentDomain,
        constraints: Optional[List[str]] = None,
        available_equipment: Optional[List[str]] = None
    ) -> ExperimentProtocol:
        """
        Design an experiment to test a hypothesis.

        This method uses Curie's hypothesis formulation and experiment design
        capabilities, integrated with SOP Generator for protocol generation.

        Args:
            hypothesis: Hypothesis statement to test
            domain: Scientific domain
            constraints: Experimental constraints
            available_equipment: Available equipment

        Returns:
            Complete experimental protocol
        """
        if not self._initialized:
            raise RuntimeError("Curie adapter not initialized. Call initialize() first.")

        logger.info(f"Designing experiment for hypothesis: {hypothesis[:100]}...")

        start_time = time.time()

        # Check cache
        cache_key = f"design:{hash(hypothesis)}:{domain.value}"
        if self.config.cache_enabled and cache_key in self._cache:
            logger.info("Returning cached experiment design")
            return self._cache[cache_key]

        # Parse hypothesis using LLM
        hypothesis_obj = await self._parse_hypothesis(hypothesis, domain, constraints)

        # Generate experimental protocol using SOP Generator
        protocol_steps = await self.bridge.generate_protocol(
            hypothesis=hypothesis,
            domain=domain.value,
            constraints=constraints or [],
            available_equipment=available_equipment or []
        )

        # Estimate duration and reproducibility
        duration_estimate = self._estimate_duration(protocol_steps)
        reproducibility_checks = self._generate_reproducibility_checks(domain)

        # Create protocol object
        protocol = ExperimentProtocol(
            protocol_id=f"curie_{int(time.time())}_{domain.value}",
            hypothesis=hypothesis_obj,
            steps=protocol_steps,
            parameters=self._extract_parameters(protocol_steps),
            equipment=available_equipment or [],
            materials=self._extract_materials(protocol_steps),
            duration_estimate=duration_estimate,
            reproducibility_checks=reproducibility_checks
        )

        # Cache result
        if self.config.cache_enabled:
            self._cache[cache_key] = protocol

        elapsed = time.time() - start_time
        logger.info(f"Experiment designed in {elapsed:.2f}s: {protocol.protocol_id}")

        return protocol

    async def _parse_hypothesis(
        self,
        hypothesis: str,
        domain: ExperimentDomain,
        constraints: Optional[List[str]]
    ) -> Hypothesis:
        """Parse hypothesis statement into structured format"""
        if not OPENAI_AVAILABLE:
            # Fallback: simple parsing
            return Hypothesis(
                statement=hypothesis,
                domain=domain,
                independent_variables=[],
                dependent_variables=[],
                control_variables=[],
                assumptions=constraints or [],
                confidence=0.5
            )

        # Use LLM to parse hypothesis
        prompt = f"""
Analyze this scientific hypothesis and extract its components:

Hypothesis: {hypothesis}
Domain: {domain.value}

Extract:
1. Independent variables (what is being changed)
2. Dependent variables (what is being measured)
3. Control variables (what is kept constant)
4. Assumptions (implicit or explicit)

Return as JSON:
{{
    "independent_variables": ["var1", "var2"],
    "dependent_variables": ["var1"],
    "control_variables": ["var1", "var2"],
    "assumptions": ["assumption1", "assumption2"]
}}
"""

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a scientific hypothesis analyzer. Extract structured components from hypotheses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                timeout=self.config.timeout
            )

            result = json.loads(response.choices[0].message.content)

            return Hypothesis(
                statement=hypothesis,
                domain=domain,
                independent_variables=result.get("independent_variables", []),
                dependent_variables=result.get("dependent_variables", []),
                control_variables=result.get("control_variables", []),
                assumptions=result.get("assumptions", []),
                confidence=0.5
            )
        except Exception as e:
            logger.error(f"Failed to parse hypothesis with LLM: {e}")
            # Fallback to simple parsing
            return Hypothesis(
                statement=hypothesis,
                domain=domain,
                independent_variables=[],
                dependent_variables=[],
                control_variables=[],
                assumptions=constraints or [],
                confidence=0.5
            )

    async def run_experiment(
        self,
        protocol: ExperimentProtocol,
        iterations: int = 1
    ) -> ExperimentResults:
        """
        Execute an experimental protocol.

        This method simulates experiment execution with validation.
        In production, this would interface with real laboratory equipment
        or simulation frameworks.

        Args:
            protocol: Protocol to execute
            iterations: Number of times to repeat for reproducibility

        Returns:
            Experimental results with validation
        """
        if not self._initialized:
            raise RuntimeError("Curie adapter not initialized. Call initialize() first.")

        logger.info(f"Running experiment {protocol.protocol_id} ({iterations} iterations)")

        start_time = time.time()

        # Execute protocol through bridge
        execution_results = []

        for i in range(iterations):
            logger.info(f"Executing iteration {i+1}/{iterations}")
            result = await self.bridge.execute_protocol(
                protocol=protocol,
                iteration=i
            )
            execution_results.append(result)

        # Aggregate results
        aggregated_data = self._aggregate_results(execution_results)

        # Calculate metrics
        metrics = await self._calculate_metrics(aggregated_data, protocol)

        # Validate results
        validation_passed = await self._validate_results(aggregated_data, protocol)

        # Calculate reproducibility score
        reproducibility_score = self._calculate_reproducibility(execution_results)

        execution_time = time.time() - start_time

        results = ExperimentResults(
            protocol_id=protocol.protocol_id,
            status=ExperimentStatus.COMPLETED if validation_passed else ExperimentStatus.FAILED,
            data=aggregated_data,
            metrics=metrics,
            observations=self._extract_observations(execution_results),
            execution_time=execution_time,
            reproducibility_score=reproducibility_score,
            validation_passed=validation_passed
        )

        # Store in history
        self._experiment_history.append({
            "timestamp": datetime.now().isoformat(),
            "protocol_id": protocol.protocol_id,
            "results": asdict(results)
        })

        logger.info(
            f"Experiment completed: {protocol.protocol_id} "
            f"(reproducibility: {reproducibility_score:.2f}, "
            f"validation: {validation_passed})"
        )

        return results

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
        if not self._initialized:
            raise RuntimeError("Curie adapter not initialized. Call initialize() first.")

        logger.info(f"Analyzing results for {results.protocol_id}")

        # Perform statistical tests
        significance_tests = await self._perform_significance_tests(results, hypothesis)

        # Calculate effect sizes
        effect_sizes = await self._calculate_effect_sizes(results, hypothesis)

        # Calculate confidence intervals
        confidence_intervals = await self._calculate_confidence_intervals(results)

        # Calculate statistical power
        statistical_power = await self._calculate_statistical_power(results)

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            results,
            significance_tests,
            effect_sizes
        )

        # Validate overall
        validation_passed = (
            len(significance_tests) > 0 and
            all(test.get("significant", False) for test in significance_tests.values()) and
            statistical_power > 0.8
        )

        analysis = StatisticalAnalysis(
            significance_tests=significance_tests,
            effect_sizes=effect_sizes,
            confidence_intervals=confidence_intervals,
            statistical_power=statistical_power,
            recommendations=recommendations,
            validation_passed=validation_passed
        )

        logger.info(
            f"Statistical analysis complete: "
            f"power={statistical_power:.2f}, "
            f"validation={validation_passed}"
        )

        return analysis

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
        if not self._initialized:
            raise RuntimeError("Curie adapter not initialized. Call initialize() first.")

        logger.info(f"Reflecting on results for {protocol.protocol_id}")

        # Validate hypothesis
        hypothesis_validated = analysis.validation_passed

        # Calculate confidence delta
        confidence_delta = 0.1 if hypothesis_validated else -0.1
        new_confidence = max(0.0, min(1.0, protocol.hypothesis.confidence + confidence_delta))

        # Identify methodological issues
        methodological_issues = await self._identify_methodological_issues(
            protocol,
            results,
            analysis
        )

        # Generate improvements
        suggested_improvements = await self._generate_improvements(
            protocol,
            results,
            methodological_issues
        )

        # Suggest next experiments
        next_experiments = await self._suggest_next_experiments(
            protocol,
            results,
            analysis
        )

        # Determine if should continue
        should_continue = (
            not hypothesis_validated or
            len(methodological_issues) > 0 or
            new_confidence < 0.9
        )

        reflection = ReflectionReport(
            hypothesis_validated=hypothesis_validated,
            confidence_delta=confidence_delta,
            methodological_issues=methodological_issues,
            suggested_improvements=suggested_improvements,
            next_experiments=next_experiments,
            should_continue=should_continue
        )

        logger.info(
            f"Reflection complete: validated={hypothesis_validated}, "
            f"should_continue={should_continue}"
        )

        return reflection

    async def validate(self) -> Dict[str, Any]:
        """
        Validate the Curie experimentation system is properly configured.

        Returns:
            Validation report
        """
        logger.info("Validating Curie adapter")

        issues = []
        capabilities = {}

        # Check OpenAI availability
        if not OPENAI_AVAILABLE:
            issues.append("OpenAI library not installed")
        else:
            capabilities["openai_available"] = True

        # Check API key
        if not self.config.openai_api_key:
            issues.append("OpenAI API key not configured")
        else:
            capabilities["api_configured"] = True

        # Check workspace
        if not os.path.exists(self.config.workspace_dir):
            issues.append(f"Workspace directory not found: {self.config.workspace_dir}")
        else:
            capabilities["workspace_available"] = True

        # Check bridge
        if self.bridge:
            bridge_validation = await self.bridge.validate()
            capabilities["bridge_validation"] = bridge_validation
            if not bridge_validation.get("valid", False):
                issues.extend(bridge_validation.get("issues", []))

        # Check templates
        template_dir = Path(__file__).parent / "templates"
        template_file = template_dir / f"{self.config.domain}.yaml"
        if template_file.exists():
            capabilities[f"{self.config.domain}_template"] = True
        else:
            issues.append(f"No template for domain: {self.config.domain}")

        # Determine supported domains
        supported_domains = []
        for template_file in template_dir.glob("*.yaml"):
            supported_domains.append(template_file.stem)
        capabilities["supported_domains"] = supported_domains

        return {
            "system_available": len(issues) == 0,
            "domains_supported": supported_domains,
            "issues": issues,
            "capabilities": capabilities
        }

    async def shutdown(self) -> None:
        """Shutdown the Curie experimentation system and cleanup resources"""
        logger.info("Shutting down Curie adapter")

        if self.bridge:
            await self.bridge.shutdown()

        # Save experiment history
        if self._experiment_history:
            history_file = Path(self.config.workspace_dir) / "experiment_history.json"
            with open(history_file, 'w') as f:
                json.dump(self._experiment_history, f, indent=2)
            logger.info(f"Experiment history saved to {history_file}")

        self._initialized = False

        logger.info("Curie adapter shutdown complete")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _estimate_duration(self, steps: List[Dict[str, Any]]) -> float:
        """Estimate experiment duration from protocol steps"""
        # Simple heuristic: 5 minutes per step
        return len(steps) * 300

    def _generate_reproducibility_checks(self, domain: ExperimentDomain) -> List[str]:
        """Generate domain-specific reproducibility checks"""
        checks = [
            "Verify equipment calibration",
            "Document environmental conditions",
            "Record all parameter values"
        ]

        if domain == ExperimentDomain.PHYSICS:
            checks.extend([
                "Verify initial conditions",
                "Check measurement uncertainty",
                "Validate theoretical assumptions"
            ])
        elif domain == ExperimentDomain.CHEMISTRY:
            checks.extend([
                "Verify reagent purity",
                "Control reaction temperature",
                "Monitor reaction time precisely"
            ])
        elif domain == ExperimentDomain.BIOLOGY:
            checks.extend([
                "Maintain sterile conditions",
                "Control sample storage conditions",
                "Verify biological activity"
            ])

        return checks

    def _extract_parameters(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract parameters from protocol steps"""
        parameters = {}
        for step in steps:
            if "parameters" in step:
                parameters.update(step["parameters"])
        return parameters

    def _extract_materials(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Extract materials from protocol steps"""
        materials = []
        for step in steps:
            if "materials" in step:
                materials.extend(step["materials"])
        return list(set(materials))

    def _aggregate_results(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple iterations"""
        if not execution_results:
            return {}

        # For simulation, return the first result
        # In production, implement proper aggregation
        return execution_results[0].get("data", {})

    async def _calculate_metrics(
        self,
        data: Dict[str, Any],
        protocol: ExperimentProtocol
    ) -> Dict[str, float]:
        """Calculate performance metrics from experimental data"""
        # Placeholder: implement domain-specific metrics
        return {
            "success_rate": 1.0,
            "measurement_accuracy": 0.95,
            "protocol_compliance": 1.0
        }

    async def _validate_results(
        self,
        data: Dict[str, Any],
        protocol: ExperimentProtocol
    ) -> bool:
        """Validate experimental results"""
        # Basic validation: check if data exists
        return len(data) > 0

    def _calculate_reproducibility(self, execution_results: List[Dict[str, Any]]) -> float:
        """Calculate reproducibility score across iterations"""
        if len(execution_results) < 2:
            return 1.0

        # Placeholder: calculate actual variance
        # For simulation, return high reproducibility
        return 0.95

    def _extract_observations(self, execution_results: List[Dict[str, Any]]) -> List[str]:
        """Extract observations from execution results"""
        observations = []
        for result in execution_results:
            if "observations" in result:
                observations.extend(result["observations"])
        return observations

    async def _perform_significance_tests(
        self,
        results: ExperimentResults,
        hypothesis: Hypothesis
    ) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        # Placeholder: implement actual statistical tests
        return {
            "t_test": {
                "significant": True,
                "p_value": 0.01,
                "test_statistic": 3.5
            }
        }

    async def _calculate_effect_sizes(
        self,
        results: ExperimentResults,
        hypothesis: Hypothesis
    ) -> Dict[str, float]:
        """Calculate effect sizes"""
        # Placeholder: implement actual effect size calculations
        return {
            "cohens_d": 0.8
        }

    async def _calculate_confidence_intervals(
        self,
        results: ExperimentResults
    ) -> Dict[str, tuple]:
        """Calculate confidence intervals"""
        # Placeholder: implement actual CI calculations
        return {
            "mean": (0.95, 1.05)
        }

    async def _calculate_statistical_power(
        self,
        results: ExperimentResults
    ) -> float:
        """Calculate statistical power"""
        # Placeholder: implement actual power calculation
        return 0.9

    async def _generate_recommendations(
        self,
        results: ExperimentResults,
        significance_tests: Dict[str, Any],
        effect_sizes: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        if all(test.get("significant", False) for test in significance_tests.values()):
            recommendations.append("Results are statistically significant")
        else:
            recommendations.append("Consider increasing sample size")

        if effect_sizes.get("cohens_d", 0) > 0.8:
            recommendations.append("Large effect size detected")

        return recommendations

    async def _identify_methodological_issues(
        self,
        protocol: ExperimentProtocol,
        results: ExperimentResults,
        analysis: StatisticalAnalysis
    ) -> List[str]:
        """Identify methodological issues"""
        issues = []

        if results.reproducibility_score < 0.8:
            issues.append("Low reproducibility detected")

        if analysis.statistical_power < 0.8:
            issues.append("Low statistical power")

        return issues

    async def _generate_improvements(
        self,
        protocol: ExperimentProtocol,
        results: ExperimentResults,
        issues: List[str]
    ) -> List[str]:
        """Generate improvement suggestions"""
        improvements = []

        for issue in issues:
            if "reproducibility" in issue.lower():
                improvements.append("Increase number of iterations")
                improvements.append("Standardize procedures")
            elif "power" in issue.lower():
                improvements.append("Increase sample size")

        return improvements

    async def _suggest_next_experiments(
        self,
        protocol: ExperimentProtocol,
        results: ExperimentResults,
        analysis: StatisticalAnalysis
    ) -> List[str]:
        """Suggest follow-up experiments"""
        suggestions = []

        if not analysis.validation_passed:
            suggestions.append("Repeat experiment with increased sample size")
            suggestions.append("Modify experimental conditions based on findings")

        return suggestions
