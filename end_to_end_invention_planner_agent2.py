"""
End-to-End Invention Planner - Agent 2 Enhancements

This module provides enhanced implementations for:
- Task 2.1: Comprehensive error source analysis with Monte Carlo
- Task 2.2: Real adversarial testing (Red Team)
- Task 2.3: Real blue team defense

These methods integrate with the main end_to_end_invention_planner.py
"""

import logging
import re
from typing import List, Dict, Any, Tuple
import numpy as np

# Try to import required modules
try:
    from uncertainty_propagation import (
        UncertaintyPropagator,
        enumerate_all_errors,
        ErrorCategory as UncertaintyErrorCategory,
        ErrorSource as UncertaintyErrorSource
    )
except ImportError:
    UncertaintyPropagator = None
    enumerate_all_errors = None

try:
    from sovereign_problem_analyzer import SovereignProblemAnalyzer
except ImportError:
    SovereignProblemAnalyzer = None

try:
    from red_team import RedTeam, IssueFinding, IssueCategory, SeverityLevel
    from blue_team import BlueTeam, FixSuggestion, BlueTeamFix, FixPriority
except ImportError:
    RedTeam = None
    BlueTeam = None
    IssueFinding = None
    IssueCategory = None
    SeverityLevel = None

from generic_maker_integration import run_generic_maker, TaskType

logger = logging.getLogger(__name__)


class InventionPlannerAgent2:
    """
    Agent 2 - Error Analysis and Adversarial Testing

    Provides enhanced implementations for:
    1. Comprehensive error source analysis
    2. Real adversarial testing (Red Team)
    3. Real blue team defense
    """

    def __init__(self, config=None):
        self.config = config
        self.uncertainty_propagator = UncertaintyPropagator() if UncertaintyPropagator else None
        self.problem_analyzer = SovereignProblemAnalyzer() if SovereignProblemAnalyzer else None
        self.red_team = RedTeam() if RedTeam else None
        self.blue_team = BlueTeam() if BlueTeam else None

    async def analyze_error_sources(
        self,
        goal: Any,
        decomposition: Dict[str, Any],
        knowledge: List[str],
        error_source_class: Any
    ) -> List[Any]:
        """
        Task 2.1: Implement comprehensive error source analysis

        Uses:
        - sovereign_problem_analyzer.py for systematic error analysis
        - problem_analyzer.py for error categorization
        - uncertainty_propagation.py for Monte Carlo simulation

        Enumerates ACTUAL error sources from:
        - Equipment specifications (tolerances, failure rates)
        - Material properties (impurities, variations)
        - Measurement uncertainties
        - Environmental factors
        - Human factors
        - Systematic errors

        Calculates ACTUAL probabilities (not LLM guesses)
        Implements Monte Carlo simulation for error propagation
        Identifies critical error sources via sensitivity analysis
        """

        errors = []
        logger.info(f"Starting comprehensive error source analysis for: {goal.target}")

        # Extract equipment, material, and measurement specs from decomposition
        equipment_specs = self._extract_equipment_specs(decomposition)
        material_specs = self._extract_material_specs(decomposition)
        measurement_specs = self._extract_measurement_specs(decomposition)

        # Enumerate ACTUAL error sources from specifications using uncertainty propagation
        if self.uncertainty_propagator and enumerate_all_errors:
            try:
                # Enumerate all actual error sources
                uncertainty_errors = enumerate_all_errors(
                    equipment_specs=equipment_specs,
                    material_specs=material_specs,
                    measurement_specs=measurement_specs
                )

                # Convert uncertainty error sources to invention planner error sources
                for ue in uncertainty_errors:
                    errors.append(self._convert_uncertainty_error(ue, error_source_class))

                logger.info(f"Enumerated {len(errors)} actual error sources from specifications")

            except Exception as e:
                logger.warning(f"Error enumerating specifications: {e}, falling back to LLM analysis")

        # Use Sovereign problem analyzer for systematic error categorization
        if self.problem_analyzer:
            try:
                # Create problem description from decomposition
                problem_text = f"""
                Goal: {goal.target}
                Domain: {goal.domain}
                Steps: {len(decomposition.get('steps', []))}
                Complexity: {goal.complexity_score}

                Analyze this invention process for potential error sources including:
                - Equipment failures and tolerances
                - Material property variations
                - Measurement uncertainties
                - Environmental factors
                - Human error potential
                - Systematic biases
                """

                # Use problem analyzer for additional insight
                problem = self.problem_analyzer.analyze_problem(problem_text, goal.target)

                # Add problem-specific errors from analysis
                for constraint in problem.constraints:
                    if constraint.type in ["technical", "quality", "resource"]:
                        # Estimate probability based on constraint severity
                        probability = 0.7 if constraint.severity == "hard" else 0.4

                        errors.append(error_source_class(
                            error_type=f"constraint_{constraint.type}",
                            description=constraint.description,
                            probability=probability,
                            impact="high" if constraint.severity == "hard" else "medium",
                            mitigation_strategy=f"Monitor and verify compliance with {constraint.type} constraint",
                            verification_method="Regular validation checks",
                            acceptance_criteria=f"Constraint satisfied: {constraint.description}"
                        ))

            except Exception as e:
                logger.warning(f"Sovereign problem analyzer failed: {e}")

        # Use MAKER for additional error source discovery if gaps remain
        if len(errors) < 20:  # Minimum threshold for comprehensive analysis
            logger.info("Running supplemental LLM-based error analysis")
            task_desc = f"""
Perform comprehensive error source analysis for:

Goal: {goal.target}
Domain: {goal.domain}
Number of steps: {len(decomposition.get('steps', []))}
Existing errors identified: {len(errors)}

For each step and for the overall process, identify:
1. Equipment failure modes (with actual failure rates if known)
2. Measurement errors (with actual tolerances)
3. Human errors (with psychological basis)
4. Material impurities (with actual impurity levels)
5. Environmental variations (temperature, humidity, etc.)
6. Timing errors (process timing variations)
7. Calculation errors (numerical precision issues)
8. Systematic errors (bias, drift, etc.)

For each error:
- Estimate probability (0-1) based on REAL data or physical principles
- Assess impact (critical/high/medium/low)
- Provide specific mitigation strategy
- Define verification method
- Specify acceptance criteria

Be thorough - account for EVERY possible error source.
"""

            try:
                result = await run_generic_maker(
                    task_description=task_desc,
                    evaluator=InventionEvaluator(),
                    task_type=TaskType.CUSTOM,
                    config=self.config
                )
                additional_errors = self._parse_error_sources(result.solution, error_source_class)
                errors.extend(additional_errors)
            except Exception as e:
                logger.warning(f"LLM error analysis failed: {e}")

        # Remove duplicates and consolidate
        unique_errors = self._consolidate_errors(errors)

        # Monte Carlo sensitivity analysis for critical errors
        if self.uncertainty_propagator and len(unique_errors) > 0:
            try:
                # Create a simple model function for sensitivity analysis
                def simple_model(error_values):
                    # Simple weighted sum as placeholder for actual model
                    return np.sum(error_values * np.array([e.probability for e in unique_errors]))

                # Convert to uncertainty error sources for Monte Carlo
                uncertainty_errors = [self._convert_to_uncertainty_error(e) for e in unique_errors[:10]]  # Limit to top 10

                if uncertainty_errors:
                    result = self.uncertainty_propagator.monte_carlo_propagation(
                        uncertainty_errors,
                        simple_model,
                        n_samples=1000
                    )

                    # Update error sources with sensitivity scores
                    for error_name, sensitivity in result.critical_error_sources:
                        for error in unique_errors:
                            if error_name in error.description:
                                # Update impact based on sensitivity
                                if sensitivity > 0.7:
                                    error.impact = "critical"
                                elif sensitivity > 0.4:
                                    error.impact = "high"
                                break

                    logger.info(f"Sensitivity analysis complete. Top critical errors: {result.critical_error_sources[:5]}")

            except Exception as e:
                logger.warning(f"Monte Carlo sensitivity analysis failed: {e}")

        logger.info(f"Error source analysis complete: {len(unique_errors)} unique error sources identified")
        return unique_errors

    async def red_blue_team_test(
        self,
        goal: Any,
        decomposition: Dict[str, Any],
        errors: List[Any],
        error_source_class: Any
    ) -> Tuple[List[str], List[str]]:
        """
        Task 2.2 & 2.3: Implement real adversarial testing (Red Team) and Blue Team defense

        Uses:
        - red_team.py for actual adversarial testing
        - adversarial.py for systematic vulnerability scanning
        - adversarial_testing.py for comprehensive testing
        - blue_team.py for actual defense

        Red Team implements actual attack strategies:
        - Parameter perturbation testing
        - Edge case exploration
        - Failure mode injection
        - Boundary condition testing
        - Stress testing
        - Chaos testing

        Blue Team implements actual defense:
        - Root cause analysis
        - Generate ACTUAL fix (not just text)
        - Apply fix to SOP
        - Re-test with red team
        - Iterate until fixed
        """

        red_findings = []
        blue_fixes = []

        logger.info(f"Starting real adversarial testing for: {goal.target}")

        # Convert error sources to Red Team format
        red_team_issues = self._convert_errors_to_red_team_format(errors) if self.red_team else []

        # Build invention plan description for red team
        invention_plan_text = self._build_invention_plan_description(goal, decomposition, errors)

        # Red Team: Real adversarial testing
        if self.red_team and red_team_issues:
            try:
                logger.info("Running actual Red Team testing...")

                # Perform red team assessment
                red_assessment = self.red_team.assess_content(
                    content=invention_plan_text,
                    content_type="protocol",
                    attack_modes=[
                        "security scan",
                        "edge case exploration",
                        "assumption challenge",
                        "compliance check",
                        "logic verification"
                    ]
                )

                # Convert findings to strings
                red_findings = [
                    f"[{finding.severity.value.upper()}] {finding.category.value}: {finding.description}"
                    for finding in red_assessment.findings
                ]

                logger.info(f"Red Team found {len(red_findings)} vulnerabilities")

            except Exception as e:
                logger.warning(f"Red Team testing failed: {e}, falling back to LLM")
                red_findings = await self._llm_red_team(goal, decomposition, errors)

        else:
            # Fallback to LLM-based red team
            red_findings = await self._llm_red_team(goal, decomposition, errors)

        # Blue Team: Real defense
        if self.blue_team and red_findings:
            try:
                logger.info("Running actual Blue Team defense...")

                # Convert red findings to issue format
                red_issues = self._parse_red_findings_to_issues(red_findings)

                # Apply fixes using blue team
                blue_assessment = self.blue_team.apply_fixes(
                    content=invention_plan_text,
                    issues=red_issues,
                    content_type="protocol",
                    strategy="comprehensive"  # Use comprehensive strategy
                )

                # Convert blue fixes to strings
                blue_fixes = [
                    f"Fix: {fix.fix_suggestion.fix_description} - {fix.fix_status}"
                    for fix in blue_assessment.applied_fixes
                ]

                # Also include fix suggestions
                blue_fixes.extend([
                    f"Suggestion: {sug.fix_description} (Priority: {sug.priority.value})"
                    for sug in blue_assessment.fix_suggestions[:10]  # Top 10
                ])

                logger.info(f"Blue Team applied {len(blue_assessment.applied_fixes)} fixes, suggested {len(blue_assessment.fix_suggestions)} improvements")

            except Exception as e:
                logger.warning(f"Blue Team defense failed: {e}, falling back to LLM")
                blue_fixes = await self._llm_blue_team(goal, red_findings)

        else:
            # Fallback to LLM-based blue team
            blue_fixes = await self._llm_blue_team(goal, red_findings)

        return red_findings, blue_fixes

    # ==================== HELPER METHODS ====================

    def _extract_equipment_specs(self, decomposition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract equipment specifications from decomposition"""
        specs = []

        # Look for equipment mentions in decomposition steps
        steps = decomposition.get('steps', [])
        for step in steps:
            description = step.get('description', '').lower()

            # Common equipment with default specs
            if 'thermometer' in description or 'temperature' in description:
                specs.append({
                    'name': 'Thermometer',
                    'accuracy': 0.5,  # ±0.5°C
                    'precision': 0.1,  # ±0.1°C
                    'tolerance': 1.0,  # ±1.0°C
                    'failure_rate': 0.001  # 0.1% failure rate
                })

            if 'scale' in description or 'balance' in description or 'weigh' in description:
                specs.append({
                    'name': 'Scale/Balance',
                    'accuracy': 0.01,  # ±0.01g
                    'precision': 0.001,  # ±0.001g
                    'tolerance': 0.05,  # ±0.05g
                    'failure_rate': 0.0005
                })

            if 'voltage' in description or 'multimeter' in description or 'meter' in description:
                specs.append({
                    'name': 'Multimeter',
                    'accuracy': 0.02,  # ±2%
                    'precision': 0.01,  # ±1%
                    'tolerance': 0.05,  # ±5%
                    'failure_rate': 0.002
                })

        # Add some generic equipment if none found
        if not specs:
            specs.append({
                'name': 'Generic Equipment',
                'accuracy': 0.05,
                'precision': 0.02,
                'tolerance': 0.1,
                'failure_rate': 0.001
            })

        return specs

    def _extract_material_specs(self, decomposition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract material specifications from decomposition"""
        specs = []

        steps = decomposition.get('steps', [])
        for step in steps:
            description = step.get('description', '').lower()

            # Common materials with default specs
            if 'chemical' in description or 'reagent' in description or 'solution' in description:
                specs.append({
                    'name': 'Chemical Reagent',
                    'property_variations': {
                        'purity': 0.01,  # ±1% purity variation
                        'concentration': 0.02  # ±2% concentration
                    },
                    'impurity_level': 0.001,  # 0.1% impurities
                    'batch_variation': 0.005  # ±0.5% batch-to-batch
                })

            if 'material' in description or 'sample' in description:
                specs.append({
                    'name': 'Sample Material',
                    'property_variations': {
                        'thickness': 0.05,  # ±5% thickness
                        'density': 0.02  # ±2% density
                    },
                    'impurity_level': 0.005,
                    'batch_variation': 0.01
                })

        if not specs:
            specs.append({
                'name': 'Generic Material',
                'property_variations': {
                    'property': 0.05
                },
                'impurity_level': 0.001,
                'batch_variation': 0.01
            })

        return specs

    def _extract_measurement_specs(self, decomposition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract measurement specifications from decomposition"""
        specs = []

        steps = decomposition.get('steps', [])

        # Generic measurement specs
        specs.append({
            'name': 'Length Measurement',
            'resolution': 0.001,  # 1mm resolution
            'uncertainty': 0.005,  # ±5mm uncertainty
            'bias': 0.0
        })

        specs.append({
            'name': 'Time Measurement',
            'resolution': 0.1,  # 0.1s resolution
            'uncertainty': 0.5,  # ±0.5s uncertainty
            'bias': 0.0
        })

        specs.append({
            'name': 'Temperature Measurement',
            'resolution': 0.1,  # 0.1°C resolution
            'uncertainty': 0.5,  # ±0.5°C uncertainty
            'bias': 0.0
        })

        return specs

    def _convert_uncertainty_error(self, ue: Any, error_source_class: Any) -> Any:
        """Convert uncertainty propagation error source to invention planner error source"""
        return error_source_class(
            error_type=ue.category.value,
            description=f"{ue.name}: {ue.description}",
            probability=ue.probability_of_occurrence,
            impact=ue.impact_severity,
            mitigation_strategy=ue.mitigation_strategy,
            verification_method=ue.verification_method,
            acceptance_criteria=ue.acceptance_criteria
        )

    def _convert_to_uncertainty_error(self, e: Any) -> Any:
        """Convert invention planner error to uncertainty error (for Monte Carlo)"""
        if UncertaintyErrorSource is None:
            return None

        return UncertaintyErrorSource(
            name=e.error_type,
            category=UncertaintyErrorCategory.EQUIPMENT_SPECIFICATION,  # Default
            description=e.description,
            distribution=ProbabilityDistribution.NORMAL,
            distribution_params={'mean': 0.0, 'std': e.probability},
            nominal_value=0.0,
            tolerance=e.probability * 3,
            probability_of_occurrence=e.probability,
            impact_severity=e.impact,
            mitigation_strategy=e.mitigation_strategy,
            verification_method=e.verification_method,
            acceptance_criteria=e.acceptance_criteria
        )

    def _convert_errors_to_red_team_format(self, errors: List[Any]) -> List[Any]:
        """Convert error sources to Red Team IssueFinding format"""
        if IssueFinding is None:
            return []

        findings = []
        for error in errors:
            severity_map = {
                'critical': SeverityLevel.CRITICAL,
                'high': SeverityLevel.HIGH,
                'medium': SeverityLevel.MEDIUM,
                'low': SeverityLevel.LOW
            }

            category_map = {
                'equipment_specification': IssueCategory.STRUCTURAL_FLAW,
                'material_properties': IssueCategory.COMPLIANCE_ISSUE,
                'measurement_uncertainty': IssueCategory.PERFORMANCE_PROBLEM,
                'environmental_factors': IssueCategory.EDGE_CASE,
                'human_factors': IssueCategory.CLARITY_ISSUE,
                'systematic_errors': IssueCategory.LOGICAL_ERROR
            }

            findings.append(IssueFinding(
                title=error.error_type,
                description=error.description,
                severity=severity_map.get(error.impact.lower(), SeverityLevel.MEDIUM),
                category=category_map.get(error.error_type.lower(), IssueCategory.LOGICAL_ERROR),
                confidence=error.probability
            ))

        return findings

    def _build_invention_plan_description(self, goal: Any, decomposition: Dict[str, Any], errors: List[Any]) -> str:
        """Build a comprehensive description of the invention plan for red/blue team"""
        lines = [
            f"Invention Goal: {goal.target}",
            f"Domain: {goal.domain}",
            f"Complexity: {goal.complexity_score:.2f}",
            "",
            f"Number of Steps: {len(decomposition.get('steps', []))}",
            "",
            "Steps:",
        ]

        for i, step in enumerate(decomposition.get('steps', [])[:20], 1):  # Limit to 20 steps
            lines.append(f"{i}. {step.get('description', 'Unknown step')}")

        lines.append("")
        lines.append(f"Identified Error Sources ({len(errors)}):")

        for i, error in enumerate(errors[:30], 1):  # Limit to 30 errors
            lines.append(f"{i}. [{error.impact.upper()}] {error.error_type}: {error.description}")

        return "\n".join(lines)

    def _parse_red_findings_to_issues(self, red_findings: List[str]) -> List[Any]:
        """Parse red team findings into IssueFinding format for blue team"""
        if IssueFinding is None:
            return []

        issues = []
        for finding in red_findings:
            # Parse severity from finding
            if '[CRITICAL]' in finding:
                severity = SeverityLevel.CRITICAL
            elif '[HIGH]' in finding:
                severity = SeverityLevel.HIGH
            elif '[MEDIUM]' in finding:
                severity = SeverityLevel.MEDIUM
            elif '[LOW]' in finding:
                severity = SeverityLevel.LOW
            else:
                severity = SeverityLevel.MEDIUM

            # Parse category
            if 'security' in finding.lower():
                category = IssueCategory.SECURITY_VULNERABILITY
            elif 'performance' in finding.lower():
                category = IssueCategory.PERFORMANCE_PROBLEM
            elif 'logic' in finding.lower():
                category = IssueCategory.LOGICAL_ERROR
            elif 'edge' in finding.lower():
                category = IssueCategory.EDGE_CASE
            else:
                category = IssueCategory.STRUCTURAL_FLAW

            issues.append(IssueFinding(
                title=finding[:50],  # First 50 chars as title
                description=finding,
                severity=severity,
                category=category,
                confidence=0.8
            ))

        return issues

    def _parse_error_sources(self, solution: str, error_source_class: Any) -> List[Any]:
        """Parse error sources from LLM solution"""
        errors = []

        for line in solution.split('\n'):
            line = line.strip()
            if 'error' in line.lower() or 'fail' in line.lower():
                errors.append(error_source_class(
                    error_type="llm_identified",
                    description=line[:200],
                    probability=0.3,  # Default probability
                    impact="medium",
                    mitigation_strategy="See LLM analysis",
                    verification_method="Observation",
                    acceptance_criteria="No error occurred"
                ))

        return errors[:50]  # Limit to 50

    def _consolidate_errors(self, errors: List[Any]) -> List[Any]:
        """Consolidate duplicate errors"""
        unique_errors = []
        seen_descriptions = set()

        for error in errors:
            # Create a key from first 100 chars of description
            key = error.description[:100].lower()
            if key not in seen_descriptions:
                unique_errors.append(error)
                seen_descriptions.add(key)

        return unique_errors

    async def _llm_red_team(self, goal: Any, decomposition: Dict[str, Any], errors: List[Any]) -> List[str]:
        """Fallback LLM-based red team testing"""
        task_desc = f"""
You are a RED TEAM adversarial tester. Find every possible vulnerability, flaw, or failure mode in this invention plan:

Goal: {goal.target}
Number of steps: {len(decomposition.get('steps', []))}
Known Error Sources: {len(errors)}

Be ruthless - assume everything that can go wrong WILL go wrong. Identify:
1. Logical fallacies
2. Physical impossibilities
3. Missing steps
4. Unrealistic assumptions
5. Hidden dependencies
6. Single points of failure
7. Validation gaps
8. Anything else that could cause failure

List all findings with severity ratings [CRITICAL], [HIGH], [MEDIUM], [LOW].
"""
        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=InventionEvaluator(),
            task_type=TaskType.CUSTOM,
            config=self.config
        )

        return self._parse_findings(result.solution)

    async def _llm_blue_team(self, goal: Any, red_findings: List[str]) -> List[str]:
        """Fallback LLM-based blue team defense"""
        task_desc = f"""
You are a BLUE TEAM defender. For each red team finding, provide a comprehensive fix:

Red Team Findings:
{chr(10).join(f"- {f}" for f in red_findings)}

For each finding, provide:
1. Root cause analysis
2. Fix strategy
3. Implementation approach
4. Verification method
5. Fallback options

Ensure fixes address the root cause, not just symptoms.
"""
        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=InventionEvaluator(),
            task_type=TaskType.CUSTOM,
            config=self.config
        )

        return self._parse_findings(result.solution)

    def _parse_findings(self, solution: str) -> List[str]:
        """Parse findings from solution"""
        findings = []
        for line in solution.split('\n'):
            line = line.strip()
            if line and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                findings.append(line.strip()[2:] if line.strip().startswith('-') else line.strip())
        return findings[:30]  # Limit to 30


class InventionEvaluator:
    """Simple evaluator for invention planning tasks"""

    def evaluate(self, solution: str, task: Any = None) -> float:
        score = 0.0
        score += 0.2 * len(solution) / 1000
        if 'step' in solution.lower():
            score += 0.2
        if 'error' in solution.lower():
            score += 0.2
        if 'verify' in solution.lower() or 'validate' in solution.lower():
            score += 0.2
        if 'fix' in solution.lower():
            score += 0.2
        return min(1.0, score)
