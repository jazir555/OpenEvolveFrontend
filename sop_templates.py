#!/usr/bin/env python3
"""
SOP Template System - Structured Evolution for Each SOP Facet

Provides specialized templates, prompts, and validators for each part of
the Magneto-Chemical Directed Assembly SOP.

Usage:
    from sop_templates import SOPEvolver, SOPTemplateRegistry

    # Evolve specific facet
    evolved = evolve_environmental_conditions(original_sop)

    # Or evolve entire SOP with facet-specific handling
    registry = SOPTemplateRegistry()
    evolved_sop = registry.evolve_entire_sop(sop_content)
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime

# Import ensemble components
from red_team import RedTeam
from blue_team import BlueTeam, BlueTeamStrategy
from evaluator_team import EvaluatorTeam


class SOPFacet(Enum):
    """Enumeration of SOP facets (sections)"""
    ENVIRONMENTAL = "Part 0: Environmental Conditions"
    EQUIPMENT = "Part 1: Equipment Specifications"
    MATERIALS = "Part 2: Materials"
    EXECUTION_PHASES = "Part 3: Execution Protocols"
    QUALITY_CONTROL = "Part 4: Quality Control"
    SAFETY = "Part 5: Safety Protocols"
    VALIDATION = "Part 6: Validation and Scalability"


@dataclass
class FacetTemplate:
    """Template for evolving a specific SOP facet"""
    facet: SOPFacet
    section_extractor: Callable[[str], str]
    red_team_attacks: List[str]
    blue_team_strategy: BlueTeamStrategy
    evaluation_criteria: Dict[str, float]
    reconstruction_template: Optional[str] = None
    facet_specific_validators: List[Callable[[str], bool]] = field(default_factory=list)


class SOPTemplateRegistry:
    """Registry of SOP facet templates and evolution functions"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.templates: Dict[SOPFacet, FacetTemplate] = {}
        self.red_team = RedTeam()
        self.blue_team = BlueTeam()
        self.evaluator = EvaluatorTeam()

        self._register_all_templates()

    def _register_all_templates(self):
        """Register templates for all SOP facets"""
        self.templates[SOPFacet.ENVIRONMENTAL] = self._environmental_template()
        self.templates[SOPFacet.EQUIPMENT] = self._equipment_template()
        self.templates[SOPFacet.MATERIALS] = self._materials_template()
        self.templates[SOPFacet.EXECUTION_PHASES] = self._execution_template()
        self.templates[SOPFacet.QUALITY_CONTROL] = self._quality_control_template()
        self.templates[SOPFacet.SAFETY] = self._safety_template()
        self.templates[SOPFacet.VALIDATION] = self._validation_template()

    # ============================================================================
    # PART 0: ENVIRONMENTAL CONDITIONS
    # ============================================================================

    def _environmental_template(self) -> FacetTemplate:
        """Template for environmental conditions (Part 0)"""
        return FacetTemplate(
            facet=SOPFacet.ENVIRONMENTAL,
            section_extractor=self._extract_part_0,
            red_team_attacks=[
                "unrealistic_tolerance",
                "missing_contingency",
                "insufficient_monitoring",
                "seasonal_variation",
                "thermal_inertia",
                "hvac_capacity_limit"
            ],
            blue_team_strategy=BlueTeamStrategy.DEFENSIVE,
            evaluation_criteria={
                "physical_realizability": 0.30,
                "verifiability": 0.25,
                "safety": 0.25,
                "operational_clarity": 0.20
            },
            facet_specific_validators=[
                self._validate_temperature_specs,
                self._validate_humidity_specs,
                self._validate_monitoring_frequency
            ]
        )

    def _extract_part_0(self, sop_content: str) -> str:
        """Extract Part 0 from SOP"""
        # Find Part 0 section
        match = re.search(
            r'PART 0.*?(?=PART 1|$)',
            sop_content,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(0) if match else ""

    def _validate_temperature_specs(self, content: str) -> bool:
        """Validate temperature specifications are realistic"""
        # Extract temperature values
        temp_pattern = r'(\d+\.?\d*)\s*degrees?\s*Celsius'
        temps = re.findall(temp_pattern, content)

        # Check for realistic ranges
        for temp_str in temps:
            temp = float(temp_str)
            if temp < 15 or temp > 30:
                return False  # Outside lab comfort range
        return True

    def _validate_humidity_specs(self, content: str) -> bool:
        """Validate humidity specifications"""
        humidity_pattern = r'(\d+\.?\d*)\s*percent'
        humidities = re.findall(humidity_pattern, content)

        for hum_str in humidities:
            hum = float(hum_str)
            if hum < 20 or hum > 60:
                return False  # Outside typical lab range
        return True

    def _validate_monitoring_frequency(self, content: str) -> bool:
        """Validate monitoring frequency is sufficient"""
        # Check for monitoring every 60 minutes or more frequent
        freq_pattern = r'Every\s*(\d+)\s*minutes'
        frequencies = re.findall(freq_pattern, content)

        for freq_str in frequencies:
            freq = int(freq_str)
            if freq > 240:  # More than 4 hours
                return False
        return True

    # ============================================================================
    # PART 1: EQUIPMENT SPECIFICATIONS
    # ============================================================================

    def _equipment_template(self) -> FacetTemplate:
        """Template for equipment specifications (Part 1)"""
        return FacetTemplate(
            facet=SOPFacet.EQUIPMENT,
            section_extractor=self._extract_part_1,
            red_team_attacks=[
                "measurement_uncertainty",
                "equipment_compatibility",
                "calibration_traceability",
                "power_requirement_mismatch",
                "cooling_capacity_insufficient",
                "interlock_ambiguity"
            ],
            blue_team_strategy=BlueTeamStrategy.COMPREHENSIVE,
            evaluation_criteria={
                "verifiability": 0.35,
                "physical_realizability": 0.30,
                "safety": 0.25,
                "scalability": 0.10
            },
            facet_specific_validators=[
                self._validate_magnetic_specs,
                self._validate_uv_specs,
                self._validate_thermal_specs
            ]
        )

    def _extract_part_1(self, sop_content: str) -> str:
        """Extract Part 1 from SOP"""
        match = re.search(
            r'PART 1.*?(?=PART 2|$)',
            sop_content,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(0) if match else ""

    def _validate_magnetic_specs(self, content: str) -> str:
        """Validate magnetic field specifications"""
        # Extract field strength and tolerance
        field_pattern = r'(\d+\.?\d*)\s*Tesla\s*±\s*(\d+\.?\d*)\s*Tesla'
        matches = re.findall(field_pattern, content)

        for field_str, tolerance_str in matches:
            field = float(field_str)
            tolerance = float(tolerance_str)

            # Check if tolerance is achievable (< 1% is very tight)
            tolerance_pct = (tolerance / field) * 100
            if tolerance_pct < 0.1:
                return False  # < 0.1% requires NMR, not Hall probe
        return True

    def _validate_uv_specs(self, content: str) -> bool:
        """Validate UV curing specifications"""
        # Check for interlock logic
        if "interlock" not in content.lower():
            return False

        # Check for power density limits
        power_pattern = r'(\d+\.?\d*)\s*milliwatts?\s*per\s*square\s*centimetre'
        power_matches = re.findall(power_pattern, content, re.IGNORECASE)

        for power_str in power_matches:
            power = float(power_str)
            if power > 100:  # > 100 mW/cm² causes heating
                return False
        return True

    def _validate_thermal_specs(self, content: str) -> bool:
        """Validate thermal stage specifications"""
        # Check for temperature range
        range_pattern = r'(\d+\.?\d*)\s*degrees?\s*Celsius\s*to\s*(\d+\.?\d*)\s*degrees?\s*Celsius'
        range_matches = re.findall(range_pattern, content, re.IGNORECASE)

        for min_str, max_str in range_matches:
            min_temp = float(min_str)
            max_temp = float(max_str)

            if max_temp - min_temp > 100:
                return False  # > 100°C range unrealistic for single stage
        return True

    # ============================================================================
    # PART 2: MATERIALS
    # ============================================================================

    def _materials_template(self) -> FacetTemplate:
        """Template for materials specifications (Part 2)"""
        return FacetTemplate(
            facet=SOPFacet.MATERIALS,
            section_extractor=self._extract_part_2,
            red_team_attacks=[
                "chemical_instability",
                "shelf_life_unrealistic",
                "purity_unachievable",
                "mixing_incompatibility",
                "contamination_risk",
                "supply_chain_risk"
            ],
            blue_team_strategy=BlueTeamStrategy.COMPREHENSIVE,
            evaluation_criteria={
                "physical_realizability": 0.35,
                "verifiability": 0.30,
                "scalability": 0.20,
                "safety": 0.15
            },
            facet_specific_validators=[
                self._validate_composition_purity,
                self._validate_shelf_life,
                self._validate_mixing_protocol
            ]
        )

    def _extract_part_2(self, sop_content: str) -> str:
        """Extract Part 2 from SOP"""
        match = re.search(
            r'PART 2.*?(?=PART 3|$)',
            sop_content,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(0) if match else ""

    def _validate_composition_purity(self, content: str) -> bool:
        """Validate composition purity specifications"""
        # Extract purity percentages
        purity_pattern = r'(\d+\.?\d*)\s*percent\s*purity'
        purities = re.findall(purity_pattern, content, re.IGNORECASE)

        for purity_str in purities:
            purity = float(purity_str)
            if purity > 99.999:
                return False  # 5 nines unrealistic for most chemicals
        return True

    def _validate_shelf_life(self, content: str) -> bool:
        """Validate shelf life claims"""
        # Find shelf life specifications
        shelf_pattern = r'shelf\s*life:\s*(\d+)\s*days?'
        shelf_matches = re.findall(shelf_pattern, content, re.IGNORECASE)

        for days_str in shelf_matches:
            days = int(days_str)
            if days > 365 and "refrigerated" not in content.lower():
                return False  # > 1 year at room temp unrealistic
        return True

    def _validate_mixing_protocol(self, content: str) -> bool:
        """Validate mixing protocol"""
        # Check for mixing equipment specified
        equipment_patterns = [
            r'planetary\s+mixer',
            r'magnetic\s+stirrer',
            r'ultrasonicator'
        ]

        content_lower = content.lower()
        has_equipment = any(re.search(pattern, content_lower) for pattern in equipment_patterns)

        if not has_equipment:
            return False
        return True

    # ============================================================================
    # PART 3: EXECUTION PROTOCOLS
    # ============================================================================

    def _execution_template(self) -> FacetTemplate:
        """Template for execution protocols (Part 3 - 4 phases)"""
        return FacetTemplate(
            facet=SOPFacet.EXECUTION_PHASES,
            section_extractor=self._extract_part_3,
            red_team_attacks=[
                "timing_conflict",
                "sequential_dependency",
                "phase_transition_risk",
                "measurement_timing",
                "equilibrium_time_insufficient",
                "thermal_gradient_issue"
            ],
            blue_team_strategy=BlueTeamStrategy.COMPREHENSIVE,
            evaluation_criteria={
                "operational_clarity": 0.35,
                "physical_realizability": 0.30,
                "verifiability": 0.20,
                "scalability": 0.15
            },
            facet_specific_validators=[
                self._validate_phase_timing,
                self._validate_phase_dependencies,
                self._validate_verification_points
            ]
        )

    def _extract_part_3(self, sop_content: str) -> str:
        """Extract Part 3 from SOP"""
        match = re.search(
            r'PART 3.*?(?=PART 4|$)',
            sop_content,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(0) if match else ""

    def _validate_phase_timing(self, content: str) -> bool:
        """Validate phase timing specifications"""
        # Extract phase durations
        duration_pattern = r'(\d+)\s+minutes?\s+exact'
        durations = re.findall(duration_pattern, content, re.IGNORECASE)

        # Check for reasonable total time (< 48 hours)
        total_minutes = sum(int(d) for d in durations)
        if total_minutes > 2880:  # > 48 hours
            return False
        return True

    def _validate_phase_dependencies(self, content: str) -> bool:
        """Validate phase dependencies are explicit"""
        # Look for dependency indicators
        dependency_patterns = [
            r'after\s+complete',
            r'once\s+\w+\s+finished',
            r'following\s+completion',
            r'verify\s+before\s+proceeding'
        ]

        content_lower = content.lower()
        has_dependencies = any(re.search(pattern, content_lower) for pattern in dependency_patterns)

        return has_dependencies

    def _validate_verification_points(self, content: str) -> bool:
        """Validate verification points exist"""
        # Count verification steps
        verification_patterns = [
            r'verif[y|ication]',
            r'confirm',
            r'check',
            r'ensure'
        ]

        verification_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in verification_patterns
        )

        # Should have at least 10 verification points
        return verification_count >= 10

    # ============================================================================
    # PART 4: QUALITY CONTROL
    # ============================================================================

    def _quality_control_template(self) -> FacetTemplate:
        """Template for quality control (Part 4)"""
        return FacetTemplate(
            facet=SOPFacet.QUALITY_CONTROL,
            section_extractor=self._extract_part_4,
            red_team_attacks=[
                "unverifiable_criteria",
                "missing_acceptance_test",
                "ambiguous_pass_fail",
                "insufficient_sampling",
                "statistical_weakness"
            ],
            blue_team_strategy=BlueTeamStrategy.DEFENSIVE,
            evaluation_criteria={
                "verifiability": 0.40,
                "operational_clarity": 0.30,
                "physical_realizability": 0.30
            },
            facet_specific_validators=[
                self._validate_acceptance_criteria,
                self._validate_statistical_sampling,
                self._validate_documentation_requirements
            ]
        )

    def _extract_part_4(self, sop_content: str) -> str:
        """Extract Part 4 from SOP"""
        match = re.search(
            r'PART 4.*?(?=PART 5|$)',
            sop_content,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(0) if match else ""

    def _validate_acceptance_criteria(self, content: str) -> bool:
        """Validate acceptance criteria are measurable"""
        # Look for specific, numeric criteria
        criteria_pattern = r'(≥|<=?|greater than|less than)\s*\d+\.?\d*\s*%'
        criteria = re.findall(criteria_pattern, content, re.IGNORECASE)

        return len(criteria) >= 5

    def _validate_statistical_sampling(self, content: str) -> bool:
        """Validate statistical sampling is adequate"""
        # Check for sample size specification
        sample_patterns = [
            r'n\s*=\s*\d+',
            r'sample\s*size',
            r'statistical\s+significance'
        ]

        has_sampling = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in sample_patterns
        )

        return has_sampling

    def _validate_documentation_requirements(self, content: str) -> bool:
        """Validate documentation requirements"""
        # Check for logging requirements
        logging_patterns = [
            r'log\s+every',
            r'record',
            r'document',
            r'track'
        ]

        logging_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in logging_patterns
        )

        return logging_count >= 5

    # ============================================================================
    # PART 5: SAFETY PROTOCOLS
    # ============================================================================

    def _safety_template(self) -> FacetTemplate:
        """Template for safety protocols (Part 5)"""
        return FacetTemplate(
            facet=SOPFacet.SAFETY,
            section_extractor=self._extract_part_5,
            red_team_attacks=[
                "missing_emergency_procedure",
                "insufficient_training",
                "unsafe_equipment",
                "chemical_exposure_risk",
                "magnetic_hazard",
                "uv_radiation_risk"
            ],
            blue_team_strategy=BlueTeamStrategy.DEFENSIVE,
            evaluation_criteria={
                "safety": 0.50,
                "operational_clarity": 0.30,
                "verifiability": 0.20
            },
            facet_specific_validators=[
                self._validate_emergency_procedures,
                self._validate_training_requirements,
                self._validate_ppe_specifications
            ]
        )

    def _extract_part_5(self, sop_content: str) -> str:
        """Extract Part 5 from SOP"""
        match = re.search(
            r'PART 5.*?(?=PART 6|$)',
            sop_content,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(0) if match else ""

    def _validate_emergency_procedures(self, content: str) -> bool:
        """Validate emergency procedures are complete"""
        required_procedures = [
            r'emergency\s+stop',
            r'evacuation',
            r'first\s+aid',
            r'fire',
            r'chemical\s+exposure'
        ]

        content_lower = content.lower()
        has_all = all(
            re.search(proc, content_lower)
            for proc in required_procedures
        )

        return has_all

    def _validate_training_requirements(self, content: str) -> bool:
        """Validate training requirements"""
        training_patterns = [
            r'training',
            r'certification',
            r'qualified',
            r'competent'
        ]

        training_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in training_patterns
        )

        return training_count >= 3

    def _validate_ppe_specifications(self, content: str) -> bool:
        """Validate PPE specifications"""
        ppe_patterns = [
            r'goggles?',
            r'gloves?',
            r'lab\s+coat',
            r'safety\s+shoes'
        ]

        has_ppe = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in ppe_patterns
        )

        return has_ppe

    # ============================================================================
    # PART 6: VALIDATION AND SCALABILITY
    # ============================================================================

    def _validation_template(self) -> FacetTemplate:
        """Template for validation and scalability (Part 6)"""
        return FacetTemplate(
            facet=SOPFacet.VALIDATION,
            section_extractor=self._extract_part_6,
            red_team_attacks=[
                "scaling_law_invalid",
                "volume_limit_unspecified",
                "batch_size_inconsistency",
                "cost_unrealistic",
                "yield_unrealistic"
            ],
            blue_team_strategy=BlueTeamStrategy.COMPREHENSIVE,
            evaluation_criteria={
                "scalability": 0.40,
                "physical_realizability": 0.35,
                "verifiability": 0.25
            },
            facet_specific_validators=[
                self._validate_scaling_laws,
                self._validate_batch_specifications,
                self._validate_yield_targets
            ]
        )

    def _extract_part_6(self, sop_content: str) -> str:
        """Extract Part 6 from SOP"""
        match = re.search(
            r'PART 6.*?$',
            sop_content,
            re.DOTALL | re.IGNORECASE
        )
        return match.group(0) if match else ""

    def _validate_scaling_laws(self, content: str) -> bool:
        """Validate scaling law specifications"""
        # Look for mathematical scaling relationships
        scaling_pattern = r'time\s*∝\s*volume?\^?\(?[\d./]?\)?'
        has_scaling = re.search(scaling_pattern, content, re.IGNORECASE)

        return has_scaling is not None

    def _validate_batch_specifications(self, content: str) -> bool:
        """Validate batch size specifications"""
        # Check for baseline batch size
        batch_pattern = r'(\d+)\s*(mL|mL|microlitre|litre)'
        batches = re.findall(batch_pattern, content, re.IGNORECASE)

        return len(batches) >= 2

    def _validate_yield_targets(self, content: str) -> bool:
        """Validate yield targets are realistic"""
        yield_pattern = r'yield:\s*≥?\s*(\d+)\s*%'
        yields = re.findall(yield_pattern, content, re.IGNORECASE)

        for yield_str in yields:
            yield_val = int(yield_str)
            if yield_val > 99:
                return False  # > 99% yield unrealistic
        return True

    # ============================================================================
    # FACET-SPECIFIC EVOLUTION FUNCTIONS
    # ============================================================================

    def evolve_facet(
        self,
        sop_content: str,
        facet: SOPFacet,
        num_models: int = 7
    ) -> Dict[str, Any]:
        """
        Evolve a specific facet of the SOP

        Args:
            sop_content: Full SOP content
            facet: Facet to evolve
            num_models: Number of ensemble models

        Returns:
            Dict with evolved facet and metadata
        """
        template = self.templates.get(facet)
        if not template:
            raise ValueError(f"No template registered for facet: {facet}")

        print(f"\n{'='*70}")
        print(f"Evolving {facet.value}")
        print(f"{'='*70}\n")

        # Extract facet section
        facet_content = template.section_extractor(sop_content)

        if not facet_content:
            raise ValueError(f"Could not extract {facet.value} from SOP")

        print(f"Extracted {len(facet_content)} characters")

        # Run red team analysis
        print(f"\nRed Team Analysis ({len(template.red_team_attacks)} attack types)...")
        red_result = self.red_team.analyze_with_ensemble(
            content=facet_content,
            content_type="technical_sop",
            api_key=self.api_key,
            num_models=num_models,
            attack_types=template.red_team_attacks
        )

        vulnerabilities_found = len(red_result.vulnerabilities)
        print(f"  ✓ Found {vulnerabilities_found} vulnerabilities")

        if vulnerabilities_found == 0:
            return {
                "facet": facet.value,
                "status": "NO_VULNERABILITIES",
                "evolved_content": facet_content,
                "confidence": 1.0
            }

        # Run blue team fixes
        print(f"\nBlue Team Fix Generation ({template.blue_team_strategy.value})...")
        from red_team import IssueFinding, IssueCategory, SeverityLevel

        issues = []
        for vuln in red_result.vulnerabilities[:10]:
            severity_map = {
                'critical': SeverityLevel.CRITICAL,
                'high': SeverityLevel.HIGH,
                'medium': SeverityLevel.MEDIUM,
                'low': SeverityLevel.LOW
            }

            issue = IssueFinding(
                title=vuln.get('title', 'Unnamed'),
                description=vuln.get('description', str(vuln)),
                severity=severity_map.get(
                    vuln.get('severity', 'medium').lower(),
                    SeverityLevel.MEDIUM
                ),
                category=IssueCategory.LOGICAL_ERROR,
                confidence=red_result.confidence
            )
            issues.append(issue)

        blue_result = self.blue_team.generate_solutions_with_ensemble(
            issues=issues,
            content=facet_content,
            content_type="technical_sop",
            api_key=self.api_key,
            num_models=num_models,
            strategy=template.blue_team_strategy
        )

        fixes_applied = len(blue_result.applied_fixes)
        print(f"  ✓ Applied {fixes_applied} fixes")

        evolved_facet = blue_result.fixed_content

        # Run validators
        print(f"\nRunning facet-specific validators...")
        validation_results = []
        for validator in template.facet_specific_validators:
            try:
                is_valid = validator(evolved_facet)
                validation_results.append({
                    "validator": validator.__name__,
                    "passed": is_valid
                })
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                validation_results.append({
                    "validator": validator.__name__,
                    "passed": False,
                    "error": str(e)
                })

        all_valid = all(vr["passed"] for vr in validation_results)
        print(f"  Validators: {len(validation_results)} checks, {'ALL PASSED' if all_valid else 'SOME FAILED'}")

        # Run evaluator
        print(f"\nEvaluator Assessment...")
        eval_result = self.evaluator.evaluate_with_ensemble(
            content=evolved_facet,
            content_type="technical_sop",
            api_key=self.api_key,
            num_models=num_models + 2
        )

        quality_score = eval_result.consensus_score
        print(f"  ✓ Quality score: {quality_score:.3f}")

        return {
            "facet": facet.value,
            "status": "EVOLVED",
            "original_content": facet_content,
            "evolved_content": evolved_facet,
            "vulnerabilities_found": vulnerabilities_found,
            "fixes_applied": fixes_applied,
            "quality_score": quality_score,
            "validation_results": validation_results,
            "all_validators_passed": all_valid,
            "consensus_reached": eval_result.consensus_reached
        }

    def evolve_entire_sop(
        self,
        sop_content: str,
        facets_to_evolve: Optional[List[SOPFacet]] = None,
        num_models: int = 7
    ) -> Dict[str, Any]:
        """
        Evolve entire SOP facet-by-facet

        Args:
            sop_content: Full SOP content
            facets_to_evolve: List of facets to evolve (default: all)
            num_models: Number of ensemble models

        Returns:
            Dict with evolved SOP and metadata for all facets
        """
        if facets_to_evolve is None:
            facets_to_evolve = list(SOPFacet)

        print(f"\n{'='*70}")
        print(f"SOP EVOLUTION: {len(facets_to_evolve)} facets")
        print(f"{'='*70}\n")

        evolution_results = {
            "timestamp": datetime.now().isoformat(),
            "num_models": num_models,
            "facets": {},
            "overall_status": "SUCCESS"
        }

        evolved_sop = sop_content
        total_vulnerabilities = 0
        total_fixes = 0

        for facet in facets_to_evolve:
            try:
                # Evolve this facet
                result = self.evolve_facet(
                    sop_content=evolved_sop,
                    facet=facet,
                    num_models=num_models
                )

                evolution_results["facets"][facet.value] = result

                # Replace facet in SOP
                if result["status"] == "EVOLVED":
                    evolved_sop = evolved_sop.replace(
                        result["original_content"],
                        result["evolved_content"]
                    )
                    total_vulnerabilities += result["vulnerabilities_found"]
                    total_fixes += result["fixes_applied"]

            except Exception as e:  # TODO: Catch specific exception instead of Exception
                print(f"\n✗ Error evolving {facet.value}: {e}")
                evolution_results["facets"][facet.value] = {
                    "status": "ERROR",
                    "error": str(e)
                }
                evolution_results["overall_status"] = "PARTIAL_SUCCESS"

        evolution_results["evolved_sop"] = evolved_sop
        evolution_results["total_vulnerabilities_found"] = total_vulnerabilities
        evolution_results["total_fixes_applied"] = total_fixes

        print(f"\n{'='*70}")
        print(f"EVOLUTION COMPLETE")
        print(f"{'='*70}\n")
        print(f"Total vulnerabilities found: {total_vulnerabilities}")
        print(f"Total fixes applied: {total_fixes}")
        print(f"Status: {evolution_results['overall_status']}")

        return evolution_results


# ============================================================================
# CONVENIENCE FUNCTIONS FOR EACH FACET
# ============================================================================

def evolve_environmental_conditions(sop_content: str, api_key: str) -> Dict[str, Any]:
    """Evolve Part 0: Environmental Conditions"""
    registry = SOPTemplateRegistry(api_key)
    return registry.evolve_facet(sop_content, SOPFacet.ENVIRONMENTAL)


def evolve_equipment_specifications(sop_content: str, api_key: str) -> Dict[str, Any]:
    """Evolve Part 1: Equipment Specifications"""
    registry = SOPTemplateRegistry(api_key)
    return registry.evolve_facet(sop_content, SOPFacet.EQUIPMENT)


def evolve_materials(sop_content: str, api_key: str) -> Dict[str, Any]:
    """Evolve Part 2: Materials"""
    registry = SOPTemplateRegistry(api_key)
    return registry.evolve_facet(sop_content, SOPFacet.MATERIALS)


def evolve_execution_protocols(sop_content: str, api_key: str) -> Dict[str, Any]:
    """Evolve Part 3: Execution Protocols"""
    registry = SOPTemplateRegistry(api_key)
    return registry.evolve_facet(sop_content, SOPFacet.EXECUTION_PHASES)


def evolve_quality_control(sop_content: str, api_key: str) -> Dict[str, Any]:
    """Evolve Part 4: Quality Control"""
    registry = SOPTemplateRegistry(api_key)
    return registry.evolve_facet(sop_content, SOPFacet.QUALITY_CONTROL)


def evolve_safety_protocols(sop_content: str, api_key: str) -> Dict[str, Any]:
    """Evolve Part 5: Safety Protocols"""
    registry = SOPTemplateRegistry(api_key)
    return registry.evolve_facet(sop_content, SOPFacet.SAFETY)


def evolve_validation_scalability(sop_content: str, api_key: str) -> Dict[str, Any]:
    """Evolve Part 6: Validation and Scalability"""
    registry = SOPTemplateRegistry(api_key)
    return registry.evolve_facet(sop_content, SOPFacet.VALIDATION)
