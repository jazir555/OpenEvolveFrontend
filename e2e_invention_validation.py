"""
COMPLETE END-TO-END VALIDATION OF RESE-E2E INVENTION SYSTEM

This script executes a full end-to-end test with a real invention example:
High-temperature superconducting wire with specific constraints.

Author: Agent Z1 (Integration Specialist)
Created: 2026-01-01
Status: PRODUCTION VALIDATION
"""

import sys
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('e2e_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import RESE pipeline
from rese_pipeline import (
    RESEPipeline,
    ProblemInput,
    PipelineStatus,
    PhaseStatus
)
from config import get_config


# =============================================================================
# Real Invention Test Case
# =============================================================================

INVENTION_PROMPT = """
Create a plan to invent high-temperature superconducting wire with:
- Critical temperature: 77 K or higher
- Current density: 10^6 A/cm² or higher
- Wire length: 10 meters
- Must use standard lab equipment
"""

INVENTION_CONSTRAINTS = [
    {
        'id': 'tc_constraint',
        'type': 'hard',
        'description': 'Critical temperature must be ≥ 77 K (liquid nitrogen temperature)',
        'formalization': 'Tc ≥ 77 K',
        'source': 'user'
    },
    {
        'id': 'jc_constraint',
        'type': 'hard',
        'description': 'Current density must be ≥ 10^6 A/cm²',
        'formalization': 'Jc ≥ 10^6 A/cm²',
        'source': 'user'
    },
    {
        'id': 'length_constraint',
        'type': 'hard',
        'description': 'Wire must be at least 10 meters in length',
        'formalization': 'L ≥ 10 m',
        'source': 'user'
    },
    {
        'id': 'equipment_constraint',
        'type': 'soft',
        'description': 'Must use standard lab equipment (no specialized fabrication)',
        'formalization': 'Equipment ∈ StandardLabSet',
        'source': 'user'
    }
]

INVENTION_VARIABLES = {
    'material_type': 'unknown',
    'critical_temperature': 0.0,  # Target: ≥ 77 K
    'current_density': 0.0,  # Target: ≥ 10^6 A/cm²
    'wire_length': 0.0,  # Target: ≥ 10 m
    'fabrication_method': 'unknown',
    'equipment_availability': 'standard'
}


# =============================================================================
# Validation Results Tracker
# =============================================================================

class ValidationResult:
    """Track validation results for each stage"""

    def __init__(self):
        self.stage_results = {}
        self.metrics = {}
        self.errors = []
        self.warnings = []
        self.start_time = datetime.now()
        self.end_time = None

    def add_stage_result(self, stage_name: str, result: Any):
        """Add result for a stage"""
        self.stage_results[stage_name] = {
            'timestamp': datetime.now().isoformat(),
            'result': result
        }

    def add_metric(self, key: str, value: Any):
        """Add performance metric"""
        self.metrics[key] = value

    def add_error(self, error: str):
        """Add error"""
        self.errors.append(error)

    def add_warning(self, warning: str):
        """Add warning"""
        self.warnings.append(warning)

    def finalize(self):
        """Finalize validation"""
        self.end_time = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'stage_results': self.stage_results,
            'metrics': self.metrics,
            'errors': self.errors,
            'warnings': self.warnings,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time else 0
            )
        }


# =============================================================================
# Stage Validators
# =============================================================================

def validate_stage1_output(output: Dict) -> tuple[bool, str]:
    """
    Validate Stage 1: Prompt Analysis

    Checks:
    - Constraints extracted
    - Assumptions identified
    - SCE state initialized
    """
    try:
        checks = []

        # Check constraints
        if 'constraints' in output:
            num_constraints = len(output['constraints'])
            checks.append(f"[OK] Extracted {num_constraints} constraints")
        else:
            return False, "[FAIL] No constraints extracted"

        # Check assumptions
        if 'assumptions' in output:
            num_assumptions = len(output['assumptions'])
            checks.append(f"[OK] Identified {num_assumptions} assumptions")
        else:
            checks.append("[WARN] No assumptions identified (optional)")

        # Check bias detection
        if 'bias_report' in output:
            checks.append("[OK] Bias analysis performed")
        else:
            checks.append("[WARN] No bias report (optional)")

        return True, " | ".join(checks)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return False, f"[ERROR] Validation error: {e}"


def validate_stage2_output(output: Dict) -> tuple[bool, str]:
    """
    Validate Stage 2: Knowledge Retrieval & Isomorphic Mapping

    Checks:
    - Domain pairs identified
    - Ontology mappings created
    - Isomorphism score calculated
    """
    try:
        checks = []

        # Check inverted constraints
        if 'inverted_constraints' in output:
            checks.append("[OK] Constraint inversion performed")
        else:
            checks.append("[WARN] No inverted constraints (optional)")

        # Check ontology mappings
        if 'ontology_mappings' in output:
            num_mappings = len(output['ontology_mappings'])
            checks.append(f"[OK] Created {num_mappings} ontology mappings")
        else:
            return False, "[FAIL] No ontology mappings"

        # Check isomorphism score
        if 'isomorphism_score' in output:
            score = output['isomorphism_score']
            checks.append(f"[OK] Isomorphism score: {score:.3f}")
        else:
            return False, "[FAIL] No isomorphism score"

        return True, " | ".join(checks)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return False, f"[ERROR] Validation error: {e}"


def validate_stage3_output(output: Dict) -> tuple[bool, str]:
    """
    Validate Stage 3: Decomposition with MCTS Search

    Checks:
    - ACI analysis performed
    - MCTS iterations executed
    - Convergence achieved
    """
    try:
        checks = []

        # Check ACI value
        if 'aci_value' in output:
            aci = output['aci_value']
            checks.append(f"[OK] ACI value: {aci:.3f}")
        else:
            return False, "[FAIL] No ACI value"

        # Check MCTS iterations
        if 'mcts_iterations' in output:
            iterations = output['mcts_iterations']
            checks.append(f"[OK] MCTS iterations: {iterations}")
        else:
            return False, "[FAIL] No MCTS iterations"

        # Check convergence
        if 'converged' in output:
            converged = output['converged']
            status = "converged" if converged else "max iterations"
            checks.append(f"[OK] Search status: {status}")
        else:
            checks.append("[WARN] No convergence status")

        return True, " | ".join(checks)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return False, f"[ERROR] Validation error: {e}"


def validate_stage4_output(output: Dict) -> tuple[bool, str]:
    """
    Validate Stage 4: Math Formalization

    Checks:
    - Architecture assembled
    - Predictions generated
    - Validation performed
    """
    try:
        checks = []

        # Check architecture
        if 'architecture' in output:
            checks.append("[OK] Architecture assembled")
        else:
            checks.append("[WARN] No architecture (optional for E2E)")

        # Check predictions
        if 'predictions' in output:
            checks.append("[OK] Predictions generated")
        else:
            checks.append("[WARN] No predictions (optional for E2E)")

        # Check validation
        if 'validation' in output:
            validation = output['validation']
            if 'is_valid' in validation:
                status = "valid" if validation['is_valid'] else "invalid"
                checks.append(f"[OK] Validation: {status}")
            if 'score' in validation:
                score = validation['score']
                checks.append(f"[OK] Validation score: {score:.3f}")
        else:
            return False, "[FAIL] No validation results"

        return True, " | ".join(checks)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return False, f"[ERROR] Validation error: {e}"


# =============================================================================
# Main Validation Function
# =============================================================================

def run_complete_e2e_validation() -> ValidationResult:
    """
    Execute complete end-to-end validation with real invention example.

    Returns:
        ValidationResult with all results
    """
    logger.info("=" * 80)
    logger.info("RESE-E2E COMPLETE END-TO-END VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Test Case: High-Temperature Superconducting Wire")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("")

    validation_result = ValidationResult()

    # Track memory and performance
    import psutil
    process = psutil.Process()

    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    validation_result.add_metric('initial_memory_mb', initial_memory)

    pipeline_start = time.time()

    try:
        # =====================================================================
        # Initialize Pipeline
        # =====================================================================
        logger.info("[INIT] Initializing RESE Pipeline...")
        config = get_config()
        pipeline = RESEPipeline(config)
        logger.info("[INIT] Pipeline initialized successfully")

        # =====================================================================
        # Create Problem Input
        # =====================================================================
        logger.info("[INPUT] Creating problem input...")
        problem = ProblemInput(
            id="hts_wire_invention",
            description=INVENTION_PROMPT.strip(),
            constraints=INVENTION_CONSTRAINTS,
            variables=INVENTION_VARIABLES.copy(),
            objective="Design high-temperature superconducting wire meeting all constraints",
            domain="materials_science/physics"
        )
        logger.info(f"[INPUT] Problem ID: {problem.id}")
        logger.info(f"[INPUT] Constraints: {len(problem.constraints)}")
        logger.info("")

        # =====================================================================
        # Execute Complete Pipeline
        # =====================================================================
        logger.info("[PIPELINE] Executing complete RESE pipeline...")
        logger.info("-" * 80)

        result = pipeline.run(
            problem=problem,
            phases=['phase1', 'phase2', 'phase3', 'phase4'],
            use_cache=False  # Don't use cache for validation
        )

        pipeline_elapsed = time.time() - pipeline_start
        validation_result.add_metric('pipeline_execution_time', pipeline_elapsed)

        logger.info("-" * 80)
        logger.info(f"[PIPELINE] Execution completed in {pipeline_elapsed:.2f}s")
        logger.info(f"[PIPELINE] Status: {result.status.value}")
        logger.info("")

        # =====================================================================
        # Validate Each Stage
        # =====================================================================
        logger.info("[VALIDATION] Validating stage outputs...")
        logger.info("")

        # Stage 1 Validation
        if 'phase1' in result.phase_results:
            phase1_result = result.phase_results['phase1']
            logger.info("Stage 1: Epistemic Audit")
            logger.info(f"  Status: {phase1_result.status.value}")
            logger.info(f"  Time: {phase1_result.elapsed_seconds:.2f}s")

            if phase1_result.status == PhaseStatus.COMPLETED:
                valid, message = validate_stage1_output(phase1_result.output)
                logger.info(f"  Validation: {message}")
                validation_result.add_stage_result('stage1', {
                    'valid': valid,
                    'message': message,
                    'output': phase1_result.output
                })
            else:
                error_msg = f"Stage 1 failed: {phase1_result.errors}"
                logger.error(f"  {error_msg}")
                validation_result.add_error(error_msg)

            logger.info("")

        # Stage 2 Validation
        if 'phase2' in result.phase_results:
            phase2_result = result.phase_results['phase2']
            logger.info("Stage 2: Isomorphic Resonance")
            logger.info(f"  Status: {phase2_result.status.value}")
            logger.info(f"  Time: {phase2_result.elapsed_seconds:.2f}s")

            if phase2_result.status == PhaseStatus.COMPLETED:
                valid, message = validate_stage2_output(phase2_result.output)
                logger.info(f"  Validation: {message}")
                validation_result.add_stage_result('stage2', {
                    'valid': valid,
                    'message': message,
                    'output': phase2_result.output
                })
            else:
                error_msg = f"Stage 2 failed: {phase2_result.errors}"
                logger.error(f"  {error_msg}")
                validation_result.add_error(error_msg)

            logger.info("")

        # Stage 3 Validation
        if 'phase3' in result.phase_results:
            phase3_result = result.phase_results['phase3']
            logger.info("Stage 3: Monte Carlo Refinement")
            logger.info(f"  Status: {phase3_result.status.value}")
            logger.info(f"  Time: {phase3_result.elapsed_seconds:.2f}s")

            if phase3_result.status == PhaseStatus.COMPLETED:
                valid, message = validate_stage3_output(phase3_result.output)
                logger.info(f"  Validation: {message}")
                validation_result.add_stage_result('stage3', {
                    'valid': valid,
                    'message': message,
                    'output': phase3_result.output
                })
            else:
                error_msg = f"Stage 3 failed: {phase3_result.errors}"
                logger.error(f"  {error_msg}")
                validation_result.add_error(error_msg)

            logger.info("")

        # Stage 4 Validation
        if 'phase4' in result.phase_results:
            phase4_result = result.phase_results['phase4']
            logger.info("Stage 4: Architectural Synthesis")
            logger.info(f"  Status: {phase4_result.status.value}")
            logger.info(f"  Time: {phase4_result.elapsed_seconds:.2f}s")

            if phase4_result.status == PhaseStatus.COMPLETED:
                valid, message = validate_stage4_output(phase4_result.output)
                logger.info(f"  Validation: {message}")
                validation_result.add_stage_result('stage4', {
                    'valid': valid,
                    'message': message,
                    'output': phase4_result.output
                })
            else:
                error_msg = f"Stage 4 failed: {phase4_result.errors}"
                logger.error(f"  {error_msg}")
                validation_result.add_error(error_msg)

            logger.info("")

        # =====================================================================
        # Final Results
        # =====================================================================
        logger.info("=" * 80)
        logger.info("FINAL VALIDATION RESULTS")
        logger.info("=" * 80)

        # Overall status
        overall_success = result.status == PipelineStatus.COMPLETED
        logger.info(f"Overall Status: {'[SUCCESS]' if overall_success else '[FAILED]'}")
        logger.info(f"Total Time: {result.elapsed_seconds:.2f}s")
        logger.info("")

        # ACI History
        if result.aci_history:
            logger.info("ACI Reduction History:")
            for i, aci in enumerate(result.aci_history):
                logger.info(f"  Stage {i+1}: ACI = {aci:.3f}")

            if len(result.aci_history) > 1:
                initial_aci = result.aci_history[0]
                final_aci = result.aci_history[-1]
                reduction = (initial_aci - final_aci) / initial_aci * 100
                logger.info(f"  Total Reduction: {reduction:.1f}%")
            logger.info("")

        # Validation Score
        if result.validation_score > 0:
            logger.info(f"Validation Score: {result.validation_score:.3f}")
        if result.confidence > 0:
            logger.info(f"Confidence: {result.confidence:.3f}")
        logger.info("")

        # Performance Metrics
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory
        validation_result.add_metric('final_memory_mb', final_memory)
        validation_result.add_metric('memory_delta_mb', memory_delta)
        validation_result.add_metric('cpu_percent', process.cpu_percent())

        logger.info("Performance Metrics:")
        logger.info(f"  Initial Memory: {initial_memory:.2f} MB")
        logger.info(f"  Final Memory: {final_memory:.2f} MB")
        logger.info(f"  Memory Delta: {memory_delta:+.2f} MB")
        logger.info(f"  CPU Usage: {process.cpu_percent():.1f}%")
        logger.info("")

        # Stage Summary
        logger.info("Stage Summary:")
        for phase_name, phase_result in result.phase_results.items():
            status_symbol = '[OK]' if phase_result.status == PhaseStatus.COMPLETED else '[FAIL]'
            logger.info(f"  {status_symbol} {phase_name}: {phase_result.status.value} ({phase_result.elapsed_seconds:.2f}s)")
        logger.info("")

        # =====================================================================
        # Generate Final Output
        # =====================================================================
        if result.final_solution:
            logger.info("Final Solution Generated:")
            logger.info(f"  Components: {len(result.final_solution) if isinstance(result.final_solution, list) else 'N/A'}")

        # Finalize validation
        validation_result.finalize()

        # Check for critical failures
        if validation_result.errors:
            logger.warning('[WARN] Warnings/Errors detected:')
            for error in validation_result.errors:
                logger.warning(f"  - {error}")

        # Final verdict
        all_stages_valid = all(
            r.get('result', {}).get('valid', False)
            for r in validation_result.stage_results.values()
        )

        logger.info("")
        logger.info("=" * 80)
        if overall_success and all_stages_valid:
            logger.info('[SUCCESS] COMPLETE E2E VALIDATION: SUCCESS')
        else:
            logger.info('[FAILED] COMPLETE E2E VALIDATION: FAILED')
        logger.info("=" * 80)

    except (RuntimeError, ValueError, TypeError, KeyError) as e:
        logger.error(f"CRITICAL ERROR: {e}")
        logger.error(traceback.format_exc())
        validation_result.add_error(f"Critical error: {e}")
        validation_result.finalize()
        return validation_result

    return validation_result


# =============================================================================
# Report Generation
# =============================================================================

def generate_validation_report(result: ValidationResult) -> str:
    """
    Generate comprehensive validation report.

    Args:
        result: ValidationResult

    Returns:
        Report as formatted string
    """
    report = []
    report.append("=" * 80)
    report.append("RESE-E2E END-TO-END VALIDATION REPORT")
    report.append("=" * 80)
    report.append("")
    report.append(f"Test Case: High-Temperature Superconducting Wire Invention")
    report.append(f"Execution Time: {result.start_time.isoformat()}")
    duration = (
        (result.end_time - result.start_time).total_seconds()
        if result.end_time else 0
    )
    report.append(f"Duration: {duration:.2f} seconds")
    report.append("")

    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 80)

    # Check validation status - stage_results has nested structure
    all_valid = all(
        r.get('result', {}).get('valid', False)
        for r in result.stage_results.values()
    )
    report.append(f"Overall Status: {'PASS' if all_valid else 'FAIL'}")
    report.append(f"Stages Validated: {len(result.stage_results)}")
    report.append(f"Errors: {len(result.errors)}")
    report.append(f"Warnings: {len(result.warnings)}")
    report.append("")

    # Stage Results
    report.append("STAGE VALIDATION RESULTS")
    report.append("-" * 80)

    stage_names = {
        'stage1': 'Stage 1: Epistemic Audit',
        'stage2': 'Stage 2: Isomorphic Resonance',
        'stage3': 'Stage 3: Monte Carlo Refinement',
        'stage4': 'Stage 4: Architectural Synthesis'
    }

    for stage_id, stage_data in result.stage_results.items():
        stage_name = stage_names.get(stage_id, stage_id)
        stage_result = stage_data.get('result', {})
        status = "PASS" if stage_result.get('valid', False) else "FAIL"
        message = stage_result.get('message', 'N/A')
        report.append(f"\n{stage_name}")
        report.append(f"  Status: {status}")
        report.append(f"  Result: {message}")
    report.append("")

    # Performance Metrics
    report.append("PERFORMANCE METRICS")
    report.append("-" * 80)
    for key, value in result.metrics.items():
        if isinstance(value, float):
            report.append(f"  {key}: {value:.3f}")
        else:
            report.append(f"  {key}: {value}")
    report.append("")

    # Errors and Warnings
    if result.errors:
        report.append("ERRORS")
        report.append("-" * 80)
        for i, error in enumerate(result.errors, 1):
            report.append(f"  {i}. {error}")
        report.append("")

    if result.warnings:
        report.append("WARNINGS")
        report.append("-" * 80)
        for i, warning in enumerate(result.warnings, 1):
            report.append(f"  {i}. {warning}")
        report.append("")

    # Conclusion
    report.append("=" * 80)
    if all_valid and not result.errors:
        report.append("CONCLUSION: ALL VALIDATIONS PASSED")
        report.append("The RESE-E2E system successfully processed the complete invention pipeline.")
    else:
        report.append("CONCLUSION: VALIDATION FAILED")
        report.append("Some validations did not pass. Review errors above.")
    report.append("=" * 80)

    return "\n".join(report)


def save_report(report: str, result: ValidationResult):
    """Save report to file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save text report
    report_file = Path(f'e2e_validation_report_{timestamp}.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report saved: {report_file}")

    # Save JSON report
    json_file = Path(f'e2e_validation_report_{timestamp}.json')
    with open(json_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    logger.info(f"JSON report saved: {json_file}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point"""
    print("""
================================================================================
                    RESE-E2E COMPLETE END-TO-END VALIDATION
                         SYSTEM - PRODUCTION RUN
                  Test: High-Temperature Superconducting Wire
================================================================================
    """)

    # Run validation
    result = run_complete_e2e_validation()

    # Generate report
    report = generate_validation_report(result)
    print("\n" + report)

    # Save report
    save_report(report, result)

    # Return exit code
    all_valid = all(r.get('valid', False) for r in result.stage_results.values())
    return 0 if all_valid and not result.errors else 1


if __name__ == '__main__':
    sys.exit(main())
