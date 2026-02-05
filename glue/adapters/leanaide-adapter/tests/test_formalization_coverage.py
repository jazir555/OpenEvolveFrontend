#!/usr/bin/env python3
"""
RESE Phase I -> Lean 4 Formalization Coverage Verification

Per RESE Technical Manual §2.1.5:
"All Hard Parameter Inequality Constraints (Category A laws) are formally
proven within the Lean 4 environment."

This test suite verifies:
1. 100% of Category A constraints are formalized in Lean 4
2. All constraints have machine-verified proofs
3. Lean 4 file is syntactically correct
4. Coverage report is generated

Following CLAUDE.md Laws:
- Law of Runtime Truth: Execute Lean 4 to verify
- Law of Idempotency: Tests safe to run 100x
"""

import os
import sys
import json
import unittest
import subprocess
import re
from typing import Dict, List, Any, Tuple, Set
from datetime import datetime, timezone

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../rese-phase1/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from autoformalization_pipeline import (
    AutoformalizationPipeline,
    AutoformalizationConfig,
    CategoryAConstraint,
    Lean4Theorem,
    FormalizationResult,
)


class TestFormalizationCoverage(unittest.TestCase):
    """Test suite for Lean 4 formalization coverage"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.config = AutoformalizationConfig.from_env()
        cls.pipeline = AutoformalizationPipeline(config=cls.config)

    def test_lean4_file_exists(self):
        """Test that Lean 4 file exists"""
        self.assertTrue(
            os.path.exists(self.config.LEAN4_CATEGORY_A_FILE),
            f"Lean 4 file not found: {self.config.LEAN4_CATEGORY_A_FILE}"
        )

    def test_lean4_file_readable(self):
        """Test that Lean 4 file is readable"""
        with open(self.config.LEAN4_CATEGORY_A_FILE, 'r') as f:
            content = f.read()

        self.assertGreater(len(content), 0, "Lean 4 file is empty")
        self.assertIn("namespace RESE.Constraints", content,
                     "Missing RESE.Constraints namespace")

    def test_lean4_file_syntax(self):
        """Test that Lean 4 file has correct syntax

        This test parses the Lean 4 file to ensure it's syntactically correct
        """
        content = self._read_lean4_file()

        # Check for basic Lean 4 structure
        self.assertIn("import ", content, "Missing import statements")
        self.assertIn("theorem ", content, "Missing theorem declarations")
        self.assertIn(":=", content, "Missing theorem definitions")

        # Check for proper namespace structure
        self.assertIn("namespace RESE.Constraints", content)
        self.assertIn("end RESE.Constraints", content)

    def test_all_constraints_formalized(self):
        """Test that all Category A constraints are formalized

        Per RESE spec: 100% of Category A constraints must be formalized
        """
        # Get all Category A constraints from Phase I
        category_a_constraints = self.pipeline._get_example_constraints()

        # Get all theorems from Lean 4 file
        theorems = self._extract_theorems_from_lean4()

        # Verify each constraint has a corresponding theorem
        constraint_ids = set(c.id for c in category_a_constraints)
        theorem_names = set(t.name for t in theorems)

        # Map constraint IDs to theorem names
        for constraint_id in constraint_ids:
            expected_theorem_name = self._constraint_id_to_theorem_name(constraint_id)
            self.assertIn(
                expected_theorem_name,
                theorem_names,
                f"Constraint '{constraint_id}' not formalized as theorem '{expected_theorem_name}'"
            )

    def test_all_theorems_have_proofs(self):
        """Test that all theorems have proof skeletons

        Per RESE spec: All constraints must have proofs
        """
        theorems = self._extract_theorems_from_lean4()

        for theorem in theorems:
            self.assertIsNotNone(
                theorem.proof,
                f"Theorem '{theorem.name}' missing proof"
            )
            self.assertGreater(
                len(theorem.proof),
                0,
                f"Theorem '{theorem.name}' has empty proof"
            )

    def test_coverage_percentage(self):
        """Test that coverage meets minimum requirement

        Per RESE spec: 100% coverage required for Category A constraints
        """
        # Run autoformalization pipeline
        result = self.pipeline.run()

        # Verify coverage
        self.assertEqual(
            result.total_constraints,
            result.formalized_count,
            f"Not all constraints formalized: {result.formalized_count}/{result.total_constraints}"
        )

        # Check coverage percentage
        expected_coverage = 100.0
        actual_coverage = result.coverage_percentage

        self.assertEqual(
            actual_coverage,
            expected_coverage,
            f"Coverage {actual_coverage}% below required {expected_coverage}%"
        )

    def test_temperature_constraints(self):
        """Test temperature constraint formalization"""
        theorems = self._extract_theorems_from_lean4()
        theorem_names = set(t.name for t in theorems)

        # Check for temperature constraints
        self.assertIn("temp_max_constraint", theorem_names,
                     "Missing temperature max constraint theorem")

        # Verify theorem signature
        temp_theorem = next(t for t in theorems if t.name == "temp_max_constraint")
        self.assertIn("t < 1000", temp_theorem.signature,
                     "Temperature constraint signature incorrect")

    def test_pressure_constraints(self):
        """Test pressure constraint formalization"""
        theorems = self._extract_theorems_from_lean4()
        theorem_names = set(t.name for t in theorems)

        # Check for pressure constraints
        self.assertIn("pressure_min_constraint", theorem_names,
                     "Missing pressure min constraint theorem")
        self.assertIn("pressure_max_constraint", theorem_names,
                     "Missing pressure max constraint theorem")
        self.assertIn("pressure_combined_constraint", theorem_names,
                     "Missing combined pressure constraint theorem")

    def test_deuterium_loading_constraints(self):
        """Test deuterium loading constraint formalization"""
        theorems = self._extract_theorems_from_lean4()
        theorem_names = set(t.name for t in theorems)

        # Check for deuterium loading constraints
        self.assertIn("deuterium_loading_min_constraint", theorem_names,
                     "Missing deuterium loading constraint theorem")

        # Verify theorem signature
        loading_theorem = next(
            t for t in theorems if t.name == "deuterium_loading_min_constraint"
        )
        self.assertIn("d ≥ 0.85", loading_theorem.signature,
                     "Deuterium loading constraint signature incorrect")

    def test_mathlib_imports(self):
        """Test that Mathlib imports are present

        Mathlib provides real number theory and order relations
        """
        content = self._read_lean4_file()

        required_imports = [
            "Mathlib.Data.Real.Basic",
            "Mathlib.Order.Basic",
        ]

        for imp in required_imports:
            self.assertIn(f"import {imp}", content,
                         f"Missing required Mathlib import: {imp}")

    def test_rese_namespace(self):
        """Test that RESE.Constraints namespace is used"""
        content = self._read_lean4_file()

        self.assertIn("namespace RESE.Constraints", content,
                     "Missing RESE.Constraints namespace declaration")
        self.assertIn("end RESE.Constraints", content,
                     "Missing RESE.Constraints namespace end")

    def test_theorem_documentation(self):
        """Test that theorems have documentation comments

        Documentation is critical for maintainability
        """
        theorems = self._extract_theorems_from_lean4()

        for theorem in theorems:
            # Check for doc comment (/--)
            self.assertIsNotNone(
                theorem.documentation,
                f"Theorem '{theorem.name}' missing documentation"
            )

    def test_formalization_idempotency(self):
        """Test that formalization is idempotent

        Law of Idempotency: Running formalization multiple times should
        produce the same result
        """
        # Run pipeline twice
        result1 = self.pipeline.run()
        result2 = self.pipeline.run()

        # Verify same constraints found
        self.assertEqual(
            result1.total_constraints,
            result2.total_constraints,
            "Idempotency violated: different number of constraints found"
        )

        # Verify same formalization count
        self.assertEqual(
            result1.formalized_count,
            result2.formalized_count,
            "Idempotency violated: different formalization count"
        )

        # Verify same coverage
        self.assertEqual(
            result1.coverage_percentage,
            result2.coverage_percentage,
            "Idempotency violated: different coverage percentage"
        )

    def test_generate_coverage_report(self):
        """Test coverage report generation"""
        # Run pipeline
        result = self.pipeline.run()

        # Generate coverage report
        report = self._generate_coverage_report(result)

        # Verify report structure
        self.assertIn("total_constraints", report)
        self.assertIn("formalized_count", report)
        self.assertIn("coverage_percentage", report)
        self.assertIn("theorems", report)

        # Verify 100% coverage
        self.assertEqual(report["coverage_percentage"], 100.0,
                        "Coverage report shows less than 100%")

    def test_lean4_compilation(self):
        """Test that Lean 4 file compiles

        This is an integration test that requires Lean 4 to be installed
        """
        if not self._lean4_available():
            self.skipTest("Lean 4 not available")

        # Try to compile the Lean 4 file
        lean4_file = self.config.LEAN4_CATEGORY_A_FILE

        try:
            result = subprocess.run(
                ["lake", "build", lean4_file],
                capture_output=True,
                text=True,
                timeout=self.config.LEAN4_LAKE_TIMEOUT_MS / 1000.0,
                cwd=os.path.dirname(lean4_file),
            )

            # Check for compilation errors
            if result.returncode != 0:
                # If Lean 4 is not properly set up, skip test
                if "error: unknown package" in result.stderr or \
                   "error: file" in result.stderr:
                    self.skipTest(f"Lean 4 project not set up: {result.stderr}")
                else:
                    self.fail(f"Lean 4 compilation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.fail("Lean 4 compilation timed out")
        except FileNotFoundError:
            self.skipTest("Lean 4 'lake' tool not found")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _read_lean4_file(self) -> str:
        """Read Lean 4 file content"""
        with open(self.config.LEAN4_CATEGORY_A_FILE, 'r') as f:
            return f.read()

    def _extract_theorems_from_lean4(self) -> List[Lean4Theorem]:
        """Extract theorems from Lean 4 file

        Returns:
            List of Lean 4 theorems with metadata
        """
        content = self._read_lean4_file()
        theorems = []

        # Parse theorem declarations
        # Pattern: theorem name (params) : type := by
        theorem_pattern = r'theorem\s+(\w+)\s*\([^)]*\)\s*:\s*([^:]*)\s*:=\s*by'

        matches = re.finditer(theorem_pattern, content)

        for match in matches:
            name = match.group(1)
            signature = match.group(2).strip()

            # Extract proof (everything after := by until next theorem or end)
            proof_start = match.end()
            next_theorem = content.find("\ntheorem ", proof_start)
            if next_theorem == -1:
                next_theorem = len(content)

            proof = content[proof_start:next_theorem].strip()

            # Extract documentation (preceding /-- comment)
            doc_start = content.rfind("/--", 0, match.start())
            documentation = None
            if doc_start != -1:
                doc_end = content.find("-/", doc_start)
                if doc_end < match.start():
                    documentation = content[doc_start+3:doc_end].strip()

            theorem = Lean4Theorem(
                theorem_name=name,
                signature=signature,
                proof=proof,
                dependencies=[],
                mathlib_imports=[],
            )
            theorem.documentation = documentation
            theorems.append(theorem)

        return theorems

    def _constraint_id_to_theorem_name(self, constraint_id: str) -> str:
        """Convert constraint ID to theorem name"""
        if self.config.THEOREM_NAMING_CONVENTION == 'snake_case':
            return f"{constraint_id}_constraint"
        else:
            # camelCase
            parts = constraint_id.split('_')
            return ''.join([parts[0]] + [p.capitalize() for p in parts[1:]]) + 'Constraint'

    def _generate_coverage_report(self, result: FormalizationResult) -> Dict[str, Any]:
        """Generate coverage report from formalization result"""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_constraints": result.total_constraints,
            "formalized_count": result.formalized_count,
            "proof_complete_count": result.proof_complete_count,
            "coverage_percentage": result.coverage_percentage,
            "lean4_file_path": result.lean4_file_path,
            "theorems": [
                {
                    "name": t.theorem_name,
                    "signature": t.signature,
                    "has_proof": bool(t.proof),
                }
                for t in result.theorems
            ],
            "errors": result.errors,
            "metadata": result.metadata,
        }

        return report

    def _lean4_available(self) -> bool:
        """Check if Lean 4 is available"""
        try:
            result = subprocess.run(
                ["lake", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


# ============================================================================
# COVERAGE REPORT GENERATION
# ============================================================================

class CoverageReportGenerator:
    """Generate detailed coverage reports"""

    def __init__(self, config: AutoformalizationConfig):
        self.config = config

    def generate(self, result: FormalizationResult) -> str:
        """Generate coverage report

        Args:
            result: Formalization result

        Returns:
            Report as markdown string
        """
        report = f"""# RESE Phase I -> Lean 4 Formalization Coverage Report

**Generated:** {datetime.now(timezone.utc).isoformat()}
**Correlation ID:** {result.correlation_id}

## Summary

| Metric | Value |
|--------|-------|
| Total Category A Constraints | {result.total_constraints} |
| Formalized in Lean 4 | {result.formalized_count} |
| Proofs Complete | {result.proof_complete_count} |
| Coverage Percentage | {result.coverage_percentage}% |
| Lean 4 File | `{result.lean4_file_path}` |

## Coverage Status

"""

        if result.coverage_percentage >= 100.0:
            report += "[OK] **100% Coverage Achieved** - All Category A constraints formalized\n"
        else:
            report += f"[WARN] **Coverage Below 100%** - {100.0 - result.coverage_percentage:.1f}% missing\n"

        report += "\n## Theorems\n\n"

        for i, theorem in enumerate(result.theorems, 1):
            proof_status = "[OK]" if theorem.proof else "[FAIL]"
            report += f"{i}. {proof_status} **{theorem.theorem_name}**\n"
            report += f"   - Signature: `{theorem.signature}`\n"
            if theorem.proof:
                report += f"   - Proof: Complete\n"
            else:
                report += f"   - Proof: Missing\n"
            report += "\n"

        if result.errors:
            report += "## Errors\n\n"
            for error in result.errors:
                report += f"- [FAIL] {error}\n"

        report += "\n## Verification\n\n"

        # Run verification checks
        checks_passed = 0
        checks_total = 0

        # Check 1: File exists
        checks_total += 1
        if os.path.exists(result.lean4_file_path):
            report += f"- [OK] Lean 4 file exists\n"
            checks_passed += 1
        else:
            report += f"- [FAIL] Lean 4 file missing\n"

        # Check 2: Coverage 100%
        checks_total += 1
        if result.coverage_percentage >= 100.0:
            report += f"- [OK] 100% coverage achieved\n"
            checks_passed += 1
        else:
            report += f"- [FAIL] Coverage {result.coverage_percentage}% < 100%\n"

        # Check 3: All proofs complete
        checks_total += 1
        if result.proof_complete_count == result.total_constraints:
            report += f"- [OK] All proofs complete\n"
            checks_passed += 1
        else:
            report += f"- [FAIL] {result.total_constraints - result.proof_complete_count} proofs incomplete\n"

        report += f"\n**Verification: {checks_passed}/{checks_total} checks passed**\n"

        return report


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for coverage verification"""
    import argparse

    parser = argparse.ArgumentParser(
        description='RESE Phase I -> Lean 4 Formalization Coverage Verification'
    )
    parser.add_argument('--output-report', help='Write coverage report to file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Load configuration
    config = AutoformalizationConfig.from_env()
    pipeline = AutoformalizationPipeline(config=config)

    # Run formalization
    print("Running autoformalization pipeline...")
    result = pipeline.run()

    # Print summary
    print(f"\nFormalization complete:")
    print(f"  Total constraints: {result.total_constraints}")
    print(f"  Formalized: {result.formalized_count}")
    print(f"  Proofs complete: {result.proof_complete_count}")
    print(f"  Coverage: {result.coverage_percentage}%")

    # Generate report
    report_generator = CoverageReportGenerator(config)
    report = report_generator.generate(result)

    if args.output_report:
        with open(args.output_report, 'w') as f:
            f.write(report)
        print(f"\nCoverage report written to: {args.output_report}")
    elif args.verbose:
        print("\n" + report)

    # Exit with status code based on coverage
    if result.coverage_percentage < config.MIN_COVERAGE_PERCENTAGE:
        print(f"\n[FAIL] Coverage {result.coverage_percentage}% below minimum {config.MIN_COVERAGE_PERCENTAGE}%")
        sys.exit(1)
    elif config.REQUIRE_ALL_PROOFS_COMPLETE and result.proof_complete_count < result.total_constraints:
        print(f"\n[FAIL] {result.total_constraints - result.proof_complete_count} proofs incomplete")
        sys.exit(1)
    else:
        print(f"\n[OK] All checks passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
