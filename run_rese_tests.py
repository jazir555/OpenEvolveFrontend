#!/usr/bin/env python3
"""
RESE Framework Comprehensive Test Runner

Runs all Phase 1 and Phase 2 tests, generates reports, and validates KEY INNOVATIONS.

Usage:
    python run_rese_tests.py --phase all --verbose
    python run_rese_tests.py --phase 1 --coverage
    python run_rese_tests.py --module phi15 --debug

Author: Claude Code (RESE Testing/QA Agent)
Created: 2025-12-31
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


class TestRunner:
    """Test runner for RESE framework"""

    def __init__(self, root_dir: Path, verbose: bool = False):
        self.root_dir = root_dir
        self.verbose = verbose
        self.results = {}
        self.start_time = datetime.now()

    def run_command(self, cmd: List[str], capture: bool = True) -> Tuple[int, str, str]:
        """Run command and return exit code, stdout, stderr"""
        if self.verbose:
            print_info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=self.root_dir,
            capture_output=capture,
            text=True
        )

        return result.returncode, result.stdout, result.stderr

    def run_pytest(
        self,
        test_path: str,
        extra_args: List[str] = None
    ) -> Tuple[int, str, str]:
        """Run pytest with specified arguments"""
        cmd = ["python", "-m", "pytest", test_path, "-v"]

        if extra_args:
            cmd.extend(extra_args)

        if self.verbose:
            cmd.append("-s")
            cmd.append("--tb=long")
        else:
            cmd.append("--tb=short")

        return self.run_command(cmd)

    def parse_pytest_output(self, output: str) -> Dict:
        """Parse pytest output to extract test results"""
        lines = output.split('\n')

        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration": 0.0,
            "failures": []
        }

        for line in lines:
            # Parse summary line
            if " passed" in line or " failed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "passed" in part:
                        results["passed"] = int(parts[i-1])
                    elif "failed" in part:
                        results["failed"] = int(parts[i-1].split('+')[0])
                    elif "skipped" in part:
                        results["skipped"] = int(parts[i-1])
                    elif "error" in part:
                        results["errors"] = int(parts[i-1].split('+')[0])

            # Parse duration
            if "in" in line and any(unit in line for unit in ["s", "ms"]):
                try:
                    duration_str = line.split("in ")[1].split()[0]
                    results["duration"] = float(duration_str)
                except (IndexError, ValueError):
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error in {__name__}", exc_info=True)
                    raise  # Re-raise the exception

        results["total"] = results["passed"] + results["failed"] + results["skipped"]

        return results

    def run_phase1_tests(self) -> Dict:
        """Run Phase 1 tests"""
        print_header("Running Phase 1 Tests")

        phase1_tests = [
            {
                "name": "Φ₁.₅ Tacit Assumption Miner",
                "path": "rese/tests/test_phi15.py"
            },
            {
                "name": "Φ₂ Cognitive Biases",
                "path": "rese/tests/phase1/test_cognitive_biases.py"
            },
            {
                "name": "Φ₂ Integration",
                "path": "rese/tests/phase1/test_phi2_integration.py"
            },
            {
                "name": "Phase 1 Integration",
                "path": "rese/tests/test_integration/test_phase1_integration.py"
            }
        ]

        results = {}

        for test in phase1_tests:
            print_info(f"Running {test['name']}...")

            returncode, stdout, stderr = self.run_pytest(test['path'])

            if returncode == 0:
                print_success(f"{test['name']} passed")
            else:
                print_error(f"{test['name']} failed")

            test_results = self.parse_pytest_output(stdout)
            results[test['name']] = test_results

            # Print summary
            print(f"  Total: {test_results['total']}, "
                  f"Passed: {test_results['passed']}, "
                  f"Failed: {test_results['failed']}, "
                  f"Skipped: {test_results['skipped']}, "
                  f"Duration: {test_results['duration']:.2f}s")

            # Show failures if any
            if test_results['failed'] > 0:
                print_error(f"  {test_results['failed']} test(s) failed")
                # Extract failure details
                for line in stdout.split('\n'):
                    if 'FAILED' in line:
                        print(f"    - {line.strip()}")

        return results

    def run_phase2_tests(self) -> Dict:
        """Run Phase 2 tests"""
        print_header("Running Phase 2 Tests")

        phase2_tests = [
            {
                "name": "I_mech Validator",
                "path": "rese/tests/test_imech/test_validator.py"
            },
            {
                "name": "I_mech Algorithms",
                "path": "rese/tests/test_imech/test_algorithms.py"
            },
            {
                "name": "I_mech Integration",
                "path": "rese/tests/test_imech/test_integration.py"
            },
            {
                "name": "I_mech Transfer",
                "path": "rese/tests/test_imech/test_transfer.py"
            },
            {
                "name": "I_mech Validation",
                "path": "rese/tests/test_imech/test_validation.py"
            },
            {
                "name": "I_mech FDG",
                "path": "rese/tests/test_imech/test_fdg.py"
            },
            {
                "name": "Ψ₃ Constraint Inverter",
                "path": "rese/phase2/psi3/tests/unit/test_constraint_inverter.py"
            },
            {
                "name": "Ψ₂ Ontology Mapper",
                "path": "rese/tests/test_ontology_mapper/test_ontology_mapper.py"
            },
            {
                "name": "Ψ₂ Integration",
                "path": "rese/tests/test_ontology_mapper/test_integration.py"
            }
        ]

        results = {}

        for test in phase2_tests:
            print_info(f"Running {test['name']}...")

            test_path = self.root_dir / test['path']
            if not test_path.exists():
                print_warning(f"  Test file not found: {test['path']}")
                continue

            returncode, stdout, stderr = self.run_pytest(str(test_path))

            if returncode == 0:
                print_success(f"{test['name']} passed")
            else:
                print_error(f"{test['name']} failed")

            test_results = self.parse_pytest_output(stdout)
            results[test['name']] = test_results

            # Print summary
            print(f"  Total: {test_results['total']}, "
                  f"Passed: {test_results['passed']}, "
                  f"Failed: {test_results['failed']}, "
                  f"Skipped: {test_results['skipped']}, "
                  f"Duration: {test_results['duration']:.2f}s")

        return results

    def run_integration_tests(self) -> Dict:
        """Run integration tests"""
        print_header("Running Integration Tests")

        integration_tests = [
            {
                "name": "Phase 1 Integration",
                "path": "rese/tests/test_integration/test_phase1_integration.py"
            },
            {
                "name": "Full Pipeline",
                "path": "rese/tests/test_integration/test_full_pipeline.py"
            }
        ]

        results = {}

        for test in integration_tests:
            print_info(f"Running {test['name']}...")

            test_path = self.root_dir / test['path']
            if not test_path.exists():
                print_warning(f"  Test file not found: {test['path']}")
                continue

            returncode, stdout, stderr = self.run_pytest(str(test_path))

            if returncode == 0:
                print_success(f"{test['name']} passed")
            else:
                print_error(f"{test['name']} failed")

            test_results = self.parse_pytest_output(stdout)
            results[test['name']] = test_results

            print(f"  Total: {test_results['total']}, "
                  f"Passed: {test_results['passed']}, "
                  f"Failed: {test_results['failed']}, "
                  f"Duration: {test_results['duration']:.2f}s")

        return results

    def run_performance_tests(self) -> Dict:
        """Run performance tests"""
        print_header("Running Performance Tests")

        perf_tests = [
            {
                "name": "Φ₁.₅ Performance",
                "path": "rese/tests/test_integration/test_phase1_integration.py::TestPhi15Performance",
                "target_runtime": 60.0  # 60 seconds max
            },
            {
                "name": "I_mech Performance",
                "path": "rese/tests/test_imech/test_integration.py::TestPerformance",
                "target_runtime": 30.0  # 30 seconds max
            }
        ]

        results = {}

        for test in perf_tests:
            print_info(f"Running {test['name']}...")

            returncode, stdout, stderr = self.run_pytest(test['path'])

            test_results = self.parse_pytest_output(stdout)
            results[test['name']] = test_results

            # Check performance
            if test_results['duration'] <= test['target_runtime']:
                print_success(f"{test['name']}: {test_results['duration']:.2f}s "
                            f"(target: {test['target_runtime']}s)")
            else:
                print_error(f"{test['name']}: {test_results['duration']:.2f}s "
                           f"EXCEEDED target {test['target_runtime']}s")

        return results

    def generate_report(self):
        """Generate final test report"""
        print_header("Test Summary Report")

        total_time = (datetime.now() - self.start_time).total_seconds()

        print(f"Total execution time: {total_time:.2f}s")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Calculate totals
        all_results = []

        for phase_results in self.results.values():
            if isinstance(phase_results, dict):
                for test_results in phase_results.values():
                    if isinstance(test_results, dict):
                        all_results.append(test_results)

        total_tests = sum(r.get('total', 0) for r in all_results)
        total_passed = sum(r.get('passed', 0) for r in all_results)
        total_failed = sum(r.get('failed', 0) for r in all_results)
        total_skipped = sum(r.get('skipped', 0) for r in all_results)
        total_duration = sum(r.get('duration', 0.0) for r in all_results)

        print(f"\n{Colors.BOLD}Overall Statistics:{Colors.ENDC}")
        print(f"  Total Tests: {total_tests}")
        print(f"  {Colors.OKGREEN}Passed: {total_passed}{Colors.ENDC}")
        print(f"  {Colors.FAIL}Failed: {total_failed}{Colors.ENDC}")
        print(f"  {Colors.WARNING}Skipped: {total_skipped}{Colors.ENDC}")
        print(f"  Total Duration: {total_duration:.2f}s")

        if total_tests > 0:
            pass_rate = (total_passed / total_tests) * 100
            print(f"  Pass Rate: {pass_rate:.1f}%")

            if pass_rate >= 90:
                print_success("Overall: EXCELLENT")
            elif pass_rate >= 70:
                print_warning("Overall: GOOD")
            else:
                print_error("Overall: NEEDS IMPROVEMENT")

        # Save results to JSON
        report_path = self.root_dir / "rese_test_results.json"
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_time_seconds': total_time,
                'summary': {
                    'total': total_tests,
                    'passed': total_passed,
                    'failed': total_failed,
                    'skipped': total_skipped,
                    'duration': total_duration
                },
                'results': self.results
            }, f, indent=2)

        print(f"\nDetailed results saved to: {report_path}")

    def run_all(self, phase: str = 'all', coverage: bool = False):
        """Run all tests"""
        print_header("RESE Framework Test Suite")
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if phase in ['all', '1']:
            self.results['phase1'] = self.run_phase1_tests()

        if phase in ['all', '2']:
            self.results['phase2'] = self.run_phase2_tests()

        if phase in ['all', 'integration']:
            self.results['integration'] = self.run_integration_tests()

        if coverage:
            print_info("\nRunning coverage report...")
            self.run_coverage()

        self.generate_report()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='RESE Framework Comprehensive Test Runner'
    )
    parser.add_argument(
        '--phase',
        choices=['all', '1', '2', 'integration'],
        default='all',
        help='Phase to test (default: all)'
    )
    parser.add_argument(
        '--module',
        type=str,
        help='Specific module to test (e.g., phi15, imech, psi3)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode (stop on first failure)'
    )

    args = parser.parse_args()

    # Get root directory
    root_dir = Path(__file__).parent

    # Create test runner
    runner = TestRunner(root_dir, verbose=args.verbose)

    if args.debug:
        print_info("Running in debug mode (will stop on first failure)")

    try:
        runner.run_all(phase=args.phase, coverage=args.coverage)
    except KeyboardInterrupt:
        print_warning("\n\nTest execution interrupted by user")
        sys.exit(1)
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print_error(f"\n\nFatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
