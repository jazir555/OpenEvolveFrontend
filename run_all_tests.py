"""
Comprehensive Test Runner for Sovereign System
Runs all tests and generates detailed report
"""

import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple


class TestRunner:
    """Runs and reports on all system tests."""
    
    def __init__(self):
        """Initialize test runner."""
        self.results = {}
        self.total_time = 0
    
    def run_test_suite(self, name: str, test_files: List[str]) -> Tuple[bool, float, str]:
        """
        Run a test suite.
        
        Args:
            name: Suite name
            test_files: List of test files
            
        Returns:
            Tuple of (success, duration, output)
        """
        print(f"\n{'='*60}")
        print(f"Running {name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest"] + test_files + ["-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            # Print output
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            return success, duration, result.stdout
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"[FAIL] Test suite timed out after {duration:.2f}s")
            return False, duration, "TIMEOUT"
        except Exception as e:
            duration = time.time() - start_time
            print(f"[FAIL] Test suite failed: {e}")
            return False, duration, str(e)
    
    def run_all_tests(self) -> bool:
        """
        Run all test suites.
        
        Returns:
            True if all tests passed
        """
        print("\n" + "="*60)
        print("SOVEREIGN SYSTEM - COMPREHENSIVE TEST SUITE")
        print("="*60)
        
        test_suites = {
            "Data Models": ["test_sovereign_data_models.py"],
            "Gauntlets": ["test_sovereign_gauntlets.py"],
            "Team Coordination": ["test_sovereign_team_coordination.py"],
            "Quality Assessment": ["test_sovereign_quality_assessment.py"],
            "Solution Orchestration": ["test_sovereign_solution_orchestration.py"],
            "Knowledge Management": ["test_sovereign_knowledge_manager.py"],
            "Refinement": ["test_sovereign_refinement.py"],
            "UI Components": ["test_sovereign_ui.py"],
            "Performance": ["test_sovereign_performance.py"],
            "Reliability": ["test_sovereign_reliability.py"],
            "Integration": ["test_sovereign_integration.py"],
        }
        
        all_passed = True
        total_start = time.time()
        
        for suite_name, test_files in test_suites.items():
            # Check if files exist
            existing_files = [f for f in test_files if Path(f).exists()]
            if not existing_files:
                print(f"\n⚠ Skipping {suite_name} - files not found")
                continue
            
            success, duration, output = self.run_test_suite(suite_name, existing_files)
            
            self.results[suite_name] = {
                'success': success,
                'duration': duration,
                'output': output
            }
            
            if not success:
                all_passed = False
        
        self.total_time = time.time() - total_start
        
        return all_passed
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in self.results.values() if r['success'])
        failed = len(self.results) - passed
        
        print(f"\nTotal Suites: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total Time: {self.total_time:.2f}s")
        
        print("\nDetailed Results:")
        print("-" * 60)
        
        for suite_name, result in self.results.items():
            status = "[OK] PASS" if result['success'] else "[FAIL] FAIL"
            duration = result['duration']
            print(f"{status:10} {suite_name:30} {duration:6.2f}s")
        
        if failed > 0:
            print("\n" + "="*60)
            print("FAILED SUITES")
            print("="*60)
            for suite_name, result in self.results.items():
                if not result['success']:
                    print(f"\n{suite_name}:")
                    print(result['output'][-500:])  # Last 500 chars
        
        print("\n" + "="*60)
        if failed == 0:
            print("[OK] ALL TESTS PASSED")
        else:
            print(f"[FAIL] {failed} SUITE(S) FAILED")
        print("="*60 + "\n")
    
    def generate_report(self, filename: str = "test_report.txt"):
        """Generate detailed test report file."""
        with open(filename, 'w') as f:
            f.write("SOVEREIGN SYSTEM - TEST REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total Suites: {len(self.results)}\n")
            f.write(f"Total Time: {self.total_time:.2f}s\n\n")
            
            for suite_name, result in self.results.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"{suite_name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Status: {'PASS' if result['success'] else 'FAIL'}\n")
                f.write(f"Duration: {result['duration']:.2f}s\n\n")
                f.write("Output:\n")
                f.write(result['output'])
                f.write("\n\n")
        
        print(f"Detailed report saved to {filename}")


def main():
    """Main entry point."""
    runner = TestRunner()
    
    try:
        all_passed = runner.run_all_tests()
        runner.print_summary()
        runner.generate_report()
        
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
