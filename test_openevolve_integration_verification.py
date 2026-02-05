#!/usr/bin/env python3
"""
OpenEvolve Integration Verification Tests

This script tests that OpenEvolve is properly integrated and functioning
correctly across the entire project.

Run: python test_openevolve_integration_verification.py
"""

import sys
import os
from typing import List, Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")

def print_test(name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = f"{Colors.GREEN}PASS{Colors.ENDC}" if passed else f"{Colors.RED}FAIL{Colors.ENDC}"
    print(f"  {name:50s} [{status}]")
    if details:
        print(f"    -> {details}")

def print_section(title: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{title}{Colors.ENDC}")
    print("-" * 70)

# Test functions

def test_openevolve_import() -> Tuple[bool, str]:
    """Test that OpenEvolve can be imported"""
    try:
        from openevolve.api import run_evolution, evolve_code, evolve_function, evolve_algorithm
        from openevolve.config import Config, LLMModelConfig
        from openevolve._version import __version__
        return True, f"OpenEvolve {__version__} imported successfully"
    except ImportError as e:
        return False, f"Import failed: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

def test_openevolve_version() -> Tuple[bool, str]:
    """Test that correct version (0.2.15) is being used"""
    try:
        from openevolve._version import __version__
        if __version__ == "0.2.15":
            return True, f"Correct version: {__version__}"
        else:
            return False, f"Wrong version: {__version__} (expected 0.2.15)"
    except Exception as e:
        return False, f"Error checking version: {e}"

def test_openevolve_api_functions() -> Tuple[bool, str]:
    """Test that OpenEvolve API functions exist"""
    try:
        from openevolve import api

        functions = [
            'run_evolution',
            'evolve_code',
            'evolve_function',
            'evolve_algorithm',
            'EvolutionResult'
        ]

        missing = []
        for func_name in functions:
            if not hasattr(api, func_name):
                missing.append(func_name)

        if missing:
            return False, f"Missing functions: {', '.join(missing)}"

        return True, f"All {len(functions)} API functions available"
    except Exception as e:
        return False, f"Error: {e}"

def test_openevolve_config_classes() -> Tuple[bool, str]:
    """Test that OpenEvolve config classes exist"""
    try:
        from openevolve import config

        classes = [
            'Config',
            'LLMModelConfig',
            'LLMConfig',
            'PromptConfig',
            'DatabaseConfig',
            'EvaluatorConfig',
            'EvolutionTraceConfig'
        ]

        missing = []
        for class_name in classes:
            if not hasattr(config, class_name):
                missing.append(class_name)

        if missing:
            return False, f"Missing classes: {', '.join(missing)}"

        return True, f"All {len(classes)} config classes available"
    except Exception as e:
        return False, f"Error: {e}"

def test_team_system_logging() -> Tuple[bool, str]:
    """Test that team system files have proper logging setup"""
    files_to_check = [
        'red_team.py',
        'blue_team.py',
        'evaluator_team.py',
        'decomposition_engine.py',
        'decomposition_engine_backup.py',
        'decomposition_mcp_tools.py',
        'openevolve_mcp_tools.py',
        'openevolve_client.py',
        'sovereign_solution_orchestration.py',
        'sovereign_quality_assessment.py',
        'sovereign_refinement.py',
        'sovereign_gauntlets.py',
        'sovereign_knowledge_manager.py',
        'sub_problem_solver.py'
    ]

    issues = []
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            issues.append(f"{filepath} not found")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        has_import = 'import logging' in content
        has_logger = 'logger = logging.getLogger' in content or 'self.logger = logging.getLogger' in content

        if not has_import:
            issues.append(f"{filepath} missing 'import logging'")
        if not has_logger:
            issues.append(f"{filepath} missing logger initialization")

    if issues:
        return False, f"Found {len(issues)} issues: {'; '.join(issues[:3])}"
    else:
        return True, f"All {len(files_to_check)} files have proper logging"

def test_evolution_py_integration() -> Tuple[bool, str]:
    """Test that evolution.py can import openevolve_integration"""
    try:
        # Try to import the integration module
        import openevolve_integration

        # Check for key functions
        functions = [
            'run_unified_evolution',
            'create_specialized_evaluator',
            'create_language_specific_evaluator'
        ]

        missing = []
        for func_name in functions:
            if not hasattr(openevolve_integration, func_name):
                missing.append(func_name)

        if missing:
            return False, f"Missing functions: {', '.join(missing)}"

        return True, "openevolve_integration.py has all required functions"
    except ImportError as e:
        return False, f"Cannot import openevolve_integration: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def test_run_evolution_signature() -> Tuple[bool, str]:
    """Test that run_evolution has the correct signature"""
    try:
        from openevolve.api import run_evolution
        import inspect

        sig = inspect.signature(run_evolution)
        params = list(sig.parameters.keys())

        expected_params = ['initial_program', 'evaluator', 'config', 'iterations', 'output_dir', 'cleanup']

        # Check if key parameters exist
        missing = []
        for param in expected_params:
            if param not in params:
                missing.append(param)

        if missing:
            return False, f"Missing parameters: {', '.join(missing)}"

        return True, f"run_evolution has correct signature ({len(params)} params)"
    except Exception as e:
        return False, f"Error: {e}"

def test_pip_installation() -> Tuple[bool, str]:
    """Test that OpenEvolve is installed correctly"""
    try:
        import subprocess
        result = subprocess.run(
            ['pip', 'show', 'openevolve'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return False, "OpenEvolve not installed via pip"

        output = result.stdout

        # Check for version
        if '0.2.15' not in output:
            return False, "Wrong version installed (not 0.2.15)"

        # Check for editable install
        if 'Editable project location' not in output:
            return False, "Not installed as editable (-e)"

        # Extract location
        for line in output.split('\n'):
            if line.startswith('Editable project location:'):
                location = line.split(':', 1)[1].strip()
                if 'openevolve' in location:
                    return True, f"Correctly installed as editable at {location}"
                else:
                    return False, f"Editable install points to wrong location: {location}"

        return True, "OpenEvolve 0.2.15 installed correctly"
    except Exception as e:
        return False, f"Error checking installation: {e}"

def test_requirements_txt() -> Tuple[bool, str]:
    """Test that requirements.txt references local OpenEvolve"""
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()

        # Check for editable install
        if '-e ./openevolve' in content or '-e ./open.evolve' in content:
            return True, "requirements.txt has editable install reference"
        elif 'openevolve==0.1.0' in content:
            return False, "requirements.txt still references old version 0.1.0"
        elif 'openevolve' in content:
            return False, "requirements.txt has OpenEvolve but not as editable"
        else:
            return False, "OpenEvolve not found in requirements.txt"
    except Exception as e:
        return False, f"Error reading requirements.txt: {e}"

def test_fallback_mechanism() -> Tuple[bool, str]:
    """Test that OPENEVOLVE_AVAILABLE flag works"""
    try:
        # The flag should be set in various modules
        modules_to_check = [
            'evolution',
            'red_team',
            'blue_team',
            'evaluator_team'
        ]

        all_have_flag = True
        for module_name in modules_to_check:
            try:
                module = __import__(module_name)
                if not hasattr(module, 'OPENEVOLVE_AVAILABLE'):
                    all_have_flag = False
                    break
            except ImportError:
                # Module might have other import issues
                pass

        if all_have_flag:
            return True, "OPENEVOLVE_AVAILABLE flag present in modules"
        else:
            return False, "Some modules missing OPENEVOLVE_AVAILABLE flag"
    except Exception as e:
        return False, f"Error: {e}"

# Main test runner

def run_all_tests() -> List[Tuple[str, bool, str]]:
    """Run all tests and return results"""
    tests = [
        ("OpenEvolve Import", test_openevolve_import),
        ("OpenEvolve Version Check", test_openevolve_version),
        ("API Functions Available", test_openevolve_api_functions),
        ("Config Classes Available", test_openevolve_config_classes),
        ("Team System Logging Setup", test_team_system_logging),
        ("evolution.py Integration", test_evolution_py_integration),
        ("run_evolution Signature", test_run_evolution_signature),
        ("Pip Installation Check", test_pip_installation),
        ("requirements.txt Check", test_requirements_txt),
        ("Fallback Mechanism", test_fallback_mechanism),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed, details = test_func()
            results.append((test_name, passed, details))
        except Exception as e:
            results.append((test_name, False, f"Test crashed: {e}"))

    return results

def main():
    """Main entry point"""
    print_header("OpenEvolve Integration Verification Tests")

    print(f"{Colors.BOLD}Testing OpenEvolve integration...{Colors.ENDC}\n")

    results = run_all_tests()

    # Print results
    for test_name, passed, details in results:
        print_test(test_name, passed, details)

    # Summary
    print_section("Test Summary")

    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)

    print(f"\nTotal Tests: {total_count}")
    print(f"Passed: {Colors.GREEN}{passed_count}{Colors.ENDC}")
    print(f"Failed: {Colors.RED}{total_count - passed_count}{Colors.ENDC}")
    print(f"Success Rate: {(passed_count/total_count)*100:.1f}%")

    if passed_count == total_count:
        print(f"\n{Colors.GREEN}{Colors.BOLD}[OK] ALL TESTS PASSED!{Colors.ENDC}")
        print(f"{Colors.GREEN}OpenEvolve is properly integrated.{Colors.ENDC}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}[FAIL] SOME TESTS FAILED{Colors.ENDC}")
        print(f"{Colors.YELLOW}Please review the failures above.{Colors.ENDC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
