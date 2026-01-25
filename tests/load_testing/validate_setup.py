"""
Setup Validation Script

Validates that the load testing framework is properly configured
and all dependencies are installed.

Usage:
    python validate_setup.py
"""

import sys
import asyncio
from pathlib import Path
from typing import List, Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text:^70}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}[OK] {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}[FAIL] {text}{Colors.RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}[WARN] {text}{Colors.RESET}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}[INFO] {text}{Colors.RESET}")


def check_python_version() -> bool:
    """Check Python version."""
    print("Checking Python version...")

    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python 3.7+ required, found {version.major}.{version.minor}.{version.micro}")
        return False


def check_imports() -> Tuple[bool, List[str]]:
    """Check required imports."""
    print("\nChecking required imports...")

    required = {
        'asyncio': 'Standard library',
        'json': 'Standard library',
        'pathlib': 'Standard library',
        'dataclasses': 'Standard library',
        'random': 'Standard library',
        'statistics': 'Standard library',
        'datetime': 'Standard library',
        'logging': 'Standard library',
    }

    optional = {
        'locust': 'Load testing framework',
        'psutil': 'System monitoring',
        'yaml': 'Configuration parsing',
        'pytest': 'Testing framework',
    }

    missing_required = []
    missing_optional = []

    # Check required
    for module, description in required.items():
        try:
            __import__(module)
            print_success(f"{module:15s} - {description}")
        except ImportError:
            print_error(f"{module:15s} - MISSING")
            missing_required.append(module)

    # Check optional
    print("\nChecking optional imports...")
    for module, description in optional.items():
        try:
            if module == 'yaml':
                __import__('yaml')
            else:
                __import__(module)
            print_success(f"{module:15s} - {description}")
        except ImportError:
            print_warning(f"{module:15s} - Not installed (optional)")
            missing_optional.append(module)

    return len(missing_required) == 0, missing_optional


def check_files() -> bool:
    """Check required files exist."""
    print("\nChecking required files...")

    required_files = [
        'kg_load_tests.py',
        'locustfile.py',
        'run_load_tests.py',
        'analyze_results.py',
        'load_test_config.yaml',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'example_usage.py',
        'test_load_tests.py',
    ]

    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print_success(f"{file}")
        else:
            print_error(f"{file} - MISSING")
            all_exist = False

    return all_exist


def check_config() -> bool:
    """Check configuration file."""
    print("\nChecking configuration...")

    try:
        import yaml

        with open('load_test_config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        required_sections = [
            'read_heavy',
            'write_heavy',
            'spike_test',
            'endurance',
            'locust'
        ]

        all_valid = True
        for section in required_sections:
            if section in config:
                print_success(f"Config section: {section}")
            else:
                print_error(f"Missing config section: {section}")
                all_valid = False

        return all_valid

    except ImportError:
        print_warning("PyYAML not installed - skipping config validation")
        return True
    except Exception as e:
        print_error(f"Config validation failed: {e}")
        return False


async def check_basic_functionality() -> bool:
    """Check basic functionality."""
    print("\nChecking basic functionality...")

    try:
        # Import framework
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from tests.load_testing.kg_load_tests import KnowledgeGraphLoadTest, LoadTestResult

        print_success("Framework imports successful")

        # Create mock engine
        class MockEngine:
            async def search(self, query, search_type="hybrid"):
                return {"results": []}

            async def add_knowledge(self, source, content, metadata=None):
                return {"id": "test_id"}

            async def get_graph_stats(self):
                return {"nodes": 100, "edges": 200}

        # Create load tester
        load_test = KnowledgeGraphLoadTest(MockEngine())
        print_success("Load tester initialization successful")

        # Test result creation
        result = LoadTestResult(
            test_name="validation_test",
            metrics={"test": True},
            passed=True
        )
        print_success("Result creation successful")

        # Test summary
        summary = load_test.get_summary()
        print_success(f"Summary generation: {summary['total_tests']} tests")

        return True

    except Exception as e:
        print_error(f"Functionality check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results: dict):
    """Print validation summary."""
    print_header("VALIDATION SUMMARY")

    all_passed = True

    if results['python_version']:
        print_success("Python version compatible")
    else:
        print_error("Python version incompatible")
        all_passed = False

    if results['imports'][0]:
        print_success("All required imports available")
    else:
        print_error("Some required imports missing")
        all_passed = False

    if results['files']:
        print_success("All required files present")
    else:
        print_error("Some required files missing")
        all_passed = False

    if results['config']:
        print_success("Configuration valid")
    else:
        print_error("Configuration invalid")
        all_passed = False

    if results['functionality']:
        print_success("Basic functionality working")
    else:
        print_error("Basic functionality broken")
        all_passed = False

    print("\n" + "="*70)

    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}[SUCCESS] ALL CHECKS PASSED - Framework ready!{Colors.RESET}\n")

        if results['imports'][1]:  # Optional imports missing
            print(f"{Colors.YELLOW}Note: Optional imports not installed:{Colors.RESET}")
            for module in results['imports'][1]:
                print(f"  - {module}")
            print(f"\nInstall with: pip install {' '.join(results['imports'][1])}")

        print("\nNext steps:")
        print("  1. Review QUICKSTART.md for usage guide")
        print("  2. Run example: python example_usage.py")
        print("  3. Run tests: pytest test_load_tests.py -v")
        print("  4. Run load test: python run_load_tests.py")

        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}[FAILURE] SOME CHECKS FAILED - Please fix issues above{Colors.RESET}\n")
        return 1


async def main():
    """Main validation routine."""
    print_header("LOAD TESTING FRAMEWORK - SETUP VALIDATION")

    results = {
        'python_version': check_python_version(),
        'imports': check_imports(),
        'files': check_files(),
        'config': check_config(),
        'functionality': await check_basic_functionality()
    }

    return print_summary(results)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
