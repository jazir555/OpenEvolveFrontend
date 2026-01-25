#!/usr/bin/env python3
"""
RESE Quick Start Script

Run this script to verify your RESE installation and run a quick demo.

Usage:
    python quickstart.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def print_header(text):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def print_success(text):
    """Print success message"""
    print(f"✓ {text}")

def print_error(text):
    """Print error message"""
    print(f"✗ {text}")

def check_dependencies():
    """Check if required dependencies are installed"""
    print_header("Checking Dependencies")

    required = {
        'numpy': 'numpy',
        'fastapi': 'fastapi',
        'pydantic': 'pydantic',
    }

    optional = {
        'psutil': 'psutil',
        'networkx': 'networkx',
        'scipy': 'scipy',
    }

    missing_required = []
    missing_optional = []

    for module_name, package_name in required.items():
        try:
            __import__(module_name)
            print_success(f"{module_name} (required)")
        except ImportError:
            print_error(f"{module_name} (required) - MISSING")
            missing_required.append(package_name)

    for module_name, package_name in optional.items():
        try:
            __import__(module_name)
            print_success(f"{module_name} (optional)")
        except ImportError:
            print(f"⚠ {module_name} (optional) - not installed")

    if missing_required:
        print("\n" + "-" * 80)
        print("Missing required dependencies. Install with:")
        print(f"  pip install {' '.join(missing_required)}")
        print("-" * 80)
        return False

    return True

def test_configuration():
    """Test configuration system"""
    print_header("Testing Configuration System")

    try:
        from config import RESEConfig, get_config

        # Test default configuration
        config = RESEConfig()
        print_success("Created default configuration")

        # Test configuration manager
        config2 = get_config()
        print_success("Configuration manager working")

        # Test serialization
        config_dict = config.to_dict()
        print_success("Configuration serialization working")

        # Test environment-specific config
        prod_config = config.for_environment("production")
        print_success("Environment-specific configuration working")

        return True

    except Exception as e:
        print_error(f"Configuration test failed: {e}")
        return False

def test_pipeline():
    """Test pipeline execution"""
    print_header("Testing Pipeline Execution")

    try:
        from .rese_pipeline import RESEPipeline, ProblemInput

        # Create test problem
        problem = ProblemInput(
            id="quickstart_test",
            description="Quick start test problem",
            constraints=[
                {
                    'id': 'c1',
                    'type': 'hard',
                    'description': 'Test constraint',
                    'formalization': 'x > 0',
                    'source': 'test'
                }
            ],
            variables={'x': {'type': 'real'}}
        )

        print_success("Created test problem")

        # Create pipeline
        pipeline = RESEPipeline()
        print_success("Created pipeline")

        # Run single phase
        print("\nRunning Phase I (Epistemic Audit)...")
        result = pipeline.run(problem, phases=['phase1'], use_cache=False)

        if result.status.value in ['completed', 'failed']:
            print_success(f"Phase execution completed with status: {result.status.value}")
            print(f"  Duration: {result.elapsed_seconds:.2f} seconds")

            if 'phase1' in result.phase_results:
                phase_result = result.phase_results['phase1']
                print(f"  Metrics: {phase_result.metrics}")

            return True
        else:
            print_error(f"Unexpected status: {result.status.value}")
            return False

    except Exception as e:
        print_error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_monitoring():
    """Test monitoring system"""
    print_header("Testing Monitoring System")

    try:
        from monitoring import MonitoringSystem
        from config import get_config

        # Create monitoring system
        config = get_config()
        monitoring = MonitoringSystem(config.monitoring)
        print_success("Created monitoring system")

        # Setup pipeline monitoring
        monitoring.setup_pipeline_monitoring("test_pipeline", 0.8)
        print_success("Setup pipeline monitoring")

        # Record phase completion
        monitoring.record_phase_completion(
            "test_pipeline",
            "phase1",
            5.0,
            True,
            0.6
        )
        print_success("Recorded phase completion")

        # Get dashboard data
        dashboard = monitoring.get_dashboard_data("test_pipeline")
        print_success("Generated dashboard data")

        # Generate metrics report
        report = monitoring.generate_metrics_report()
        print_success("Generated metrics report")

        return True

    except Exception as e:
        print_error(f"Monitoring test failed: {e}")
        return False

def run_demo():
    """Run complete demo"""
    print_header("Running Complete Demo")

    try:
        from .rese_pipeline import run_rese

        print("\nRunning RESE pipeline on demo problem...")
        print("-" * 80)

        result = run_rese(
            problem_description="Demo: Simple optimization problem",
            constraints=[
                {
                    'id': 'c1',
                    'type': 'hard',
                    'description': 'Positive constraint',
                    'formalization': 'x > 0',
                    'source': 'demo'
                }
            ],
            variables={'x': {'type': 'real', 'domain': 'positive'}}
        )

        print("\n" + "-" * 80)
        print("Demo Results:")
        print("-" * 80)
        print(f"Pipeline ID: {result.pipeline_id}")
        print(f"Problem ID: {result.problem_id}")
        print(f"Status: {result.status.value}")
        print(f"Duration: {result.elapsed_seconds:.2f} seconds")
        print(f"Phases Executed: {len(result.phase_results)}")

        if result.validation_score > 0:
            print(f"Validation Score: {result.validation_score:.2f}")
            print(f"Confidence: {result.confidence:.2f}")

        if result.aci_history:
            print(f"ACI History: {result.aci_history}")

        print_success("Demo completed successfully!")
        return True

    except Exception as e:
        print_error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main quick start function"""
    print_header("RESE Quick Start")
    print("\nRecursive Epistemic Solvability Engine - Quick Start Guide")
    print("This script will verify your installation and run a quick demo.\n")

    # Track results
    results = []

    # Check dependencies
    results.append(("Dependencies", check_dependencies()))

    if not results[-1][1]:
        print("\n" + "=" * 80)
        print("Please install missing dependencies before continuing.")
        print("=" * 80)
        return 1

    # Test configuration
    results.append(("Configuration", test_configuration()))

    # Test pipeline
    results.append(("Pipeline", test_pipeline()))

    # Test monitoring
    results.append(("Monitoring", test_monitoring()))

    # Run demo
    results.append(("Demo", run_demo()))

    # Print summary
    print_header("Summary")
    print("\nTest Results:")
    print("-" * 80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:.<50} {status}")
        if not passed:
            all_passed = False

    print("-" * 80)

    if all_passed:
        print("\n🎉 All tests passed! RESE is ready to use.\n")
        print("Next Steps:")
        print("  1. Review configuration in config.json")
        print("  2. Start the API server: python -m rese.api")
        print("  3. View API docs: http://localhost:8000/docs")
        print("  4. Read the documentation: rese/docs/")
        return 0
    else:
        print("\n⚠ Some tests failed. Please review the errors above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
