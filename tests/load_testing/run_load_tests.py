"""
Load Test Runner

Orchestrates the execution of all load tests for the knowledge graph system.

Usage:
    python run_load_tests.py

Or run specific tests:
    python run_load_tests.py --test read_heavy
    python run_load_tests.py --test spike --users 200
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.load_testing.kg_load_tests import KnowledgeGraphLoadTest, LoadTestResult
from tests.load_testing.analyze_results import LoadTestAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('load_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


def print_result(result: LoadTestResult):
    """Print test result."""
    status = "✓ PASSED" if result.passed else "✗ FAILED"
    print(f"\n{status}: {result.test_name}")

    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


async def run_all_tests(engine, config: dict):
    """
    Run all load tests.

    Args:
        engine: Knowledge graph engine instance
        config: Test configuration

    Returns:
        List of test results
    """
    print_header("KNOWLEDGE GRAPH LOAD TESTING")

    load_test = KnowledgeGraphLoadTest(engine)

    # Read-heavy test
    print_header("TEST 1: Read-Heavy Workload")
    await load_test.run_read_heavy_test(
        num_users=config["read_heavy"]["users"][0],
        spawn_rate=config["read_heavy"]["spawn_rate"],
        test_duration=config["read_heavy"]["duration"],
        config=config["read_heavy"]
    )

    await asyncio.sleep(5)  # Cool down

    # Write-heavy test
    print_header("TEST 2: Write-Heavy Workload")
    await load_test.run_write_heavy_test(
        num_users=config["write_heavy"]["users"][0],
        spawn_rate=config["write_heavy"]["spawn_rate"],
        test_duration=config["write_heavy"]["duration"],
        config=config["write_heavy"]
    )

    await asyncio.sleep(5)

    # Spike test
    print_header("TEST 3: Spike Test")
    await load_test.run_spike_test(
        base_users=config["spike_test"]["base_users"],
        spike_users=config["spike_test"]["spike_users"][0],
        spike_duration=config["spike_test"]["spike_duration"],
        config=config["spike_test"]
    )

    await asyncio.sleep(5)

    # Endurance test
    print_header("TEST 4: Endurance Test")
    await load_test.run_endurance_test(
        num_users=config["endurance"]["users"],
        test_duration=config["endurance"]["duration"],
        config=config["endurance"]
    )

    return load_test


async def run_single_test(engine, test_name: str, config: dict):
    """
    Run a single load test.

    Args:
        engine: Knowledge graph engine
        test_name: Name of test to run
        config: Test configuration

    Returns:
        Test result
    """
    print_header(f"LOAD TEST: {test_name}")

    load_test = KnowledgeGraphLoadTest(engine)

    if test_name == "read_heavy":
        result = await load_test.run_read_heavy_test(
            num_users=config["read_heavy"]["users"][0],
            spawn_rate=config["read_heavy"]["spawn_rate"],
            test_duration=config["read_heavy"]["duration"],
            config=config["read_heavy"]
        )
    elif test_name == "write_heavy":
        result = await load_test.run_write_heavy_test(
            num_users=config["write_heavy"]["users"][0],
            spawn_rate=config["write_heavy"]["spawn_rate"],
            test_duration=config["write_heavy"]["duration"],
            config=config["write_heavy"]
        )
    elif test_name == "spike":
        result = await load_test.run_spike_test(
            base_users=config["spike_test"]["base_users"],
            spike_users=config["spike_test"]["spike_users"][0],
            spike_duration=config["spike_test"]["spike_duration"],
            config=config["spike_test"]
        )
    elif test_name == "endurance":
        result = await load_test.run_endurance_test(
            num_users=config["endurance"]["users"],
            test_duration=config["endurance"]["duration"],
            config=config["endurance"]
        )
    else:
        logger.error(f"Unknown test: {test_name}")
        return None

    return load_test


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run load tests for knowledge graph system"
    )
    parser.add_argument(
        "--test",
        choices=["read_heavy", "write_heavy", "spike", "endurance", "all"],
        default="all",
        help="Test to run (default: all)"
    )
    parser.add_argument(
        "--users",
        type=int,
        help="Number of users (overrides config)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Test duration in seconds (overrides config)"
    )
    parser.add_argument(
        "--config",
        default="tests/load_testing/load_test_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        default="load_test_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze results after testing"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}")
        logger.info("Using default configuration")
        config = get_default_config()
    except ImportError:
        logger.warning("PyYAML not installed, using default config")
        config = get_default_config()

    # Override config with command line args
    if args.users:
        if args.test in ["read_heavy", "all"]:
            config["read_heavy"]["users"] = [args.users]
        if args.test in ["write_heavy", "all"]:
            config["write_heavy"]["users"] = [args.users]
        if args.test in ["spike", "all"]:
            config["spike_test"]["spike_users"] = [args.users]
        if args.test in ["endurance", "all"]:
            config["endurance"]["users"] = args.users

    if args.duration:
        if args.test in ["read_heavy", "write_heavy", "all"]:
            config["read_heavy"]["duration"] = args.duration
            config["write_heavy"]["duration"] = args.duration
        if args.test == "spike":
            config["spike_test"]["spike_duration"] = args.duration
        if args.test == "endurance":
            config["endurance"]["duration"] = args.duration

    # Initialize knowledge engine
    try:
        from knowledge_engine.engine import KnowledgeEngine
        engine = KnowledgeEngine()
        logger.info("Knowledge engine initialized")
    except ImportError as e:
        logger.error(f"Failed to import KnowledgeEngine: {e}")
        logger.error("Please ensure knowledge_engine is available")
        return 1
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        return 1

    # Run tests
    try:
        if args.test == "all":
            load_test = await run_all_tests(engine, config)
        else:
            load_test = await run_single_test(engine, args.test, config)

        if not load_test:
            logger.error("Test execution failed")
            return 1

        # Print summary
        print_header("TEST SUMMARY")
        summary = load_test.get_summary()

        print(f"\nTotal Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")

        print("\n" + "-"*60)
        print("Test Results:")
        print("-"*60)

        for test in summary['tests']:
            status = "✓" if test['passed'] else "✗"
            print(f"{status} {test['name']}")
            if test['errors']:
                for error in test['errors']:
                    print(f"  - {error}")

        # Save results
        load_test.save_results(args.output)
        print(f"\nResults saved to: {args.output}")

        # Analyze results
        if args.analyze:
            print_header("RESULT ANALYSIS")
            analyzer = LoadTestAnalyzer(args.output)
            analyzer.generate_report(args.output.replace('.json', '_report.txt'))
            print(f"Report generated: {args.output.replace('.json', '_report.txt')}")

        # Return exit code
        return 0 if summary['failed'] == 0 else 1

    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)
        return 1


def get_default_config() -> dict:
    """Get default configuration."""
    return {
        "read_heavy": {
            "users": [100],
            "spawn_rate": 10,
            "duration": 60,
            "target_throughput": 100,
            "max_error_rate": 0.01
        },
        "write_heavy": {
            "users": [50],
            "spawn_rate": 5,
            "duration": 60,
            "target_throughput": 50,
            "max_error_rate": 0.05
        },
        "spike_test": {
            "base_users": 10,
            "spike_users": [100],
            "spike_duration": 30,
            "max_response_time_degradation": 0.5
        },
        "endurance": {
            "users": 20,
            "duration": 300,
            "max_memory_growth": 0.5,
            "max_performance_degradation": 0.2
        }
    }


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
