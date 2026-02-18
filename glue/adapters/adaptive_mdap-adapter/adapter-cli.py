#!/usr/bin/env python3
"""
Adaptive MDAP/MAKER Adapter CLI

Command-line interface for managing and interacting with the
Adaptive MDAP/MAKER adapter.

Usage:
    python adapter-cli.py analyze --description "Problem description"
    python adapter-cli.py health
    python adapter-cli.py metrics
    python adapter-cli.py dashboard
    python adapter-cli.py validate-config
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from adaptive_mdap_adapter import (
    get_adapter,
    CanonicalSubProblem,
    TaskStatus,
    AdaptiveMDAPAdapterConfig
)

from maker_adapter import (
    get_maker_adapter,
    CanonicalMakerStep
)

from bubblelab_api_client import (
    get_bubblelab_client,
    BubbleLabAPIClientConfig
)


def cmd_analyze(args) -> int:
    """Analyze a problem's complexity."""
    print(f"Analyzing problem: {args.description}")
    print("-" * 60)

    adapter = get_adapter()

    subproblem = CanonicalSubProblem(
        id=f"cli-{args.id or 'analyze'}",
        description=args.description,
        domain=args.domain or "general",
        depth=args.depth or 1,
        dependencies=[],
        metadata={}
    )

    response = adapter.analyze_complexity(subproblem)

    if response.status == TaskStatus.COMPLETED:
        print(f"✓ Complexity Analysis Complete")
        print(f"  Overall Score: {response.complexity_score.overall_score:.2f}")
        print(f"  Text Length: {response.complexity_score.text_length_score:.2f}")
        print(f"  Dependencies: {response.complexity_score.dependency_score:.2f}")
        print(f"  Depth: {response.complexity_score.depth_score:.2f}")
        print(f"  Execution Time: {response.execution_time_ms}ms")

        if args.json:
            result = {
                "status": "completed",
                "complexity_score": response.complexity_score.overall_score,
                "breakdown": {
                    "text_length": response.complexity_score.text_length_score,
                    "dependencies": response.complexity_score.dependency_score,
                    "depth": response.complexity_score.depth_score,
                },
                "execution_time_ms": response.execution_time_ms
            }
            print(json.dumps(result, indent=2))
        return 0
    else:
        print(f"✗ Analysis Failed: {response.error}")
        return 1


def cmd_allocate(args) -> int:
    """Allocate resources based on complexity."""
    print(f"Allocating resources for complexity: {args.complexity}")
    print("-" * 60)

    adapter = get_adapter()

    from adaptive_mdap_adapter import CanonicalComplexityScore
    complexity = CanonicalComplexityScore(overall_score=float(args.complexity))

    response = adapter.allocate_resources(complexity_score=complexity)

    if response.status == TaskStatus.COMPLETED:
        strategy = response.strategy
        print(f"✓ Resource Allocation Complete")
        print(f"  Strategy: {strategy.strategy}")
        print(f"  Agents: {strategy.n_agents}")
        print(f"  K-Ahead: {strategy.k_ahead}")
        print(f"  Max Retries: {strategy.max_retries}")
        print(f"  Timeout: {strategy.timeout_ms}ms")

        if args.json:
            result = {
                "status": "completed",
                "strategy": strategy.strategy,
                "n_agents": strategy.n_agents,
                "k_ahead": strategy.k_ahead,
                "max_retries": strategy.max_retries,
                "timeout_ms": strategy.timeout_ms
            }
            print(json.dumps(result, indent=2))
        return 0
    else:
        print(f"✗ Allocation Failed: {response.error}")
        return 1


def cmd_health(args) -> int:
    """Check adapter health."""
    print("Checking Adapter Health")
    print("-" * 60)

    adapter = get_adapter()
    mdap_health = adapter.health_check()

    print(f"MDAP Adapter:")
    print(f"  Status: {mdap_health['status']}")
    print(f"  Circuit Breaker: {mdap_health['circuit_breaker_state']}")
    print(f"  MDAP Available: {mdap_health['mdap_available']}")
    print(f"  Metrics: {mdap_health['metrics']}")

    maker_adapter = get_maker_adapter()
    maker_health = maker_adapter.health_check()

    print(f"\nMAKER Adapter:")
    print(f"  Status: {maker_health['status']}")
    print(f"  Circuit Breaker: {maker_health['circuit_breaker_state']}")
    print(f"  MAKER Available: {maker_health['maker_available']}")
    print(f"  Metrics: {maker_health['metrics']}")

    # Overall health
    overall_healthy = (
        mdap_health['status'] == 'healthy' and
        maker_health['status'] == 'healthy'
    )

    print(f"\nOverall: {'✓ HEALTHY' if overall_healthy else '✗ DEGRADED'}")

    if args.json:
        result = {
            "mdap_adapter": mdap_health,
            "maker_adapter": maker_health,
            "overall": "healthy" if overall_healthy else "degraded"
        }
        print(json.dumps(result, indent=2))

    return 0 if overall_healthy else 1


def cmd_metrics(args) -> int:
    """Show adapter metrics."""
    print("Adapter Metrics")
    print("-" * 60)

    adapter = get_adapter()
    health = adapter.health_check()
    metrics = health['metrics']

    print(f"MDAP Adapter Metrics:")
    print(f"  Total Requests: {metrics['requests_total']}")
    print(f"  Successful: {metrics['requests_success']}")
    print(f"  Failed: {metrics['requests_failed']}")
    print(f"  Circuit Breaker Trips: {metrics['circuit_breaker_trips']}")

    if args.watch:
        import time
        try:
            while True:
            time.sleep(args.watch_interval)
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Adapter Metrics (Ctrl+C to exit)")
            print("-" * 60)
            cmd_metrics(argparse.Namespace(json=False, watch=False))
        except KeyboardInterrupt:
            print("\nMetrics monitoring stopped.")
    return 0


def cmd_dashboard(args) -> int:
    """Launch monitoring dashboard."""
    print("Launching Monitoring Dashboard...")
    print("-" * 60)

    from monitoring_dashboard import (
        AdapterMonitor,
        DashboardConfig
    )

    monitor = AdapterMonitor(
        config=DashboardConfig(
            refresh_interval_seconds=args.refresh_interval or 5,
            enable_metrics_export=args.export_metrics,
            metrics_export_path=args.export_path
        )
    )

    monitor.run_dashboard_loop(iterations=args.iterations)
    return 0


def cmd_validate_config(args) -> int:
    """Validate adapter configuration."""
    print("Validating Adapter Configuration")
    print("-" * 60)

    errors = []
    warnings = []

    # Check required environment variables
    timeout_ms = os.getenv("ADAPTIVE_MDAP_TIMEOUT_MS")
    if timeout_ms is None:
        errors.append("ADAPTIVE_MDAP_TIMEOUT_MS is not set (REQUIRED)")
    else:
        try:
            timeout = int(timeout_ms)
            if timeout <= 0:
                errors.append(f"ADAPTIVE_MDAP_TIMEOUT_MS must be positive, got {timeout}")
            elif timeout > 300000:
                warnings.append(f"ADAPTIVE_MDAP_TIMEOUT_MS is very high ({timeout}ms)")
        except ValueError:
            errors.append(f"ADAPTIVE_MDAP_TIMEOUT_MS must be an integer, got {timeout_ms}")

    # Check optional variables
    max_retries = os.getenv("ADAPTIVE_MDAP_MAX_RETRIES")
    if max_retries is not None:
        try:
            retries = int(max_retries)
            if retries < 0:
                errors.append(f"ADAPTIVE_MDAP_MAX_RETRIES must be non-negative, got {retries}")
            elif retries > 10:
                warnings.append(f"ADAPTIVE_MDAP_MAX_RETRIES is high ({retries}), may delay failures")
        except ValueError:
            errors.append(f"ADAPTIVE_MDAP_MAX_RETRIES must be an integer, got {max_retries}")

    # Check BubbleLab API config
    bubblelab_url = os.getenv("BUBBLELAB_API_URL")
    if bubblelab_url is not None:
        if not bubblelab_url.startswith(('http://', 'https://')):
            errors.append(f"BUBBLELAB_API_URL must be a valid URL, got {bubblelab_url}")

    # Print results
    if not errors and not warnings:
        print("✓ Configuration is valid")
        return 0

    if warnings:
        print("⚠ Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("✗ Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    return 0


def cmd_bubblelab(args) -> int:
    """Interact with BubbleLab API."""
    print(f"BubbleLab API: {args.action}")
    print("-" * 60)

    try:
        client = get_bubblelab_client()

        if args.action == "status":
            result = client.get_mdap_maker_status()
            print(json.dumps(result, indent=2))
            return 0

        elif args.action == "health":
            health = client.health_check()
            print(json.dumps(health, indent=2))
            return 0 if health['status'] == 'healthy' else 1

        elif args.action == "solve":
            result = client.solve_with_mdap_maker(
                problem_statement=args.problem,
                use_mdap=args.use_mdap,
                num_mdap_agents=args.num_agents
            )
            print(json.dumps(result, indent=2))
            return 0 if result.get('success') else 1

        else:
            print(f"✗ Unknown action: {args.action}")
            return 1

    except Exception as e:
        print(f"✗ BubbleLab API Error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Adaptive MDAP/MAKER Adapter CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a problem
  python adapter-cli.py analyze --description "Build ML pipeline" --domain ml

  # Check health
  python adapter-cli.py health

  # Show metrics
  python adapter-cli.py metrics

  # Launch dashboard
  python adapter-cli.py dashboard --iterations 10

  # Validate configuration
  python adapter-cli.py validate-config

  # BubbleLab API status
  python adapter-cli.py bubblelab status
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze problem complexity')
    analyze_parser.add_argument('--description', required=True, help='Problem description')
    analyze_parser.add_argument('--domain', default='general', help='Problem domain')
    analyze_parser.add_argument('--depth', type=int, default=1, help='Problem depth')
    analyze_parser.add_argument('--id', help='Problem ID')
    analyze_parser.add_argument('--json', action='store_true', help='Output JSON')

    # Allocate command
    allocate_parser = subparsers.add_parser('allocate', help='Allocate resources')
    allocate_parser.add_argument('--complexity', type=float, required=True, help='Complexity score (0-1)')
    allocate_parser.add_argument('--json', action='store_true', help='Output JSON')

    # Health command
    health_parser = subparsers.add_parser('health', help='Check adapter health')
    health_parser.add_argument('--json', action='store_true', help='Output JSON')

    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show adapter metrics')
    metrics_parser.add_argument('--watch', action='store_true', help='Watch mode (continuous)')
    metrics_parser.add_argument('--watch-interval', type=int, default=5, help='Watch interval (seconds)')
    metrics_parser.add_argument('--json', action='store_true', help='Output JSON')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch monitoring dashboard')
    dashboard_parser.add_argument('--refresh-interval', type=int, help='Refresh interval (seconds)')
    dashboard_parser.add_argument('--iterations', type=int, help='Number of iterations')
    dashboard_parser.add_argument('--export-metrics', action='store_true', help='Enable metrics export')
    dashboard_parser.add_argument('--export-path', default='metrics.json', help='Metrics export path')

    # Validate config command
    validate_parser = subparsers.add_parser('validate-config', help='Validate configuration')

    # BubbleLab command
    bubblelab_parser = subparsers.add_parser('bubblelab', help='Interact with BubbleLab API')
    bubblelab_parser.add_argument('action', choices=['status', 'health', 'solve'], help='Action to perform')
    bubblelab_parser.add_argument('--problem', help='Problem statement (for solve)')
    bubblelab_parser.add_argument('--use-mdap', action='store_true', help='Use MDAP (for solve)')
    bubblelab_parser.add_argument('--num-agents', type=int, help='Number of MDAP agents (for solve)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch command
    commands = {
        'analyze': cmd_analyze,
        'allocate': cmd_allocate,
        'health': cmd_health,
        'metrics': cmd_metrics,
        'dashboard': cmd_dashboard,
        'validate-config': cmd_validate_config,
        'bubblelab': cmd_bubblelab,
    }

    command_func = commands.get(args.command)
    if command_func:
        return command_func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
