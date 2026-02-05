#!/usr/bin/env python3
"""
Quick Start Script for Gauntlet Monitoring System

This script demonstrates the fastest way to get started with monitoring.

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from glue.adapters.gauntlet_adapter.monitoring import (
    get_metrics_collector,
    get_health_checker,
    get_alerting_engine
)


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_section(text):
    """Print a section header"""
    print(f"\n--- {text} ---\n")


def quick_start():
    """Run quick start demonstration"""
    print_header("Gauntlet Monitoring System - Quick Start")

    # 1. Initialize monitoring components
    print_section("1. Initialize Monitoring Components")
    print("Getting metrics collector, health checker, and alerting engine...")

    metrics = get_metrics_collector()
    health = get_health_checker()
    alerts = get_alerting_engine()

    print("[OK] All components initialized successfully!")

    # 2. Record a gauntlet execution
    print_section("2. Record Gauntlet Execution")
    print("Recording a sample gauntlet execution...")

    metrics.record_execution(
        domain="finance",
        passed=True,
        duration_ms=2345.67,
        score=0.87,
        rounds_completed=3
    )

    print("[OK] Execution recorded!")
    print("  Domain: finance")
    print("  Passed: True")
    print("  Duration: 2345.67ms")
    print("  Score: 0.87")

    # 3. Check system health
    print_section("3. Check System Health")
    print("Running health checks...")

    is_healthy = health.is_healthy()
    is_ready = health.is_ready()

    print(f"[OK] System Healthy: {is_healthy}")
    print(f"[OK] System Ready: {is_ready}")

    # Get health report
    report = health.get_health_report()
    print(f"\nOverall Status: {report['overall_status']}")
    print(f"Uptime: {report['uptime_seconds']:.1f} seconds")

    # 4. View metrics summary
    print_section("4. View Metrics Summary")
    print("Getting metrics summary...")

    summary = metrics.get_metric_summary()
    print(f"[OK] Total Executions: {summary['total_executions']}")
    print(f"[OK] Total Passes: {summary['total_passes']}")
    print(f"[OK] Total Failures: {summary['total_failures']}")
    print(f"[OK] Pass Rate: {summary['global_pass_rate']:.1%}")

    # 5. Export metrics (Prometheus format)
    print_section("5. Export Metrics (Prometheus Format)")
    print("First 500 characters of Prometheus export:\n")

    prometheus = metrics.export_prometheus()
    print(prometheus[:500] + "...\n")

    # 6. Evaluate alerts
    print_section("6. Evaluate Alerts")
    print("Evaluating alert rules...")

    triggered = alerts.evaluate()

    if triggered:
        print(f"⚠ {len(triggered)} alert(s) triggered:")
        for alert in triggered:
            print(f"  - [{alert.severity.value.upper()}] {alert.message}")
    else:
        print("[OK] No alerts triggered - system looking good!")

    # 7. Next steps
    print_section("7. Next Steps")
    print("""
You've successfully completed the quick start! Here's what you can do next:

1. INTEGRATE WITH YOUR GAUNTLET:
   ```python
   from glue.adapters.gauntlet_adapter.monitoring import record_execution

   # After each gauntlet execution
   record_execution(
       domain=your_domain,
       passed=result.passed,
       duration_ms=result.duration_ms,
       score=result.score
   )
   ```

2. SET UP PROMETHEUS:
   - Add to prometheus.yml:
     scrape_configs:
       - job_name: 'gauntlet'
         static_configs:
           - targets: ['localhost:9090']

3. CREATE GRAFANA DASHBOARD:
   - Import GRAFANA_DASHBOARD.json into Grafana
   - Select your Prometheus data source

4. CONFIGURE ALERTS:
   - Edit PROMETHEUS_ALERTS.yml with your thresholds
   - Load into Prometheus: prometheus --config.file=prometheus.yml

5. CUSTOMIZE MONITORING:
   - Add custom health checks
   - Create custom alert rules
   - Set up webhook notifications

For more details, see README.md
For code examples, see example_usage.py
    """)

    print_header("Quick Start Complete!")
    print("For more information, check out the documentation:")
    print("  - README.md: Complete documentation")
    print("  - example_usage.py: Code examples")
    print("  - config.py: Configuration options")


def interactive_demo():
    """Run an interactive demonstration"""
    print_header("Interactive Gauntlet Monitoring Demo")

    metrics = get_metrics_collector()
    health = get_health_checker()
    alerts = get_alerting_engine()

    while True:
        print("\nOptions:")
        print("  1. Record a gauntlet execution")
        print("  2. View metrics summary")
        print("  3. Check system health")
        print("  4. Evaluate alerts")
        print("  5. Export metrics (Prometheus)")
        print("  6. Export metrics (JSON)")
        print("  7. View alert statistics")
        print("  0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "0":
            print("\nGoodbye!")
            break

        elif choice == "1":
            domain = input("Enter domain (finance/science/web): ").strip() or "general"
            passed = input("Passed? (y/n): ").strip().lower() == "y"
            duration = float(input("Duration (ms): ").strip() or "1000")
            score = float(input("Score (0-1): ").strip() or "0.5")

            metrics.record_execution(
                domain=domain,
                passed=passed,
                duration_ms=duration,
                score=score
            )
            print("\n[OK] Execution recorded!")

        elif choice == "2":
            summary = metrics.get_metric_summary()
            print(f"\nTotal Executions: {summary['total_executions']}")
            print(f"Pass Rate: {summary['global_pass_rate']:.1%}")
            print(f"Uptime: {summary['uptime_seconds']:.1f}s")

            domain_stats = metrics.get_execution_stats()
            if domain_stats:
                print("\nDomain Statistics:")
                for domain, stats in domain_stats.items():
                    print(f"  {domain}: {stats['pass_rate']:.1%} pass rate")

        elif choice == "3":
            is_healthy = health.is_healthy()
            is_ready = health.is_ready()
            print(f"\nHealthy: {is_healthy}")
            print(f"Ready: {is_ready}")

        elif choice == "4":
            triggered = alerts.evaluate()
            if triggered:
                print(f"\n{len(triggered)} alert(s) triggered:")
                for alert in triggered:
                    print(f"  - {alert.message}")
            else:
                print("\n[OK] No alerts triggered")

        elif choice == "5":
            prometheus = metrics.export_prometheus()
            print("\n" + prometheus[:500] + "...")

        elif choice == "6":
            json_export = metrics.export_json()
            print("\n" + json_export[:500] + "...")

        elif choice == "7":
            stats = alerts.get_alert_statistics()
            print(f"\nTotal Alerts: {stats['total_alerts']}")
            print(f"Active: {stats['active_alerts']}")
            print(f"Rules: {stats['total_rules']}")

        else:
            print("\nInvalid choice")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick start for Gauntlet Monitoring System"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive demo"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_demo()
    else:
        quick_start()
