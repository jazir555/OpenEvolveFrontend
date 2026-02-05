"""
Example Usage of Gauntlet Monitoring System

Demonstrates how to integrate the monitoring system with gauntlet execution.

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import time
import random
from glue.adapters.gauntlet_adapter.monitoring import (
    get_metrics_collector,
    get_health_checker,
    get_alerting_engine,
    AlertRule,
    AlertSeverity,
    WebhookNotificationChannel
)


def example_metrics_collection():
    """Example: Collecting and exporting metrics"""
    print("\n=== Metrics Collection Example ===\n")

    metrics = get_metrics_collector()

    # Simulate some gauntlet executions
    domains = ["finance", "science", "web", "general"]

    for i in range(10):
        domain = random.choice(domains)
        passed = random.random() > 0.3  # 70% pass rate
        duration_ms = random.uniform(500, 5000)
        score = random.uniform(0.4, 0.95) if passed else 0.0

        print(f"Execution {i+1}: domain={domain}, passed={passed}, "
              f"duration={duration_ms:.0f}ms, score={score:.2f}")

        metrics.record_execution(
            domain=domain,
            passed=passed,
            duration_ms=duration_ms,
            score=score,
            rounds_completed=3,
            artifact_id=f"artifact_{i}"
        )

        time.sleep(0.1)  # Small delay

    # Record ML metrics
    print("\nRecording ML metrics...")
    metrics.record_optimization_iteration(
        strategy="q_learning",
        iteration=50,
        score=0.87,
        improvement=0.23
    )

    metrics.record_prediction(
        success_probability=0.75,
        confidence=0.80,
        actual_outcome=True,
        domain="finance"
    )

    metrics.record_training_metrics(
        loss=0.123,
        converged=True,
        epoch=100
    )

    # Get metrics summary
    print("\n=== Metrics Summary ===")
    summary = metrics.get_metric_summary()
    print(f"Total executions: {summary['total_executions']}")
    print(f"Pass rate: {summary['global_pass_rate']:.1%}")
    print(f"Uptime: {summary['uptime_seconds']:.1f}s")

    # Get domain-specific stats
    print("\n=== Domain Statistics ===")
    domain_stats = metrics.get_execution_stats()
    for domain, stats in domain_stats.items():
        print(f"{domain}: {stats['pass_rate']:.1%} pass rate, "
              f"{stats['average_duration_ms']:.0f}ms avg duration")

    # Export Prometheus format
    print("\n=== Prometheus Export ===")
    prometheus = metrics.export_prometheus()
    print(prometheus[:500] + "..." if len(prometheus) > 500 else prometheus)

    # Export JSON format
    print("\n=== JSON Export ===")
    json_export = metrics.export_json()
    print(json_export[:300] + "..." if len(json_export) > 300 else json_export)


def example_health_checks():
    """Example: Running health checks"""
    print("\n=== Health Checks Example ===\n")

    health = get_health_checker()

    # Run all health checks
    results = health.check_all()

    print("Health Check Results:")
    for component, result in results.items():
        status_icon = "[OK]" if result.is_healthy() else "[FAIL]"
        print(f"{status_icon} {component}: {result.status.value} - {result.message}")

    # Get overall status
    print(f"\nOverall status: {health.get_overall_status().value}")
    print(f"Is healthy: {health.is_healthy()}")
    print(f"Is ready: {health.is_ready()}")
    print(f"Uptime: {health.get_uptime_seconds():.1f}s")

    # Get comprehensive report
    print("\n=== Full Health Report ===")
    report = health.get_health_report()
    import json
    print(json.dumps(report, indent=2))


def example_alerting():
    """Example: Setting up and evaluating alerts"""
    print("\n=== Alerting Example ===\n")

    alerts = get_alerting_engine()

    # Add a custom alert rule
    def custom_condition(metrics):
        # Trigger if more than 5 failures
        return metrics.get("total_failures", 0) > 5

    alerts.add_rule(AlertRule(
        name="many_failures",
        severity=AlertSeverity.WARNING,
        condition_fn=custom_condition,
        message_template="Too many failures: {value} (threshold: {threshold})",
        threshold=5,
        cooldown_seconds=60
    ))

    print("Registered custom alert rule: many_failures")

    # Get current rules
    rules = alerts.get_rules()
    print(f"\nTotal alert rules: {len(rules)}")
    for rule in rules:
        print(f"  - {rule.name} ({rule.severity.value}): "
              f"{'enabled' if rule.enabled else 'disabled'}")

    # Evaluate alerts
    print("\nEvaluating alerts...")
    triggered = alerts.evaluate()

    if triggered:
        print(f"\nTriggered {len(triggered)} alerts:")
        for alert in triggered:
            print(f"  [{alert.severity.value.upper()}] {alert.name}: {alert.message}")
    else:
        print("No alerts triggered")

    # Get alert statistics
    stats = alerts.get_alert_statistics()
    print(f"\n=== Alert Statistics ===")
    print(f"Total alerts: {stats['total_alerts']}")
    print(f"Active alerts: {stats['active_alerts']}")
    print(f"Active by severity: {stats['active_by_severity']}")

    # Get active alerts
    active = alerts.get_active_alerts()
    if active:
        print(f"\n=== Active Alerts ===")
        for alert in active:
            print(f"  - {alert.name}: {alert.message}")
            print(f"    Severity: {alert.severity.value}")
            print(f"    Age: {alert.age_seconds():.0f}s")


def example_custom_health_check():
    """Example: Adding custom health check"""
    print("\n=== Custom Health Check Example ===\n")

    from glue.adapters.gauntlet_adapter.monitoring import (
        get_health_checker,
        HealthCheckResult,
        HealthStatus
    )

    health = get_health_checker()

    # Define custom check
    def check_database() -> HealthCheckResult:
        """Simulated database health check"""
        import random
        start_time = time.time()

        # Simulate database check
        is_healthy = random.random() > 0.1  # 90% healthy

        duration_ms = (time.time() - start_time) * 1000

        return HealthCheckResult(
            component="database",
            status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
            message="Database connection OK" if is_healthy else "Database connection failed",
            details={
                "connection_pool": "10/20",
                "latency_ms": duration_ms
            },
            duration_ms=duration_ms
        )

    # Register custom check
    health.register_custom_check("database", check_database)
    print("Registered custom health check: database")

    # Run checks
    results = health.check_all()
    if "database" in results:
        result = results["database"]
        print(f"\nDatabase health: {result.status.value}")
        print(f"Message: {result.message}")


def example_notification_channel():
    """Example: Adding webhook notification"""
    print("\n=== Notification Channel Example ===\n")

    from glue.adapters.gauntlet_adapter.monitoring import (
        get_alerting_engine,
        LogNotificationChannel
    )

    alerts = get_alerting_engine()

    # Log channel is already added by default
    # Create a custom log channel to demonstrate
    log_channel = LogNotificationChannel()
    alerts.add_notification_channel(log_channel)

    print("Added log notification channel")

    # You could add a webhook channel:
    # webhook = WebhookNotificationChannel(
    #     url="https://your-webhook-url.com/alerts",
    #     timeout=5
    # )
    # alerts.add_notification_channel(webhook)

    channels = alerts._notification_channels
    print(f"Total notification channels: {len(channels)}")
    for channel in channels:
        print(f"  - {type(channel).__name__}")


def example_complete_workflow():
    """Example: Complete monitoring workflow"""
    print("\n=== Complete Monitoring Workflow ===\n")

    # Initialize components
    metrics = get_metrics_collector()
    health = get_health_checker()
    alerts = get_alerting_engine()

    # 1. Check health before starting
    print("1. Checking system health...")
    if not health.is_ready():
        print("WARNING: System is not ready!")
        return

    print("System is ready, proceeding...")

    # 2. Simulate gauntlet execution
    print("\n2. Running gauntlet execution...")
    domain = "finance"
    passed = True
    duration_ms = 2345.6
    score = 0.87

    metrics.record_execution(
        domain=domain,
        passed=passed,
        duration_ms=duration_ms,
        score=score
    )

    print(f"Execution recorded: domain={domain}, passed={passed}, "
          f"duration={duration_ms}ms, score={score}")

    # 3. Record ML metrics
    print("\n3. Recording ML metrics...")
    metrics.record_prediction(
        success_probability=0.80,
        confidence=0.75,
        actual_outcome=passed,
        domain=domain
    )

    # 4. Check for alerts
    print("\n4. Evaluating alerts...")
    triggered_alerts = alerts.evaluate()

    if triggered_alerts:
        print(f"Triggered {len(triggered_alerts)} alerts:")
        for alert in triggered_alerts:
            print(f"  - [{alert.severity.value}] {alert.message}")
    else:
        print("No alerts triggered")

    # 5. Export metrics
    print("\n5. Exporting metrics...")
    summary = metrics.get_metric_summary()
    print(f"Metrics summary: {summary['total_executions']} executions, "
          f"{summary['global_pass_rate']:.1%} pass rate")

    print("\nWorkflow complete!")


if __name__ == "__main__":
    print("=" * 60)
    print("Gauntlet Monitoring System - Usage Examples")
    print("=" * 60)

    # Run all examples
    example_metrics_collection()
    example_health_checks()
    example_alerting()
    example_custom_health_check()
    example_notification_channel()
    example_complete_workflow()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
