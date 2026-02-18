"""
Adapter Monitoring Dashboard

Provides real-time monitoring and metrics for the Adaptive MDAP/MAKER adapter.
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adaptive_mdap_adapter import (
    AdaptiveMDAPAdapter,
    AdaptiveMDAPAdapterConfig,
    get_adapter,
    TaskStatus
)

from maker_adapter import (
    MakerAdapter,
    get_maker_adapter
)


@dataclass
class AdapterMetrics:
    """Metrics for a single adapter."""
    adapter_name: str
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    requests_rate_per_second: float = 0.0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    circuit_breaker_state: str = "closed"
    circuit_breaker_trips: int = 0
    uptime_seconds: float = 0.0
    last_request_time: Optional[str] = None
    health_status: str = "unknown"


@dataclass
class DashboardConfig:
    """Configuration for the monitoring dashboard."""
    refresh_interval_seconds: int = 5
    history_size: int = 100
    enable_metrics_export: bool = True
    metrics_export_path: Optional[str] = None


class AdapterMonitor:
    """
    Monitor for tracking adapter metrics and health.

    Collects metrics from adapters and provides dashboards/alerts.
    """

    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize the adapter monitor."""
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger("AdapterMonitor")
        self.mdap_adapter: Optional[AdaptiveMDAPAdapter] = None
        self.maker_adapter: Optional[MakerAdapter] = None

        # Metrics history
        self.metrics_history: List[Dict[str, Any]] = []
        self.latency_samples: List[float] = []

        # Start time
        self.start_time = time.time()

    def get_mdap_adapter(self) -> AdaptiveMDAPAdapter:
        """Get or create MDAP adapter."""
        if self.mdap_adapter is None:
            self.mdap_adapter = get_adapter()
        return self.mdap_adapter

    def get_maker_adapter(self) -> MakerAdapter:
        """Get or create MAKER adapter."""
        if self.maker_adapter is None:
            self.maker_adapter = get_maker_adapter()
        return self.maker_adapter

    def collect_metrics(self) -> Dict[str, AdapterMetrics]:
        """
        Collect metrics from all adapters.

        Returns:
            Dictionary mapping adapter names to their metrics
        """
        metrics = {}

        # Collect MDAP adapter metrics
        try:
            mdap_adapter = self.get_mdap_adapter()
            mdap_health = mdap_adapter.health_check()

            mdap_metrics = AdapterMetrics(
                adapter_name="adaptive_mdap",
                requests_total=mdap_health['metrics']['requests_total'],
                requests_success=mdap_health['metrics']['requests_success'],
                requests_failed=mdap_health['metrics']['requests_failed'],
                circuit_breaker_state=mdap_health['circuit_breaker_state'],
                circuit_breaker_trips=mdap_health['metrics']['circuit_breaker_trips'],
                uptime_seconds=time.time() - self.start_time,
                health_status=mdap_health['status']
            )
            metrics["adaptive_mdap"] = mdap_metrics
        except Exception as e:
            self.logger.error(f"Failed to collect MDAP metrics: {e}")
            metrics["adaptive_mdap"] = AdapterMetrics(
                adapter_name="adaptive_mdap",
                health_status="error"
            )

        # Collect MAKER adapter metrics
        try:
            maker_adapter = self.get_maker_adapter()
            maker_health = maker_adapter.health_check()

            maker_metrics = AdapterMetrics(
                adapter_name="maker",
                requests_total=maker_health['metrics']['maker_runs_total'],
                requests_success=maker_health['metrics']['maker_runs_success'],
                requests_failed=maker_health['metrics']['maker_runs_failed'],
                circuit_breaker_state=maker_health['circuit_breaker_state'],
                circuit_breaker_trips=maker_health.get('circuit_breaker_trips', 0),
                uptime_seconds=time.time() - self.start_time,
                health_status=maker_health['status']
            )
            metrics["maker"] = maker_metrics
        except Exception as e:
            self.logger.error(f"Failed to collect MAKER metrics: {e}")
            metrics["maker"] = AdapterMetrics(
                adapter_name="maker",
                health_status="error"
            )

        return metrics

    def calculate_derived_metrics(self, metrics: Dict[str, AdapterMetrics]) -> Dict[str, Any]:
        """
        Calculate derived metrics from collected metrics.

        Args:
            metrics: Collected adapter metrics

        Returns:
            Dictionary of derived metrics
        """
        total_requests = sum(m.requests_total for m in metrics.values())
        total_success = sum(m.requests_success for m in metrics.values())
        total_failed = sum(m.requests_failed for m in metrics.values())

        success_rate = total_success / total_requests if total_requests > 0 else 0.0
        failure_rate = total_failed / total_requests if total_requests > 0 else 0.0

        # Calculate request rate
        uptime = time.time() - self.start_time
        request_rate = total_requests / uptime if uptime > 0 else 0.0

        # Circuit breaker status
        any_open = any(m.circuit_breaker_state == "open" for m in metrics.values())
        overall_status = "healthy" if not any_open and all(
            m.health_status == "healthy" for m in metrics.values()
        ) else "degraded"

        return {
            "total_requests": total_requests,
            "total_success": total_success,
            "total_failed": total_failed,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "request_rate_per_second": request_rate,
            "overall_status": overall_status,
            "uptime_seconds": uptime,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def export_metrics(self, metrics: Dict[str, AdapterMetrics], derived: Dict[str, Any]):
        """
        Export metrics to file or external system.

        Args:
            metrics: Collected adapter metrics
            derived: Derived metrics
        """
        if not self.config.enable_metrics_export:
            return

        export_data = {
            "adapters": {
                name: {
                    "requests_total": m.requests_total,
                    "requests_success": m.requests_success,
                    "requests_failed": m.requests_failed,
                    "circuit_breaker_state": m.circuit_breaker_state,
                    "circuit_breaker_trips": m.circuit_breaker_trips,
                    "uptime_seconds": m.uptime_seconds,
                    "health_status": m.health_status
                }
                for name, m in metrics.items()
            },
            "derived": derived,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Export to file if path specified
        if self.config.metrics_export_path:
            try:
                import json
                with open(self.config.metrics_export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                self.logger.info(f"Metrics exported to {self.config.metrics_export_path}")
            except Exception as e:
                self.logger.error(f"Failed to export metrics: {e}")

        # Add to history
        self.metrics_history.append(export_data)
        if len(self.metrics_history) > self.config.history_size:
            self.metrics_history.pop(0)

    def generate_dashboard(self) -> str:
        """
        Generate a text-based dashboard.

        Returns:
            Dashboard as a formatted string
        """
        metrics = self.collect_metrics()
        derived = self.calculate_derived_metrics(metrics)

        # Export metrics
        self.export_metrics(metrics, derived)

        # Build dashboard
        lines = [
            "=" * 70,
            "ADAPTIVE MDAP/MAKER ADAPTER MONITORING DASHBOARD",
            "=" * 70,
            "",
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Uptime: {derived['uptime_seconds']:.0f} seconds ({derived['uptime_seconds']/86400:.2f} days)",
            f"Overall Status: {derived['overall_status'].upper()}",
            "",
            "-" * 70,
            "DERIVED METRICS",
            "-" * 70,
            f"Total Requests: {derived['total_requests']}",
            f"Success Rate: {derived['success_rate']*100:.1f}%",
            f"Failure Rate: {derived['failure_rate']*100:.1f}%",
            f"Request Rate: {derived['request_rate_per_second']:.2f} req/s",
            "",
            "-" * 70,
            "ADAPTER METRICS",
            "-" * 70,
        ]

        for name, m in metrics.items():
            lines.extend([
                f"",
                f"Adapter: {name.upper()}",
                f"  Status: {m.health_status.upper()}",
                f"  Requests: {m.requests_total} (success: {m.requests_success}, failed: {m.requests_failed})",
                f"  Circuit Breaker: {m.circuit_breaker_state.upper()} (trips: {m.circuit_breaker_trips})",
                f"  Uptime: {m.uptime_seconds:.0f} seconds",
            ])

        lines.extend([
            "",
            "=" * 70,
            "",
        ])

        return "\n".join(lines)

    def run_dashboard_loop(self, iterations: Optional[int] = None):
        """
        Run the monitoring dashboard loop.

        Args:
            iterations: Number of iterations (None for infinite)
        """
        self.logger.info("Starting monitoring dashboard...")

        iteration = 0
        try:
            while iterations is None or iteration < iterations:
                # Clear screen (platform dependent)
                os.system('cls' if os.name == 'nt' else 'clear')

                # Generate and print dashboard
                dashboard = self.generate_dashboard()
                print(dashboard)

                iteration += 1

                if iterations is None or iteration < iterations:
                    time.sleep(self.config.refresh_interval_seconds)

        except KeyboardInterrupt:
            print("\nDashboard stopped by user.")


# ============================================================================
# Convenience Functions
# ============================================================================

_default_monitor: Optional[AdapterMonitor] = None


def get_monitor(config: Optional[DashboardConfig] = None) -> AdapterMonitor:
    """Get or create the singleton monitor instance."""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = AdapterMonitor(config)
    return _default_monitor


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Create monitor
    monitor = AdapterMonitor(
        config=DashboardConfig(
            refresh_interval_seconds=5,
            enable_metrics_export=True,
            metrics_export_path="metrics.json"
        )
    )

    # Run dashboard loop (infinite)
    monitor.run_dashboard_loop()
