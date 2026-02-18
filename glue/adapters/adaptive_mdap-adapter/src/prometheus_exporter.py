"""
Prometheus Metrics Exporter for Adaptive MDAP/MAKER Adapter

Exposes adapter metrics in Prometheus format for scraping.
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available, metrics export disabled")

from adaptive_mdap_adapter import get_adapter
from maker_adapter import get_maker_adapter


@dataclass
class PrometheusConfig:
    """Configuration for Prometheus metrics exporter."""
    enabled: bool = True
    port: int = 9090
    metrics_path: str = "/metrics"
    update_interval_seconds: int = 15


class PrometheusMetricsExporter:
    """
    Exports adapter metrics to Prometheus format.

    Metrics exposed:
    - adaptive_mdap_requests_total: Total number of requests
    - adaptive_mdap_requests_success: Successful requests
    - adaptive_mdap_requests_failed: Failed requests
    - adaptive_mdap_request_duration_seconds: Request latency histogram
    - adaptive_mdap_circuit_breaker_state: Circuit breaker state (0=closed, 1=open, 2=half_open)
    - maker_runs_total: Total MAKER runs
    - maker_runs_success: Successful MAKER runs
    - maker_runs_failed: Failed MAKER runs
    - maker_votes_cast_total: Total votes cast
    - maker_red_flags_total: Total red flags detected
    """

    def __init__(self, config: Optional[PrometheusConfig] = None):
        """Initialize the Prometheus exporter."""
        self.config = config or PrometheusConfig()
        self.logger = logging.getLogger("PrometheusExporter")

        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available, exporter disabled")
            self.enabled = False
            return

        self.enabled = self.config.enabled

        # Initialize metrics
        self._init_metrics()

        # Start HTTP server
        if self.enabled:
            try:
                start_http_server(self.config.port)
                self.logger.info(f"Prometheus metrics server started on port {self.config.port}")
            except Exception as e:
                self.logger.error(f"Failed to start Prometheus server: {e}")
                self.enabled = False

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        # MDAP Adapter metrics
        self.mdap_requests_total = Counter(
            'adaptive_mdap_requests_total',
            'Total number of MDAP adapter requests',
            ['adapter', 'operation']
        )

        self.mdap_request_duration = Histogram(
            'adaptive_mdap_request_duration_seconds',
            'MDAP adapter request latency in seconds',
            ['adapter', 'operation']
        )

        self.mdap_circuit_breaker_state = Gauge(
            'adaptive_mdap_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half_open)',
            ['adapter']
        )

        # MAKER Adapter metrics
        self.maker_runs_total = Counter(
            'maker_runs_total',
            'Total number of MAKER runs',
            ['adapter']
        )

        self.maker_votes_cast = Counter(
            'maker_votes_cast_total',
            'Total number of votes cast',
            ['adapter']
        )

        self.maker_red_flags = Counter(
            'maker_red_flags_total',
            'Total number of red flags detected',
            ['adapter']
        )

    def update_metrics(self):
        """
        Update metrics from current adapter state.

        Should be called periodically to scrape metrics from adapters.
        """
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return

        try:
            # Update MDAP adapter metrics
            mdap_adapter = get_adapter()
            mdap_health = mdap_adapter.health_check()
            mdap_metrics = mdap_health['metrics']

            # Circuit breaker state
            state_map = {"closed": 0, "open": 1, "half_open": 2}
            cb_state = mdap_health['circuit_breaker_state']
            self.mdap_circuit_breaker_state.labels(adapter='adaptive_mdap').set(
                state_map.get(cb_state, 0)
            )

            # Request counts (increment only changes since last update)
            # Note: In production, track deltas instead of absolute values
            self.mdap_requests_total.labels(
                adapter='adaptive_mdap',
                operation='all'
            ).inc(mdap_metrics['requests_total'])

            # Update MAKER adapter metrics
            maker_adapter = get_maker_adapter()
            maker_health = maker_adapter.health_check()
            maker_metrics = maker_health['metrics']

            self.maker_runs_total.labels(adapter='maker').inc(
                maker_metrics.get('maker_runs_total', 0)
            )

            self.maker_votes_cast.labels(adapter='maker').inc(
                maker_metrics.get('total_votes_cast', 0)
            )

            self.maker_red_flags.labels(adapter='maker').inc(
                maker_metrics.get('total_red_flags', 0)
            )

            self.logger.debug("Metrics updated successfully")

        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")

    def start_metrics_loop(self):
        """
        Start the metrics update loop.

        Periodically scrapes metrics from adapters and updates Prometheus.
        Runs until interrupted.
        """
        if not self.enabled:
            self.logger.warning("Metrics exporter is disabled")
            return

        self.logger.info("Starting metrics update loop...")

        try:
            while True:
                self.update_metrics()
                time.sleep(self.config.update_interval_seconds)

        except KeyboardInterrupt:
            self.logger.info("Metrics loop stopped by user")


def export_metrics_text() -> str:
    """
    Export metrics in Prometheus text format.

    Returns:
        Metrics formatted as Prometheus exposition format
    """
    try:
        mdap_adapter = get_adapter()
        maker_adapter = get_maker_adapter()

        mdap_health = mdap_adapter.health_check()
        maker_health = maker_adapter.health_check()

        lines = [
            "# Prometheus metrics for Adaptive MDAP/MAKER Adapter",
            f"# Generated at {datetime.now(timezone.utc).isoformat()}",
            "",
            "# MDAP Adapter metrics",
            f"adaptive_mdap_requests_total {mdap_health['metrics']['requests_total']}",
            f"adaptive_mdap_requests_success {mdap_health['metrics']['requests_success']}",
            f"adaptive_mdap_requests_failed {mdap_health['metrics']['requests_failed']}",
            f"adaptive_mdap_circuit_breaker_trips {mdap_health['metrics']['circuit_breaker_trips']}",
            "",
            "# MAKER Adapter metrics",
            f"maker_runs_total {maker_health['metrics'].get('maker_runs_total', 0)}",
            f"maker_runs_success {maker_health['metrics'].get('maker_runs_success', 0)}",
            f"maker_runs_failed {maker_health['metrics'].get('maker_runs_failed', 0)}",
            f"maker_votes_cast_total {maker_health['metrics'].get('total_votes_cast', 0)}",
            f"maker_red_flags_total {maker_health['metrics'].get('total_red_flags', 0)}",
            "",
            "# Health status",
            f"adaptive_mdap_up {1 if mdap_health['status'] == 'healthy' else 0}",
            f"maker_up {1 if maker_health['status'] == 'healthy' else 0}",
            "",
        ]

        return "\n".join(lines)

    except Exception as e:
        return f"# Error generating metrics: {e}"


# ============================================================================
# Convenience Functions
# ============================================================================

_default_exporter: Optional[PrometheusMetricsExporter] = None


def get_prometheus_exporter(config: Optional[PrometheusConfig] = None) -> PrometheusMetricsExporter:
    """Get or create the singleton Prometheus exporter."""
    global _default_exporter
    if _default_exporter is None:
        _default_exporter = PrometheusMetricsExporter(config)
    return _default_exporter


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prometheus Metrics Exporter")
    parser.add_argument("--port", type=int, default=9090, help="Metrics port")
    parser.add_argument("--text", action="store_true", help="Export as text format")
    parser.add_argument("--loop", action="store_true", help="Run metrics update loop")

    args = parser.parse_args()

    if args.text:
        print(export_metrics_text())
    elif args.loop:
        exporter = PrometheusMetricsExporter(
            config=PrometheusConfig(port=args.port)
        )
        exporter.start_metrics_loop()
    else:
        print("Use --text to export metrics or --loop to run the update loop")
