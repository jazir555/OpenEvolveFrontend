"""
Configuration for Gauntlet Monitoring System

Provides configuration management via environment variables
and sensible defaults.

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    # Prometheus configuration
    prometheus_port: int = 9090
    prometheus_enabled: bool = True

    # Metrics export
    export_interval_seconds: int = 15
    retain_hours: int = 24  # How long to keep metrics

    # System metrics collection
    collect_cpu: bool = True
    collect_memory: bool = True
    collect_disk: bool = True

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Create configuration from environment variables"""
        return cls(
            prometheus_port=int(os.getenv("GAUNTLET_PROMETHEUS_PORT", "9090")),
            prometheus_enabled=os.getenv("GAUNTLET_PROMETHEUS_ENABLED", "true").lower() == "true",
            export_interval_seconds=int(os.getenv("GAUNTLET_METRICS_EXPORT_INTERVAL", "15")),
            retain_hours=int(os.getenv("GAUNTLET_METRICS_RETAIN_HOURS", "24")),
            collect_cpu=os.getenv("GAUNTLET_COLLECT_CPU", "true").lower() == "true",
            collect_memory=os.getenv("GAUNTLET_COLLECT_MEMORY", "true").lower() == "true",
            collect_disk=os.getenv("GAUNTLET_COLLECT_DISK", "true").lower() == "true",
        )


@dataclass
class HealthCheckConfig:
    """Configuration for health checks"""
    # Thresholds
    cpu_threshold_percent: float = 80.0
    memory_threshold_percent: float = 85.0
    disk_threshold_percent: float = 85.0

    # Check intervals
    liveness_interval_seconds: int = 10
    readiness_interval_seconds: int = 5

    # Timeout
    check_timeout_seconds: int = 5

    @classmethod
    def from_env(cls) -> "HealthCheckConfig":
        """Create configuration from environment variables"""
        return cls(
            cpu_threshold_percent=float(os.getenv("GAUNTLET_CPU_THRESHOLD", "80.0")),
            memory_threshold_percent=float(os.getenv("GAUNTLET_MEMORY_THRESHOLD", "85.0")),
            disk_threshold_percent=float(os.getenv("GAUNTLET_DISK_THRESHOLD", "85.0")),
            liveness_interval_seconds=int(os.getenv("GAUNTLET_LIVENESS_INTERVAL", "10")),
            readiness_interval_seconds=int(os.getenv("GAUNTLET_READINESS_INTERVAL", "5")),
            check_timeout_seconds=int(os.getenv("GAUNTLET_HEALTH_CHECK_TIMEOUT", "5")),
        )


@dataclass
class AlertingConfig:
    """Configuration for alerting"""
    # Alert thresholds
    error_rate_threshold: float = 0.1  # 10%
    latency_threshold_ms: float = 5000.0
    pass_rate_threshold: float = 0.5  # 50%
    prediction_accuracy_threshold: float = 0.6  # 60%

    # Cooldown periods
    default_cooldown_seconds: int = 300
    critical_cooldown_seconds: int = 60

    # Notification settings
    webhook_enabled: bool = False
    webhook_url: Optional[str] = None
    webhook_timeout_seconds: int = 5

    # Alert retention
    retain_resolved_hours: int = 24

    @classmethod
    def from_env(cls) -> "AlertingConfig":
        """Create configuration from environment variables"""
        return cls(
            error_rate_threshold=float(os.getenv("GAUNTLET_ERROR_RATE_THRESHOLD", "0.1")),
            latency_threshold_ms=float(os.getenv("GAUNTLET_LATENCY_THRESHOLD_MS", "5000.0")),
            pass_rate_threshold=float(os.getenv("GAUNTLET_PASS_RATE_THRESHOLD", "0.5")),
            prediction_accuracy_threshold=float(os.getenv("GAUNTLET_PREDICTION_ACCURACY_THRESHOLD", "0.6")),
            default_cooldown_seconds=int(os.getenv("GAUNTLET_ALERT_COOLDOWN", "300")),
            critical_cooldown_seconds=int(os.getenv("GAUNTLET_CRITICAL_ALERT_COOLDOWN", "60")),
            webhook_enabled=os.getenv("GAUNTLET_WEBHOOK_ENABLED", "false").lower() == "true",
            webhook_url=os.getenv("GAUNTLET_WEBHOOK_URL"),
            webhook_timeout_seconds=int(os.getenv("GAUNTLET_WEBHOOK_TIMEOUT", "5")),
            retain_resolved_hours=int(os.getenv("GAUNTLET_ALERT_RETAIN_HOURS", "24")),
        )


@dataclass
class MonitoringConfig:
    """Complete monitoring configuration"""
    metrics: MetricsConfig
    health: HealthCheckConfig
    alerting: AlertingConfig

    # Global settings
    enabled: bool = True
    debug: bool = False

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        """Create complete configuration from environment variables"""
        enabled = os.getenv("GAUNTLET_MONITORING_ENABLED", "true").lower() == "true"
        debug = os.getenv("GAUNTLET_MONITORING_DEBUG", "false").lower() == "true"

        return cls(
            metrics=MetricsConfig.from_env(),
            health=HealthCheckConfig.from_env(),
            alerting=AlertingConfig.from_env(),
            enabled=enabled,
            debug=debug,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "enabled": self.enabled,
            "debug": self.debug,
            "metrics": {
                "prometheus_port": self.metrics.prometheus_port,
                "prometheus_enabled": self.metrics.prometheus_enabled,
                "export_interval_seconds": self.metrics.export_interval_seconds,
                "retain_hours": self.metrics.retain_hours,
            },
            "health": {
                "cpu_threshold_percent": self.health.cpu_threshold_percent,
                "memory_threshold_percent": self.health.memory_threshold_percent,
                "disk_threshold_percent": self.health.disk_threshold_percent,
                "liveness_interval_seconds": self.health.liveness_interval_seconds,
                "readiness_interval_seconds": self.health.readiness_interval_seconds,
            },
            "alerting": {
                "error_rate_threshold": self.alerting.error_rate_threshold,
                "latency_threshold_ms": self.alerting.latency_threshold_ms,
                "pass_rate_threshold": self.alerting.pass_rate_threshold,
                "prediction_accuracy_threshold": self.alerting.prediction_accuracy_threshold,
                "default_cooldown_seconds": self.alerting.default_cooldown_seconds,
                "webhook_enabled": self.alerting.webhook_enabled,
                "webhook_url": self.alerting.webhook_url,
            }
        }


# Global configuration instance
_config: Optional[MonitoringConfig] = None


def get_config() -> MonitoringConfig:
    """Get the global monitoring configuration"""
    global _config
    if _config is None:
        _config = MonitoringConfig.from_env()
    return _config


def set_config(config: MonitoringConfig) -> None:
    """Set the global monitoring configuration"""
    global _config
    _config = config


def reload_config() -> MonitoringConfig:
    """Reload configuration from environment variables"""
    global _config
    _config = MonitoringConfig.from_env()
    return _config


# Example configuration file (not used directly, but useful for documentation)
EXAMPLE_ENV = """
# Gauntlet Monitoring Configuration

# Enable/disable monitoring
GAUNTLET_MONITORING_ENABLED=true
GAUNTLET_MONITORING_DEBUG=false

# Metrics Configuration
GAUNTLET_PROMETHEUS_PORT=9090
GAUNTLET_PROMETHEUS_ENABLED=true
GAUNTLET_METRICS_EXPORT_INTERVAL=15
GAUNTLET_METRICS_RETAIN_HOURS=24
GAUNTLET_COLLECT_CPU=true
GAUNTLET_COLLECT_MEMORY=true
GAUNTLET_COLLECT_DISK=true

# Health Check Configuration
GAUNTLET_CPU_THRESHOLD=80.0
GAUNTLET_MEMORY_THRESHOLD=85.0
GAUNTLET_DISK_THRESHOLD=85.0
GAUNTLET_LIVENESS_INTERVAL=10
GAUNTLET_READINESS_INTERVAL=5
GAUNTLET_HEALTH_CHECK_TIMEOUT=5

# Alerting Configuration
GAUNTLET_ERROR_RATE_THRESHOLD=0.1
GAUNTLET_LATENCY_THRESHOLD_MS=5000.0
GAUNTLET_PASS_RATE_THRESHOLD=0.5
GAUNTLET_PREDICTION_ACCURACY_THRESHOLD=0.6
GAUNTLET_ALERT_COOLDOWN=300
GAUNTLET_CRITICAL_ALERT_COOLDOWN=60
GAUNTLET_WEBHOOK_ENABLED=false
GAUNTLET_WEBHOOK_URL=https://your-webhook-url.com/alerts
GAUNTLET_WEBHOOK_TIMEOUT=5
GAUNTLET_ALERT_RETAIN_HOURS=24
"""


def print_example_env():
    """Print example environment configuration"""
    print(EXAMPLE_ENV)


if __name__ == "__main__":
    import json

    # Load and print current configuration
    config = get_config()
    print("Current Monitoring Configuration:")
    print(json.dumps(config.to_dict(), indent=2))

    print("\n" + "=" * 60)
    print("Example Environment Configuration:")
    print("=" * 60)
    print_example_env()
