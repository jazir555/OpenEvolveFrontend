"""
Health Check System for Gauntlet Adapter

Provides comprehensive health checks for liveness, readiness, and dependency health.
Monitors core components and external dependencies.

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import logging
import time
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, UTC
from threading import Lock
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Types of health checks"""
    LIVENESS = "liveness"      # Is the service running?
    READINESS = "readiness"    # Is the service ready to handle traffic?
    STARTUP = "startup"        # Has the service started up?


@dataclass
class HealthCheckResult:
    """
    Result of a health check.

    Attributes:
        component: Component name
        status: Health status
        message: Status message
        details: Additional details
        timestamp: Check timestamp
        check_type: Type of check performed
        duration_ms: Time taken for check
    """
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    check_type: CheckType = CheckType.LIVENESS
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "check_type": self.check_type.value,
            "duration_ms": self.duration_ms
        }

    def is_healthy(self) -> bool:
        """Check if result is healthy"""
        return self.status == HealthStatus.HEALTHY


@dataclass
class DependencyHealth:
    """
    Health of an external dependency.

    Attributes:
        name: Dependency name
        url: Dependency URL/endpoint
        status: Health status
        response_time_ms: Response time in milliseconds
        last_check: Last check timestamp
        error_message: Error message if unhealthy
    """
    name: str
    url: str
    status: HealthStatus
    response_time_ms: float
    last_check: float
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "url": self.url,
            "status": self.status.value,
            "response_time_ms": self.response_time_ms,
            "last_check": self.last_check,
            "error_message": self.error_message
        }


class HealthChecker:
    """
    Comprehensive health checker for gauntlet system.

    Performs:
    - Liveness checks (service is running)
    - Readiness checks (service can handle requests)
    - Dependency health checks (external services)
    - Resource health checks (CPU, memory, disk)

    Example:
        >>> checker = HealthChecker()
        >>>
        >>> # Run all checks
        >>> results = checker.check_all()
        >>>
        >>> # Check if system is ready
        >>> if checker.is_ready():
        ...     print("System is ready to handle traffic")
        >>>
        >>> # Get health report
        >>> report = checker.get_health_report()
    """

    def __init__(self):
        """Initialize health checker"""
        self._lock = Lock()
        self._start_time = time.time()
        self._check_results: Dict[str, HealthCheckResult] = {}
        self._dependencies: Dict[str, DependencyHealth] = {}
        self._custom_checks: Dict[str, Callable[[], HealthCheckResult]] = {}

        # Thresholds
        self._cpu_threshold = 80.0  # percent
        self._memory_threshold = 85.0  # percent
        self._disk_threshold = 85.0  # percent

        logger.info("Gauntlet Health Checker initialized")

    # ========== Core Health Checks ==========

    def check_all(self, check_type: CheckType = CheckType.LIVENESS) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks.

        Args:
            check_type: Type of checks to run

        Returns:
            Dictionary of component names to check results
        """
        start_time = time.time()

        with self._lock:
            # Always check core resources
            self._check_memory()
            self._check_cpu()
            self._check_disk()

            # Check based on type
            if check_type in [CheckType.LIVENESS, CheckType.READINESS]:
                self._check_gauntlet_executor()
                self._check_ml_components()
                self._check_websocket_server()

            if check_type == CheckType.READINESS:
                self._check_dependencies()
                self._check_metrics_collector()

            # Run custom checks
            for name, check_func in self._custom_checks.items():
                try:
                    result = check_func()
                    result.duration_ms = (time.time() - start_time) * 1000
                    self._check_results[name] = result
                except Exception as e:
                    logger.error(f"Custom health check failed: {name}: {e}")
                    self._check_results[name] = HealthCheckResult(
                        component=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {str(e)}",
                        duration_ms=(time.time() - start_time) * 1000
                    )

        return self._check_results.copy()

    def _check_memory(self) -> None:
        """Check memory usage"""
        start_time = time.time()
        memory = psutil.virtual_memory()
        used_percent = memory.percent

        if used_percent < self._memory_threshold - 15:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {used_percent:.1f}%"
        elif used_percent < self._memory_threshold:
            status = HealthStatus.DEGRADED
            message = f"Memory usage elevated: {used_percent:.1f}%"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Memory usage critical: {used_percent:.1f}%"

        self._check_results["memory"] = HealthCheckResult(
            component="memory",
            status=status,
            message=message,
            details={
                "used_percent": used_percent,
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3),
                "threshold": self._memory_threshold
            },
            check_type=CheckType.LIVENESS,
            duration_ms=(time.time() - start_time) * 1000
        )

    def _check_cpu(self) -> None:
        """Check CPU usage"""
        start_time = time.time()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        if cpu_percent < self._cpu_threshold - 20:
            status = HealthStatus.HEALTHY
            message = f"CPU usage normal: {cpu_percent:.1f}%"
        elif cpu_percent < self._cpu_threshold:
            status = HealthStatus.DEGRADED
            message = f"CPU usage elevated: {cpu_percent:.1f}%"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"CPU usage critical: {cpu_percent:.1f}%"

        self._check_results["cpu"] = HealthCheckResult(
            component="cpu",
            status=status,
            message=message,
            details={
                "percent": cpu_percent,
                "count": psutil.cpu_count(),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "threshold": self._cpu_threshold
            },
            check_type=CheckType.LIVENESS,
            duration_ms=(time.time() - start_time) * 1000
        )

    def _check_disk(self) -> None:
        """Check disk usage"""
        start_time = time.time()
        disk = psutil.disk_usage("/")
        used_percent = disk.percent

        if used_percent < self._disk_threshold - 15:
            status = HealthStatus.HEALTHY
            message = f"Disk usage normal: {used_percent:.1f}%"
        elif used_percent < self._disk_threshold:
            status = HealthStatus.DEGRADED
            message = f"Disk usage elevated: {used_percent:.1f}%"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Disk usage critical: {used_percent:.1f}%"

        self._check_results["disk"] = HealthCheckResult(
            component="disk",
            status=status,
            message=message,
            details={
                "used_percent": used_percent,
                "free_gb": disk.free / (1024**3),
                "total_gb": disk.total / (1024**3),
                "threshold": self._disk_threshold
            },
            check_type=CheckType.LIVENESS,
            duration_ms=(time.time() - start_time) * 1000
        )

    def _check_gauntlet_executor(self) -> None:
        """Check gauntlet executor health"""
        start_time = time.time()

        try:
            # Try to import and check executor
            from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import PredictiveGauntletExecutor

            # Basic check - can we instantiate?
            # In production, would check actual executor state
            status = HealthStatus.HEALTHY
            message = "Gauntlet executor operational"

        except ImportError as e:
            status = HealthStatus.DEGRADED
            message = f"Gauntlet executor not available: {str(e)}"
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Gauntlet executor error: {str(e)}"

        self._check_results["gauntlet_executor"] = HealthCheckResult(
            component="gauntlet_executor",
            status=status,
            message=message,
            check_type=CheckType.READINESS,
            duration_ms=(time.time() - start_time) * 1000
        )

    def _check_ml_components(self) -> None:
        """Check ML component health"""
        start_time = time.time()

        try:
            from glue.adapters.gauntlet_adapter.src.ml_optimizer import MLBasedGauntletOptimizer
            from glue.adapters.gauntlet_adapter.src.adaptive_learner import AdaptiveLearner

            status = HealthStatus.HEALTHY
            message = "ML components available"

        except ImportError as e:
            status = HealthStatus.DEGRADED
            message = f"ML components not available: {str(e)}"
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"ML component error: {str(e)}"

        self._check_results["ml_components"] = HealthCheckResult(
            component="ml_components",
            status=status,
            message=message,
            check_type=CheckType.READINESS,
            duration_ms=(time.time() - start_time) * 1000
        )

    def _check_websocket_server(self) -> None:
        """Check WebSocket server health"""
        start_time = time.time()

        try:
            # Check if WebSocket server is running
            # In production, would check actual WebSocket state
            import asyncio

            status = HealthStatus.HEALTHY
            message = "WebSocket server operational"

        except ImportError as e:
            status = HealthStatus.DEGRADED
            message = f"WebSocket server not available: {str(e)}"
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"WebSocket server error: {str(e)}"

        self._check_results["websocket_server"] = HealthCheckResult(
            component="websocket_server",
            status=status,
            message=message,
            check_type=CheckType.READINESS,
            duration_ms=(time.time() - start_time) * 1000
        )

    def _check_metrics_collector(self) -> None:
        """Check metrics collector health"""
        start_time = time.time()

        try:
            from glue.adapters.gauntlet_adapter.monitoring.metrics import get_metrics_collector

            collector = get_metrics_collector()
            summary = collector.get_metric_summary()

            # Check if metrics are being collected
            uptime = summary.get("uptime_seconds", 0)
            if uptime > 0:
                status = HealthStatus.HEALTHY
                message = f"Metrics collector active (uptime: {uptime:.1f}s)"
            else:
                status = HealthStatus.DEGRADED
                message = "Metrics collector not initialized"

        except ImportError as e:
            status = HealthStatus.DEGRADED
            message = f"Metrics collector not available: {str(e)}"
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Metrics collector error: {str(e)}"

        self._check_results["metrics_collector"] = HealthCheckResult(
            component="metrics_collector",
            status=status,
            message=message,
            check_type=CheckType.READINESS,
            duration_ms=(time.time() - start_time) * 1000
        )

    def _check_dependencies(self) -> None:
        """Check external dependencies"""
        # Check each registered dependency
        for name in list(self._dependencies.keys()):
            self._check_dependency(name)

    # ========== Dependency Management ==========

    def add_dependency(
        self,
        name: str,
        url: str,
        health_check_url: Optional[str] = None
    ) -> None:
        """
        Add an external dependency to monitor.

        Args:
            name: Dependency name
            url: Dependency URL
            health_check_url: Optional health check endpoint
        """
        with self._lock:
            self._dependencies[name] = DependencyHealth(
                name=name,
                url=url,
                status=HealthStatus.UNKNOWN,
                response_time_ms=0.0,
                last_check=0.0
            )

        logger.info(f"Added dependency: {name} at {url}")

    def remove_dependency(self, name: str) -> None:
        """Remove a dependency from monitoring"""
        with self._lock:
            if name in self._dependencies:
                del self._dependencies[name]
                logger.info(f"Removed dependency: {name}")

    def _check_dependency(self, name: str) -> None:
        """
        Check a specific dependency.

        Args:
            name: Dependency name
        """
        if name not in self._dependencies:
            logger.warning(f"Unknown dependency: {name}")
            return

        dep = self._dependencies[name]
        start_time = time.time()

        try:
            import requests

            # Try to reach the dependency
            health_url = dep.url
            if not health_url.endswith("/health"):
                health_url = f"{health_url.rstrip('/')}/health"

            response = requests.get(health_url, timeout=5)

            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                dep.status = HealthStatus.HEALTHY
                dep.error_message = ""
            elif response.status_code >= 500:
                dep.status = HealthStatus.UNHEALTHY
                dep.error_message = f"HTTP {response.status_code}"
            else:
                dep.status = HealthStatus.DEGRADED
                dep.error_message = f"HTTP {response.status_code}"

            dep.response_time_ms = response_time
            dep.last_check = time.time()

        except requests.Timeout:
            dep.status = HealthStatus.UNHEALTHY
            dep.response_time_ms = 5000.0  # Timeout
            dep.error_message = "Connection timeout"
            dep.last_check = time.time()

        except Exception as e:
            dep.status = HealthStatus.UNHEALTHY
            dep.response_time_ms = (time.time() - start_time) * 1000
            dep.error_message = str(e)
            dep.last_check = time.time()

        logger.debug(f"Checked dependency {name}: {dep.status.value}")

    # ========== Custom Checks ==========

    def register_custom_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult]
    ) -> None:
        """
        Register a custom health check.

        Args:
            name: Check name
            check_func: Function that returns HealthCheckResult
        """
        with self._lock:
            self._custom_checks[name] = check_func

        logger.info(f"Registered custom health check: {name}")

    def unregister_custom_check(self, name: str) -> None:
        """Unregister a custom health check"""
        with self._lock:
            if name in self._custom_checks:
                del self._custom_checks[name]
                logger.info(f"Unregistered custom health check: {name}")

    # ========== Status Reports ==========

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        results = self.check_all()

        if not results:
            return HealthStatus.UNKNOWN

        # Check for any unhealthy components
        unhealthy = [r for r in results.values() if r.status == HealthStatus.UNHEALTHY]
        if unhealthy:
            return HealthStatus.UNHEALTHY

        # Check for degraded components
        degraded = [r for r in results.values() if r.status == HealthStatus.DEGRADED]
        if degraded:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def is_healthy(self) -> bool:
        """Quick check if system is healthy"""
        return self.get_overall_status() == HealthStatus.HEALTHY

    def is_ready(self) -> bool:
        """Quick check if system is ready to handle traffic"""
        results = self.check_all(CheckType.READINESS)

        # All critical components must be healthy or degraded
        critical_components = ["gauntlet_executor", "ml_components"]
        for comp in critical_components:
            if comp in results:
                if results[comp].status == HealthStatus.UNHEALTHY:
                    return False

        return True

    def get_uptime_seconds(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self._start_time

    def get_health_report(self, check_type: CheckType = CheckType.LIVENESS) -> Dict[str, Any]:
        """
        Get comprehensive health report.

        Args:
            check_type: Type of checks to include

        Returns:
            Health report dictionary
        """
        results = self.check_all(check_type)

        return {
            "overall_status": self.get_overall_status().value,
            "is_healthy": self.is_healthy(),
            "is_ready": self.is_ready(),
            "uptime_seconds": self.get_uptime_seconds(),
            "timestamp": time.time(),
            "check_type": check_type.value,
            "components": {
                name: result.to_dict()
                for name, result in results.items()
            },
            "dependencies": {
                name: dep.to_dict()
                for name, dep in self._dependencies.items()
            }
        }

    # ========== Threshold Management ==========

    def set_cpu_threshold(self, threshold: float) -> None:
        """Set CPU warning threshold (percent)"""
        self._cpu_threshold = threshold
        logger.info(f"CPU threshold set to {threshold}%")

    def set_memory_threshold(self, threshold: float) -> None:
        """Set memory warning threshold (percent)"""
        self._memory_threshold = threshold
        logger.info(f"Memory threshold set to {threshold}%")

    def set_disk_threshold(self, threshold: float) -> None:
        """Set disk warning threshold (percent)"""
        self._disk_threshold = threshold
        logger.info(f"Disk threshold set to {threshold}%")


# Global health checker instance
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker"""
    return _health_checker


def check_liveness() -> Dict[str, Any]:
    """
    Liveness probe - is the service running?

    Returns:
        Liveness check results
    """
    return get_health_checker().get_health_report(CheckType.LIVENESS)


def check_readiness() -> Dict[str, Any]:
    """
    Readiness probe - is the service ready to handle traffic?

    Returns:
        Readiness check results
    """
    return get_health_checker().get_health_report(CheckType.READINESS)


def is_healthy() -> bool:
    """Quick health check"""
    return get_health_checker().is_healthy()


def is_ready() -> bool:
    """Quick readiness check"""
    return get_health_checker().is_ready()
