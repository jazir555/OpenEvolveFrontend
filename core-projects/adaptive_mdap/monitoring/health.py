"""Health check system for Adaptive MDAP."""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from adaptive_mdap.utils.logger import get_logger
    logger = get_logger("monitoring.health")
except ImportError:
    import logging
    logger = logging.getLogger("monitoring.health")


class ComponentStatus(Enum):
    """Status of a component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: ComponentStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class HealthChecker:
    """Health check system for Adaptive MDAP components."""
    
    def __init__(self):
        self._start_time = time.time()
        self._last_check: Optional[datetime] = None
        self._check_results: Dict[str, HealthCheckResult] = {}
    
    def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        self._last_check = datetime.utcnow()
        
        # Core checks
        self._check_memory()
        self._check_cpu()
        self._check_disk()
        
        # Component checks
        self._check_cache()
        self._check_metrics()
        
        return self._check_results
    
    def _check_memory(self) -> None:
        """Check memory usage."""
        if not PSUTIL_AVAILABLE:
            self._check_results["memory"] = HealthCheckResult(
                component="memory",
                status=ComponentStatus.UNKNOWN,
                message="psutil not available - cannot check memory",
                details={},
                timestamp=time.time(),
            )
            return
            
        memory = psutil.virtual_memory()
        used_percent = memory.percent
        
        if used_percent < 70:
            status = ComponentStatus.HEALTHY
            message = f"Memory usage normal: {used_percent:.1f}%"
        elif used_percent < 85:
            status = ComponentStatus.DEGRADED
            message = f"Memory usage elevated: {used_percent:.1f}%"
        else:
            status = ComponentStatus.UNHEALTHY
            message = f"Memory usage critical: {used_percent:.1f}%"
        
        self._check_results["memory"] = HealthCheckResult(
            component="memory",
            status=status,
            message=message,
            details={
                "used_percent": used_percent,
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3),
            },
            timestamp=time.time(),
        )
    
    def _check_cpu(self) -> None:
        """Check CPU usage."""
        if not PSUTIL_AVAILABLE:
            self._check_results["cpu"] = HealthCheckResult(
                component="cpu",
                status=ComponentStatus.UNKNOWN,
                message="psutil not available - cannot check CPU",
                details={},
                timestamp=time.time(),
            )
            return
            
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent < 50:
            status = ComponentStatus.HEALTHY
            message = f"CPU usage normal: {cpu_percent:.1f}%"
        elif cpu_percent < 75:
            status = ComponentStatus.DEGRADED
            message = f"CPU usage elevated: {cpu_percent:.1f}%"
        else:
            status = ComponentStatus.UNHEALTHY
            message = f"CPU usage critical: {cpu_percent:.1f}%"
        
        self._check_results["cpu"] = HealthCheckResult(
            component="cpu",
            status=status,
            message=message,
            details={
                "percent": cpu_percent,
                "count": psutil.cpu_count(),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            },
            timestamp=time.time(),
        )
    
    def _check_disk(self) -> None:
        """Check disk usage."""
        if not PSUTIL_AVAILABLE:
            self._check_results["disk"] = HealthCheckResult(
                component="disk",
                status=ComponentStatus.UNKNOWN,
                message="psutil not available - cannot check disk",
                details={},
                timestamp=time.time(),
            )
            return
            
        disk = psutil.disk_usage("/")
        used_percent = disk.percent
        
        if used_percent < 70:
            status = ComponentStatus.HEALTHY
            message = f"Disk usage normal: {used_percent:.1f}%"
        elif used_percent < 85:
            status = ComponentStatus.DEGRADED
            message = f"Disk usage elevated: {used_percent:.1f}%"
        else:
            status = ComponentStatus.UNHEALTHY
            message = f"Disk usage critical: {used_percent:.1f}%"
        
        self._check_results["disk"] = HealthCheckResult(
            component="disk",
            status=status,
            message=message,
            details={
                "used_percent": used_percent,
                "free_gb": disk.free / (1024**3),
                "total_gb": disk.total / (1024**3),
            },
            timestamp=time.time(),
        )
    
    def _check_cache(self) -> None:
        """Check cache health."""
        try:
            from adaptive_mdap.utils.cache import get_cache_stats
            stats = get_cache_stats()
            
            hit_rate = stats.get("hit_rate", 0)
            if hit_rate > 0.7:
                status = ComponentStatus.HEALTHY
                message = f"Cache hit rate good: {hit_rate:.1%}"
            elif hit_rate > 0.4:
                status = ComponentStatus.DEGRADED
                message = f"Cache hit rate moderate: {hit_rate:.1%}"
            else:
                status = ComponentStatus.UNHEALTHY
                message = f"Cache hit rate low: {hit_rate:.1%}"
            
            self._check_results["cache"] = HealthCheckResult(
                component="cache",
                status=status,
                message=message,
                details=stats,
                timestamp=time.time(),
            )
        except Exception as e:
            self._check_results["cache"] = HealthCheckResult(
                component="cache",
                status=ComponentStatus.UNHEALTHY,
                message=f"Cache check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
            )
    
    def _check_metrics(self) -> None:
        """Check metrics collector health."""
        try:
            from adaptive_mdap.utils.metrics import get_metrics
            metrics = get_metrics()
            all_metrics = metrics.get_all_metrics()
            
            # Basic check - ensure metrics are being collected
            counters = all_metrics.get("counters", {})
            
            if len(counters) > 0:
                status = ComponentStatus.HEALTHY
                message = f"Metrics collection active: {len(counters)} counters"
            else:
                status = ComponentStatus.DEGRADED
                message = "No metrics collected yet"
            
            self._check_results["metrics"] = HealthCheckResult(
                component="metrics",
                status=status,
                message=message,
                details=all_metrics,
                timestamp=time.time(),
            )
        except Exception as e:
            self._check_results["metrics"] = HealthCheckResult(
                component="metrics",
                status=ComponentStatus.UNHEALTHY,
                message=f"Metrics check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
            )
    
    def get_overall_status(self) -> ComponentStatus:
        """Get overall system status."""
        results = self.check_all()
        
        if not results:
            return ComponentStatus.UNKNOWN
        
        # Check for any unhealthy components
        unhealthy = [r for r in results.values() if r.status == ComponentStatus.UNHEALTHY]
        if unhealthy:
            return ComponentStatus.UNHEALTHY
        
        # Check for degraded components
        degraded = [r for r in results.values() if r.status == ComponentStatus.DEGRADED]
        if degraded:
            return ComponentStatus.DEGRADED
        
        return ComponentStatus.HEALTHY
    
    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self._start_time
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        results = self.check_all()
        
        return {
            "overall_status": self.get_overall_status().value,
            "uptime_seconds": self.get_uptime_seconds(),
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "components": {name: result.to_dict() for name, result in results.items()},
        }


# Global health checker instance
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker."""
    return _health_checker


def check_health() -> Dict[str, Any]:
    """Quick health check for API endpoints."""
    return get_health_checker().get_status_report()


__all__ = [
    "HealthChecker",
    "HealthCheckResult",
    "ComponentStatus",
    "get_health_checker",
    "check_health",
]
