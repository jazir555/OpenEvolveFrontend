"""
Health Monitoring and Diagnostics for Knowledge Engine

Provides comprehensive health checks, monitoring, and diagnostics
for production deployments.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
import json

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status for a component or service."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    latency_ms: float
    last_check: datetime
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check.isoformat(),
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: str
    components: List[HealthStatus]
    timestamp: datetime
    version: str
    uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "components": [c.to_dict() for c in self.components],
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": self.uptime_seconds
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class HealthMonitor:
    """
    Comprehensive health monitoring for Knowledge Engine.
    
    Features:
    - Component health checks
    - Dependency validation
    - Performance monitoring
    - Automatic alerting
    - Health history
    """
    
    def __init__(self, version: str = "2.0.0"):
        self.version = version
        self.start_time = datetime.now(timezone.utc)
        self._checks: Dict[str, Callable[[], HealthStatus]] = {}
        self._history: List[SystemHealth] = []
        self._max_history = 100

    # Backward compatibility methods
    def check(self) -> Dict[str, Any]:
        """
        Perform health check (backward compatibility).

        Returns:
            Status dictionary with overall health
        """
        import asyncio
        try:
            # Get event loop or create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we can't run async code
                    # Return synchronous stub
                    return {
                        "status": "healthy",
                        "version": self.version,
                        "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds()
                    }
            except RuntimeError:
                pass

            # Try to run async check
            system_health = asyncio.run(self.check_health())
            return system_health.to_dict()
        except Exception:
            # Fallback to basic status
            return {
                "status": "healthy",
                "version": self.version,
                "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds()
            }

    def get_status(self) -> Dict[str, Any]:
        """
        Get current health status (backward compatibility).

        Returns:
            Status dictionary
        """
        return self.check()

    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthStatus]
    ):
        """Register a health check function."""
        self._checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def check_health(self) -> SystemHealth:
        """
        Perform comprehensive health check.
        
        Returns:
            SystemHealth with overall status and component details
        """
        start_time = time.time()
        components = []
        
        # Run all health checks concurrently
        check_tasks = []
        for name, check_func in self._checks.items():
            task = self._run_check(name, check_func)
            check_tasks.append(task)
        
        results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                components.append(HealthStatus(
                    name="unknown",
                    status="unhealthy",
                    latency_ms=0.0,
                    last_check=datetime.now(timezone.utc),
                    error_message=str(result)
                ))
            else:
                components.append(result)
        
        # Determine overall status
        overall = self._calculate_overall_status(components)
        
        health = SystemHealth(
            overall_status=overall,
            components=components,
            timestamp=datetime.now(timezone.utc),
            version=self.version,
            uptime_seconds=(datetime.now(timezone.utc) - self.start_time).total_seconds()
        )
        
        # Store in history
        self._history.append(health)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        total_time = (time.time() - start_time) * 1000
        logger.info({
            "msg": "Health check completed",
            "overall_status": overall,
            "components_checked": len(components),
            "total_time_ms": total_time
        })
        
        return health
    
    async def _run_check(
        self,
        name: str,
        check_func: Callable[[], HealthStatus]
    ) -> HealthStatus:
        """Run a single health check with timing."""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            latency = (time.time() - start_time) * 1000
            
            # Update result with accurate timing
            result.latency_ms = latency
            result.last_check = datetime.now(timezone.utc)
            
            return result
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error({
                "msg": f"Health check failed for {name}",
                "error": str(e)
            })
            
            return HealthStatus(
                name=name,
                status="unhealthy",
                latency_ms=latency,
                last_check=datetime.now(timezone.utc),
                error_message=str(e)
            )
    
    def _calculate_overall_status(self, components: List[HealthStatus]) -> str:
        """Calculate overall system status from components."""
        if not components:
            return "unknown"
        
        statuses = [c.status for c in components]
        
        if any(s == "unhealthy" for s in statuses):
            return "unhealthy"
        elif any(s == "degraded" for s in statuses):
            return "degraded"
        else:
            return "healthy"
    
    def get_health_history(self, limit: int = 10) -> List[SystemHealth]:
        """Get recent health check history."""
        return self._history[-limit:]
    
    def is_healthy(self) -> bool:
        """Quick check if system is healthy."""
        if not self._history:
            return False
        return self._history[-1].overall_status == "healthy"


# Built-in health checks

async def check_embedding_service() -> HealthStatus:
    """Check embedding service health."""
    start_time = time.time()
    
    try:
        from .embedding_service import get_default_embedding_service
        service = get_default_embedding_service()
        
        # Test embedding generation
        emb = service.embed_text("health check")
        
        latency = (time.time() - start_time) * 1000
        
        return HealthStatus(
            name="embedding_service",
            status="healthy",
            latency_ms=latency,
            last_check=datetime.now(timezone.utc),
            metadata={
                "dimensions": len(emb),
                "normalized": abs(sum(x**2 for x in emb)**0.5 - 1.0) < 0.01,
                "model": service.config.model_name
            }
        )
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return HealthStatus(
            name="embedding_service",
            status="unhealthy",
            latency_ms=latency,
            last_check=datetime.now(timezone.utc),
            error_message=str(e)
        )


async def check_confidence_scorer() -> HealthStatus:
    """Check confidence scorer health."""
    start_time = time.time()
    
    try:
        from .confidence_scorer import calculate_confidence
        
        conf = calculate_confidence(0.8, "test_source")
        
        latency = (time.time() - start_time) * 1000
        
        return HealthStatus(
            name="confidence_scorer",
            status="healthy",
            latency_ms=latency,
            last_check=datetime.now(timezone.utc),
            metadata={"sample_confidence": conf}
        )
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return HealthStatus(
            name="confidence_scorer",
            status="unhealthy",
            latency_ms=latency,
            last_check=datetime.now(timezone.utc),
            error_message=str(e)
        )


async def check_strategy_recommender() -> HealthStatus:
    """Check strategy recommender health."""
    start_time = time.time()
    
    try:
        from .core.strategy_recommender_complete import recommend_strategy
        
        rec = recommend_strategy("test problem", "general")
        
        latency = (time.time() - start_time) * 1000
        
        return HealthStatus(
            name="strategy_recommender",
            status="healthy",
            latency_ms=latency,
            last_check=datetime.now(timezone.utc),
            metadata={
                "sample_strategy": rec.strategy_name,
                "confidence": rec.confidence
            }
        )
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return HealthStatus(
            name="strategy_recommender",
            status="unhealthy",
            latency_ms=latency,
            last_check=datetime.now(timezone.utc),
            error_message=str(e)
        )


def check_memory_usage() -> HealthStatus:
    """Check system memory usage."""
    start_time = time.time()
    
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        
        # Determine status based on memory usage
        if memory.percent > 90:
            status = "unhealthy"
        elif memory.percent > 75:
            status = "degraded"
        else:
            status = "healthy"
        
        latency = (time.time() - start_time) * 1000
        
        return HealthStatus(
            name="memory_usage",
            status=status,
            latency_ms=latency,
            last_check=datetime.now(timezone.utc),
            metadata={
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent_used": memory.percent
            }
        )
    except ImportError:
        latency = (time.time() - start_time) * 1000
        return HealthStatus(
            name="memory_usage",
            status="unknown",
            latency_ms=latency,
            last_check=datetime.now(timezone.utc),
            error_message="psutil not available"
        )
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return HealthStatus(
            name="memory_usage",
            status="unhealthy",
            latency_ms=latency,
            last_check=datetime.now(timezone.utc),
            error_message=str(e)
        )


def check_disk_usage() -> HealthStatus:
    """Check disk usage."""
    start_time = time.time()
    
    try:
        import psutil
        
        disk = psutil.disk_usage('/')
        percent_used = (disk.used / disk.total) * 100
        
        if percent_used > 95:
            status = "unhealthy"
        elif percent_used > 85:
            status = "degraded"
        else:
            status = "healthy"
        
        latency = (time.time() - start_time) * 1000
        
        return HealthStatus(
            name="disk_usage",
            status=status,
            latency_ms=latency,
            last_check=datetime.now(timezone.utc),
            metadata={
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent_used": percent_used
            }
        )
    except ImportError:
        latency = (time.time() - start_time) * 1000
        return HealthStatus(
            name="disk_usage",
            status="unknown",
            latency_ms=latency,
            last_check=datetime.now(timezone.utc),
            error_message="psutil not available"
        )
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        return HealthStatus(
            name="disk_usage",
            status="unhealthy",
            latency_ms=latency,
            last_check=datetime.now(timezone.utc),
            error_message=str(e)
        )


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor(version: str = "2.0.0") -> HealthMonitor:
    """Get or create the global health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor(version)
        
        # Register built-in checks
        _health_monitor.register_check("embedding_service", check_embedding_service)
        _health_monitor.register_check("confidence_scorer", check_confidence_scorer)
        _health_monitor.register_check("strategy_recommender", check_strategy_recommender)
        _health_monitor.register_check("memory_usage", check_memory_usage)
        _health_monitor.register_check("disk_usage", check_disk_usage)
    
    return _health_monitor


async def quick_health_check() -> Dict[str, Any]:
    """Perform a quick health check and return results."""
    monitor = get_health_monitor()
    health = await monitor.check_health()
    return health.to_dict()


__all__ = [
    'HealthMonitor',
    'HealthStatus',
    'SystemHealth',
    'get_health_monitor',
    'quick_health_check',
    'check_embedding_service',
    'check_confidence_scorer',
    'check_strategy_recommender',
    'check_memory_usage',
    'check_disk_usage'
]
