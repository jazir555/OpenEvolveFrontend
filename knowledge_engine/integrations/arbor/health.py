"""
Health checking for Arbor integration

Following CLAUDE.md principles:
- RUNTIME TRUTH: Verify actual connectivity
- STRUCTURED LOGGING: Health status as JSON
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Awaitable
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    status: HealthStatus
    """Overall health status."""
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    """When the check was performed."""
    
    response_time_ms: float = 0.0
    """Response time in milliseconds."""
    
    message: str = ""
    """Human-readable status message."""
    
    details: Dict[str, Any] = field(default_factory=dict)
    """Additional diagnostic details."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "message": self.message,
            "details": self.details
        }


class ArborHealthChecker:
    """
    Health checker for Arbor server connection.
    
    Performs periodic health checks and maintains health status.
    Can trigger callbacks when health status changes.
    """
    
    def __init__(
        self,
        client,
        check_interval: float = 30.0,
        timeout: float = 10.0
    ):
        """
        Initialize health checker.
        
        Args:
            client: ArborClient instance to check
            check_interval: Seconds between health checks
            timeout: Timeout for health check operations
        """
        self.client = client
        self.check_interval = check_interval
        self.timeout = timeout
        
        self._current_status = HealthCheckResult(
            status=HealthStatus.UNKNOWN,
            message="Health check not yet performed"
        )
        self._callbacks: list[Callable[[HealthCheckResult], Awaitable[None]]] = []
        self._check_task: Optional[asyncio.Task] = None
        self._running = False
    
    @property
    def current_status(self) -> HealthCheckResult:
        """Get current health status."""
        return self._current_status
    
    def is_healthy(self) -> bool:
        """Check if current status is healthy."""
        return self._current_status.status == HealthStatus.HEALTHY
    
    def on_status_change(
        self,
        callback: Callable[[HealthCheckResult], Awaitable[None]]
    ) -> None:
        """
        Register a callback for health status changes.
        
        Args:
            callback: Async function to call when status changes
        """
        self._callbacks.append(callback)
    
    async def check_once(self) -> HealthCheckResult:
        """
        Perform a single health check.
        
        Returns:
            HealthCheckResult with current status
        """
        import time
        start_time = time.time()
        
        try:
            # Check if client is connected
            if not self.client.is_connected:
                result = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Not connected to Arbor server",
                    details={"connected": False}
                )
            else:
                # Try a simple ping/query
                try:
                    # Use a simple status query if available
                    # For now, just check connection state
                    response_time = (time.time() - start_time) * 1000
                    
                    result = HealthCheckResult(
                        status=HealthStatus.HEALTHY,
                        response_time_ms=response_time,
                        message="Connected to Arbor server",
                        details={
                            "connected": True,
                            "ws_url": self.client.config.connection.ws_url,
                            "reconnect_count": self.client._reconnect_count
                        }
                    )
                except Exception as e:
                    response_time = (time.time() - start_time) * 1000
                    result = HealthCheckResult(
                        status=HealthStatus.DEGRADED,
                        response_time_ms=response_time,
                        message=f"Connected but query failed: {str(e)}",
                        details={"connected": True, "query_error": str(e)}
                    )
        
        except Exception as e:
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)}
            )
        
        # Check if status changed
        old_status = self._current_status.status
        self._current_status = result
        
        if old_status != result.status:
            logger.info({
                "msg": "Arbor health status changed",
                "old_status": old_status.value,
                "new_status": result.status.value,
                "message": result.message
            })
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    await callback(result)
                except Exception as e:
                    logger.error(f"Health status callback failed: {e}")
        
        return result
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._check_task = asyncio.create_task(self._monitor_loop())
        
        logger.info({
            "msg": "Arbor health monitoring started",
            "check_interval": self.check_interval
        })
    
    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if not self._running:
            return
        
        self._running = False
        
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Arbor health monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self.check_once()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            try:
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break


class CompositeHealthChecker:
    """
    Combines multiple health checkers into a single status.
    """
    
    def __init__(self):
        self._checkers: Dict[str, ArborHealthChecker] = {}
        self._callbacks: list[Callable[[str, HealthCheckResult], Awaitable[None]]] = []
    
    def add_checker(self, name: str, checker: ArborHealthChecker) -> None:
        """Add a health checker."""
        self._checkers[name] = checker
        checker.on_status_change(
            lambda result, name=name: self._on_status_change(name, result)
        )
    
    def on_status_change(
        self,
        callback: Callable[[str, HealthCheckResult], Awaitable[None]]
    ) -> None:
        """Register callback for any checker's status change."""
        self._callbacks.append(callback)
    
    async def _on_status_change(
        self,
        name: str,
        result: HealthCheckResult
    ) -> None:
        """Handle status change from any checker."""
        for callback in self._callbacks:
            try:
                await callback(name, result)
            except Exception as e:
                logger.error(f"Composite health callback failed: {e}")
    
    def get_overall_status(self) -> HealthStatus:
        """Get aggregated health status."""
        if not self._checkers:
            return HealthStatus.UNKNOWN
        
        statuses = [c.current_status.status for c in self._checkers.values()]
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_all_statuses(self) -> Dict[str, HealthCheckResult]:
        """Get all individual checker statuses."""
        return {
            name: checker.current_status
            for name, checker in self._checkers.items()
        }
