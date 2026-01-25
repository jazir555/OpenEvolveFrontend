"""
Circuit Breaker Dashboard API

Provides HTTP endpoints for monitoring and managing circuit breakers.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from .circuit_breakers import (
    CircuitBreakerState,
    CircuitBreakerConfig,
    LevelCircuitBreaker,
    HierarchicalCircuitBreakerManager,
)

logger = logging.getLogger(__name__)


@dataclass
class BreakerStatus:
    """Status of a circuit breaker"""
    level: int
    state: str
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    opened_count: int
    config: Dict[str, Any]


@dataclass
class DashboardSummary:
    """Summary of all circuit breakers"""
    total_breakers: int
    open_breakers: int
    half_open_breakers: int
    closed_breakers: int
    breaker_statuses: List[BreakerStatus]
    generated_at: datetime


class CircuitBreakerDashboard:
    """
    Dashboard for monitoring circuit breaker status.
    """

    def __init__(self, manager: HierarchicalCircuitBreakerManager):
        self.manager = manager
        self.history: Dict[int, List[Dict]] = {}  # Per-level history
        self.max_history_size = 1000

    def get_breaker_status(self, level: int) -> BreakerStatus:
        """
        Get current status of a breaker.

        Args:
            level: Hierarchy level

        Returns:
            BreakerStatus with current state
        """
        breaker = self.manager.get_breaker(level)
        metrics = breaker.get_metrics()

        return BreakerStatus(
            level=level,
            state=breaker.state.value,
            failure_count=breaker.failure_count,
            success_count=metrics.get('success_count', 0),
            last_failure_time=breaker.last_failure_time,
            last_success_time=breaker.last_success_time,
            opened_count=metrics.get('opened_count', 0),
            config={
                'failure_threshold': breaker.config.failure_threshold,
                'recovery_timeout_seconds': breaker.config.recovery_timeout_seconds,
                'half_open_max_calls': breaker.config.half_open_max_calls,
                'strategy': breaker.config.strategy.value,
            }
        )

    def get_all_breaker_statuses(self) -> List[BreakerStatus]:
        """Get status of all breakers"""
        statuses = []
        states = self.manager.get_all_states()

        for level in states.keys():
            status = self.get_breaker_status(level)
            statuses.append(status)

        return statuses

    def get_dashboard_summary(self) -> DashboardSummary:
        """
        Get dashboard summary.

        Returns:
            DashboardSummary with overview
        """
        statuses = self.get_all_breaker_statuses()

        open_count = sum(1 for s in statuses if s.state == CircuitBreakerState.OPEN.value)
        half_open_count = sum(1 for s in statuses if s.state == CircuitBreakerState.HALF_OPEN.value)
        closed_count = sum(1 for s in statuses if s.state == CircuitBreakerState.CLOSED.value)

        return DashboardSummary(
            total_breakers=len(statuses),
            open_breakers=open_count,
            half_open_breakers=half_open_count,
            closed_breakers=closed_count,
            breaker_statuses=statuses,
            generated_at=datetime.utcnow()
        )

    def get_breaker_history(
        self,
        level: int,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get historical state changes for a breaker.

        Args:
            level: Hierarchy level
            limit: Max number of history entries

        Returns:
            List of historical state changes
        """
        if level not in self.history:
            return []

        return self.history[level][-limit:]

    def record_state_change(
        self,
        level: int,
        old_state: CircuitBreakerState,
        new_state: CircuitBreakerState,
        reason: str = None
    ):
        """
        Record a state change in history.

        Args:
            level: Hierarchy level
            old_state: Previous state
            new_state: New state
            reason: Optional reason for change
        """
        if level not in self.history:
            self.history[level] = []

        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'old_state': old_state.value,
            'new_state': new_state.value,
            'reason': reason,
        }

        self.history[level].append(entry)

        # Trim history if needed
        if len(self.history[level]) > self.max_history_size:
            self.history[level] = self.history[level][-self.max_history_size:]

    def get_breaker_metrics(self, level: int) -> Dict[str, Any]:
        """
        Get metrics for a breaker.

        Args:
            level: Hierarchy level

        Returns:
            Metrics dictionary
        """
        breaker = self.manager.get_breaker(level)
        return breaker.get_metrics()

    def get_all_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Get metrics for all breakers"""
        metrics = {}
        statuses = self.manager.get_all_states()

        for level in statuses.keys():
            metrics[level] = self.get_breaker_metrics(level)

        return metrics

    def reset_breaker(self, level: int) -> bool:
        """
        Reset a breaker to closed state.

        Args:
            level: Hierarchy level

        Returns:
            True if reset successful
        """
        try:
            breaker = self.manager.get_breaker(level)
            breaker.state = CircuitBreakerState.CLOSED
            breaker.failure_count = 0
            breaker.last_failure_time = None

            self.record_state_change(
                level,
                CircuitBreakerState.OPEN,  # Assume it was open
                CircuitBreakerState.CLOSED,
                reason="Manual reset"
            )

            logger.info(f"Reset circuit breaker for level {level}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset breaker for level {level}: {e}")
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall system health status.

        Returns:
            Health status dictionary
        """
        summary = self.get_dashboard_summary()

        # Calculate health score
        total = summary.total_breakers
        if total == 0:
            health_score = 100
        else:
            # Weight open breakers heavily
            health_score = (
                (summary.closed_breakers * 100) +
                (summary.half_open_breakers * 50) +
                (summary.open_breakers * 0)
            ) / total

        return {
            'health_score': round(health_score, 2),
            'status': 'healthy' if health_score >= 80 else 'degraded' if health_score >= 50 else 'unhealthy',
            'total_breakers': total,
            'open_breakers': summary.open_breakers,
            'timestamp': datetime.utcnow().isoformat(),
        }


class CircuitBreakerAPI:
    """
    HTTP API for circuit breaker dashboard.
    """

    def __init__(self, dashboard: CircuitBreakerDashboard):
        self.dashboard = dashboard

    def get_status(self, level: int) -> Dict[str, Any]:
        """GET /api/circuit-breakers/{level}/status"""
        status = self.dashboard.get_breaker_status(level)
        return asdict(status)

    def get_all_statuses(self) -> Dict[str, Any]:
        """GET /api/circuit-breakers/status"""
        summary = self.dashboard.get_dashboard_summary()
        return {
            'total_breakers': summary.total_breakers,
            'open_breakers': summary.open_breakers,
            'half_open_breakers': summary.half_open_breakers,
            'closed_breakers': summary.closed_breakers,
            'breakers': [asdict(s) for s in summary.breaker_statuses],
            'generated_at': summary.generated_at.isoformat(),
        }

    def get_history(self, level: int, limit: int = 100) -> List[Dict[str, Any]]:
        """GET /api/circuit-breakers/{level}/history"""
        return self.dashboard.get_breaker_history(level, limit)

    def get_metrics(self, level: int) -> Dict[str, Any]:
        """GET /api/circuit-breakers/{level}/metrics"""
        return self.dashboard.get_breaker_metrics(level)

    def get_all_metrics(self) -> Dict[int, Dict[str, Any]]:
        """GET /api/circuit-breakers/metrics"""
        return self.dashboard.get_all_metrics()

    def get_health(self) -> Dict[str, Any]:
        """GET /api/circuit-breakers/health"""
        return self.dashboard.get_health_status()

    def reset_breaker(self, level: int) -> Dict[str, Any]:
        """POST /api/circuit-breakers/{level}/reset"""
        success = self.dashboard.reset_breaker(level)
        return {
            'success': success,
            'message': f"Breaker {level} reset" if success else f"Failed to reset breaker {level}",
            'timestamp': datetime.utcnow().isoformat(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """GET /api/circuit-breakers/summary"""
        summary = self.dashboard.get_dashboard_summary()

        return {
            'summary': {
                'total': summary.total_breakers,
                'open': summary.open_breakers,
                'half_open': summary.half_open_breakers,
                'closed': summary.closed_breakers,
            },
            'breakers': [
                {
                    'level': s.level,
                    'state': s.state,
                    'failures': s.failure_count,
                    'config': s.config,
                }
                for s in summary.breaker_statuses
            ],
            'health': self.dashboard.get_health_status(),
            'generated_at': summary.generated_at.isoformat(),
        }


# Integration with existing API server
def setup_circuit_breaker_routes(api_app, manager: HierarchicalCircuitBreakerManager):
    """
    Setup circuit breaker routes on an API app.

    Args:
        api_app: API application (e.g., FastAPI, Flask)
        manager: Circuit breaker manager
    """
    dashboard = CircuitBreakerDashboard(manager)
    api = CircuitBreakerAPI(dashboard)

    # Example for FastAPI
    try:
        from fastapi import APIRouter

        router = APIRouter(prefix="/api/circuit-breakers", tags=["circuit-breakers"])

        @router.get("/status")
        async def get_all_statuses():
            return api.get_all_statuses()

        @router.get("/{level}/status")
        async def get_status(level: int):
            return api.get_status(level)

        @router.get("/{level}/history")
        async def get_history(level: int, limit: int = 100):
            return api.get_history(level, limit)

        @router.get("/{level}/metrics")
        async def get_metrics(level: int):
            return api.get_metrics(level)

        @router.get("/metrics")
        async def get_all_metrics():
            return api.get_all_metrics()

        @router.get("/health")
        async def get_health():
            return api.get_health()

        @router.post("/{level}/reset")
        async def reset_breaker(level: int):
            return api.reset_breaker(level)

        @router.get("/summary")
        async def get_summary():
            return api.get_summary()

        api_app.include_router(router)

    except ImportError:
        logger.warning("FastAPI not available, skipping route setup")


# Convenience function
def create_dashboard(manager: HierarchicalCircuitBreakerManager) -> CircuitBreakerDashboard:
    """Create a circuit breaker dashboard"""
    return CircuitBreakerDashboard(manager)
