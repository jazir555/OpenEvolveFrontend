"""
Per-Level Circuit Breakers for OpenEvolve Gauntlet System

Implements hierarchical circuit breakers that operate at each level
of the problem decomposition hierarchy, providing fault isolation
and preventing cascading failures.

Key Features:
- Hierarchical breaker management
- Level-specific threshold calculation
- Dynamic threshold adjustment
- Circuit breaker state tracking
- Performance metrics per level
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """States of a circuit breaker"""
    CLOSED = "closed"  # Operating normally
    OPEN = "open"  # Failing, not allowing requests
    HALF_OPEN = "half_open"  # Testing if system has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker"""
    level: int  # Hierarchy level (0 = root)
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close after half-open
    timeout: float = 60.0  # Seconds to wait before trying half-open
    half_open_attempts: int = 3  # Attempts allowed in half-open state

    def __post_init__(self):
        # Adjust thresholds based on level
        # Higher levels (more decomposed) are more lenient
        level_multiplier = max(1, self.level + 1)
        self.failure_threshold = self.failure_threshold * level_multiplier
        self.success_threshold = max(1, self.success_threshold)
        self.half_open_attempts = max(1, self.half_open_attempts)


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker"""
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    opened_count: int = 0  # How many times breaker has opened
    total_requests: int = 0
    last_state_change: Optional[datetime] = None


class LevelCircuitBreaker:
    """
    Circuit breaker for a specific hierarchy level.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.stats = CircuitBreakerStats()
        self.half_open_attempts_used = 0

    async def execute(
        self,
        operation: callable,
        context: Dict[str, Any] = None
    ) -> tuple[bool, Any, Optional[str]]:
        """
        Execute an operation through the circuit breaker.

        Args:
            operation: Async function to execute
            context: Execution context

        Returns:
            Tuple of (success, result, error_message)
        """
        self.stats.total_requests += 1

        # Check if breaker is open and should transition to half-open
        if self.stats.state == CircuitBreakerState.OPEN:
            if self._should_attempt_half_open():
                self._transition_to_half_open()
            else:
                return (False, None, f"Circuit breaker OPEN for level {self.config.level}")

        # Execute operation
        try:
            result = await operation(context or {})

            # Success
            self._record_success()
            return (True, result, None)

        except Exception as e:
            # Failure
            self._record_failure(str(e))
            return (False, None, f"Operation failed: {str(e)}")

    def _should_attempt_half_open(self) -> bool:
        """Check if enough time has passed to try half-open"""
        if self.stats.last_failure_time is None:
            return True

        elapsed = (datetime.utcnow() - self.stats.last_failure_time).total_seconds()
        return elapsed >= self.config.timeout

    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN"""
        logger.info(
            f"Level {self.config.level} circuit breaker transitioning "
            f"from OPEN to HALF_OPEN"
        )
        self.stats.state = CircuitBreakerState.HALF_OPEN
        self.stats.last_state_change = datetime.utcnow()
        self.half_open_attempts_used = 0

    def _transition_to_open(self):
        """Transition from CLOSED or HALF_OPEN to OPEN"""
        logger.warning(
            f"Level {self.config.level} circuit breaker OPENING "
            f"(failures: {self.stats.failure_count})"
        )
        self.stats.state = CircuitBreakerState.OPEN
        self.stats.last_state_change = datetime.utcnow()
        self.stats.opened_count += 1

    def _transition_to_closed(self):
        """Transition from HALF_OPEN to CLOSED"""
        logger.info(
            f"Level {self.config.level} circuit breaker CLOSING "
            f"(system recovered)"
        )
        self.stats.state = CircuitBreakerState.CLOSED
        self.stats.last_state_change = datetime.utcnow()
        self.stats.failure_count = 0
        self.stats.half_open_attempts_used = 0

    def _record_success(self):
        """Record a successful operation"""
        self.stats.success_count += 1
        self.stats.last_success_time = datetime.utcnow()

        if self.stats.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_attempts_used += 1

            # Check if we should close the breaker
            if self.stats.success_count >= self.config.success_threshold:
                self._transition_to_closed()

    def _record_failure(self, error: str):
        """Record a failed operation"""
        self.stats.failure_count += 1
        self.stats.last_failure_time = datetime.utcnow()

        if self.stats.state == CircuitBreakerState.CLOSED:
            # Check if we should open the breaker
            if self.stats.failure_count >= self.config.failure_threshold:
                self._transition_to_open()

        elif self.stats.state == CircuitBreakerState.HALF_OPEN:
            # Immediate back to open on failure in half-open
            self._transition_to_open()

    def get_state(self) -> CircuitBreakerState:
        """Get current breaker state"""
        return self.stats.state

    def get_stats(self) -> CircuitBreakerStats:
        """Get breaker statistics"""
        return self.stats

    def reset(self):
        """Reset the breaker to initial state"""
        logger.info(f"Resetting level {self.config.level} circuit breaker")
        self.stats = CircuitBreakerStats()
        self.half_open_attempts_used = 0


class HierarchicalCircuitBreakerManager:
    """
    Manages circuit breakers across all hierarchy levels.

    Ensures that failures at one level don't cascade to other levels
    while allowing independent recovery.
    """

    def __init__(self, default_config: CircuitBreakerConfig = None):
        self.default_config = default_config or CircuitBreakerConfig(level=0)
        self.breakers: Dict[int, LevelCircuitBreaker] = {}

    def get_breaker(self, level: int, config: CircuitBreakerConfig = None) -> LevelCircuitBreaker:
        """
        Get or create a circuit breaker for a level.

        Args:
            level: Hierarchy level
            config: Optional custom configuration

        Returns:
            LevelCircuitBreaker instance
        """
        if level not in self.breakers:
            config = config or CircuitBreakerConfig(level=level)
            self.breakers[level] = LevelCircuitBreaker(config)

        return self.breakers[level]

    async def execute_at_level(
        self,
        level: int,
        operation: callable,
        context: Dict[str, Any] = None
    ) -> tuple[bool, Any, Optional[str]]:
        """
        Execute an operation at a specific level with circuit breaking.

        Args:
            level: Hierarchy level
            operation: Operation to execute
            context: Execution context

        Returns:
            Tuple of (success, result, error_message)
        """
        breaker = self.get_breaker(level)
        return await breaker.execute(operation, context)

    def get_all_states(self) -> Dict[int, CircuitBreakerState]:
        """Get states of all breakers"""
        return {
            level: breaker.get_state()
            for level, breaker in self.breakers.items()
        }

    def get_all_stats(self) -> Dict[int, CircuitBreakerStats]:
        """Get statistics for all breakers"""
        return {
            level: breaker.get_stats()
            for level, breaker in self.breakers.items()
        }

    def reset_level(self, level: int):
        """Reset a specific level's breaker"""
        if level in self.breakers:
            self.breakers[level].reset()

    def reset_all(self):
        """Reset all breakers"""
        for breaker in self.breakers.values():
            breaker.reset()

    def cleanup_stale(self, max_age_hours: int = 24):
        """Remove breakers that haven't been used recently"""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        stale_levels = [
            level for level, breaker in self.breakers.items()
            if breaker.stats.last_state_change
            and breaker.stats.last_state_change < cutoff
            and breaker.stats.total_requests == 0
        ]

        for level in stale_levels:
            del self.breakers[level]
            logger.info(f"Cleaned up stale breaker for level {level}")


class CircuitBreakerDashboard:
    """
    Provides dashboard-style reporting for circuit breakers.
    """

    def __init__(self, manager: HierarchicalCircuitBreakerManager):
        self.manager = manager

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard report"""
        states = self.manager.get_all_states()
        stats = self.manager.get_all_stats()

        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_breakers': len(self.manager.breakers),
            'states': {
                'closed': 0,
                'open': 0,
                'half_open': 0,
            },
            'levels': {},
            'summary': self._generate_summary(stats),
        }

        # Count states
        for state in states.values():
            report['states'][state.value] += 1

        # Per-level details
        for level, breaker_stats in stats.items():
            report['levels'][level] = {
                'state': breaker_stats.state.value,
                'failures': breaker_stats.failure_count,
                'successes': breaker_stats.success_count,
                'total_requests': breaker_stats.total_requests,
                'opened_count': breaker_stats.opened_count,
                'failure_rate': self._calculate_failure_rate(breaker_stats),
                'last_state_change': breaker_stats.last_state_change.isoformat()
                if breaker_stats.last_state_change else None,
            }

        return report

    def _generate_summary(self, stats: Dict[int, CircuitBreakerStats]) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_requests = sum(s.total_requests for s in stats.values())
        total_failures = sum(s.failure_count for s in stats.values())
        total_opened = sum(s.opened_count for s in stats.values())

        open_breakers = sum(
            1 for s in stats.values()
            if s.state == CircuitBreakerState.OPEN
        )

        return {
            'total_requests': total_requests,
            'total_failures': total_failures,
            'overall_failure_rate': total_failures / total_requests if total_requests > 0 else 0,
            'total_breaker_opens': total_opened,
            'breakers_currently_open': open_breakers,
            'health_status': self._assess_health(open_breakers, len(stats)),
        }

    def _calculate_failure_rate(self, stats: CircuitBreakerStats) -> float:
        """Calculate failure rate for a breaker"""
        if stats.total_requests == 0:
            return 0.0
        return stats.failure_count / stats.total_requests

    def _assess_health(self, open_count: int, total_count: int) -> str:
        """Assess overall system health"""
        if total_count == 0:
            return "unknown"

        open_ratio = open_count / total_count

        if open_ratio == 0:
            return "healthy"
        elif open_ratio < 0.2:
            return "degraded"
        elif open_ratio < 0.5:
            return "poor"
        else:
            return "critical"

    def format_text_report(self) -> str:
        """Format dashboard report as text"""
        report = self.generate_report()
        lines = []

        lines.append("=" * 60)
        lines.append("CIRCUIT BREAKER DASHBOARD")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {report['timestamp']}")
        lines.append(f"Total Breakers: {report['total_breakers']}")
        lines.append("")
        lines.append("State Summary:")
        lines.append(f"  Closed:    {report['states']['closed']}")
        lines.append(f"  Open:      {report['states']['open']}")
        lines.append(f"  Half-Open: {report['states']['half_open']}")
        lines.append("")
        lines.append("Overall Health:")
        summary = report['summary']
        lines.append(f"  Status: {summary['health_status'].upper()}")
        lines.append(f"  Total Requests: {summary['total_requests']}")
        lines.append(f"  Failure Rate: {summary['overall_failure_rate']:.1%}")
        lines.append(f"  Breakers Open: {summary['breakers_currently_open']}")
        lines.append("")

        if report['levels']:
            lines.append("Per-Level Details:")
            for level in sorted(report['levels'].keys()):
                level_data = report['levels'][level]
                lines.append(f"  Level {level}:")
                lines.append(f"    State: {level_data['state'].upper()}")
                lines.append(f"    Requests: {level_data['total_requests']}")
                lines.append(f"    Failures: {level_data['failures']}")
                lines.append(f"    Failure Rate: {level_data['failure_rate']:.1%}")
                lines.append(f"    Opened: {level_data['opened_count']} times")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def create_hierarchical_breaker_manager(
    base_failure_threshold: int = 5,
    base_timeout: float = 60.0
) -> HierarchicalCircuitBreakerManager:
    """
    Factory function to create hierarchical breaker manager.

    Args:
        base_failure_threshold: Base failure threshold for level 0
        base_timeout: Base timeout in seconds

    Returns:
        HierarchicalCircuitBreakerManager instance
    """
    default_config = CircuitBreakerConfig(
        level=0,
        failure_threshold=base_failure_threshold,
        timeout=base_timeout
    )

    return HierarchicalCircuitBreakerManager(default_config=default_config)


# Example usage
async def demo_circuit_breakers():
    """Demonstration of hierarchical circuit breakers"""

    manager = create_hierarchical_breaker_manager()
    dashboard = CircuitBreakerDashboard(manager)

    # Simulate operations at different levels
    async def failing_operation(context):
        raise Exception("Simulated failure")

    async def successful_operation(context):
        return "Success"

    print("\n" + "=" * 60)
    print("Circuit Breaker Demo")
    print("=" * 60)

    # Level 0 - Root problem
    print("\nLevel 0 (Root):")
    for i in range(10):
        success, result, error = await manager.execute_at_level(
            0,
            successful_operation if i < 2 else failing_operation
        )
        print(f"  Attempt {i+1}: {'[OK]' if success else '[FAIL]'}")

    # Level 1 - Subproblems
    print("\nLevel 1 (Subproblems):")
    for i in range(8):
        success, result, error = await manager.execute_at_level(
            1,
            successful_operation if i < 3 else failing_operation
        )
        print(f"  Attempt {i+1}: {'[OK]' if success else '[FAIL]'}")

    # Level 2 - Atomic
    print("\nLevel 2 (Atomic):")
    for i in range(3):
        success, result, error = await manager.execute_at_level(
            2,
            successful_operation
        )
        print(f"  Attempt {i+1}: {'[OK]' if success else '[FAIL]'}")

    # Show dashboard
    print("\n" + dashboard.format_text_report())


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_circuit_breakers())
