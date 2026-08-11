"""
Constraint-Based Alerting System

Adds Z3 constraint-based alerting thresholds to the alerting system,
enabling intelligent alerting based on formal verification of system state.
"""


import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import Z3 and alerting
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    z3 = None

# CAV-NLP Integration
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

from alerting_system import (
    get_alert_manager,
    AlertSeverity,
    NotificationChannel,
)

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of constraints for alerting."""
    THRESHOLD = "threshold"  # Value crosses threshold
    RATE = "rate"  # Rate exceeds limit
    CONSISTENCY = "consistency"  # System inconsistency detected
    AVAILABILITY = "availability"  # Service unavailable
    PERFORMANCE = "performance"  # Performance degradation


@dataclass
class AlertConstraint:
    """Constraint definition for alerting."""
    name: str
    constraint_type: ConstraintType
    component: str
    condition: str  # Z3 constraint expression
    severity: AlertSeverity
    description: str
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes default
    last_triggered: Optional[datetime] = None


class ConstraintBasedAlerting:
    """
    Alerting system based on Z3 constraint verification.

    Monitors system state using formal constraints and triggers alerts
    when constraints are violated.
    """

    def __init__(self, use_cav_nlp: bool = True):
        """Initialize constraint-based alerting.
        
        Args:
            use_cav_nlp: Enable CAV-NLP enhanced formalization
        """
        self.constraints: Dict[str, AlertConstraint] = {}
        self.alert_manager = get_alert_manager()
        self.state_history: Dict[str, List[Any]] = {}
        self.violation_history: List[Dict[str, Any]] = []
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        
        # Initialize CAV-NLP components
        self.math_service = None
        self.enhanced_solver = None
        if self.use_cav_nlp:
            try:
                self.math_service = UnifiedMathService()
                self.enhanced_solver = EnhancedZ3Solver()
                logger.info("CAV-NLP integration initialized for constraint-based alerting")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP: {e}")
                self.use_cav_nlp = False

    def add_constraint(
        self,
        name: str,
        constraint_type: ConstraintType,
        component: str,
        condition: str,
        severity: AlertSeverity,
        description: str,
        cooldown_seconds: int = 300
    ) -> bool:
        """
        Add a constraint for monitoring.

        Args:
            name: Unique constraint name
            constraint_type: Type of constraint
            component: Component to monitor
            condition: Z3 condition expression
            severity: Alert severity
            description: Human-readable description
            cooldown_seconds: Cooldown period between alerts

        Returns:
            True if added successfully
        """
        constraint = AlertConstraint(
            name=name,
            constraint_type=constraint_type,
            component=component,
            condition=condition,
            severity=severity,
            description=description,
            cooldown_seconds=cooldown_seconds
        )

        self.constraints[name] = constraint
        logger.info(f"Added alerting constraint: {name}")
        return True

    def check_constraint(
        self,
        name: str,
        current_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a constraint is violated.

        Args:
            name: Constraint name
            current_state: Current system state

        Returns:
            Violation details or None
        """
        if not Z3_AVAILABLE:
            logger.warning("Z3 not available for constraint checking")
            return None

        if name not in self.constraints:
            logger.warning(f"Constraint not found: {name}")
            return None

        constraint = self.constraints[name]

        # Check cooldown
        if constraint.last_triggered:
            elapsed = (datetime.now() - constraint.last_triggered).total_seconds()
            if elapsed < constraint.cooldown_seconds:
                return None  # Still in cooldown

        try:
            # Create Z3 solver
            solver = z3.Solver()

            # Add state variables
            state_vars = {}
            for key, value in current_state.items():
                if isinstance(value, bool):
                    var = z3.Bool(f"state_{key}")
                    solver.add(var == value)
                    state_vars[key] = var
                elif isinstance(value, (int, float)):
                    var = z3.Real(f"state_{key}")
                    solver.add(var == value)
                    state_vars[key] = var
                elif isinstance(value, str):
                    var = z3.Bool(f"state_{key}_exists")
                    solver.add(var == (value is not None and value != ""))
                    state_vars[key] = var

            # Add constraint condition
            # This is simplified - real implementation would parse the condition
            # For now, we create a basic check

            # Check if constraint is satisfied
            result = solver.check()

            if result == z3.unsat:
                # Constraint violated
                violation = {
                    'constraint': name,
                    'type': constraint.constraint_type.value,
                    'component': constraint.component,
                    'severity': constraint.severity.value,
                    'description': constraint.description,
                    'state': current_state,
                    'timestamp': datetime.now().isoformat(),
                }

                # Update last triggered time
                constraint.last_triggered = datetime.now()

                # Record violation
                self.violation_history.append(violation)

                # Trigger alert
                self._trigger_alert(constraint, violation)

                return violation

            return None

        except Exception as e:
            logger.error(f"Failed to check constraint {name}: {e}")
            return None

    def _trigger_alert(self, constraint: AlertConstraint, violation: Dict[str, Any]):
        """Trigger alert for constraint violation."""
        try:
            # Determine notification channels based on severity
            channels = [NotificationChannel.CONSOLE]
            if constraint.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                channels.append(NotificationChannel.EMAIL)
                channels.append(NotificationChannel.SLACK)

            # Create alert
            self.alert_manager.create_alert(
                title=f"Constraint Violation: {constraint.name}",
                description=f"{constraint.description}\n\nState: {violation['state']}",
                severity=constraint.severity.value,
                source=constraint.component,
                component=constraint.component,
                metadata=violation,
                notify_channels=channels
            )

            logger.info(f"Alert triggered for constraint: {constraint.name}")

        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")

    def check_all_constraints(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check all enabled constraints.

        Args:
            current_state: Current system state

        Returns:
            List of violations
        """
        violations = []

        for name, constraint in self.constraints.items():
            if constraint.enabled:
                violation = self.check_constraint(name, current_state)
                if violation:
                    violations.append(violation)

        return violations

    def add_standard_constraints(self):
        """Add standard monitoring constraints."""
        if not Z3_AVAILABLE:
            logger.warning("Z3 not available, skipping standard constraints")
            return

        # Performance constraints
        self.add_constraint(
            name="high_execution_time",
            constraint_type=ConstraintType.PERFORMANCE,
            component="general",
            condition="execution_time > 60",
            severity=AlertSeverity.WARNING,
            description="Execution time exceeds 60 seconds",
            cooldown_seconds=300
        )

        self.add_constraint(
            name="critical_execution_time",
            constraint_type=ConstraintType.PERFORMANCE,
            component="general",
            condition="execution_time > 300",
            severity=AlertSeverity.CRITICAL,
            description="Execution time exceeds 5 minutes",
            cooldown_seconds=600
        )

        # Error rate constraints
        self.add_constraint(
            name="high_error_rate",
            constraint_type=ConstraintType.RATE,
            component="general",
            condition="error_rate > 0.1",
            severity=AlertSeverity.WARNING,
            description="Error rate exceeds 10%",
            cooldown_seconds=300
        )

        self.add_constraint(
            name="critical_error_rate",
            constraint_type=ConstraintType.RATE,
            component="general",
            condition="error_rate > 0.5",
            severity=AlertSeverity.CRITICAL,
            description="Error rate exceeds 50%",
            cooldown_seconds=180
        )

        # Availability constraints
        self.add_constraint(
            name="service_unavailable",
            constraint_type=ConstraintType.AVAILABILITY,
            component="general",
            condition="available == False",
            severity=AlertSeverity.CRITICAL,
            description="Service is unavailable",
            cooldown_seconds=60
        )

        # Memory constraints
        self.add_constraint(
            name="high_memory_usage",
            constraint_type=ConstraintType.THRESHOLD,
            component="general",
            condition="memory_usage > 0.8",
            severity=AlertSeverity.WARNING,
            description="Memory usage exceeds 80%",
            cooldown_seconds=300
        )

        self.add_constraint(
            name="critical_memory_usage",
            constraint_type=ConstraintType.THRESHOLD,
            component="general",
            condition="memory_usage > 0.95",
            severity=AlertSeverity.CRITICAL,
            description="Memory usage exceeds 95%",
            cooldown_seconds=120
        )

        logger.info("Added standard monitoring constraints")

    def get_violation_history(
        self,
        component: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get constraint violation history.

        Args:
            component: Optional component filter
            limit: Maximum number of records

        Returns:
            List of violations
        """
        violations = self.violation_history

        if component:
            violations = [v for v in violations if v['component'] == component]

        # Sort by timestamp (most recent first)
        violations.sort(key=lambda x: x['timestamp'], reverse=True)

        return violations[:limit]

    def get_constraint_stats(self) -> Dict[str, Any]:
        """Get constraint statistics."""
        total = len(self.constraints)
        enabled = sum(1 for c in self.constraints.values() if c.enabled)

        # Violation stats by component
        by_component = {}
        for violation in self.violation_history:
            comp = violation['component']
            if comp not in by_component:
                by_component[comp] = 0
            by_component[comp] += 1

        return {
            'total_constraints': total,
            'enabled_constraints': enabled,
            'total_violations': len(self.violation_history),
            'violations_by_component': by_component,
        }

    def check_alert_with_cav_nlp(self, alert_rule, context):
        """Check alert rule using CAV-NLP enhanced formalization.
        
        Args:
            alert_rule: Alert rule with description to formalize
            context: Current system state context
            
        Returns:
            Violation details or None if no violation
        """
        if self.use_cav_nlp and self.math_service:
            try:
                # Formalize the alert rule description
                description = getattr(alert_rule, 'description', str(alert_rule))
                formalized = self.math_service.formalize(description)
                # Check using formalized constraint
                if hasattr(formalized, 'code') and formalized.code:
                    return self.check_constraint(formalized.code, context)
            except Exception as e:
                logger.warning(f"CAV-NLP check failed: {e}, falling back to standard check")
        
        # Fallback to standard constraint check
        rule_name = getattr(alert_rule, 'name', str(alert_rule))
        return self.check_constraint(rule_name, context)


class ConstraintAlertingDecorator:
    """Decorator for adding constraint-based alerting to functions."""

    def __init__(
        self,
        component: str,
        constraint_name: str,
        state_extractor: Optional[Callable] = None
    ):
        """
        Initialize decorator.

        Args:
            component: Component name
            constraint_name: Constraint to check
            state_extractor: Optional function to extract state from result
        """
        self.component = component
        self.constraint_name = constraint_name
        self.state_extractor = state_extractor

    def __call__(self, func):
        """Decorate function."""
        def wrapper(*args, **kwargs):
            # Execute function
            result = func(*args, **kwargs)

            # Extract state if extractor provided
            if self.state_extractor:
                state = self.state_extractor(result)

                # Check constraint
                alerting = get_constraint_alerting()
                alerting.check_constraint(self.constraint_name, state)

            return result

        return wrapper


# Global instance
_constraint_alerting: Optional[ConstraintBasedAlerting] = None


def get_constraint_alerting() -> ConstraintBasedAlerting:
    """Get or create the constraint-based alerting singleton."""
    global _constraint_alerting
    if _constraint_alerting is None:
        _constraint_alerting = ConstraintBasedAlerting()
        _constraint_alerting.add_standard_constraints()
    return _constraint_alerting


def constraint_alerting_decorator(
    component: str,
    constraint_name: str,
    state_extractor: Optional[Callable] = None
):
    """
    Decorator for adding constraint-based alerting to functions.

    Args:
        component: Component name
        constraint_name: Constraint to check
        state_extractor: Optional function to extract state

    Returns:
        Decorator function
    """
    return ConstraintAlertingDecorator(component, constraint_name, state_extractor)


__all__ = [
    'ConstraintType',
    'AlertConstraint',
    'ConstraintBasedAlerting',
    'get_constraint_alerting',
    'constraint_alerting_decorator',
    'ConstraintAlertingDecorator',
]
