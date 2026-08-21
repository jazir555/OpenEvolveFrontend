"""
Universal Alerting Integration for OpenEvolve Frontend

Provides centralized alerting integration for all major system components:
- ROMA-MDAP-MAKER
- Decomposition Engine
- Z3 Verification
- CrewAI Workflows
- Knowledge Graph
- Caching Systems
- DataPizza
- ICR (Iterative Contextual Refinement)

This module wraps all critical operations with multi-channel alerting,
ensuring proactive monitoring and rapid incident response.
"""
from __future__ import annotations


import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
from contextlib import contextmanager
import traceback as tb

# Import alerting system
try:
    from alerting_system import (
        get_alert_manager,
        AlertManager,
        NotificationChannel,
        AlertSeverity
    )
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False
    # Create placeholder classes
    class AlertManager:
        def create_alert(self, **kwargs): return None
        def get_alert(self, alert_id): return None
    class NotificationChannel:
        EMAIL = "email"
        SLACK = "slack"
        WEBHOOK = "webhook"
        CONSOLE = "console"
    class AlertSeverity:
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"

logger = logging.getLogger(__name__)


class UniversalAlertingIntegration:
    """
    Centralized alerting integration for all OpenEvolve components.

    Provides decorators, context managers, and direct methods for
    wrapping operations with alerting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize universal alerting integration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._default_config()
        self.alert_manager: Optional[AlertManager] = None
        self.component_stats: Dict[str, Dict[str, int]] = {}

        if ALERTING_AVAILABLE:
            try:
                self.alert_manager = get_alert_manager()
                logger.info("Universal alerting integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize alert manager: {e}")

        # Initialize component statistics
        self._init_component_stats()

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enable_email': os.getenv('ALERT_EMAIL_ENABLED', 'false').lower() == 'true',
            'enable_slack': os.getenv('ALERT_SLACK_ENABLED', 'false').lower() == 'true',
            'enable_webhook': os.getenv('ALERT_WEBHOOK_ENABLED', 'false').lower() == 'true',
            'enable_console': True,  # Always enable console for debugging
            'default_severity': 'warning',
            'alert_on_success': False,
            'alert_on_failure': True,
            'include_traceback': True,
        }

    def _init_component_stats(self):
        """Initialize statistics tracking for components."""
        components = [
            'roma_mdap_maker',
            'decomposition_engine',
            'z3_verification',
            'crewai_workflows',
            'knowledge_graph',
            'caching_systems',
            'datapizza',
            'icr_refinement',
            'ace_engine',
            'claudiomiro',
            'adaptive_strategies',
        ]
        for component in components:
            self.component_stats[component] = {
                'total_operations': 0,
                'successful': 0,
                'failed': 0,
                'alerts_created': 0,
            }

    def track_operation(self, component: str, success: bool):
        """
        Track operation statistics for a component.

        Args:
            component: Component name
            success: Whether operation was successful
        """
        if component not in self.component_stats:
            self.component_stats[component] = {
                'total_operations': 0,
                'successful': 0,
                'failed': 0,
                'alerts_created': 0,
            }

        self.component_stats[component]['total_operations'] += 1
        if success:
            self.component_stats[component]['successful'] += 1
        else:
            self.component_stats[component]['failed'] += 1

    def create_alert(
        self,
        component: str,
        title: str,
        description: str,
        severity: str = 'warning',
        metadata: Optional[Dict[str, Any]] = None,
        notify_channels: Optional[List[str]] = None
    ):
        """
        Create an alert for a component.

        Args:
            component: Component name
            title: Alert title
            description: Alert description
            severity: Alert severity (info, warning, error, critical)
            metadata: Optional metadata
            notify_channels: Notification channels to use
        """
        if not self.alert_manager:
            logger.warning(f"Alert manager not available, cannot create alert: {title}")
            return None

        try:
            # Determine notification channels
            if notify_channels is None:
                notify_channels = []
                if self.config['enable_console']:
                    notify_channels.append(NotificationChannel.CONSOLE)
                if self.config['enable_email'] and severity in ['error', 'critical']:
                    notify_channels.append(NotificationChannel.EMAIL)
                if self.config['enable_slack'] and severity in ['error', 'critical']:
                    notify_channels.append(NotificationChannel.SLACK)
                if self.config['enable_webhook'] and severity in ['critical']:
                    notify_channels.append(NotificationChannel.WEBHOOK)

            # Add component to metadata
            if metadata is None:
                metadata = {}
            metadata['component'] = component
            metadata['timestamp'] = datetime.now().isoformat()

            # Create alert
            alert = self.alert_manager.create_alert(
                title=title,
                description=description,
                severity=severity,
                source=component,
                component=component,
                metadata=metadata,
                notify_channels=notify_channels
            )

            # Update statistics
            if component in self.component_stats:
                self.component_stats[component]['alerts_created'] += 1

            return alert

        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return None

    def alert_decorator(
        self,
        component: str,
        operation_name: Optional[str] = None,
        severity_on_error: str = 'error',
        alert_on_success: bool = False,
        include_args: bool = False,
        include_result: bool = False,
        custom_channels: Optional[List[str]] = None
    ):
        """
        Decorator for adding alerting to any function.

        Args:
            component: Component name
            operation_name: Optional operation name (defaults to function name)
            severity_on_error: Severity level when error occurs
            alert_on_success: Whether to alert on successful operations
            include_args: Whether to include function arguments in alert
            include_result: Whether to include function result in alert
            custom_channels: Custom notification channels

        Returns:
            Decorated function
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or func.__name__
                start_time = datetime.now()

                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Track success
                    self.track_operation(component, True)

                    # Alert on success if configured
                    if alert_on_success or self.config['alert_on_success']:
                        metadata = {
                            'operation': op_name,
                            'execution_time': (datetime.now() - start_time).total_seconds(),
                        }
                        if include_args:
                            metadata['args'] = str(args)[:500]
                            metadata['kwargs'] = str(kwargs)[:500]
                        if include_result:
                            metadata['result'] = str(result)[:500]

                        self.create_alert(
                            component=component,
                            title=f"{op_name} completed successfully",
                            description=f"Operation {op_name} in {component} completed successfully",
                            severity='info',
                            metadata=metadata,
                            notify_channels=custom_channels
                        )

                    return result

                except Exception as e:
                    # Track failure
                    self.track_operation(component, False)

                    # Build error metadata
                    metadata = {
                        'operation': op_name,
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'execution_time': (datetime.now() - start_time).total_seconds(),
                    }

                    if include_args:
                        metadata['args'] = str(args)[:500]
                        metadata['kwargs'] = str(kwargs)[:500]

                    if self.config['include_traceback']:
                        metadata['traceback'] = tb.format_exc()[:2000]

                    # Create alert
                    self.create_alert(
                        component=component,
                        title=f"{op_name} failed in {component}",
                        description=f"Operation {op_name} failed: {str(e)}",
                        severity=severity_on_error,
                        metadata=metadata,
                        notify_channels=custom_channels
                    )

                    # Re-raise exception
                    raise

            return wrapper
        return decorator

    @contextmanager
    def alert_context(
        self,
        component: str,
        operation_name: str,
        severity_on_error: str = 'error',
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for alerting on code blocks.

        Usage:
            with integration.alert_context('component', 'operation'):
                # Your code here
                pass

        Args:
            component: Component name
            operation_name: Operation name
            severity_on_error: Severity level when error occurs
            metadata: Optional additional metadata
        """
        start_time = datetime.now()
        success = False

        try:
            yield
            success = True
            self.track_operation(component, True)

        except Exception as e:
            self.track_operation(component, False)

            error_metadata = {
                'operation': operation_name,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds(),
            }

            if metadata:
                error_metadata.update(metadata)

            if self.config['include_traceback']:
                error_metadata['traceback'] = tb.format_exc()[:2000]

            self.create_alert(
                component=component,
                title=f"{operation_name} failed in {component}",
                description=f"Operation {operation_name} failed: {str(e)}",
                severity=severity_on_error,
                metadata=error_metadata
            )

            raise

    def get_component_stats(self, component: str) -> Dict[str, int]:
        """Get statistics for a component."""
        return self.component_stats.get(component, {})

    def get_all_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all components."""
        return self.component_stats.copy()


# Global singleton instance
_universal_integration: Optional[UniversalAlertingIntegration] = None


def get_universal_alerting() -> UniversalAlertingIntegration:
    """Get or create the universal alerting integration singleton."""
    global _universal_integration
    if _universal_integration is None:
        _universal_integration = UniversalAlertingIntegration()
    return _universal_integration


# Component-specific alerting helpers

def alert_roma_operation(operation_name: str = None):
    """Decorator for ROMA-MDAP-MAKER operations."""
    return get_universal_alerting().alert_decorator(
        component='roma_mdap_maker',
        operation_name=operation_name,
        severity_on_error='error'
    )


def alert_decomposition_operation(operation_name: str = None):
    """Decorator for Decomposition Engine operations."""
    return get_universal_alerting().alert_decorator(
        component='decomposition_engine',
        operation_name=operation_name,
        severity_on_error='error'
    )


def alert_z3_operation(operation_name: str = None):
    """Decorator for Z3 Verification operations."""
    return get_universal_alerting().alert_decorator(
        component='z3_verification',
        operation_name=operation_name,
        severity_on_error='warning'
    )


def alert_crewai_operation(operation_name: str = None):
    """Decorator for CrewAI Workflow operations."""
    return get_universal_alerting().alert_decorator(
        component='crewai_workflows',
        operation_name=operation_name,
        severity_on_error='error'
    )


def alert_knowledge_graph_operation(operation_name: str = None):
    """Decorator for Knowledge Graph operations."""
    return get_universal_alerting().alert_decorator(
        component='knowledge_graph',
        operation_name=operation_name,
        severity_on_error='warning'
    )


def alert_cache_operation(operation_name: str = None):
    """Decorator for Caching System operations."""
    return get_universal_alerting().alert_decorator(
        component='caching_systems',
        operation_name=operation_name,
        severity_on_error='warning'
    )


def alert_datapizza_operation(operation_name: str = None):
    """Decorator for DataPizza operations."""
    return get_universal_alerting().alert_decorator(
        component='datapizza',
        operation_name=operation_name,
        severity_on_error='error'
    )


def alert_icr_operation(operation_name: str = None):
    """Decorator for ICR Refinement operations."""
    return get_universal_alerting().alert_decorator(
        component='icr_refinement',
        operation_name=operation_name,
        severity_on_error='warning'
    )


def alert_ace_operation(operation_name: str = None):
    """Decorator for ACE Engine operations."""
    return get_universal_alerting().alert_decorator(
        component='ace_engine',
        operation_name=operation_name,
        severity_on_error='warning'
    )


def alert_claudiomiro_operation(operation_name: str = None):
    """Decorator for Claudiomiro operations."""
    return get_universal_alerting().alert_decorator(
        component='claudiomiro',
        operation_name=operation_name,
        severity_on_error='error'
    )


def alert_adaptive_strategy_operation(operation_name: str = None):
    """Decorator for Adaptive Strategy operations."""
    return get_universal_alerting().alert_decorator(
        component='adaptive_strategies',
        operation_name=operation_name,
        severity_on_error='warning'
    )


# Export key components
__all__ = [
    'UniversalAlertingIntegration',
    'get_universal_alerting',
    'alert_roma_operation',
    'alert_decomposition_operation',
    'alert_z3_operation',
    'alert_crewai_operation',
    'alert_knowledge_graph_operation',
    'alert_cache_operation',
    'alert_datapizza_operation',
    'alert_icr_operation',
    'alert_ace_operation',
    'alert_claudiomiro_operation',
    'alert_adaptive_strategy_operation',
]
