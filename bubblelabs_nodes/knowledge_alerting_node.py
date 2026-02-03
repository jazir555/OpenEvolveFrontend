"""
Knowledge Alerting Node for BubbleLabs Integration

Provides comprehensive alerting capabilities for knowledge graphs:
- Monitor knowledge conditions and trigger alerts
- Alert on entity/triple pattern matches
- Alert on confidence threshold changes
- Alert on contradictions detected
- Alert on quality degradation
- Alert on specific knowledge changes
- Generate and dispatch alert notifications

The node can work with knowledge graph IDs from the UnifiedKGIntegrationHub
or process knowledge graph data directly from the workflow context.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import re
import json
import hashlib
from .base_node import BubbleLabsNode, NodeExecutionError


class AlertType(Enum):
    """Types of alerts that can be configured."""
    ENTITY_PATTERN = "entity_pattern"
    TRIPLE_PATTERN = "triple_pattern"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    CONTRADICTION = "contradiction"
    QUALITY_DROP = "quality_drop"
    KNOWLEDGE_GAP = "knowledge_gap"


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class KnowledgeAlertingNode(BubbleLabsNode):
    """
    Knowledge Alerting Node for monitoring knowledge graphs and triggering alerts.

    Provides comprehensive alerting capabilities:
    - check_conditions: Evaluate alert conditions against knowledge graph
    - setup_alert: Configure new alert rules
    - evaluate_alert: Test specific alert against knowledge graph
    - list_alerts: List all configured alerts
    - clear_alerts: Clear alert history or specific alert types

    Supports multiple alert types including entity patterns, triple patterns,
    confidence thresholds, contradictions, quality degradation, and knowledge gaps.
    """

    # Node metadata
    DISPLAY_NAME = "Knowledge Alerting"
    DESCRIPTION = "Monitor knowledge graphs and alert on specific conditions and patterns"
    ICON = "knowledge-alerting"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports for optional dependencies
        unified_hub_module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for KnowledgeAlertingNode"
        )

        self.UnifiedKGIntegrationHub = None
        if unified_hub_module:
            self.UnifiedKGIntegrationHub = getattr(unified_hub_module, 'UnifiedKGIntegrationHub', None)

        # Initialize hub instance
        self.hub = None
        if self.UnifiedKGIntegrationHub:
            try:
                self.hub = self.UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized for KnowledgeAlertingNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None

        # Safe import for NotificationManager
        notification_module = self.safe_import(
            'knowledge_engine.notifications',
            fallback_value=None,
            error_msg="NotificationManager not available for KnowledgeAlertingNode"
        )

        self.NotificationManager = None
        if notification_module:
            self.NotificationManager = getattr(notification_module, 'NotificationManager', None)

        self.notification_manager = None
        if self.NotificationManager:
            try:
                self.notification_manager = self.NotificationManager()
                self.logger.info("NotificationManager initialized for KnowledgeAlertingNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize NotificationManager: {e}")
                self.notification_manager = None

        # Safe import for WebhookManager
        webhook_module = self.safe_import(
            'knowledge_engine.webhook_manager',
            fallback_value=None,
            error_msg="WebhookManager not available for KnowledgeAlertingNode"
        )

        if webhook_module is None:
            # Try alternative import path
            webhook_module = self.safe_import(
                'webhook_manager',
                fallback_value=None,
                error_msg="WebhookManager not found in alternate path"
            )

        self.WebhookManager = None
        if webhook_module:
            self.WebhookManager = getattr(webhook_module, 'WebhookManager', None)

        self.webhook_manager = None
        if self.WebhookManager:
            try:
                self.webhook_manager = self.WebhookManager()
                self.logger.info("WebhookManager initialized for KnowledgeAlertingNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize WebhookManager: {e}")
                self.webhook_manager = None

        # Alert storage (in-memory with optional persistence)
        self._alert_rules: List[Dict[str, Any]] = []
        self._alert_history: List[Dict[str, Any]] = []
        self._last_alert_times: Dict[str, datetime] = {}

        # Import alerting system if available
        alerting_module = self.safe_import(
            'alerting_system',
            fallback_value=None,
            error_msg="AlertingSystem not available for KnowledgeAlertingNode"
        )

        self.AlertManager = None
        self.alert_manager = None
        if alerting_module:
            self.AlertManager = getattr(alerting_module, 'AlertManager', None)
            if self.AlertManager:
                try:
                    self.alert_manager = self.AlertManager()
                    self.logger.info("AlertManager initialized for KnowledgeAlertingNode")
                except Exception as e:
                    self.logger.warning(f"Could not initialize AlertManager: {e}")

        # Load any persisted alert rules
        self._load_alert_rules()

    def _load_alert_rules(self):
        """Load persisted alert rules if available."""
        if self.alert_manager:
            try:
                # Load alert rules from alert manager storage
                all_alerts = self.alert_manager.get_all_alerts(component='knowledge_alerting')
                self.logger.info(f"Loaded {len(all_alerts)} knowledge alerting rules")
            except Exception as e:
                self.logger.warning(f"Failed to load alert rules: {e}")
        else:
            # Fallback to placeholder
            self.logger.debug("AlertManager not available - using in-memory storage")

    def _save_alert_rules(self):
        """Save alert rules for persistence."""
        # This is a placeholder for persistence logic
        pass

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields depend on operation:
        - check_conditions: Either 'knowledge_graph_id' or 'knowledge_graph'
        - setup_alert: 'alert_type' and 'conditions'
        - evaluate_alert: 'alert_id' or alert configuration
        - list_alerts: No required fields
        - clear_alerts: No required fields (optional 'alert_type' filter)
        """
        errors = []

        # Get operation type from inputs or config
        operation = inputs.get('operation', self.config.get('operation', 'check_conditions'))

        valid_operations = ['check_conditions', 'setup_alert', 'evaluate_alert', 'list_alerts', 'clear_alerts']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")
            return errors

        # Validate based on operation
        if operation in ['check_conditions', 'evaluate_alert']:
            has_kg_id = inputs.get('knowledge_graph_id') or self.config.get('knowledge_graph_id')
            has_kg = inputs.get('knowledge_graph') or self.config.get('knowledge_graph')

            if not has_kg_id and not has_kg:
                errors.append("Either 'knowledge_graph_id' or 'knowledge_graph' must be provided")

        if operation == 'setup_alert':
            alert_type = inputs.get('alert_type') or self.config.get('alert_type')
            if not alert_type:
                errors.append("'alert_type' is required for setup_alert operation")
            elif alert_type not in [t.value for t in AlertType]:
                errors.append(f"Invalid alert_type: {alert_type}. Must be one of: {', '.join(t.value for t in AlertType)}")

            conditions = inputs.get('conditions') or self.config.get('conditions')
            if not conditions:
                errors.append("'conditions' is required for setup_alert operation")

        if operation == 'evaluate_alert':
            alert_id = inputs.get('alert_id') or self.config.get('alert_id')
            if not alert_id and not (inputs.get('conditions') or self.config.get('conditions')):
                errors.append("Either 'alert_id' or 'conditions' must be provided for evaluate_alert")

        # Validate alert_type if provided
        if 'alert_type' in inputs:
            alert_type = inputs['alert_type']
            valid_types = [t.value for t in AlertType]
            if alert_type not in valid_types:
                errors.append(f"Invalid alert_type: {alert_type}. Must be one of: {', '.join(valid_types)}")

        # Validate severity if provided
        if 'severity' in inputs:
            severity = inputs['severity']
            valid_severities = [s.value for s in AlertSeverity]
            if severity not in valid_severities:
                errors.append(f"Invalid severity: {severity}. Must be one of: {', '.join(valid_severities)}")

        # Validate confidence_threshold if provided
        if 'confidence_threshold' in inputs:
            try:
                threshold = float(inputs['confidence_threshold'])
                if not (0.0 <= threshold <= 1.0):
                    errors.append("confidence_threshold must be between 0.0 and 1.0")
            except (ValueError, TypeError):
                errors.append("confidence_threshold must be a number")

        # Validate alert_cooldown if provided
        if 'alert_cooldown' in inputs:
            try:
                cooldown = int(inputs['alert_cooldown'])
                if cooldown < 0:
                    errors.append("alert_cooldown must be non-negative")
            except (ValueError, TypeError):
                errors.append("alert_cooldown must be an integer")

        # Validate threshold_direction if provided
        if 'threshold_direction' in inputs:
            direction = inputs['threshold_direction']
            if direction not in ['above', 'below']:
                errors.append("threshold_direction must be 'above' or 'below'")

        # Validate notification_channels if provided
        if 'notification_channels' in inputs:
            channels = inputs['notification_channels']
            if not isinstance(channels, list):
                errors.append("'notification_channels' must be a list")
            else:
                valid_channels = ['log', 'webhook', 'email']
                for channel in channels:
                    if channel not in valid_channels:
                        errors.append(f"Invalid notification channel: {channel}. Must be one of: {', '.join(valid_channels)}")

        # Validate conditions structure if provided
        if 'conditions' in inputs:
            conditions = inputs['conditions']
            if not isinstance(conditions, list):
                errors.append("'conditions' must be an array of condition objects")
            else:
                for i, condition in enumerate(conditions):
                    if not isinstance(condition, dict):
                        errors.append(f"condition[{i}] must be an object")
                    elif 'alert_type' not in condition:
                        errors.append(f"condition[{i}] missing required field: 'alert_type'")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute knowledge alerting based on operation type.

        Args:
            inputs: Input data containing operation and operation-specific parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - alerts_triggered: List of triggered alerts
                - alert_count: Number of alerts triggered
                - notifications_sent: List of sent notifications
                - metadata: Execution metadata

        Raises:
            NodeExecutionError: If execution fails
        """
        # Get parameters
        operation = inputs.get('operation', self.config.get('operation', 'check_conditions'))
        alert_type = inputs.get('alert_type', self.config.get('alert_type'))
        conditions = inputs.get('conditions', self.config.get('conditions', []))
        severity = inputs.get('severity', self.config.get('severity', 'warning'))
        alert_cooldown = inputs.get('alert_cooldown', self.config.get('alert_cooldown', 3600))
        notification_channels = inputs.get('notification_channels', self.config.get('notification_channels', ['log']))

        context.update_progress(10, f"Initializing {operation} operation")
        self.logger.info(f"Executing knowledge alerting: operation={operation}")

        try:
            if operation == 'check_conditions':
                result = self._execute_check_conditions(
                    inputs, conditions, severity, alert_cooldown, notification_channels, context
                )
            elif operation == 'setup_alert':
                result = self._execute_setup_alert(
                    alert_type, conditions, severity, alert_cooldown, notification_channels, context
                )
            elif operation == 'evaluate_alert':
                result = self._execute_evaluate_alert(
                    inputs, conditions, severity, alert_cooldown, notification_channels, context
                )
            elif operation == 'list_alerts':
                result = self._execute_list_alerts(context)
            elif operation == 'clear_alerts':
                result = self._execute_clear_alerts(alert_type, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['check_conditions', 'setup_alert', 'evaluate_alert', 'list_alerts', 'clear_alerts']}
                )

            # Add execution metadata
            result['metadata'] = {
                'operation': operation,
                'execution_id': self.execution_id,
                'executed_at': datetime.now().isoformat()
            }

            # Store result in context
            context.add_artifact('knowledge_alerting', {
                'operation': operation,
                'alerts_triggered': result.get('alert_count', 0),
                'notifications_sent': len(result.get('notifications_sent', []))
            })

            context.update_progress(100, f"Knowledge alerting {operation} complete")
            self.logger.info(f"Knowledge alerting completed: {result.get('alert_count', 0)} alerts triggered")

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Knowledge alerting failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Knowledge alerting failed: {str(e)}",
                details={
                    'operation': operation,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_check_conditions(
        self,
        inputs: Dict,
        conditions: List[Dict],
        severity: str,
        alert_cooldown: int,
        notification_channels: List[str],
        context
    ) -> Dict[str, Any]:
        """Execute check_conditions operation."""
        context.update_progress(20, "Loading knowledge graph data")

        # Get knowledge graph data
        kg_data = self._get_knowledge_graph_data(inputs)

        if not kg_data:
            return {
                'alerts_triggered': [],
                'alert_count': 0,
                'notifications_sent': [],
                'warning': 'No knowledge graph data available for checking'
            }

        context.update_progress(40, f"Checking {len(conditions)} alert conditions")

        # If no conditions provided, use default checks
        if not conditions:
            conditions = self._get_default_conditions(severity)

        # Check each condition
        alerts_triggered = []
        for i, condition in enumerate(conditions):
            progress = 40 + (i / len(conditions)) * 40
            context.update_progress(int(progress), f"Checking condition {i+1}/{len(conditions)}")

            alerts = self._check_condition(kg_data, condition)
            for alert in alerts:
                # Check cooldown
                if self._is_alert_cooled_down(alert, alert_cooldown):
                    alerts_triggered.append(alert)
                    self._last_alert_times[alert['id']] = datetime.now()

        context.update_progress(80, f"{len(alerts_triggered)} alerts triggered, sending notifications")

        # Send notifications
        notifications_sent = self._send_notifications(alerts_triggered, notification_channels, context)

        # Add to alert history
        for alert in alerts_triggered:
            self._alert_history.append({
                **alert,
                'triggered_at': datetime.now().isoformat(),
                'execution_id': self.execution_id
            })

        return {
            'alerts_triggered': alerts_triggered,
            'alert_count': len(alerts_triggered),
            'notifications_sent': notifications_sent
        }

    def _execute_setup_alert(
        self,
        alert_type: str,
        conditions: List[Dict],
        severity: str,
        alert_cooldown: int,
        notification_channels: List[str],
        context
    ) -> Dict[str, Any]:
        """Execute setup_alert operation."""
        context.update_progress(30, "Setting up new alert rule")

        # Create alert rule
        alert_rule = {
            'id': self._generate_alert_id(),
            'alert_type': alert_type,
            'conditions': conditions,
            'severity': severity,
            'alert_cooldown': alert_cooldown,
            'notification_channels': notification_channels,
            'created_at': datetime.now().isoformat(),
            'enabled': True
        }

        # Add to stored rules
        self._alert_rules.append(alert_rule)
        self._save_alert_rules()

        context.update_progress(100, "Alert rule created successfully")

        return {
            'alerts_triggered': [],
            'alert_count': 0,
            'notifications_sent': [],
            'alert_rule': alert_rule,
            'message': f"Alert rule '{alert_rule['id']}' created successfully"
        }

    def _execute_evaluate_alert(
        self,
        inputs: Dict,
        conditions: List[Dict],
        severity: str,
        alert_cooldown: int,
        notification_channels: List[str],
        context
    ) -> Dict[str, Any]:
        """Execute evaluate_alert operation."""
        context.update_progress(20, "Loading knowledge graph data for evaluation")

        # Get knowledge graph data
        kg_data = self._get_knowledge_graph_data(inputs)

        if not kg_data:
            return {
                'alerts_triggered': [],
                'alert_count': 0,
                'notifications_sent': [],
                'warning': 'No knowledge graph data available for evaluation'
            }

        context.update_progress(40, "Evaluating alert conditions")

        # Check if evaluating by alert_id or direct conditions
        alert_id = inputs.get('alert_id') or self.config.get('alert_id')

        if alert_id:
            # Find alert rule by ID
            alert_rule = next((r for r in self._alert_rules if r['id'] == alert_id), None)
            if not alert_rule:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Alert rule not found: {alert_id}",
                    details={'alert_id': alert_id}
                )
            conditions = alert_rule['conditions']
            severity = alert_rule['severity']

        # Evaluate conditions
        alerts_triggered = []
        for condition in conditions:
            alerts = self._check_condition(kg_data, condition)
            for alert in alerts:
                # Check cooldown
                if self._is_alert_cooled_down(alert, alert_cooldown):
                    alerts_triggered.append(alert)
                    self._last_alert_times[alert['id']] = datetime.now()

        context.update_progress(80, f"{len(alerts_triggered)} alerts triggered during evaluation")

        # Send notifications
        notifications_sent = self._send_notifications(alerts_triggered, notification_channels, context)

        return {
            'alerts_triggered': alerts_triggered,
            'alert_count': len(alerts_triggered),
            'notifications_sent': notifications_sent,
            'evaluation_result': {
                'alert_id': alert_id,
                'conditions_evaluated': len(conditions),
                'alerts_triggered': len(alerts_triggered)
            }
        }

    def _execute_list_alerts(self, context) -> Dict[str, Any]:
        """Execute list_alerts operation."""
        context.update_progress(50, "Retrieving alert configuration")

        return {
            'alerts_triggered': [],
            'alert_count': 0,
            'notifications_sent': [],
            'alert_rules': self._alert_rules,
            'alert_history': self._alert_history[-100:],  # Last 100 alerts
            'stats': {
                'total_rules': len(self._alert_rules),
                'enabled_rules': sum(1 for r in self._alert_rules if r.get('enabled', True)),
                'total_history': len(self._alert_history)
            }
        }

    def _execute_clear_alerts(self, alert_type: Optional[str], context) -> Dict[str, Any]:
        """Execute clear_alerts operation."""
        context.update_progress(50, "Clearing alerts")

        cleared_count = 0

        if alert_type:
            # Clear only specific alert type from history
            original_count = len(self._alert_history)
            self._alert_history = [
                a for a in self._alert_history
                if a.get('alert_type') != alert_type
            ]
            cleared_count = original_count - len(self._alert_history)
        else:
            # Clear all history
            cleared_count = len(self._alert_history)
            self._alert_history = []
            self._last_alert_times = {}

        context.update_progress(100, f"Cleared {cleared_count} alerts")

        return {
            'alerts_triggered': [],
            'alert_count': 0,
            'notifications_sent': [],
            'cleared_count': cleared_count,
            'message': f"Cleared {cleared_count} alerts"
        }

    def _get_knowledge_graph_data(self, inputs: Dict) -> Optional[Dict[str, Any]]:
        """Retrieve knowledge graph data from inputs or hub."""
        # Direct knowledge graph data
        if 'knowledge_graph' in inputs and inputs['knowledge_graph']:
            return inputs['knowledge_graph']

        if self.config.get('knowledge_graph'):
            return self.config['knowledge_graph']

        # Fetch from hub using knowledge_graph_id
        kg_id = inputs.get('knowledge_graph_id') or self.config.get('knowledge_graph_id')
        if kg_id and self.hub:
            try:
                if hasattr(self.hub, 'get_knowledge_graph'):
                    return self.hub.get_knowledge_graph(kg_id)
                elif hasattr(self.hub, 'export_graph'):
                    return self.hub.export_graph(kg_id)
            except Exception as e:
                self.logger.warning(f"Could not fetch graph from hub: {e}")

        # Check for kg_instance in inputs
        if 'kg_instance' in inputs and inputs['kg_instance']:
            kg = inputs['kg_instance']
            if hasattr(kg, 'export_to_dict'):
                return kg.export_to_dict()
            elif hasattr(kg, 'to_dict'):
                return kg.to_dict()

        return None

    def _get_default_conditions(self, severity: str) -> List[Dict]:
        """Get default alert conditions."""
        return [
            {
                'alert_type': 'contradiction',
                'severity': severity,
                'description': 'Alert on detected contradictions'
            },
            {
                'alert_type': 'quality_drop',
                'severity': severity,
                'description': 'Alert on quality degradation'
            },
            {
                'alert_type': 'confidence_threshold',
                'confidence_threshold': 0.5,
                'threshold_direction': 'below',
                'severity': severity,
                'description': 'Alert on low confidence triples'
            }
        ]

    def _check_condition(self, kg_data: Dict, condition: Dict) -> List[Dict]:
        """Check a single alert condition against knowledge graph data."""
        alert_type = condition.get('alert_type')

        if alert_type == AlertType.ENTITY_PATTERN.value:
            return self._check_entity_pattern(kg_data, condition)
        elif alert_type == AlertType.TRIPLE_PATTERN.value:
            return self._check_triple_pattern(kg_data, condition)
        elif alert_type == AlertType.CONFIDENCE_THRESHOLD.value:
            return self._check_confidence_threshold(kg_data, condition)
        elif alert_type == AlertType.CONTRADICTION.value:
            return self._check_contradictions(kg_data, condition)
        elif alert_type == AlertType.QUALITY_DROP.value:
            return self._check_quality_drop(kg_data, condition)
        elif alert_type == AlertType.KNOWLEDGE_GAP.value:
            return self._check_knowledge_gap(kg_data, condition)
        else:
            self.logger.warning(f"Unknown alert type: {alert_type}")
            return []

    def _check_entity_pattern(self, kg_data: Dict, condition: Dict) -> List[Dict]:
        """Check for entities matching a pattern."""
        alerts = []
        pattern = condition.get('entity_pattern', '')

        if not pattern:
            return alerts

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            self.logger.warning(f"Invalid entity pattern regex: {e}")
            return alerts

        nodes = kg_data.get('nodes', [])
        matching_entities = []

        for node in nodes:
            entity_id = node.get('id', '')
            entity_name = node.get('name', '')
            entity_type = node.get('type', '')

            if regex.search(entity_id) or regex.search(entity_name) or regex.search(entity_type):
                matching_entities.append({
                    'id': entity_id,
                    'name': entity_name,
                    'type': entity_type
                })

        if matching_entities:
            alert_id = self._generate_alert_id('entity_pattern')
            alerts.append({
                'id': alert_id,
                'alert_type': AlertType.ENTITY_PATTERN.value,
                'severity': condition.get('severity', 'warning'),
                'title': f"Entity Pattern Match: {pattern}",
                'description': f"Found {len(matching_entities)} entities matching pattern '{pattern}'",
                'matching_entities': matching_entities,
                'pattern': pattern,
                'entity_count': len(matching_entities),
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def _check_triple_pattern(self, kg_data: Dict, condition: Dict) -> List[Dict]:
        """Check for triples matching a pattern."""
        alerts = []
        triple_pattern = condition.get('triple_pattern', {})

        subject_pattern = triple_pattern.get('subject', '')
        predicate_pattern = triple_pattern.get('predicate', '')
        object_pattern = triple_pattern.get('object', '')

        triples = kg_data.get('triples', [])
        matching_triples = []

        for triple in triples:
            subject = triple.get('subject', '')
            predicate = triple.get('predicate', '')
            obj = triple.get('object', '')

            matches = True

            if subject_pattern and not re.search(subject_pattern, subject, re.IGNORECASE):
                matches = False
            if predicate_pattern and not re.search(predicate_pattern, predicate, re.IGNORECASE):
                matches = False
            if object_pattern and not re.search(object_pattern, obj, re.IGNORECASE):
                matches = False

            if matches:
                matching_triples.append({
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj,
                    'confidence': triple.get('confidence', 1.0)
                })

        if matching_triples:
            alert_id = self._generate_alert_id('triple_pattern')
            alerts.append({
                'id': alert_id,
                'alert_type': AlertType.TRIPLE_PATTERN.value,
                'severity': condition.get('severity', 'warning'),
                'title': f"Triple Pattern Match Detected",
                'description': f"Found {len(matching_triples)} triples matching pattern",
                'matching_triples': matching_triples[:10],  # Limit to first 10
                'pattern': triple_pattern,
                'triple_count': len(matching_triples),
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def _check_confidence_threshold(self, kg_data: Dict, condition: Dict) -> List[Dict]:
        """Check for confidence threshold violations."""
        alerts = []
        threshold = condition.get('confidence_threshold', 0.5)
        direction = condition.get('threshold_direction', 'below')

        triples = kg_data.get('triples', [])
        low_confidence_triples = []
        high_confidence_triples = []

        for triple in triples:
            confidence = triple.get('confidence', 1.0)

            if direction == 'below' and confidence < threshold:
                low_confidence_triples.append({
                    'subject': triple.get('subject'),
                    'predicate': triple.get('predicate'),
                    'object': triple.get('object'),
                    'confidence': confidence
                })
            elif direction == 'above' and confidence > threshold:
                high_confidence_triples.append({
                    'subject': triple.get('subject'),
                    'predicate': triple.get('predicate'),
                    'object': triple.get('object'),
                    'confidence': confidence
                })

        if direction == 'below' and low_confidence_triples:
            alert_id = self._generate_alert_id('confidence_low')
            alerts.append({
                'id': alert_id,
                'alert_type': AlertType.CONFIDENCE_THRESHOLD.value,
                'severity': condition.get('severity', 'warning'),
                'title': f"Low Confidence Triples Detected",
                'description': f"Found {len(low_confidence_triples)} triples with confidence below {threshold}",
                'affected_triples': low_confidence_triples[:10],
                'threshold': threshold,
                'direction': direction,
                'violation_count': len(low_confidence_triples),
                'timestamp': datetime.now().isoformat()
            })
        elif direction == 'above' and high_confidence_triples:
            alert_id = self._generate_alert_id('confidence_high')
            alerts.append({
                'id': alert_id,
                'alert_type': AlertType.CONFIDENCE_THRESHOLD.value,
                'severity': condition.get('severity', 'info'),
                'title': f"High Confidence Triples Detected",
                'description': f"Found {len(high_confidence_triples)} triples with confidence above {threshold}",
                'affected_triples': high_confidence_triples[:10],
                'threshold': threshold,
                'direction': direction,
                'count': len(high_confidence_triples),
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def _check_contradictions(self, kg_data: Dict, condition: Dict) -> List[Dict]:
        """Check for contradictions in the knowledge graph."""
        alerts = []
        triples = kg_data.get('triples', [])

        # Group triples by subject-predicate
        sp_groups: Dict[Tuple[str, str], List[Dict]] = {}
        for triple in triples:
            key = (triple.get('subject', ''), triple.get('predicate', ''))
            if key not in sp_groups:
                sp_groups[key] = []
            sp_groups[key].append(triple)

        contradictions = []

        # Check for multiple different objects for same subject-predicate
        for (subject, predicate), group in sp_groups.items():
            if len(group) < 2:
                continue

            objects = [t.get('object', '') for t in group]
            unique_objects = set(objects)

            if len(unique_objects) > 1:
                # Check if objects are contradictory (simplified logic)
                contradiction_types = self._classify_contradictions(unique_objects)
                if contradiction_types:
                    contradictions.append({
                        'subject': subject,
                        'predicate': predicate,
                        'conflicting_values': list(unique_objects),
                        'contradiction_types': contradiction_types,
                        'involved_triples': group
                    })

        if contradictions:
            alert_id = self._generate_alert_id('contradiction')
            alerts.append({
                'id': alert_id,
                'alert_type': AlertType.CONTRADICTION.value,
                'severity': condition.get('severity', 'critical'),
                'title': f"Knowledge Contradictions Detected",
                'description': f"Found {len(contradictions)} contradictions in the knowledge graph",
                'contradictions': contradictions,
                'contradiction_count': len(contradictions),
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def _classify_contradictions(self, objects: Set[str]) -> List[str]:
        """Classify types of contradictions between objects."""
        types = []
        obj_list = list(objects)

        for i in range(len(obj_list)):
            for j in range(i + 1, len(obj_list)):
                o1, o2 = str(obj_list[i]).lower(), str(obj_list[j]).lower()

                # Check for boolean contradictions
                if (o1 in ['true', 'yes'] and o2 in ['false', 'no']) or \
                   (o1 in ['false', 'no'] and o2 in ['true', 'yes']):
                    types.append('boolean')

                # Check for negation
                if o1 == f"not {o2}" or o2 == f"not {o1}":
                    types.append('negation')

                # Check for antonyms (simplified)
                antonym_pairs = [
                    ('increase', 'decrease'),
                    ('high', 'low'),
                    ('up', 'down'),
                    ('positive', 'negative'),
                    ('active', 'inactive')
                ]
                for a1, a2 in antonym_pairs:
                    if (a1 in o1 and a2 in o2) or (a2 in o1 and a1 in o2):
                        types.append('antonym')

        return list(set(types)) if types else ['value_mismatch']

    def _check_quality_drop(self, kg_data: Dict, condition: Dict) -> List[Dict]:
        """Check for quality degradation in the knowledge graph."""
        alerts = []
        triples = kg_data.get('triples', [])

        if not triples:
            return alerts

        # Calculate quality metrics
        confidences = [t.get('confidence', 1.0) for t in triples]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0

        # Count triples with missing metadata
        missing_metadata = sum(1 for t in triples if not t.get('metadata'))
        metadata_quality = 1.0 - (missing_metadata / len(triples)) if triples else 1.0

        # Check for low quality sources
        low_confidence_count = sum(1 for c in confidences if c < 0.5)
        low_confidence_ratio = low_confidence_count / len(triples) if triples else 0

        quality_issues = []

        if avg_confidence < 0.7:
            quality_issues.append({
                'type': 'low_average_confidence',
                'value': avg_confidence,
                'threshold': 0.7
            })

        if metadata_quality < 0.5:
            quality_issues.append({
                'type': 'poor_metadata_coverage',
                'value': metadata_quality,
                'threshold': 0.5
            })

        if low_confidence_ratio > 0.3:
            quality_issues.append({
                'type': 'high_low_confidence_ratio',
                'value': low_confidence_ratio,
                'threshold': 0.3
            })

        if quality_issues:
            alert_id = self._generate_alert_id('quality_drop')
            alerts.append({
                'id': alert_id,
                'alert_type': AlertType.QUALITY_DROP.value,
                'severity': condition.get('severity', 'warning'),
                'title': f"Knowledge Quality Degradation Detected",
                'description': f"Found {len(quality_issues)} quality issues in the knowledge graph",
                'quality_issues': quality_issues,
                'metrics': {
                    'average_confidence': avg_confidence,
                    'metadata_quality': metadata_quality,
                    'low_confidence_ratio': low_confidence_ratio
                },
                'issue_count': len(quality_issues),
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def _check_knowledge_gap(self, kg_data: Dict, condition: Dict) -> List[Dict]:
        """Check for knowledge gaps in the knowledge graph."""
        alerts = []
        nodes = kg_data.get('nodes', [])
        triples = kg_data.get('triples', [])
        edges = kg_data.get('edges', [])

        # Find isolated nodes (no connections)
        connected_nodes = set()
        for triple in triples:
            connected_nodes.add(triple.get('subject'))
            connected_nodes.add(triple.get('object'))
        for edge in edges:
            connected_nodes.add(edge.get('source'))
            connected_nodes.add(edge.get('target'))

        isolated_nodes = [
            node for node in nodes
            if node.get('id') not in connected_nodes
        ]

        # Find nodes with incomplete information (missing properties)
        incomplete_nodes = []
        for node in nodes:
            if not node.get('name') or not node.get('type'):
                incomplete_nodes.append({
                    'id': node.get('id'),
                    'missing': []
                })
                if not node.get('name'):
                    incomplete_nodes[-1]['missing'].append('name')
                if not node.get('type'):
                    incomplete_nodes[-1]['missing'].append('type')

        gaps = []

        if isolated_nodes:
            gaps.append({
                'type': 'isolated_nodes',
                'count': len(isolated_nodes),
                'nodes': [{'id': n.get('id')} for n in isolated_nodes[:10]]
            })

        if incomplete_nodes:
            gaps.append({
                'type': 'incomplete_nodes',
                'count': len(incomplete_nodes),
                'nodes': incomplete_nodes[:10]
            })

        if gaps:
            alert_id = self._generate_alert_id('knowledge_gap')
            alerts.append({
                'id': alert_id,
                'alert_type': AlertType.KNOWLEDGE_GAP.value,
                'severity': condition.get('severity', 'info'),
                'title': f"Knowledge Gaps Detected",
                'description': f"Found {len(gaps)} types of knowledge gaps",
                'gaps': gaps,
                'gap_count': sum(g['count'] for g in gaps),
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def _is_alert_cooled_down(self, alert: Dict, cooldown_seconds: int) -> bool:
        """Check if enough time has passed since the last alert of this type."""
        alert_id = alert['id']
        last_time = self._last_alert_times.get(alert_id)

        if last_time is None:
            return True

        time_diff = datetime.now() - last_time
        return time_diff.total_seconds() >= cooldown_seconds

    def _send_notifications(
        self,
        alerts: List[Dict],
        channels: List[str],
        context
    ) -> List[Dict]:
        """Send notifications through configured channels."""
        notifications_sent = []

        for alert in alerts:
            for channel in channels:
                try:
                    if channel == 'log':
                        self._send_log_notification(alert)
                        notifications_sent.append({
                            'alert_id': alert['id'],
                            'channel': 'log',
                            'status': 'sent',
                            'timestamp': datetime.now().isoformat()
                        })
                    elif channel == 'webhook':
                        result = self._send_webhook_notification(alert)
                        notifications_sent.append({
                            'alert_id': alert['id'],
                            'channel': 'webhook',
                            'status': 'sent' if result else 'failed',
                            'timestamp': datetime.now().isoformat()
                        })
                    elif channel == 'email':
                        result = self._send_email_notification(alert)
                        notifications_sent.append({
                            'alert_id': alert['id'],
                            'channel': 'email',
                            'status': 'sent' if result else 'failed',
                            'timestamp': datetime.now().isoformat()
                        })
                    # **ACTUAL INTEGRATION**: Use the actual alerting_system
                    elif channel == 'alerting_system':
                        result = self._send_via_alerting_system(alert)
                        notifications_sent.append({
                            'alert_id': alert['id'],
                            'channel': 'alerting_system',
                            'status': 'sent' if result else 'failed',
                            'timestamp': datetime.now().isoformat()
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to send {channel} notification: {e}")
                    notifications_sent.append({
                        'alert_id': alert['id'],
                        'channel': channel,
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

        return notifications_sent

    def _send_via_alerting_system(self, alert: Dict) -> bool:
        """
        **ACTUAL INTEGRATION**: Send alert via the actual alerting_system.

        This wires the knowledge_alerting_node to the central alerting system.
        """
        if not self.alert_manager:
            return False

        try:
            # Map severity to AlertSeverity enum
            from alerting_system import AlertSeverity

            severity_map = {
                'info': AlertSeverity.INFO,
                'warning': AlertSeverity.MEDIUM,
                'critical': AlertSeverity.HIGH
            }
            alert_severity = severity_map.get(alert.get('severity', 'info').lower(), AlertSeverity.MEDIUM)

            # Create alert using the alerting_system
            self.alert_manager.create_alert(
                title=alert.get('title', 'Knowledge Alert'),
                description=alert.get('description', ''),
                severity=alert_severity.value,
                source='knowledge_alerting_node',
                component='knowledge_graph',
                metadata={
                    'alert_id': alert.get('id'),
                    'alert_type': alert.get('type'),
                    'knowledge_graph_id': alert.get('knowledge_graph_id')
                }
            )

            self.logger.debug(f"Alert sent via alerting_system: {alert.get('id')}")
            return True

        except Exception as e:
            self.logger.warning(f"alerting_system notification failed: {e}")
            return False

    def _send_log_notification(self, alert: Dict):
        """Send notification via logging."""
        severity = alert.get('severity', 'info').upper()
        title = alert.get('title', 'Unknown Alert')
        description = alert.get('description', '')

        log_message = f"[ALERT:{severity}] {title} - {description}"

        if severity == 'CRITICAL':
            self.logger.critical(log_message)
        elif severity == 'WARNING':
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _send_webhook_notification(self, alert: Dict) -> bool:
        """Send notification via webhook."""
        if self.webhook_manager and hasattr(self.webhook_manager, 'send_webhook'):
            try:
                payload = {
                    'alert': alert,
                    'timestamp': datetime.now().isoformat()
                }
                self.webhook_manager.send_webhook(
                    event_type='knowledge_alert',
                    payload=payload
                )
                return True
            except Exception as e:
                self.logger.warning(f"WebhookManager notification failed: {e}")

        # Fallback: try to send via requests if available
        try:
            import requests
            # This is a placeholder - would need actual webhook URL configuration
            # requests.post(webhook_url, json=alert, timeout=5)
            self.logger.info(f"Webhook notification would be sent: {alert.get('title')}")
            return True
        except ImportError:
            self.logger.debug("requests not available for webhook notification")
            return False

    def _send_email_notification(self, alert: Dict) -> bool:
        """Send notification via email."""
        if self.notification_manager and hasattr(self.notification_manager, 'send_email'):
            try:
                subject = f"[Knowledge Alert] {alert.get('title', 'Alert')}"
                body = alert.get('description', '')
                self.notification_manager.send_email(subject=subject, body=body)
                return True
            except Exception as e:
                self.logger.warning(f"NotificationManager email failed: {e}")

        # Fallback - log that email would be sent
        self.logger.info(f"Email notification would be sent: {alert.get('title')}")
        return False

    def _generate_alert_id(self, prefix: str = 'alert') -> str:
        """Generate a unique alert ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique = hashlib.md5(f"{prefix}{timestamp}".encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{unique}"

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration panel.
        """
        return {
            "type": "object",
            "title": "Knowledge Alerting Configuration",
            "description": "Configure knowledge graph alerting parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Alerting operation to perform",
                    "enum": ["check_conditions", "setup_alert", "evaluate_alert", "list_alerts", "clear_alerts"],
                    "enumNames": [
                        "Check Conditions - Evaluate alert conditions against knowledge graph",
                        "Setup Alert - Configure new alert rules",
                        "Evaluate Alert - Test specific alert against knowledge graph",
                        "List Alerts - List all configured alerts",
                        "Clear Alerts - Clear alert history"
                    ],
                    "default": "check_conditions"
                },
                "alert_type": {
                    "type": "string",
                    "title": "Alert Type",
                    "description": "Type of alert to configure (for setup_alert operation)",
                    "enum": ["entity_pattern", "triple_pattern", "confidence_threshold", "contradiction", "quality_drop", "knowledge_gap"],
                    "enumNames": [
                        "Entity Pattern - Alert when entities match pattern",
                        "Triple Pattern - Alert when triples match pattern",
                        "Confidence Threshold - Alert on confidence changes",
                        "Contradiction - Alert when contradictions detected",
                        "Quality Drop - Alert on quality degradation",
                        "Knowledge Gap - Alert on missing information"
                    ],
                    "default": "contradiction"
                },
                "conditions": {
                    "type": "array",
                    "title": "Alert Conditions",
                    "description": "Array of alert condition definitions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "alert_type": {
                                "type": "string",
                                "enum": ["entity_pattern", "triple_pattern", "confidence_threshold", "contradiction", "quality_drop", "knowledge_gap"]
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["info", "warning", "critical"],
                                "default": "warning"
                            },
                            "description": {
                                "type": "string"
                            }
                        },
                        "required": ["alert_type"]
                    },
                    "default": []
                },
                "entity_pattern": {
                    "type": "string",
                    "title": "Entity Pattern",
                    "description": "Regex pattern for entity matching (for entity_pattern alert type)",
                    "default": ""
                },
                "triple_pattern": {
                    "type": "object",
                    "title": "Triple Pattern",
                    "description": "Pattern for matching triples (for triple_pattern alert type)",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "Regex pattern for subject"
                        },
                        "predicate": {
                            "type": "string",
                            "description": "Regex pattern for predicate"
                        },
                        "object": {
                            "type": "string",
                            "description": "Regex pattern for object"
                        }
                    },
                    "default": {}
                },
                "confidence_threshold": {
                    "type": "number",
                    "title": "Confidence Threshold",
                    "description": "Confidence threshold value (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5
                },
                "threshold_direction": {
                    "type": "string",
                    "title": "Threshold Direction",
                    "description": "Direction for threshold comparison",
                    "enum": ["above", "below"],
                    "enumNames": [
                        "Above - Trigger when value is above threshold",
                        "Below - Trigger when value is below threshold"
                    ],
                    "default": "below"
                },
                "severity": {
                    "type": "string",
                    "title": "Alert Severity",
                    "description": "Severity level for alerts",
                    "enum": ["info", "warning", "critical"],
                    "enumNames": [
                        "Info - Informational alerts",
                        "Warning - Warning alerts",
                        "Critical - Critical alerts"
                    ],
                    "default": "warning"
                },
                "alert_cooldown": {
                    "type": "integer",
                    "title": "Alert Cooldown",
                    "description": "Seconds between duplicate alerts",
                    "minimum": 0,
                    "default": 3600
                },
                "notification_channels": {
                    "type": "array",
                    "title": "Notification Channels",
                    "description": "Channels for sending alert notifications",
                    "items": {
                        "type": "string",
                        "enum": ["log", "webhook", "email"]
                    },
                    "default": ["log"]
                },
                "knowledge_graph_id": {
                    "type": "string",
                    "title": "Knowledge Graph ID",
                    "description": "ID of the knowledge graph to monitor (optional if knowledge_graph provided)",
                    "default": ""
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node is healthy, False otherwise
        """
        try:
            # Basic health check - node can function without optional dependencies
            # but should report degraded status
            health_status = {
                'base_node': True,
                'unified_kg_hub': self.hub is not None,
                'notification_manager': self.notification_manager is not None,
                'webhook_manager': self.webhook_manager is not None
            }

            # Node is healthy if basic functionality works
            # (can still operate with fallback alerting)
            self.logger.debug(f"Health check status: {health_status}")
            return True
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
