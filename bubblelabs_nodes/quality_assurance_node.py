"""
Quality Assurance Node for BubbleLabs Integration

Provides continuous quality monitoring with automated checks, trend analysis,
and alerting for knowledge quality degradation.

Features:
- Automated quality checks (completeness, accuracy, consistency, timeliness, validity)
- Confidence score monitoring over time
- Quality degradation detection
- Quality report generation
- Alert triggering on quality thresholds
- Quality metrics history tracking
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import json
import re
from .base_node import BubbleLabsNode, NodeExecutionError


class QualityAssuranceNode(BubbleLabsNode):
    """
    Continuous quality assurance monitoring node for knowledge graphs.

    Supports multiple operations:
    - monitor: Continuous monitoring of quality metrics
    - check: One-time quality check execution
    - report: Generate comprehensive quality reports
    - trend_analysis: Analyze quality trends over time
    - alert_setup: Configure quality alerts and thresholds

    Quality check types:
    - completeness: Missing properties, null values
    - accuracy: Confidence scores, source reliability
    - consistency: Contradictions, duplicate detection
    - timeliness: Freshness, update frequency
    - validity: Schema compliance, format correctness
    """

    # Node metadata
    DISPLAY_NAME = "Quality Assurance"
    DESCRIPTION = "Continuous quality monitoring with automated checks and quality reporting"
    ICON = "quality-assurance"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    # Check types supported
    CHECK_TYPES = ["completeness", "accuracy", "consistency", "timeliness", "validity"]

    # Operations supported
    OPERATIONS = ["monitor", "check", "report", "trend_analysis", "alert_setup"]

    # Time window parsing
    TIME_WINDOW_PATTERN = re.compile(r'^(\d+)([hdwmy])$', re.IGNORECASE)

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports from knowledge_engine
        unified_hub_module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for QualityAssuranceNode"
        )

        self.UnifiedKGIntegrationHub = None
        self.UnifiedKGConfig = None
        if unified_hub_module:
            self.UnifiedKGIntegrationHub = getattr(unified_hub_module, 'UnifiedKGIntegrationHub', None)
            self.UnifiedKGConfig = getattr(unified_hub_module, 'UnifiedKGConfig', None)

        # Safe import of QualityAssurance
        qa_module = self.safe_import(
            'quality_assurance',
            fallback_value=None,
            error_msg="QualityAssurance module not available"
        )

        self.QualityAssurance = None
        if qa_module:
            self.QualityAssurance = getattr(qa_module, 'QualityAssurance', None)

        # Safe import of QualityTracker
        qt_module = self.safe_import(
            'quality_tracker',
            fallback_value=None,
            error_msg="QualityTracker module not available"
        )

        self.QualityTracker = None
        if qt_module:
            self.QualityTracker = getattr(qt_module, 'QualityTracker', None)

        # Initialize hub instance if available
        self.hub = None
        if self.UnifiedKGIntegrationHub:
            try:
                self.hub = self.UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None

        # Initialize quality tracker if available
        self.quality_tracker = None
        if self.QualityTracker:
            try:
                storage_path = self.config.get('storage_path', './quality_metrics_history.json')
                self.quality_tracker = self.QualityTracker(storage_path=storage_path)
                self.logger.info("QualityTracker initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize QualityTracker: {e}")
                self.quality_tracker = None

        # Initialize internal metrics storage
        self._metrics_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required:
            - knowledge_graph_id: str - ID of knowledge graph to monitor

        Optional:
            - check_types: List[str] - Override configured check types
            - operation: str - Override configured operation
        """
        errors = []

        # Check for knowledge_graph_id
        has_kg_id = 'knowledge_graph_id' in inputs and inputs['knowledge_graph_id']
        has_entities = 'entities' in inputs and isinstance(inputs.get('entities'), list) and len(inputs['entities']) > 0

        if not has_kg_id and not has_entities:
            errors.append("Must provide either 'knowledge_graph_id' or 'entities'")

        # Validate knowledge_graph_id if provided
        if 'knowledge_graph_id' in inputs:
            if not isinstance(inputs['knowledge_graph_id'], str):
                errors.append("'knowledge_graph_id' must be a string")
            elif len(inputs['knowledge_graph_id'].strip()) == 0:
                errors.append("'knowledge_graph_id' cannot be empty")

        # Validate operation if provided
        if 'operation' in inputs:
            if inputs['operation'] not in self.OPERATIONS:
                errors.append(
                    f"Invalid operation: '{inputs['operation']}'. "
                    f"Must be one of: {', '.join(self.OPERATIONS)}"
                )

        # Validate check_types if provided
        if 'check_types' in inputs:
            check_types = inputs['check_types']
            if not isinstance(check_types, list):
                errors.append("'check_types' must be a list of strings")
            else:
                invalid_types = [ct for ct in check_types if ct not in self.CHECK_TYPES]
                if invalid_types:
                    errors.append(
                        f"Invalid check types: {', '.join(invalid_types)}. "
                        f"Valid types: {', '.join(self.CHECK_TYPES)}"
                    )

        # Validate quality_threshold if provided
        if 'quality_threshold' in inputs:
            try:
                threshold = float(inputs['quality_threshold'])
                if not 0.0 <= threshold <= 1.0:
                    errors.append("'quality_threshold' must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                errors.append("'quality_threshold' must be a number")

        # Validate degradation_threshold if provided
        if 'degradation_threshold' in inputs:
            try:
                threshold = float(inputs['degradation_threshold'])
                if not 0.0 <= threshold <= 1.0:
                    errors.append("'degradation_threshold' must be between 0.0 and 1.0")
            except (TypeError, ValueError):
                errors.append("'degradation_threshold' must be a number")

        # Validate time_window if provided
        if 'time_window' in inputs:
            time_window = inputs['time_window']
            if not isinstance(time_window, str):
                errors.append("'time_window' must be a string")
            elif not self.TIME_WINDOW_PATTERN.match(time_window) and time_window not in ['24h', '7d', '30d']:
                # Allow common time windows even if they don't match pattern exactly
                pass

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute quality assurance operation.

        Args:
            inputs: Contains knowledge_graph_id, check_types, and operation parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - quality_score: Overall quality score (0.0-1.0)
                - checks: List of individual check results
                - trends: Quality trend analysis
                - alerts: List of triggered alerts
                - report: Comprehensive quality report

        Raises:
            NodeExecutionError: If quality assurance fails
        """
        # Get configuration
        operation = inputs.get('operation', self.config.get('operation', 'monitor'))
        check_types = inputs.get('check_types', self.config.get('check_types', self.CHECK_TYPES))
        quality_threshold = inputs.get(
            'quality_threshold',
            self.config.get('quality_threshold', 0.8)
        )
        alert_on_degradation = inputs.get(
            'alert_on_degradation',
            self.config.get('alert_on_degradation', True)
        )
        degradation_threshold = inputs.get(
            'degradation_threshold',
            self.config.get('degradation_threshold', 0.1)
        )
        time_window = inputs.get('time_window', self.config.get('time_window', '24h'))
        entity_types = inputs.get('entity_types', self.config.get('entity_types', []))
        compare_baseline = inputs.get(
            'compare_baseline',
            self.config.get('compare_baseline', True)
        )

        context.update_progress(10, f"Initializing quality assurance: {operation}")
        self.logger.info(f"Starting quality assurance: operation={operation}")

        try:
            # Retrieve knowledge graph data
            kg_id = inputs.get('knowledge_graph_id', 'unknown')
            context.update_progress(20, "Retrieving knowledge graph data")
            kg_data = self._get_knowledge_graph_data(inputs, context)

            if not kg_data:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message="Could not retrieve knowledge graph data",
                    details={'inputs': list(inputs.keys())}
                )

            # Filter by entity types if specified
            if entity_types:
                kg_data = self._filter_by_entity_types(kg_data, entity_types)

            context.update_progress(30, f"Processing {len(kg_data.get('nodes', []))} entities")

            # Execute operation
            if operation == 'monitor':
                result = self._monitor_quality(
                    kg_data=kg_data,
                    kg_id=kg_id,
                    check_types=check_types,
                    quality_threshold=quality_threshold,
                    alert_on_degradation=alert_on_degradation,
                    degradation_threshold=degradation_threshold,
                    compare_baseline=compare_baseline,
                    context=context
                )
            elif operation == 'check':
                result = self._run_quality_check(
                    kg_data=kg_data,
                    check_types=check_types,
                    quality_threshold=quality_threshold,
                    context=context
                )
            elif operation == 'report':
                result = self._generate_quality_report(
                    kg_data=kg_data,
                    kg_id=kg_id,
                    check_types=check_types,
                    time_window=time_window,
                    context=context
                )
            elif operation == 'trend_analysis':
                result = self._analyze_trends(
                    kg_id=kg_id,
                    check_types=check_types,
                    time_window=time_window,
                    context=context
                )
            elif operation == 'alert_setup':
                result = self._setup_alerts(
                    kg_id=kg_id,
                    quality_threshold=quality_threshold,
                    degradation_threshold=degradation_threshold,
                    context=context
                )
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': self.OPERATIONS}
                )

            # Add metadata
            result['operation'] = operation
            result['timestamp'] = datetime.now().isoformat()
            result['execution_id'] = self.execution_id

            # Store artifact in context
            context.add_artifact('quality_assurance', {
                'operation': operation,
                'quality_score': result.get('quality_score', 0),
                'alert_count': len(result.get('alerts', [])),
                'check_count': len(result.get('checks', [])),
                'kg_id': kg_id
            })

            context.update_progress(100, f"Quality assurance {operation} completed")

            self.logger.info(
                f"Quality assurance completed: operation={operation}, "
                f"score={result.get('quality_score', 0):.2f}, "
                f"alerts={len(result.get('alerts', []))}"
            )

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Quality assurance failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Quality assurance failed: {str(e)}",
                details={
                    'operation': operation,
                    'exception_type': type(e).__name__,
                    'kg_id': inputs.get('knowledge_graph_id', 'unknown')
                }
            ) from e

    def _get_knowledge_graph_data(self, inputs: Dict, context) -> Optional[Dict[str, Any]]:
        """Retrieve knowledge graph data from inputs or hub."""
        # Direct entities data
        if 'entities' in inputs and inputs['entities']:
            return {
                'nodes': inputs['entities'],
                'edges': inputs.get('edges', []),
                'triples': inputs.get('triples', [])
            }

        # Direct knowledge graph data
        if 'knowledge_graph' in inputs and inputs['knowledge_graph']:
            return inputs['knowledge_graph']

        # Fetch from hub using knowledge_graph_id
        kg_id = inputs.get('knowledge_graph_id')
        if kg_id and self.hub:
            try:
                if hasattr(self.hub, 'get_knowledge_graph'):
                    return self.hub.get_knowledge_graph(kg_id)
                elif hasattr(self.hub, 'export_graph'):
                    return self.hub.export_graph(kg_id)
                elif hasattr(self.hub, 'get_entities'):
                    entities = self.hub.get_entities(kg_id)
                    return {'nodes': entities, 'edges': [], 'triples': []}
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

    def _filter_by_entity_types(self, kg_data: Dict[str, Any], entity_types: List[str]) -> Dict[str, Any]:
        """Filter knowledge graph by entity types."""
        if not entity_types:
            return kg_data

        nodes = kg_data.get('nodes', [])
        filtered_nodes = [
            node for node in nodes
            if node.get('type') in entity_types or node.get('entity_type') in entity_types
        ]

        result = dict(kg_data)
        result['nodes'] = filtered_nodes
        return result

    def _monitor_quality(
        self,
        kg_data: Dict[str, Any],
        kg_id: str,
        check_types: List[str],
        quality_threshold: float,
        alert_on_degradation: bool,
        degradation_threshold: float,
        compare_baseline: bool,
        context
    ) -> Dict[str, Any]:
        """Monitor quality metrics and detect degradation."""
        context.update_progress(40, "Running quality checks")

        # Run quality checks
        check_results = self._run_checks(kg_data, check_types, context)

        # Calculate overall quality score
        quality_score = self._calculate_overall_score(check_results)

        context.update_progress(60, "Analyzing trends and detecting degradation")

        # Get historical data for comparison
        historical_scores = self._get_historical_scores(kg_id)

        # Detect degradation
        alerts = []
        if compare_baseline and historical_scores:
            degradation_detected = self._detect_degradation(
                current_score=quality_score,
                historical_scores=historical_scores,
                degradation_threshold=degradation_threshold
            )
            if degradation_detected and alert_on_degradation:
                alerts.append({
                    'type': 'degradation',
                    'severity': 'warning',
                    'message': f"Quality degradation detected: score dropped by {degradation_detected:.2f}",
                    'current_score': quality_score,
                    'previous_score': historical_scores[-1] if historical_scores else quality_score,
                    'threshold': degradation_threshold
                })

        # Check quality threshold
        if quality_score < quality_threshold:
            alerts.append({
                'type': 'threshold',
                'severity': 'critical' if quality_score < quality_threshold * 0.8 else 'warning',
                'message': f"Quality score {quality_score:.2f} below threshold {quality_threshold:.2f}",
                'current_score': quality_score,
                'threshold': quality_threshold
            })

        # Check individual check scores
        for check in check_results:
            if check['score'] < quality_threshold:
                alerts.append({
                    'type': 'check_failure',
                    'severity': 'warning',
                    'message': f"Check '{check['type']}' score {check['score']:.2f} below threshold",
                    'check_type': check['type'],
                    'score': check['score']
                })

        # Store current metrics
        self._store_metrics(kg_id, {
            'timestamp': datetime.now().isoformat(),
            'quality_score': quality_score,
            'checks': {c['type']: c['score'] for c in check_results},
            'alert_count': len(alerts)
        })

        context.update_progress(80, "Generating monitoring report")

        return {
            'quality_score': round(quality_score, 4),
            'checks': check_results,
            'alerts': alerts,
            'trends': {
                'historical_scores': historical_scores[-10:] if historical_scores else [],
                'trend_direction': self._calculate_trend(historical_scores + [quality_score]),
                'baseline_comparison': self._compare_to_baseline(quality_score, historical_scores)
            },
            'report': {
                'entities_checked': len(kg_data.get('nodes', [])),
                'checks_performed': len(check_results),
                'alerts_triggered': len(alerts),
                'monitoring_status': 'healthy' if not alerts else 'degraded'
            }
        }

    def _run_quality_check(
        self,
        kg_data: Dict[str, Any],
        check_types: List[str],
        quality_threshold: float,
        context
    ) -> Dict[str, Any]:
        """Run one-time quality check."""
        context.update_progress(50, "Executing quality checks")

        check_results = self._run_checks(kg_data, check_types, context)
        quality_score = self._calculate_overall_score(check_results)

        # Generate alerts for failed checks
        alerts = []
        if quality_score < quality_threshold:
            alerts.append({
                'type': 'quality_failure',
                'severity': 'error',
                'message': f"Quality check failed: score {quality_score:.2f} below threshold {quality_threshold:.2f}",
                'score': quality_score,
                'threshold': quality_threshold
            })

        for check in check_results:
            if check['score'] < 0.5:
                alerts.append({
                    'type': 'check_failure',
                    'severity': 'error',
                    'message': f"Check '{check['type']}' failed with score {check['score']:.2f}",
                    'check_type': check['type'],
                    'score': check['score']
                })

        return {
            'quality_score': round(quality_score, 4),
            'checks': check_results,
            'alerts': alerts,
            'trends': {},
            'report': {
                'passed': quality_score >= quality_threshold,
                'entities_checked': len(kg_data.get('nodes', [])),
                'checks_performed': len(check_results),
                'failed_checks': sum(1 for c in check_results if c['score'] < quality_threshold)
            }
        }

    def _generate_quality_report(
        self,
        kg_data: Dict[str, Any],
        kg_id: str,
        check_types: List[str],
        time_window: str,
        context
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        context.update_progress(40, "Running quality checks for report")

        check_results = self._run_checks(kg_data, check_types, context)
        quality_score = self._calculate_overall_score(check_results)

        context.update_progress(60, "Gathering historical data")

        # Get historical metrics
        historical_metrics = self._get_metrics_history(kg_id, time_window)

        context.update_progress(80, "Compiling report")

        # Build detailed report
        report = {
            'summary': {
                'quality_score': round(quality_score, 4),
                'grade': self._score_to_grade(quality_score),
                'entities_checked': len(kg_data.get('nodes', [])),
                'edges_checked': len(kg_data.get('edges', [])),
                'report_generated_at': datetime.now().isoformat(),
                'time_window': time_window
            },
            'check_details': check_results,
            'historical_comparison': {
                'average_score': round(statistics.mean([m['quality_score'] for m in historical_metrics]), 4) if historical_metrics else quality_score,
                'score_trend': self._calculate_trend([m['quality_score'] for m in historical_metrics] + [quality_score]),
                'data_points': len(historical_metrics)
            },
            'recommendations': self._generate_recommendations(check_results, quality_score),
            'check_breakdown': {
                check['type']: {
                    'score': check['score'],
                    'issues_count': len(check.get('issues', [])),
                    'status': 'passed' if check['score'] >= 0.8 else 'warning' if check['score'] >= 0.5 else 'failed'
                }
                for check in check_results
            }
        }

        return {
            'quality_score': round(quality_score, 4),
            'checks': check_results,
            'alerts': [],
            'trends': {
                'historical_data': historical_metrics,
                'trend_direction': report['historical_comparison']['score_trend']
            },
            'report': report
        }

    def _analyze_trends(
        self,
        kg_id: str,
        check_types: List[str],
        time_window: str,
        context
    ) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        context.update_progress(50, "Retrieving historical metrics")

        historical_metrics = self._get_metrics_history(kg_id, time_window)

        if not historical_metrics:
            return {
                'quality_score': 0.0,
                'checks': [],
                'alerts': [{
                    'type': 'no_data',
                    'severity': 'info',
                    'message': f'No historical data available for time window: {time_window}'
                }],
                'trends': {
                    'historical_data': [],
                    'trend_direction': 'unknown',
                    'available_data_points': 0
                },
                'report': {
                    'note': 'Insufficient data for trend analysis'
                }
            }

        context.update_progress(70, "Calculating trend statistics")

        scores = [m['quality_score'] for m in historical_metrics]

        trend_analysis = {
            'score_statistics': {
                'mean': round(statistics.mean(scores), 4),
                'median': round(statistics.median(scores), 4),
                'min': round(min(scores), 4),
                'max': round(max(scores), 4),
                'std_dev': round(statistics.stdev(scores), 4) if len(scores) > 1 else 0
            },
            'trend_direction': self._calculate_trend(scores),
            'data_points': len(historical_metrics),
            'time_span': {
                'start': historical_metrics[0].get('timestamp'),
                'end': historical_metrics[-1].get('timestamp')
            }
        }

        # Analyze trends for each check type
        check_trends = {}
        for check_type in check_types:
            check_scores = []
            for metric in historical_metrics:
                if 'checks' in metric and check_type in metric['checks']:
                    check_scores.append(metric['checks'][check_type])
            if check_scores:
                check_trends[check_type] = {
                    'mean': round(statistics.mean(check_scores), 4),
                    'trend': self._calculate_trend(check_scores)
                }

        trend_analysis['check_trends'] = check_trends

        context.update_progress(90, "Detecting anomalies")

        # Detect anomalies (significant drops)
        alerts = []
        if len(scores) >= 2:
            recent_avg = statistics.mean(scores[-5:]) if len(scores) >= 5 else statistics.mean(scores)
            older_avg = statistics.mean(scores[:-5]) if len(scores) >= 10 else statistics.mean(scores[:max(1, len(scores)//2)])

            if recent_avg < older_avg - 0.1:
                alerts.append({
                    'type': 'declining_trend',
                    'severity': 'warning',
                    'message': f'Declining quality trend detected: recent avg {recent_avg:.2f} vs older avg {older_avg:.2f}',
                    'recent_average': round(recent_avg, 4),
                    'older_average': round(older_avg, 4)
                })

        return {
            'quality_score': round(scores[-1], 4) if scores else 0.0,
            'checks': [],
            'alerts': alerts,
            'trends': trend_analysis,
            'report': {
                'analysis_summary': f"Analyzed {len(historical_metrics)} data points over {time_window}",
                'overall_trend': trend_analysis['trend_direction'],
                'stability': 'stable' if trend_analysis['score_statistics']['std_dev'] < 0.1 else 'variable'
            }
        }

    def _setup_alerts(
        self,
        kg_id: str,
        quality_threshold: float,
        degradation_threshold: float,
        context
    ) -> Dict[str, Any]:
        """Configure quality alerts."""
        context.update_progress(50, "Configuring alert thresholds")

        # Store alert configuration
        alert_config = {
            'kg_id': kg_id,
            'quality_threshold': quality_threshold,
            'degradation_threshold': degradation_threshold,
            'enabled': True,
            'created_at': datetime.now().isoformat()
        }

        # Store in metrics history for reference
        self._store_metrics(f"{kg_id}_alert_config", alert_config)

        context.update_progress(100, "Alert configuration saved")

        return {
            'quality_score': 1.0,
            'checks': [],
            'alerts': [],
            'trends': {},
            'report': {
                'alert_configuration': alert_config,
                'status': 'configured',
                'message': f'Alerts configured for knowledge graph: {kg_id}'
            }
        }

    def _run_checks(self, kg_data: Dict[str, Any], check_types: List[str], context) -> List[Dict[str, Any]]:
        """Run all specified quality checks."""
        results = []
        progress_per_check = 40 // max(len(check_types), 1)

        for i, check_type in enumerate(check_types):
            context.update_progress(40 + i * progress_per_check, f"Running {check_type} check")

            if check_type == 'completeness':
                result = self._check_completeness(kg_data)
            elif check_type == 'accuracy':
                result = self._check_accuracy(kg_data)
            elif check_type == 'consistency':
                result = self._check_consistency(kg_data)
            elif check_type == 'timeliness':
                result = self._check_timeliness(kg_data)
            elif check_type == 'validity':
                result = self._check_validity(kg_data)
            else:
                continue

            results.append(result)

        return results

    def _check_completeness(self, kg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check completeness: missing properties, null values."""
        nodes = kg_data.get('nodes', [])
        issues = []
        total_checks = 0
        passed_checks = 0

        # Required properties
        required_props = ['id', 'type', 'name']

        for node in nodes:
            entity_id = node.get('id', 'unknown')

            # Check required properties
            for prop in required_props:
                total_checks += 1
                if prop not in node or node[prop] is None:
                    issues.append({
                        'entity_id': entity_id,
                        'issue': f"Missing required property: {prop}",
                        'severity': 'error'
                    })
                else:
                    passed_checks += 1

            # Check for null values
            for key, value in node.items():
                total_checks += 1
                if value is None:
                    issues.append({
                        'entity_id': entity_id,
                        'issue': f"Null value for property: {key}",
                        'severity': 'warning'
                    })
                elif isinstance(value, str) and not value.strip():
                    issues.append({
                        'entity_id': entity_id,
                        'issue': f"Empty string for property: {key}",
                        'severity': 'warning'
                    })
                else:
                    passed_checks += 1

            # Check for description
            total_checks += 1
            if 'description' not in node or not node.get('description'):
                issues.append({
                    'entity_id': entity_id,
                    'issue': "Missing description",
                    'severity': 'info'
                })
            else:
                passed_checks += 1

        score = passed_checks / total_checks if total_checks > 0 else 1.0

        return {
            'type': 'completeness',
            'score': round(score, 4),
            'issues': issues,
            'details': {
                'entities_checked': len(nodes),
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': total_checks - passed_checks
            }
        }

    def _check_accuracy(self, kg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check accuracy: confidence scores, source reliability."""
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])
        issues = []
        total_score = 0.0
        count = 0

        # Check node confidence scores
        for node in nodes:
            entity_id = node.get('id', 'unknown')
            confidence = node.get('confidence')

            if confidence is not None:
                try:
                    conf_val = float(confidence)
                    total_score += conf_val
                    count += 1

                    if conf_val < 0.5:
                        issues.append({
                            'entity_id': entity_id,
                            'issue': f"Low confidence score: {conf_val:.2f}",
                            'severity': 'warning'
                        })
                except (TypeError, ValueError):
                    issues.append({
                        'entity_id': entity_id,
                        'issue': f"Invalid confidence value: {confidence}",
                        'severity': 'error'
                    })
            else:
                issues.append({
                    'entity_id': entity_id,
                    'issue': "Missing confidence score",
                    'severity': 'info'
                })

            # Check source attribution
            source = node.get('source')
            if not source:
                issues.append({
                    'entity_id': entity_id,
                    'issue': "Missing source attribution",
                    'severity': 'info'
                })

        # Check edge confidence
        for edge in edges:
            confidence = edge.get('confidence')
            if confidence is not None:
                try:
                    total_score += float(confidence)
                    count += 1
                except (TypeError, ValueError):
                    pass

        avg_confidence = total_score / count if count > 0 else 0.8

        return {
            'type': 'accuracy',
            'score': round(avg_confidence, 4),
            'issues': issues,
            'details': {
                'entities_checked': len(nodes),
                'edges_checked': len(edges),
                'average_confidence': round(avg_confidence, 4),
                'entities_with_confidence': count
            }
        }

    def _check_consistency(self, kg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency: contradictions, duplicate detection."""
        nodes = kg_data.get('nodes', [])
        issues = []

        # Check for duplicate IDs
        seen_ids = set()
        duplicate_ids = set()
        for node in nodes:
            node_id = node.get('id')
            if node_id:
                if node_id in seen_ids:
                    duplicate_ids.add(node_id)
                seen_ids.add(node_id)

        for dup_id in duplicate_ids:
            issues.append({
                'entity_id': dup_id,
                'issue': f"Duplicate entity ID: {dup_id}",
                'severity': 'error'
            })

        # Check for duplicate content (simplified)
        node_signatures = {}
        for node in nodes:
            node_id = node.get('id', 'unknown')
            # Create signature from name + type
            sig = f"{node.get('name', '')}:{node.get('type', '')}"
            if sig in node_signatures:
                issues.append({
                    'entity_id': node_id,
                    'issue': f"Potential duplicate of {node_signatures[sig]}: {sig}",
                    'severity': 'warning'
                })
            else:
                node_signatures[sig] = node_id

        # Check for type consistency
        type_patterns = defaultdict(set)
        for node in nodes:
            node_type = node.get('type', 'unknown')
            for prop in node.keys():
                type_patterns[node_type].add(prop)

        # Find entities missing common properties for their type
        for node in nodes:
            node_id = node.get('id', 'unknown')
            node_type = node.get('type', 'unknown')
            if node_type in type_patterns:
                common_props = type_patterns[node_type]
                missing = common_props - set(node.keys())
                if len(missing) > len(common_props) * 0.5:
                    issues.append({
                        'entity_id': node_id,
                        'issue': f"Inconsistent properties for type '{node_type}': missing {len(missing)} common properties",
                        'severity': 'warning'
                    })

        score = 1.0 - (len(duplicate_ids) * 0.3 + len(issues) * 0.05) / max(len(nodes), 1)
        score = max(0.0, min(1.0, score))

        return {
            'type': 'consistency',
            'score': round(score, 4),
            'issues': issues,
            'details': {
                'entities_checked': len(nodes),
                'duplicate_ids_found': len(duplicate_ids),
                'consistency_issues': len(issues)
            }
        }

    def _check_timeliness(self, kg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check timeliness: freshness, update frequency."""
        nodes = kg_data.get('nodes', [])
        issues = []
        timestamps = []
        now = datetime.now()

        for node in nodes:
            entity_id = node.get('id', 'unknown')
            timestamp = node.get('timestamp') or node.get('created_at') or node.get('updated_at')

            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        ts = timestamp
                    timestamps.append(ts)

                    age_days = (now - ts).days
                    if age_days > 365:
                        issues.append({
                            'entity_id': entity_id,
                            'issue': f"Entity is stale: {age_days} days old",
                            'severity': 'warning'
                        })
                    elif age_days > 180:
                        issues.append({
                            'entity_id': entity_id,
                            'issue': f"Entity may need review: {age_days} days old",
                            'severity': 'info'
                        })
                except (ValueError, TypeError):
                    issues.append({
                        'entity_id': entity_id,
                        'issue': f"Invalid timestamp format: {timestamp}",
                        'severity': 'warning'
                    })
            else:
                issues.append({
                    'entity_id': entity_id,
                    'issue': "Missing timestamp",
                    'severity': 'info'
                })

        # Calculate freshness score
        if timestamps:
            avg_age_days = sum((now - ts).days for ts in timestamps) / len(timestamps)
            freshness_score = max(0.0, 1.0 - (avg_age_days / 365))
        else:
            freshness_score = 0.5  # Neutral if no timestamps

        return {
            'type': 'timeliness',
            'score': round(freshness_score, 4),
            'issues': issues,
            'details': {
                'entities_checked': len(nodes),
                'entities_with_timestamps': len(timestamps),
                'average_freshness': round(freshness_score, 4)
            }
        }

    def _check_validity(self, kg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check validity: schema compliance, format correctness."""
        nodes = kg_data.get('nodes', [])
        issues = []
        total_checks = 0
        passed_checks = 0

        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)

        for node in nodes:
            entity_id = node.get('id', 'unknown')

            # Check ID format
            total_checks += 1
            if isinstance(entity_id, str):
                if uuid_pattern.match(entity_id) or entity_id.replace('_', '').replace('-', '').isalnum():
                    passed_checks += 1
                else:
                    issues.append({
                        'entity_id': entity_id,
                        'issue': f"Invalid ID format: {entity_id}",
                        'severity': 'warning'
                    })

            # Check type field
            total_checks += 1
            if 'type' in node and isinstance(node['type'], str):
                passed_checks += 1
            else:
                issues.append({
                    'entity_id': entity_id,
                    'issue': "Missing or invalid 'type' field",
                    'severity': 'error'
                })

            # Check confidence range if present
            if 'confidence' in node:
                total_checks += 1
                try:
                    conf = float(node['confidence'])
                    if 0.0 <= conf <= 1.0:
                        passed_checks += 1
                    else:
                        issues.append({
                            'entity_id': entity_id,
                            'issue': f"Confidence out of range: {conf}",
                            'severity': 'error'
                        })
                except (TypeError, ValueError):
                    issues.append({
                        'entity_id': entity_id,
                        'issue': f"Invalid confidence value: {node['confidence']}",
                        'severity': 'error'
                    })

            # Check timestamp format if present
            if 'timestamp' in node and node['timestamp']:
                total_checks += 1
                try:
                    ts = node['timestamp']
                    if isinstance(ts, str):
                        datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    passed_checks += 1
                except (ValueError, TypeError):
                    issues.append({
                        'entity_id': entity_id,
                        'issue': f"Invalid timestamp format: {node['timestamp']}",
                        'severity': 'warning'
                    })

        score = passed_checks / total_checks if total_checks > 0 else 1.0

        return {
            'type': 'validity',
            'score': round(score, 4),
            'issues': issues,
            'details': {
                'entities_checked': len(nodes),
                'total_checks': total_checks,
                'passed_checks': passed_checks
            }
        }

    def _calculate_overall_score(self, check_results: List[Dict[str, Any]]) -> float:
        """Calculate overall quality score from individual checks."""
        if not check_results:
            return 0.0

        # Weighted average of check scores
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.20,
            'timeliness': 0.15,
            'validity': 0.15
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for check in check_results:
            check_type = check['type']
            weight = weights.get(check_type, 0.2)
            weighted_sum += check['score'] * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _get_historical_scores(self, kg_id: str) -> List[float]:
        """Get historical quality scores for comparison."""
        history = self._metrics_history.get(kg_id, [])
        return [m['quality_score'] for m in history]

    def _store_metrics(self, kg_id: str, metrics: Dict[str, Any]):
        """Store metrics for history tracking."""
        self._metrics_history[kg_id].append(metrics)

        # Keep history manageable
        if len(self._metrics_history[kg_id]) > 1000:
            self._metrics_history[kg_id] = self._metrics_history[kg_id][-500:]

        # Also store via QualityTracker if available
        if self.quality_tracker and hasattr(self.quality_tracker, 'record_metric'):
            try:
                self.quality_tracker.record_metric(kg_id, metrics)
            except Exception as e:
                self.logger.warning(f"Could not store metrics via QualityTracker: {e}")

    def _get_metrics_history(self, kg_id: str, time_window: str) -> List[Dict[str, Any]]:
        """Get metrics history for a given time window."""
        # Parse time window
        delta = self._parse_time_window(time_window)
        cutoff = datetime.now() - delta

        history = self._metrics_history.get(kg_id, [])
        filtered = []

        for metric in history:
            try:
                ts = metric.get('timestamp')
                if ts:
                    if isinstance(ts, str):
                        metric_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    else:
                        metric_time = ts
                    if metric_time >= cutoff:
                        filtered.append(metric)
            except (ValueError, TypeError):
                continue

        return filtered

    def _parse_time_window(self, time_window: str) -> timedelta:
        """Parse time window string to timedelta."""
        match = self.TIME_WINDOW_PATTERN.match(time_window)
        if match:
            value = int(match.group(1))
            unit = match.group(2).lower()

            if unit == 'h':
                return timedelta(hours=value)
            elif unit == 'd':
                return timedelta(days=value)
            elif unit == 'w':
                return timedelta(weeks=value)
            elif unit == 'm':
                return timedelta(days=value * 30)
            elif unit == 'y':
                return timedelta(days=value * 365)

        # Default to 24 hours
        return timedelta(hours=24)

    def _detect_degradation(
        self,
        current_score: float,
        historical_scores: List[float],
        degradation_threshold: float
    ) -> Optional[float]:
        """Detect if quality has degraded significantly."""
        if not historical_scores:
            return None

        # Compare to recent average
        recent_scores = historical_scores[-5:] if len(historical_scores) >= 5 else historical_scores
        recent_avg = statistics.mean(recent_scores)

        drop = recent_avg - current_score
        if drop > degradation_threshold:
            return drop

        return None

    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend direction from scores."""
        if len(scores) < 2:
            return 'insufficient_data'

        # Simple linear trend
        n = len(scores)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(scores)

        numerator = sum((i - x_mean) * (score - y_mean) for i, score in enumerate(scores))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 'stable'

        slope = numerator / denominator

        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'

    def _compare_to_baseline(self, current_score: float, historical_scores: List[float]) -> Dict[str, Any]:
        """Compare current score to historical baseline."""
        if not historical_scores:
            return {'status': 'no_baseline', 'difference': 0}

        baseline = statistics.mean(historical_scores)
        difference = current_score - baseline

        return {
            'status': 'improved' if difference > 0.05 else 'degraded' if difference < -0.05 else 'stable',
            'difference': round(difference, 4),
            'baseline_score': round(baseline, 4),
            'current_score': round(current_score, 4)
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'

    def _generate_recommendations(self, check_results: List[Dict[str, Any]], overall_score: float) -> List[str]:
        """Generate improvement recommendations based on check results."""
        recommendations = []

        # Overall quality recommendation
        if overall_score < 0.6:
            recommendations.append("Critical: Overall quality is very low. Comprehensive review recommended.")
        elif overall_score < 0.8:
            recommendations.append("Quality improvement needed. Focus on addressing failed checks.")

        # Check-specific recommendations
        for check in check_results:
            if check['score'] < 0.5:
                if check['type'] == 'completeness':
                    recommendations.append("Add missing required properties (id, type, name) and fill null values.")
                elif check['type'] == 'accuracy':
                    recommendations.append("Review and update confidence scores. Add source attribution.")
                elif check['type'] == 'consistency':
                    recommendations.append("Remove duplicate entities. Standardize property usage across types.")
                elif check['type'] == 'timeliness':
                    recommendations.append("Update stale entities. Add timestamps to entities missing them.")
                elif check['type'] == 'validity':
                    recommendations.append("Fix format errors. Ensure schema compliance for all entities.")

        return recommendations

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration panel.
        """
        return {
            "type": "object",
            "title": "Quality Assurance Configuration",
            "description": "Configure quality assurance monitoring parameters",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Quality assurance operation to perform",
                    "enum": ["monitor", "check", "report", "trend_analysis", "alert_setup"],
                    "enumNames": [
                        "Monitor - Continuous quality monitoring with alerts",
                        "Check - One-time quality check",
                        "Report - Generate comprehensive quality report",
                        "Trend Analysis - Analyze quality trends over time",
                        "Alert Setup - Configure quality alert thresholds"
                    ],
                    "default": "monitor"
                },
                "check_types": {
                    "type": "array",
                    "title": "Check Types",
                    "description": "Types of quality checks to perform",
                    "items": {
                        "type": "string",
                        "enum": ["completeness", "accuracy", "consistency", "timeliness", "validity"]
                    },
                    "default": ["completeness", "accuracy", "consistency", "timeliness", "validity"]
                },
                "quality_threshold": {
                    "type": "number",
                    "title": "Quality Threshold",
                    "description": "Minimum quality score for passing (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8
                },
                "alert_on_degradation": {
                    "type": "boolean",
                    "title": "Alert on Degradation",
                    "description": "Trigger alerts when quality degrades",
                    "default": True
                },
                "degradation_threshold": {
                    "type": "number",
                    "title": "Degradation Threshold",
                    "description": "Quality drop amount that triggers alert (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.1
                },
                "time_window": {
                    "type": "string",
                    "title": "Time Window",
                    "description": "Time window for trend analysis (e.g., '24h', '7d', '30d')",
                    "default": "24h"
                },
                "entity_types": {
                    "type": "array",
                    "title": "Entity Types",
                    "description": "Filter by specific entity types (empty = all types)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "compare_baseline": {
                    "type": "boolean",
                    "title": "Compare with Baseline",
                    "description": "Compare current quality to historical baseline",
                    "default": True
                },
                "storage_path": {
                    "type": "string",
                    "title": "Storage Path",
                    "description": "Path for storing quality metrics history",
                    "default": "./quality_metrics_history.json"
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node is healthy (can run quality checks without external dependencies)
        """
        try:
            # Node can work without external dependencies (has internal quality check logic)
            return True
        except Exception:
            return False

    def get_supported_operations(self) -> List[str]:
        """
        Get list of supported operations.

        Returns:
            List of operation names
        """
        return self.OPERATIONS.copy()

    def get_supported_check_types(self) -> List[str]:
        """
        Get list of supported check types.

        Returns:
            List of check type names
        """
        return self.CHECK_TYPES.copy()
