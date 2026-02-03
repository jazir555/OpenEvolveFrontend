"""
Knowledge Engine Monitoring and Observability System for OpenEvolve

This module implements a comprehensive monitoring and observability framework that provides:
- Real-time performance monitoring
- Health status tracking
- Comprehensive logging and analytics
- Alerting and notification system
- Integration with all knowledge engine components
- Dashboard-ready metrics and visualizations
"""

import json
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import threading

# Import knowledge engine components
try:
    from .knowledge_extractor import KnowledgeExtractor
    from .knowledge_processor import KnowledgeProcessor
    from .knowledge_validator import KnowledgeValidator
    from .knowledge_storage import KnowledgeStorage
    from .knowledge_retriever import KnowledgeRetriever
except ImportError:
    from knowledge_extractor import KnowledgeExtractor
    from knowledge_processor import KnowledgeProcessor
    from knowledge_validator import KnowledgeValidator
    from knowledge_storage import KnowledgeStorage
    from knowledge_retriever import KnowledgeRetriever

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeMonitor:
    """
    Advanced Monitoring and Observability System for OpenEvolve Knowledge Engine.
    
    This class implements a comprehensive monitoring framework with:
    - Real-time performance monitoring
    - Health status tracking
    - Comprehensive metrics collection
    - Alerting and notification system
    - Historical trend analysis
    - Integration with all knowledge engine components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge monitor.
        
        Args:
            config: Configuration dictionary with monitoring parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Monitoring configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 60)  # seconds
        self.metrics_retention = self.config.get('metrics_retention', 3600)  # seconds
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.health_thresholds = self._initialize_health_thresholds()
        
        # Metrics storage
        self.performance_metrics = defaultdict(deque)
        self.health_metrics = defaultdict(deque)
        self.system_metrics = defaultdict(deque)
        self.alert_history = deque(maxlen=100)
        self.event_log = deque(maxlen=1000)
        
        # Component monitoring
        self.component_status = {
            'knowledge_extractor': {'status': 'unknown', 'last_check': None, 'metrics': {}},
            'knowledge_processor': {'status': 'unknown', 'last_check': None, 'metrics': {}},
            'knowledge_validator': {'status': 'unknown', 'last_check': None, 'metrics': {}},
            'knowledge_storage': {'status': 'unknown', 'last_check': None, 'metrics': {}},
            'knowledge_retriever': {'status': 'unknown', 'last_check': None, 'metrics': {}}
        }
        
        # Monitoring statistics
        self.monitoring_cycles = 0
        self.alerts_triggered = 0
        self.events_logged = 0
        self.monitoring_start_time = datetime.now()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Initialize monitoring
        self._setup_monitoring()
        
        self.logger.info("Knowledge monitor initialized with comprehensive observability framework")
    
    def _setup_monitoring(self):
        """Setup monitoring infrastructure"""
        # Initialize metrics storage with retention limits
        for metrics_store in [self.performance_metrics, self.health_metrics, self.system_metrics]:
            for key in metrics_store.keys():
                metrics_store[key] = deque(maxlen=self.metrics_retention // self.monitoring_interval)
        
        # Setup initial health checks
        self._perform_initial_health_checks()
    
    def _initialize_alert_thresholds(self) -> Dict[str, Any]:
        """Initialize alert thresholds for monitoring"""
        return {
            'performance': {
                'extraction_time': {'warning': 1.0, 'critical': 2.0},
                'processing_time': {'warning': 0.5, 'critical': 1.0},
                'validation_time': {'warning': 0.2, 'critical': 0.5},
                'retrieval_time': {'warning': 0.3, 'critical': 0.8}
            },
            'quality': {
                'success_rate': {'warning': 0.85, 'critical': 0.70},
                'quality_score': {'warning': 0.75, 'critical': 0.60},
                'compliance_rate': {'warning': 0.80, 'critical': 0.65}
            },
            'system': {
                'memory_usage': {'warning': 0.75, 'critical': 0.90},
                'cpu_usage': {'warning': 0.80, 'critical': 0.95},
                'error_rate': {'warning': 0.05, 'critical': 0.10}
            }
        }
    
    def _initialize_health_thresholds(self) -> Dict[str, Any]:
        """Initialize health thresholds for component monitoring"""
        return {
            'knowledge_extractor': {
                'success_rate': {'healthy': 0.95, 'degraded': 0.85, 'unhealthy': 0.70},
                'quality_distribution': {'excellent': 0.70, 'good': 0.20, 'fair': 0.10}
            },
            'knowledge_processor': {
                'enhancement_rate': {'healthy': 0.90, 'degraded': 0.75, 'unhealthy': 0.60},
                'processing_time': {'healthy': 0.1, 'degraded': 0.3, 'unhealthy': 0.5}
            },
            'knowledge_validator': {
                'validation_success_rate': {'healthy': 0.98, 'degraded': 0.90, 'unhealthy': 0.80},
                'compliance_rate': {'healthy': 0.95, 'degraded': 0.85, 'unhealthy': 0.70}
            },
            'knowledge_storage': {
                'storage_latency': {'healthy': 0.05, 'degraded': 0.15, 'unhealthy': 0.30},
                'retrieval_success_rate': {'healthy': 0.99, 'degraded': 0.95, 'unhealthy': 0.90}
            },
            'knowledge_retriever': {
                'search_accuracy': {'healthy': 0.95, 'degraded': 0.85, 'unhealthy': 0.70},
                'response_time': {'healthy': 0.1, 'degraded': 0.3, 'unhealthy': 0.5}
            }
        }
    
    def _perform_initial_health_checks(self):
        """Perform initial health checks on all components"""
        for component in self.component_status.keys():
            self._check_component_health(component)
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_start_time = datetime.now()
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("Started continuous monitoring")
            self._log_event('monitoring_started', {'status': 'active'})
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("Stopped continuous monitoring")
            self._log_event('monitoring_stopped', {'status': 'inactive'})
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Perform monitoring cycle
                self._perform_monitoring_cycle()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring cycle failed: {str(e)}")
                self._log_event('monitoring_error', {'error': str(e)})
                time.sleep(min(10, self.monitoring_interval))  # Wait before retry
    
    def _perform_monitoring_cycle(self):
        """Perform a complete monitoring cycle"""
        cycle_start = datetime.now()
        self.monitoring_cycles += 1
        
        self.logger.debug(f"Starting monitoring cycle {self.monitoring_cycles}")
        
        try:
            # Monitor each component
            for component in self.component_status.keys():
                self._monitor_component(component)
            
            # Check system health
            self._check_system_health()
            
            # Analyze trends
            self._analyze_trends()
            
            # Clean up old metrics
            self._cleanup_old_metrics()
            
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            self._log_metric('monitoring_cycle_time', cycle_time, 'system')
            
            self.logger.debug(f"Completed monitoring cycle {self.monitoring_cycles} in {cycle_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Monitoring cycle error: {str(e)}")
            self._log_event('monitoring_cycle_error', {'cycle': self.monitoring_cycles, 'error': str(e)})
            raise
    
    def _monitor_component(self, component_name: str):
        """Monitor a specific knowledge engine component"""
        try:
            # Check component health
            health_status = self._check_component_health(component_name)
            
            # Collect performance metrics
            self._collect_component_metrics(component_name)
            
            # Check for alerts
            self._check_component_alerts(component_name)
            
            self.logger.debug(f"Monitored {component_name}: {health_status}")
            
        except Exception as e:
            self.logger.error(f"Failed to monitor {component_name}: {str(e)}")
            self._log_event('component_monitoring_error', {'component': component_name, 'error': str(e)})
    
    def _check_component_health(self, component_name: str) -> str:
        """Check the health status of a component"""
        component = self.component_status[component_name]
        
        # Get current metrics (in real implementation, this would query the actual component)
        # For this example, we'll use simulated metrics
        if component_name == 'knowledge_extractor':
            # Simulate extraction metrics
            success_rate = 0.96
            quality_score = 0.88
            error_rate = 0.02
            
            # Determine health status
            if success_rate >= self.health_thresholds[component_name]['success_rate']['healthy']:
                status = 'healthy'
            elif success_rate >= self.health_thresholds[component_name]['success_rate']['degraded']:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            component['metrics'] = {
                'success_rate': success_rate,
                'quality_score': quality_score,
                'error_rate': error_rate,
                'last_extraction_time': 0.85
            }
            
        elif component_name == 'knowledge_processor':
            # Simulate processing metrics
            enhancement_rate = 0.92
            processing_time = 0.08
            error_rate = 0.01
            
            if enhancement_rate >= self.health_thresholds[component_name]['enhancement_rate']['healthy']:
                status = 'healthy'
            elif enhancement_rate >= self.health_thresholds[component_name]['enhancement_rate']['degraded']:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            component['metrics'] = {
                'enhancement_rate': enhancement_rate,
                'processing_time': processing_time,
                'error_rate': error_rate
            }
            
        elif component_name == 'knowledge_validator':
            # Simulate validation metrics
            validation_success_rate = 0.97
            compliance_rate = 0.94
            error_rate = 0.01
            
            if validation_success_rate >= self.health_thresholds[component_name]['validation_success_rate']['healthy']:
                status = 'healthy'
            elif validation_success_rate >= self.health_thresholds[component_name]['validation_success_rate']['degraded']:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            component['metrics'] = {
                'validation_success_rate': validation_success_rate,
                'compliance_rate': compliance_rate,
                'error_rate': error_rate
            }
            
        elif component_name == 'knowledge_storage':
            # Simulate storage metrics
            storage_latency = 0.03
            retrieval_success_rate = 0.99
            error_rate = 0.005
            
            if storage_latency <= self.health_thresholds[component_name]['storage_latency']['healthy']:
                status = 'healthy'
            elif storage_latency <= self.health_thresholds[component_name]['storage_latency']['degraded']:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            component['metrics'] = {
                'storage_latency': storage_latency,
                'retrieval_success_rate': retrieval_success_rate,
                'error_rate': error_rate
            }
            
        elif component_name == 'knowledge_retriever':
            # Simulate retrieval metrics
            search_accuracy = 0.93
            response_time = 0.07
            error_rate = 0.01
            
            if search_accuracy >= self.health_thresholds[component_name]['search_accuracy']['healthy']:
                status = 'healthy'
            elif search_accuracy >= self.health_thresholds[component_name]['search_accuracy']['degraded']:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            component['metrics'] = {
                'search_accuracy': search_accuracy,
                'response_time': response_time,
                'error_rate': error_rate
            }
        
        # Update component status
        component['status'] = status
        component['last_check'] = datetime.now().isoformat()
        
        # Log health status change
        if component.get('previous_status') and component['previous_status'] != status:
            self._log_event('component_health_change', {
                'component': component_name,
                'from': component['previous_status'],
                'to': status
            })
        
        component['previous_status'] = status
        
        return status
    
    def _collect_component_metrics(self, component_name: str):
        """Collect performance metrics for a component"""
        component = self.component_status[component_name]
        metrics = component.get('metrics', {})
        
        # Store metrics in time series
        for metric_name, metric_value in metrics.items():
            self.performance_metrics[f"{component_name}_{metric_name}"].append({
                'timestamp': datetime.now().isoformat(),
                'value': metric_value
            })
        
        # Log metrics collection
        self._log_event('metrics_collected', {
            'component': component_name,
            'metrics_count': len(metrics)
        })
    
    def _check_component_alerts(self, component_name: str):
        """Check for alert conditions in component metrics"""
        component = self.component_status[component_name]
        metrics = component.get('metrics', {})
        
        # Check performance alerts
        if component_name == 'knowledge_extractor':
            if metrics.get('last_extraction_time', 0) > self.alert_thresholds['performance']['extraction_time']['warning']:
                self._trigger_alert('high_extraction_time', component_name, 'warning', 
                                   f"Extraction time {metrics['last_extraction_time']:.3f}s exceeds warning threshold")
            
            if metrics.get('success_rate', 1.0) < self.alert_thresholds['quality']['success_rate']['warning']:
                self._trigger_alert('low_extraction_success', component_name, 'warning',
                                   f"Success rate {metrics['success_rate']:.2f} below warning threshold")
        
        elif component_name == 'knowledge_processor':
            if metrics.get('processing_time', 0) > self.alert_thresholds['performance']['processing_time']['warning']:
                self._trigger_alert('high_processing_time', component_name, 'warning',
                                   f"Processing time {metrics['processing_time']:.3f}s exceeds warning threshold")
            
            if metrics.get('enhancement_rate', 1.0) < self.alert_thresholds['quality']['success_rate']['warning']:
                self._trigger_alert('low_enhancement_rate', component_name, 'warning',
                                   f"Enhancement rate {metrics['enhancement_rate']:.2f} below warning threshold")
        
        elif component_name == 'knowledge_validator':
            if metrics.get('validation_success_rate', 1.0) < self.alert_thresholds['quality']['success_rate']['critical']:
                self._trigger_alert('low_validation_success', component_name, 'critical',
                                   f"Validation success rate {metrics['validation_success_rate']:.2f} below critical threshold")
            
            if metrics.get('compliance_rate', 1.0) < self.alert_thresholds['quality']['compliance_rate']['warning']:
                self._trigger_alert('low_compliance_rate', component_name, 'warning',
                                   f"Compliance rate {metrics['compliance_rate']:.2f} below warning threshold")
        
        # Check health status alerts
        if component['status'] == 'unhealthy':
            self._trigger_alert('component_unhealthy', component_name, 'critical',
                               f"Component {component_name} is in unhealthy state")
        elif component['status'] == 'degraded':
            self._trigger_alert('component_degraded', component_name, 'warning',
                               f"Component {component_name} is in degraded state")
    
    def _check_system_health(self):
        """Check overall system health"""
        try:
            # Simulate system metrics (in real implementation, use actual system monitoring)
            memory_usage = 0.65
            cpu_usage = 0.72
            error_rate = 0.03
            
            # Store system metrics
            self.system_metrics['memory_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'value': memory_usage
            })
            
            self.system_metrics['cpu_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'value': cpu_usage
            })
            
            self.system_metrics['error_rate'].append({
                'timestamp': datetime.now().isoformat(),
                'value': error_rate
            })
            
            # Check system alerts
            if memory_usage > self.alert_thresholds['system']['memory_usage']['warning']:
                self._trigger_alert('high_memory_usage', 'system', 'warning',
                                   f"Memory usage {memory_usage:.2f} exceeds warning threshold")
            
            if cpu_usage > self.alert_thresholds['system']['cpu_usage']['warning']:
                self._trigger_alert('high_cpu_usage', 'system', 'warning',
                                   f"CPU usage {cpu_usage:.2f} exceeds warning threshold")
            
            if error_rate > self.alert_thresholds['system']['error_rate']['warning']:
                self._trigger_alert('high_error_rate', 'system', 'warning',
                                   f"Error rate {error_rate:.2f} exceeds warning threshold")
            
            self.logger.debug("System health check completed")
            
        except Exception as e:
            self.logger.error(f"System health check failed: {str(e)}")
            self._log_event('system_health_check_error', {'error': str(e)})
    
    def _analyze_trends(self):
        """Analyze performance and quality trends"""
        try:
            trends = {
                'performance': {},
                'quality': {},
                'system': {}
            }
            
            # Analyze performance trends
            for metric_name, metric_data in self.performance_metrics.items():
                if len(metric_data) >= 2:
                    values = [item['value'] for item in metric_data]
                    recent_avg = statistics.mean(values[-3:])
                    historical_avg = statistics.mean(values[:-3]) if len(values) > 3 else values[0]
                    
                    if recent_avg > historical_avg * 1.1:
                        trend = 'increasing'
                    elif recent_avg < historical_avg * 0.9:
                        trend = 'decreasing'
                    else:
                        trend = 'stable'
                    
                    trends['performance'][metric_name] = trend
            
            # Analyze system trends
            for metric_name, metric_data in self.system_metrics.items():
                if len(metric_data) >= 2:
                    values = [item['value'] for item in metric_data]
                    recent_avg = statistics.mean(values[-3:])
                    historical_avg = statistics.mean(values[:-3]) if len(values) > 3 else values[0]
                    
                    if recent_avg > historical_avg * 1.1:
                        trend = 'increasing'
                    elif recent_avg < historical_avg * 0.9:
                        trend = 'decreasing'
                    else:
                        trend = 'stable'
                    
                    trends['system'][metric_name] = trend
            
            # Log trend analysis
            self._log_event('trend_analysis_completed', {'trends': trends})
            
            self.logger.debug("Trend analysis completed")
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {str(e)}")
            self._log_event('trend_analysis_error', {'error': str(e)})
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics data"""
        try:
            current_time = datetime.now()
            
            # Clean up performance metrics
            for metric_name, metric_data in list(self.performance_metrics.items()):
                if metric_data and len(metric_data) > 0:
                    oldest_time = datetime.fromisoformat(metric_data[0]['timestamp'])
                    if (current_time - oldest_time).total_seconds() > self.metrics_retention:
                        # Remove old metrics
                        while metric_data and (current_time - datetime.fromisoformat(metric_data[0]['timestamp'])).total_seconds() > self.metrics_retention:
                            metric_data.popleft()
            
            self.logger.debug("Metrics cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Metrics cleanup failed: {str(e)}")
            self._log_event('metrics_cleanup_error', {'error': str(e)})
    
    def _trigger_alert(self, alert_type: str, source: str, severity: str, message: str):
        """Trigger an alert"""
        alert = {
            'alert_id': hashlib.md5(f"{alert_type}_{source}_{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            'alert_type': alert_type,
            'source': source,
            'severity': severity,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'status': 'triggered',
            'acknowledged': False
        }
        
        self.alert_history.append(alert)
        self.alerts_triggered += 1
        
        # Log alert
        self._log_event('alert_triggered', {
            'alert_id': alert['alert_id'],
            'alert_type': alert_type,
            'severity': severity
        })
        
        # Log based on severity
        if severity == 'critical':
            self.logger.critical(f"CRITICAL ALERT: {message} (Source: {source})")
        elif severity == 'warning':
            self.logger.warning(f"WARNING ALERT: {message} (Source: {source})")
        else:
            self.logger.info(f"INFO ALERT: {message} (Source: {source})")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alert_history:
            if alert['alert_id'] == alert_id:
                alert['status'] = 'acknowledged'
                alert['acknowledged'] = True
                alert['acknowledged_timestamp'] = datetime.now().isoformat()
                
                self._log_event('alert_acknowledged', {'alert_id': alert_id})
                self.logger.info(f"Alert acknowledged: {alert_id}")
                return True
        
        return False
    
    def _log_metric(self, metric_name: str, metric_value: float, metric_type: str = 'performance'):
        """Log a metric"""
        metric = {
            'metric_name': metric_name,
            'value': metric_value,
            'type': metric_type,
            'timestamp': datetime.now().isoformat()
        }
        
        if metric_type == 'performance':
            self.performance_metrics[metric_name].append(metric)
        elif metric_type == 'health':
            self.health_metrics[metric_name].append(metric)
        elif metric_type == 'system':
            self.system_metrics[metric_name].append(metric)
    
    def _log_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log an event"""
        event = {
            'event_id': hashlib.md5(f"{event_type}_{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': event_data
        }
        
        self.event_log.append(event)
        self.events_logged += 1
        
        self.logger.debug(f"Event logged: {event_type}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'uptime': (datetime.now() - self.monitoring_start_time).total_seconds(),
            'monitoring_status': 'active' if self.is_monitoring else 'inactive',
            'component_status': {},
            'alerts': {
                'active': sum(1 for alert in self.alert_history if not alert['acknowledged']),
                'total': len(self.alert_history),
                'critical': sum(1 for alert in self.alert_history if alert['severity'] == 'critical' and not alert['acknowledged']),
                'warning': sum(1 for alert in self.alert_history if alert['severity'] == 'warning' and not alert['acknowledged'])
            },
            'metrics': {
                'monitoring_cycles': self.monitoring_cycles,
                'events_logged': self.events_logged,
                'alerts_triggered': self.alerts_triggered
            }
        }
        
        # Get component status summary
        for component_name, component_data in self.component_status.items():
            status['component_status'][component_name] = {
                'status': component_data['status'],
                'last_check': component_data['last_check'],
                'healthy': component_data['status'] == 'healthy'
            }
        
        # Calculate overall system health
        healthy_components = sum(1 for component in status['component_status'].values() if component['healthy'])
        total_components = len(status['component_status'])
        
        if healthy_components == total_components:
            status['overall_health'] = 'healthy'
        elif healthy_components >= total_components * 0.7:
            status['overall_health'] = 'degraded'
        else:
            status['overall_health'] = 'unhealthy'
        
        return status
    
    def get_performance_metrics(self, time_range: str = '1h') -> Dict[str, Any]:
        """Get performance metrics for a specific time range"""
        metrics = {'timestamp': datetime.now().isoformat(), 'metrics': {}}
        
        # Parse time range
        if time_range == '1h':
            cutoff = datetime.now() - timedelta(hours=1)
        elif time_range == '6h':
            cutoff = datetime.now() - timedelta(hours=6)
        elif time_range == '24h':
            cutoff = datetime.now() - timedelta(hours=24)
        elif time_range == '7d':
            cutoff = datetime.now() - timedelta(days=7)
        else:
            cutoff = datetime.now() - timedelta(hours=1)
        
        # Filter metrics by time range
        for metric_name, metric_data in self.performance_metrics.items():
            filtered_data = [
                item for item in metric_data
                if datetime.fromisoformat(item['timestamp']) >= cutoff
            ]
            
            if filtered_data:
                values = [item['value'] for item in filtered_data]
                metrics['metrics'][metric_name] = {
                    'count': len(values),
                    'average': statistics.mean(values),
                    'minimum': min(values),
                    'maximum': max(values),
                    'trend': self._calculate_trend(values)
                }
        
        return metrics
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from a list of values"""
        if len(values) >= 2:
            recent_avg = statistics.mean(values[-3:])
            historical_avg = statistics.mean(values[:-3]) if len(values) > 3 else values[0]
            
            if recent_avg > historical_avg * 1.05:
                return 'increasing'
            elif recent_avg < historical_avg * 0.95:
                return 'decreasing'
            else:
                return 'stable'
        return 'insufficient_data'
    
    def get_alert_history(self, limit: int = 10, severity: str = None) -> List[Dict[str, Any]]:
        """Get alert history"""
        alerts = list(self.alert_history)
        
        # Filter by severity if specified
        if severity:
            alerts = [alert for alert in alerts if alert['severity'] == severity]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return alerts[:limit] if limit else alerts
    
    def get_event_log(self, limit: int = 20, event_type: str = None) -> List[Dict[str, Any]]:
        """Get event log"""
        events = list(self.event_log)
        
        # Filter by event type if specified
        if event_type:
            events = [event for event in events if event['event_type'] == event_type]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return events[:limit] if limit else events
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        stats = {
            'monitoring_cycles': self.monitoring_cycles,
            'alerts_triggered': self.alerts_triggered,
            'events_logged': self.events_logged,
            'uptime': (datetime.now() - self.monitoring_start_time).total_seconds(),
            'performance_metrics_count': sum(len(metrics) for metrics in self.performance_metrics.values()),
            'health_metrics_count': sum(len(metrics) for metrics in self.health_metrics.values()),
            'system_metrics_count': sum(len(metrics) for metrics in self.system_metrics.values()),
            'alert_history_count': len(self.alert_history),
            'event_log_count': len(self.event_log)
        }
        
        # Calculate alert rates
        if stats['monitoring_cycles'] > 0:
            stats['alerts_per_cycle'] = stats['alerts_triggered'] / stats['monitoring_cycles']
            stats['events_per_cycle'] = stats['events_logged'] / stats['monitoring_cycles']
        
        return stats
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'system_status': self.get_system_status(),
            'performance_metrics': self.get_performance_metrics('1h'),
            'alert_summary': {
                'active_alerts': sum(1 for alert in self.alert_history if not alert['acknowledged']),
                'recent_alerts': self.get_alert_history(5),
                'alert_trends': self._analyze_alert_trends()
            },
            'component_health': {},
            'recommendations': []
        }
        
        # Add component health details
        for component_name, component_data in self.component_status.items():
            report['component_health'][component_name] = {
                'status': component_data['status'],
                'last_check': component_data['last_check'],
                'metrics': component_data.get('metrics', {}),
                'healthy': component_data['status'] == 'healthy'
            }
        
        # Generate recommendations
        if report['system_status']['overall_health'] == 'healthy':
            report['recommendations'].append("System is operating normally - maintain current monitoring")
        elif report['system_status']['overall_health'] == 'degraded':
            report['recommendations'].append("System showing signs of degradation - investigate degraded components")
        else:
            report['recommendations'].append("System in unhealthy state - immediate attention required")
        
        if report['alert_summary']['active_alerts'] > 0:
            report['recommendations'].append(f"Address {report['alert_summary']['active_alerts']} active alerts promptly")
        
        # Add component-specific recommendations
        for component_name, component_data in report['component_health'].items():
            if not component_data['healthy']:
                report['recommendations'].append(f"Investigate {component_name} health issues")
        
        return report
    
    def _analyze_alert_trends(self) -> Dict[str, Any]:
        """Analyze alert trends"""
        trends = {'overall': 'stable', 'by_severity': {}, 'by_type': {}}
        
        # Analyze overall alert trend
        if len(self.alert_history) >= 2:
            recent_alerts = sum(1 for alert in list(self.alert_history)[-5:] if not alert['acknowledged'])
            historical_alerts = sum(1 for alert in list(self.alert_history)[:-5] if not alert['acknowledged']) if len(self.alert_history) > 5 else 0
            
            if recent_alerts > historical_alerts * 1.5:
                trends['overall'] = 'increasing'
            elif recent_alerts < historical_alerts * 0.5:
                trends['overall'] = 'decreasing'
        
        # Analyze by severity
        severities = defaultdict(list)
        for alert in self.alert_history:
            severities[alert['severity']].append(alert)
        
        for severity, alerts in severities.items():
            if len(alerts) >= 2:
                recent = sum(1 for alert in alerts[-5:] if not alert['acknowledged'])
                historical = sum(1 for alert in alerts[:-5] if not alert['acknowledged']) if len(alerts) > 5 else 0
                
                if recent > historical * 1.5:
                    trends['by_severity'][severity] = 'increasing'
                elif recent < historical * 0.5:
                    trends['by_severity'][severity] = 'decreasing'
                else:
                    trends['by_severity'][severity] = 'stable'
        
        # Analyze by type
        alert_types = defaultdict(list)
        for alert in self.alert_history:
            alert_types[alert['alert_type']].append(alert)
        
        for alert_type, alerts in alert_types.items():
            if len(alerts) >= 2:
                recent = sum(1 for alert in alerts[-3:] if not alert['acknowledged'])
                historical = sum(1 for alert in alerts[:-3] if not alert['acknowledged']) if len(alerts) > 3 else 0
                
                if recent > historical * 2:
                    trends['by_type'][alert_type] = 'increasing'
                elif recent < historical * 0.5:
                    trends['by_type'][alert_type] = 'decreasing'
                else:
                    trends['by_type'][alert_type] = 'stable'
        
        return trends
    
    def reset_monitoring(self):
        """Reset monitoring data"""
        self.monitoring_cycles = 0
        self.alerts_triggered = 0
        self.events_logged = 0
        self.performance_metrics = defaultdict(deque)
        self.health_metrics = defaultdict(deque)
        self.system_metrics = defaultdict(deque)
        self.alert_history.clear()
        self.event_log.clear()
        
        # Reinitialize component status
        for component in self.component_status.keys():
            self.component_status[component] = {'status': 'unknown', 'last_check': None, 'metrics': {}}
        
        self.logger.info("Monitoring data reset completed")

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create knowledge monitor
    monitor = KnowledgeMonitor({
        'monitoring_interval': 10,  # Faster interval for testing
        'metrics_retention': 300
    })
    
    print("Starting knowledge engine monitoring...")
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some monitoring cycles
    print("\nSimulating monitoring cycles...")
    for i in range(3):
        print(f"  Cycle {i+1}...")
        # Simulate a monitoring cycle by calling the monitoring loop directly
        monitor._perform_monitoring_cycle()
        time.sleep(2)  # Short delay between cycles
    
    # Get system status
    status = monitor.get_system_status()
    print(f"\nSystem Status:")
    print(f"  - Overall health: {status['overall_health']}")
    print(f"  - Monitoring status: {status['monitoring_status']}")
    print(f"  - Uptime: {status['uptime']:.1f}s")
    print(f"  - Active alerts: {status['alerts']['active']}")
    print(f"  - Component status:")
    for component, data in status['component_status'].items():
        print(f"    - {component}: {data['status']}")
    
    # Get performance metrics
    metrics = monitor.get_performance_metrics('1h')
    print(f"\nPerformance Metrics ({len(metrics['metrics'])} metrics):")
    for metric_name, metric_data in list(metrics['metrics'].items())[:3]:  # Show first 3
        print(f"  - {metric_name}:")
        print(f"    Average: {metric_data['average']:.3f}")
        print(f"    Trend: {metric_data['trend']}")
    
    # Get alert history
    alerts = monitor.get_alert_history()
    print(f"\nAlert History ({len(alerts)} alerts):")
    for alert in alerts:
        print(f"  - [{alert['severity']}] {alert['alert_type']}: {alert['message']}")
    
    # Generate health report
    health_report = monitor.generate_health_report()
    print(f"\nHealth Report Summary:")
    print(f"  - Report timestamp: {health_report['report_timestamp']}")
    print(f"  - System health: {health_report['system_status']['overall_health']}")
    print(f"  - Active alerts: {health_report['alert_summary']['active_alerts']}")
    print(f"  - Recommendations: {len(health_report['recommendations'])}")
    for i, recommendation in enumerate(health_report['recommendations'], 1):
        print(f"    {i}. {recommendation}")
    
    # Get monitoring statistics
    stats = monitor.get_monitoring_stats()
    print(f"\nMonitoring Statistics:")
    print(f"  - Monitoring cycles: {stats['monitoring_cycles']}")
    print(f"  - Alerts triggered: {stats['alerts_triggered']}")
    print(f"  - Events logged: {stats['events_logged']}")
    print(f"  - Performance metrics: {stats['performance_metrics_count']}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print(f"\nMonitoring stopped successfully")
    
    print(f"\nKnowledge engine monitoring demonstration completed!")