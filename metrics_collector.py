"""
Metrics Collector - Collects, aggregates, and exports OpenEvolve metrics
Tracks evolution performance, resource usage, and quality metrics
"""

import time
import json
import csv
import logging
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import statistics


@dataclass
class EvolutionMetrics:
    """Metrics from a single evolution operation"""
    operation_id: str
    timestamp: float
    evolution_mode: str
    content_type: str
    
    # Evolution metrics
    iterations_completed: int
    best_fitness: float
    final_fitness: float
    fitness_improvement: float
    
    # Population metrics
    population_size: int
    population_diversity: float
    elite_count: int
    
    # Optional evolution metrics
    convergence_iteration: Optional[int] = None
    
    # Quality Diversity metrics
    archive_size: Optional[int] = None
    archive_coverage: Optional[float] = None
    behavior_diversity: Optional[Dict[str, float]] = None
    
    # Multi-Objective metrics
    pareto_front_size: Optional[int] = None
    hypervolume: Optional[float] = None
    spread: Optional[float] = None
    
    # Adversarial metrics
    attack_success_rate: Optional[float] = None
    defense_success_rate: Optional[float] = None
    adversarial_rounds: Optional[int] = None
    
    # Resource metrics
    api_calls: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    cost_usd: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_avg_percent: float = 0.0
    
    # Performance metrics
    duration: float = 0.0
    iterations_per_second: float = 0.0
    evaluations_per_second: float = 0.0
    time_per_iteration: float = 0.0
    
    # Error metrics
    errors_count: int = 0
    retries_count: int = 0
    fallback_used: bool = False
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationMetrics:
    """Metrics for tracking an active operation"""
    operation_id: str
    start_time: float
    evolution_mode: str
    max_iterations: int
    population_size: int
    content_type: str = ""
    file_name: str = ""
    component: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    end_time: Optional[float] = None
    status: str = "running"
    current_iteration: int = 0
    best_fitness: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self):
        """Finalize the operation metrics"""
        self.end_time = time.time()
        self.status = "completed"


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple operations"""
    total_operations: int
    total_iterations: int
    total_api_calls: int
    total_tokens: int
    total_cost_usd: float
    
    avg_fitness: float
    avg_improvement: float
    avg_duration: float
    avg_iterations_per_op: float
    
    success_rate: float
    fallback_rate: float
    
    by_mode: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_content_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    time_range: Dict[str, float] = field(default_factory=dict)


class MetricsStore:
    """Stores metrics in memory and optionally persists to disk"""
    
    def __init__(self, persist_path: Optional[str] = None):
        self.metrics: Dict[str, EvolutionMetrics] = {}
        self.persist_path = persist_path
    
    def add(self, metrics: EvolutionMetrics):
        """Add metrics to store"""
        self.metrics[metrics.operation_id] = metrics
        
        if self.persist_path:
            self._persist(metrics)
    
    def get(self, operation_id: str) -> Optional[EvolutionMetrics]:
        """Get metrics by operation ID"""
        return self.metrics.get(operation_id)
    
    def get_all(self) -> List[EvolutionMetrics]:
        """Get all metrics"""
        return list(self.metrics.values())
    
    def get_by_mode(self, evolution_mode: str) -> List[EvolutionMetrics]:
        """Get metrics by evolution mode"""
        return [m for m in self.metrics.values() if m.evolution_mode == evolution_mode]
    
    def get_by_content_type(self, content_type: str) -> List[EvolutionMetrics]:
        """Get metrics by content type"""
        return [m for m in self.metrics.values() if m.content_type == content_type]
    
    def get_recent(self, count: int = 10) -> List[EvolutionMetrics]:
        """Get most recent metrics"""
        sorted_metrics = sorted(self.metrics.values(), key=lambda m: m.timestamp, reverse=True)
        return sorted_metrics[:count]
    
    def clear(self):
        """Clear all metrics"""
        self.metrics.clear()
    
    def _persist(self, metrics: EvolutionMetrics):
        """Persist metrics to disk"""
        if not self.persist_path:
            return
        
        try:
            with open(self.persist_path, 'a') as f:
                f.write(json.dumps(asdict(metrics)) + '\n')
        except (OSError, IOError, TypeError) as e:
            print(f"Failed to persist metrics: {e}")


class MetricsAggregator:
    """Aggregates metrics across operations"""
    
    def __init__(self, store=None):
        """Initialize aggregator with optional store"""
        self.store = store
    
    def aggregate(self, metrics_list: List[EvolutionMetrics]) -> AggregatedMetrics:
        """Aggregate metrics"""
        if not metrics_list:
            return AggregatedMetrics(
                total_operations=0,
                total_iterations=0,
                total_api_calls=0,
                total_tokens=0,
                total_cost_usd=0.0,
                avg_fitness=0.0,
                avg_improvement=0.0,
                avg_duration=0.0,
                avg_iterations_per_op=0.0,
                success_rate=0.0,
                fallback_rate=0.0
            )
        
        # Calculate totals
        total_operations = len(metrics_list)
        total_iterations = sum(m.iterations_completed for m in metrics_list)
        total_api_calls = sum(m.api_calls for m in metrics_list)
        total_tokens = sum(m.tokens_total for m in metrics_list)
        total_cost_usd = sum(m.cost_usd for m in metrics_list)
        
        # Calculate averages
        avg_fitness = statistics.mean(m.best_fitness for m in metrics_list)
        avg_improvement = statistics.mean(m.fitness_improvement for m in metrics_list)
        avg_duration = statistics.mean(m.duration for m in metrics_list)
        avg_iterations_per_op = total_iterations / total_operations
        
        # Calculate rates
        success_count = sum(1 for m in metrics_list if m.errors_count == 0)
        success_rate = success_count / total_operations
        
        fallback_count = sum(1 for m in metrics_list if m.fallback_used)
        fallback_rate = fallback_count / total_operations
        
        # Aggregate by mode
        by_mode = self._aggregate_by_field(metrics_list, 'evolution_mode')
        
        # Aggregate by content type
        by_content_type = self._aggregate_by_field(metrics_list, 'content_type')
        
        # Time range
        timestamps = [m.timestamp for m in metrics_list]
        time_range = {
            'start': min(timestamps),
            'end': max(timestamps),
            'span': max(timestamps) - min(timestamps)
        }
        
        return AggregatedMetrics(
            total_operations=total_operations,
            total_iterations=total_iterations,
            total_api_calls=total_api_calls,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            avg_fitness=avg_fitness,
            avg_improvement=avg_improvement,
            avg_duration=avg_duration,
            avg_iterations_per_op=avg_iterations_per_op,
            success_rate=success_rate,
            fallback_rate=fallback_rate,
            by_mode=by_mode,
            by_content_type=by_content_type,
            time_range=time_range
        )
    
    def _aggregate_by_field(self, metrics_list: List[EvolutionMetrics], field: str) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics by a specific field"""
        grouped = defaultdict(list)
        for m in metrics_list:
            key = getattr(m, field)
            grouped[key].append(m)
        
        result = {}
        for key, group in grouped.items():
            result[key] = {
                'count': len(group),
                'avg_fitness': statistics.mean(m.best_fitness for m in group),
                'avg_duration': statistics.mean(m.duration for m in group),
                'total_cost': sum(m.cost_usd for m in group)
            }
        
        return result


class MetricsExporter:
    """Exports metrics in various formats"""
    
    def __init__(self, store=None):
        """Initialize exporter with optional store"""
        self.store = store
    
    def export_json(self, metrics_list: List[EvolutionMetrics]) -> str:
        """Export metrics as JSON"""
        data = [asdict(m) for m in metrics_list]
        return json.dumps(data, indent=2)
    
    def export_csv(self, metrics_list: List[EvolutionMetrics]) -> str:
        """Export metrics as CSV"""
        if not metrics_list:
            return ""
        
        # Get all field names from first metrics object
        fieldnames = list(asdict(metrics_list[0]).keys())
        
        # Create CSV string
        import io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for m in metrics_list:
            row = asdict(m)
            # Convert complex types to strings
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value)
            writer.writerow(row)
        
        return output.getvalue()
    
    def export_summary(self, aggregated: AggregatedMetrics) -> str:
        """Export aggregated metrics as summary text"""
        lines = [
            "OpenEvolve Metrics Summary",
            "=" * 50,
            f"Total Operations: {aggregated.total_operations}",
            f"Total Iterations: {aggregated.total_iterations}",
            f"Total API Calls: {aggregated.total_api_calls}",
            f"Total Tokens: {aggregated.total_tokens:,}",
            f"Total Cost: ${aggregated.total_cost_usd:.2f}",
            "",
            "Averages:",
            f"  Fitness: {aggregated.avg_fitness:.4f}",
            f"  Improvement: {aggregated.avg_improvement:.4f}",
            f"  Duration: {aggregated.avg_duration:.2f}s",
            f"  Iterations/Op: {aggregated.avg_iterations_per_op:.1f}",
            "",
            f"Success Rate: {aggregated.success_rate * 100:.1f}%",
            f"Fallback Rate: {aggregated.fallback_rate * 100:.1f}%",
            "",
            "By Evolution Mode:",
        ]
        
        for mode, stats in aggregated.by_mode.items():
            lines.append(f"  {mode}: {stats['count']} ops, avg fitness {stats['avg_fitness']:.4f}")
        
        return "\n".join(lines)


class MetricsVisualizer:
    """Creates visualizations from metrics"""
    
    def create_fitness_chart_data(self, metrics_list: List[EvolutionMetrics]) -> Dict[str, Any]:
        """Create data for fitness evolution chart"""
        sorted_metrics = sorted(metrics_list, key=lambda m: m.timestamp)
        
        return {
            'labels': [datetime.fromtimestamp(m.timestamp).strftime('%H:%M:%S') for m in sorted_metrics],
            'datasets': [{
                'label': 'Best Fitness',
                'data': [m.best_fitness for m in sorted_metrics]
            }, {
                'label': 'Final Fitness',
                'data': [m.final_fitness for m in sorted_metrics]
            }]
        }
    
    def create_resource_chart_data(self, metrics_list: List[EvolutionMetrics]) -> Dict[str, Any]:
        """Create data for resource usage chart"""
        sorted_metrics = sorted(metrics_list, key=lambda m: m.timestamp)
        
        return {
            'labels': [datetime.fromtimestamp(m.timestamp).strftime('%H:%M:%S') for m in sorted_metrics],
            'datasets': [{
                'label': 'API Calls',
                'data': [m.api_calls for m in sorted_metrics]
            }, {
                'label': 'Tokens (thousands)',
                'data': [m.tokens_total / 1000 for m in sorted_metrics]
            }, {
                'label': 'Cost ($)',
                'data': [m.cost_usd for m in sorted_metrics]
            }]
        }
    
    def create_mode_distribution(self, metrics_list: List[EvolutionMetrics]) -> Dict[str, int]:
        """Create distribution of evolution modes"""
        distribution = defaultdict(int)
        for m in metrics_list:
            distribution[m.evolution_mode] += 1
        return dict(distribution)


class MetricsCollector:
    """Main metrics collection class"""
    
    def __init__(self, persist_path: Optional[str] = None):
        self.store = MetricsStore(persist_path)
        self.aggregator = MetricsAggregator()
        self.exporter = MetricsExporter()
        self.visualizer = MetricsVisualizer()
    
    def collect(self, operation_id: str, metrics_data: Dict[str, Any]):
        """Collect metrics from an operation"""
        # Create EvolutionMetrics object
        metrics = EvolutionMetrics(
            operation_id=operation_id,
            timestamp=metrics_data.get('timestamp', time.time()),
            evolution_mode=metrics_data.get('evolution_mode', 'standard'),
            content_type=metrics_data.get('content_type', 'general'),
            iterations_completed=metrics_data.get('iterations_completed', 0),
            best_fitness=metrics_data.get('best_fitness', 0.0),
            final_fitness=metrics_data.get('final_fitness', 0.0),
            fitness_improvement=metrics_data.get('fitness_improvement', 0.0),
            convergence_iteration=metrics_data.get('convergence_iteration'),
            population_size=metrics_data.get('population_size', 0),
            population_diversity=metrics_data.get('population_diversity', 0.0),
            elite_count=metrics_data.get('elite_count', 0),
            archive_size=metrics_data.get('archive_size'),
            archive_coverage=metrics_data.get('archive_coverage'),
            behavior_diversity=metrics_data.get('behavior_diversity'),
            pareto_front_size=metrics_data.get('pareto_front_size'),
            hypervolume=metrics_data.get('hypervolume'),
            spread=metrics_data.get('spread'),
            attack_success_rate=metrics_data.get('attack_success_rate'),
            defense_success_rate=metrics_data.get('defense_success_rate'),
            adversarial_rounds=metrics_data.get('adversarial_rounds'),
            api_calls=metrics_data.get('api_calls', 0),
            tokens_prompt=metrics_data.get('tokens_prompt', 0),
            tokens_completion=metrics_data.get('tokens_completion', 0),
            tokens_total=metrics_data.get('tokens_total', 0),
            cost_usd=metrics_data.get('cost_usd', 0.0),
            memory_peak_mb=metrics_data.get('memory_peak_mb', 0.0),
            cpu_avg_percent=metrics_data.get('cpu_avg_percent', 0.0),
            duration=metrics_data.get('duration', 0.0),
            iterations_per_second=metrics_data.get('iterations_per_second', 0.0),
            evaluations_per_second=metrics_data.get('evaluations_per_second', 0.0),
            time_per_iteration=metrics_data.get('time_per_iteration', 0.0),
            errors_count=metrics_data.get('errors_count', 0),
            retries_count=metrics_data.get('retries_count', 0),
            fallback_used=metrics_data.get('fallback_used', False),
            metadata=metrics_data.get('metadata', {})
        )
        
        self.store.add(metrics)
    
    def get_operation_metrics(self, operation_id: str) -> Optional[EvolutionMetrics]:
        """Get metrics for a specific operation"""
        return self.store.get(operation_id)
    
    def get_all_metrics(self) -> List[EvolutionMetrics]:
        """Get all collected metrics"""
        return self.store.get_all()
    
    def aggregate(self, filter_dict: Optional[Dict[str, Any]] = None) -> AggregatedMetrics:
        """Aggregate metrics with optional filtering"""
        metrics_list = self.store.get_all()
        
        # Apply filters
        if filter_dict:
            if 'evolution_mode' in filter_dict:
                metrics_list = [m for m in metrics_list if m.evolution_mode == filter_dict['evolution_mode']]
            if 'content_type' in filter_dict:
                metrics_list = [m for m in metrics_list if m.content_type == filter_dict['content_type']]
            if 'min_timestamp' in filter_dict:
                metrics_list = [m for m in metrics_list if m.timestamp >= filter_dict['min_timestamp']]
            if 'max_timestamp' in filter_dict:
                metrics_list = [m for m in metrics_list if m.timestamp <= filter_dict['max_timestamp']]
        
        return self.aggregator.aggregate(metrics_list)
    
    def export(self, format: str = 'json', filter_dict: Optional[Dict[str, Any]] = None) -> str:
        """Export metrics in specified format"""
        metrics_list = self.store.get_all()
        
        # Apply filters
        if filter_dict:
            if 'evolution_mode' in filter_dict:
                metrics_list = [m for m in metrics_list if m.evolution_mode == filter_dict['evolution_mode']]
            if 'content_type' in filter_dict:
                metrics_list = [m for m in metrics_list if m.content_type == filter_dict['content_type']]
        
        if format == 'json':
            return self.exporter.export_json(metrics_list)
        elif format == 'csv':
            return self.exporter.export_csv(metrics_list)
        elif format == 'summary':
            aggregated = self.aggregator.aggregate(metrics_list)
            return self.exporter.export_summary(aggregated)
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def get_chart_data(self, chart_type: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get data for visualization charts"""
        metrics_list = self.store.get_all()
        
        # Apply filters
        if filter_dict:
            if 'evolution_mode' in filter_dict:
                metrics_list = [m for m in metrics_list if m.evolution_mode == filter_dict['evolution_mode']]
        
        if chart_type == 'fitness':
            return self.visualizer.create_fitness_chart_data(metrics_list)
        elif chart_type == 'resource':
            return self.visualizer.create_resource_chart_data(metrics_list)
        elif chart_type == 'mode_distribution':
            return self.visualizer.create_mode_distribution(metrics_list)
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
    
    def clear(self):
        """Clear all collected metrics"""
        self.store.clear()


class MetricsCollector:
    """Main metrics collection class"""
    
    def __init__(self, db_path: str = "./openevolve_metrics.db"):
        self.store = MetricsStore(db_path)
        self.aggregator = MetricsAggregator(self.store)
        self.exporter = MetricsExporter(self.store)
        self.visualizer = MetricsVisualizer()
        self.logger = logging.getLogger(__name__)
        
        # Active operations tracking
        self._active_operations: Dict[str, OperationMetrics] = {}
        self._lock = threading.Lock()
    
    def start_operation(
        self,
        operation_id: str,
        evolution_mode: str = "standard",
        max_iterations: int = 10,
        population_size: int = 20,
        content_type: str = "general",
        file_name: str = "",
        component: str = "",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> OperationMetrics:
        """Start tracking a new operation"""
        with self._lock:
            metrics = OperationMetrics(
                operation_id=operation_id,
                start_time=time.time(),
                evolution_mode=evolution_mode,
                max_iterations=max_iterations,
                population_size=population_size,
                content_type=content_type,
                file_name=file_name,
                component=component,
                user_id=user_id,
                session_id=session_id
            )
            self._active_operations[operation_id] = metrics
            self.logger.info(f"Started tracking operation {operation_id}")
            return metrics
    
    def update_operation(
        self,
        operation_id: str,
        **kwargs
    ):
        """Update metrics for an active operation"""
        with self._lock:
            if operation_id not in self._active_operations:
                self.logger.warning(f"Operation {operation_id} not found")
                return
            
            metrics = self._active_operations[operation_id]
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
    
    def end_operation(self, operation_id: str) -> Optional[OperationMetrics]:
        """End tracking an operation and store metrics"""
        with self._lock:
            if operation_id not in self._active_operations:
                self.logger.warning(f"Operation {operation_id} not found")
                return None
            
            metrics = self._active_operations.pop(operation_id)
            metrics.finalize()
            
            # Convert to EvolutionMetrics and store
            evolution_metrics = EvolutionMetrics(
                operation_id=metrics.operation_id,
                timestamp=metrics.start_time,
                evolution_mode=metrics.evolution_mode,
                content_type=metrics.content_type,
                iterations_completed=metrics.current_iteration,
                best_fitness=metrics.best_fitness,
                final_fitness=metrics.best_fitness,
                fitness_improvement=metrics.best_fitness,
                population_size=metrics.population_size,
                population_diversity=0.5,  # Default value
                elite_count=int(metrics.population_size * 0.1),  # Default 10%
                duration=(metrics.end_time or time.time()) - metrics.start_time
            )
            self.store.add(evolution_metrics)
            
            self.logger.info(f"Completed tracking operation {operation_id}")
            return metrics
    
    def get_operation_metrics(self, operation_id: str) -> Optional[OperationMetrics]:
        """Get metrics for a specific operation"""
        # Check active operations first
        with self._lock:
            if operation_id in self._active_operations:
                return self._active_operations[operation_id]
        
        # Check stored metrics
        return self.store.get_metrics(operation_id)
    
    def get_active_operations(self) -> List[OperationMetrics]:
        """Get all active operations"""
        with self._lock:
            return list(self._active_operations.values())
    
    def aggregate_metrics(
        self,
        evolution_mode: Optional[str] = None,
        component: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> AggregatedMetrics:
        """Aggregate metrics with filters"""
        # Get all metrics from store
        all_metrics = self.store.get_all()
        
        # Apply filters
        filtered_metrics = all_metrics
        if evolution_mode:
            filtered_metrics = [m for m in filtered_metrics if m.evolution_mode == evolution_mode]
        if component:
            filtered_metrics = [m for m in filtered_metrics if m.metadata.get('component') == component]
        if start_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]
        if end_time:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]
        
        return self.aggregator.aggregate(filtered_metrics)
    
    def export_json(
        self,
        filepath: str,
        evolution_mode: Optional[str] = None,
        component: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """Export metrics to JSON"""
        # Get filtered metrics
        all_metrics = self.store.get_all()
        filtered_metrics = all_metrics
        if evolution_mode:
            filtered_metrics = [m for m in filtered_metrics if m.evolution_mode == evolution_mode]
        
        # Export to JSON string
        json_data = self.exporter.export_json(filtered_metrics)
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write(json_data)
    
    def export_csv(
        self,
        filepath: str,
        evolution_mode: Optional[str] = None,
        component: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """Export metrics to CSV"""
        # Get filtered metrics
        all_metrics = self.store.get_all()
        filtered_metrics = all_metrics
        if evolution_mode:
            filtered_metrics = [m for m in filtered_metrics if m.evolution_mode == evolution_mode]
        
        # Export to CSV string
        csv_data = self.exporter.export_csv(filtered_metrics)
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write(csv_data)
    
    def export_excel(self, filepath: str, aggregated: AggregatedMetrics):
        """Export aggregated metrics to Excel"""
        self.exporter.export_excel(filepath, aggregated)
    
    def create_chart(
        self,
        chart_type: str,
        evolution_mode: Optional[str] = None,
        component: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ):
        """Create visualization chart"""
        metrics_list = self.store.get_metrics_by_filter(
            evolution_mode=evolution_mode,
            component=component,
            start_time=start_time,
            end_time=end_time
        )
        
        return self.visualizer.create_chart(chart_type, metrics_list)
    
    def get_summary_stats(
        self,
        evolution_mode: Optional[str] = None,
        component: Optional[str] = None,
        time_range: str = "all"
    ) -> Dict[str, Any]:
        """Get summary statistics"""
        # Calculate time range
        end_time = time.time()
        start_time = None
        
        if time_range == "hour":
            start_time = end_time - 3600
        elif time_range == "day":
            start_time = end_time - 86400
        elif time_range == "week":
            start_time = end_time - 604800
        elif time_range == "month":
            start_time = end_time - 2592000
        
        # Get aggregated metrics
        aggregated = self.aggregate_metrics(
            evolution_mode=evolution_mode,
            component=component,
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "total_operations": aggregated.total_operations,
            "total_duration": aggregated.total_duration,
            "avg_duration": aggregated.avg_duration,
            "avg_fitness_improvement": aggregated.avg_fitness_improvement,
            "success_rate": aggregated.success_rate,
            "total_cost": aggregated.total_cost,
            "total_tokens": aggregated.total_tokens,
            "avg_memory_usage": aggregated.avg_memory_usage,
            "avg_cpu_usage": aggregated.avg_cpu_usage,
            "convergence_rate": aggregated.convergence_rate,
            "total_errors": aggregated.total_errors,
            "fallback_rate": aggregated.fallback_rate
        }
    
    def clear_old_metrics(self, days: int = 30):
        """Clear metrics older than specified days"""
        cutoff_time = time.time() - (days * 86400)
        
        with sqlite3.connect(self.store.db_path) as conn:
            result = conn.execute(
                "DELETE FROM operation_metrics WHERE start_time < ?",
                (cutoff_time,)
            )
            deleted_count = result.rowcount
        
        self.logger.info(f"Deleted {deleted_count} metrics older than {days} days")
        return deleted_count


# Convenience functions for quick access
_default_collector: Optional[MetricsCollector] = None


def get_default_collector() -> MetricsCollector:
    """Get or create default metrics collector"""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector


def start_operation(operation_id: str, **kwargs) -> OperationMetrics:
    """Start tracking operation using default collector"""
    return get_default_collector().start_operation(operation_id, **kwargs)


def update_operation(operation_id: str, **kwargs):
    """Update operation using default collector"""
    get_default_collector().update_operation(operation_id, **kwargs)


def end_operation(operation_id: str) -> Optional[OperationMetrics]:
    """End operation using default collector"""
    return get_default_collector().end_operation(operation_id)


def get_summary_stats(**kwargs) -> Dict[str, Any]:
    """Get summary stats using default collector"""
    return get_default_collector().get_summary_stats(**kwargs)
