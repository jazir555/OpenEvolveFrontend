"""
Analytics Z3 Connector

Feeds Z3 solving metrics into the analytics system for performance tracking
and pattern analysis over time.

Integrates with:
- analytics_dashboard.py
- analytics_manager.py
- z3_performance_monitor.py

Author: OpenEvolve
Created: 2026-02-02
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from z3prover_integration import Z3SolverEngine, Z3Config
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

try:
    from z3_performance_monitor import get_z3_performance_monitor
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False


@dataclass
class Z3AnalyticsEvent:
    """A Z3 solving event for analytics."""
    event_id: str
    timestamp: datetime
    operation_type: str  # "solve", "optimize", "prove", "verify"
    problem_category: str
    execution_time_ms: float
    result_status: str
    constraint_count: int
    variable_count: int
    memory_usage_mb: float
    solver_version: str = "unknown"


@dataclass
class Z3MetricsAggregation:
    """Aggregated Z3 metrics."""
    period_start: datetime
    period_end: datetime
    total_operations: int
    avg_execution_time_ms: float
    success_rate: float
    sat_rate: float
    unsat_rate: float
    timeout_rate: float
    error_rate: float
    top_problem_categories: List[Dict[str, Any]]
    performance_trends: Dict[str, float]


class AnalyticsZ3Connector:
    """
    Connects Z3 solving metrics to the analytics system.
    
    Tracks:
    - Solver performance over time
    - Problem type distribution
    - Success/failure patterns
    - Resource utilization
    - Constraint solving trends
    """
    
    def __init__(self):
        self.event_buffer: List[Z3AnalyticsEvent] = []
        self.aggregations: Dict[str, Z3MetricsAggregation] = {}
        self.daily_stats = defaultdict(lambda: {
            "count": 0,
            "total_time_ms": 0.0,
            "successes": 0,
            "failures": 0,
            "sat_count": 0,
            "unsat_count": 0
        })
    
    def record_solving_event(
        self,
        operation_type: str,
        result_status: str,
        execution_time_ms: float,
        constraint_count: int = 0,
        variable_count: int = 0,
        problem_category: str = "unknown",
        memory_usage_mb: float = 0.0
    ) -> None:
        """Record a Z3 solving event."""
        event = Z3AnalyticsEvent(
            event_id=f"z3_{int(time.time() * 1000)}",
            timestamp=datetime.utcnow(),
            operation_type=operation_type,
            problem_category=problem_category,
            execution_time_ms=execution_time_ms,
            result_status=result_status,
            constraint_count=constraint_count,
            variable_count=variable_count,
            memory_usage_mb=memory_usage_mb
        )
        
        self.event_buffer.append(event)
        self._update_daily_stats(event)
    
    def _update_daily_stats(self, event: Z3AnalyticsEvent) -> None:
        """Update daily statistics."""
        date_key = event.timestamp.strftime("%Y-%m-%d")
        stats = self.daily_stats[date_key]
        
        stats["count"] += 1
        stats["total_time_ms"] += event.execution_time_ms
        
        if event.result_status in ["sat", "proven", "verified"]:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        
        if event.result_status == "sat":
            stats["sat_count"] += 1
        elif event.result_status == "unsat":
            stats["unsat_count"] += 1
    
    def get_daily_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get daily Z3 solving report."""
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")
        
        stats = self.daily_stats.get(date, {
            "count": 0,
            "total_time_ms": 0.0,
            "successes": 0,
            "failures": 0,
            "sat_count": 0,
            "unsat_count": 0
        })
        
        count = stats["count"]
        if count == 0:
            return {
                "date": date,
                "operations": 0,
                "avg_execution_time_ms": 0.0,
                "success_rate": 0.0
            }
        
        return {
            "date": date,
            "operations": count,
            "avg_execution_time_ms": stats["total_time_ms"] / count,
            "success_rate": stats["successes"] / count,
            "sat_rate": stats["sat_count"] / count,
            "unsat_rate": stats["unsat_count"] / count
        }
    
    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get Z3 performance trends over time."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        daily_reports = []
        current = start_date
        
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            report = self.get_daily_report(date_str)
            daily_reports.append(report)
            current += timedelta(days=1)
        
        # Calculate trends
        if len(daily_reports) >= 2:
            first_avg = daily_reports[0].get("avg_execution_time_ms", 0)
            last_avg = daily_reports[-1].get("avg_execution_time_ms", 0)
            
            first_success = daily_reports[0].get("success_rate", 0)
            last_success = daily_reports[-1].get("success_rate", 0)
            
            return {
                "period_days": days,
                "daily_reports": daily_reports,
                "execution_time_trend": last_avg - first_avg,
                "success_rate_trend": last_success - first_success,
                "trend_direction": "improving" if last_avg < first_avg else "degrading"
            }
        
        return {
            "period_days": days,
            "daily_reports": daily_reports,
            "trend_direction": "insufficient_data"
        }
    
    def export_to_analytics_dashboard(self) -> Dict[str, Any]:
        """Export metrics for analytics dashboard."""
        today_report = self.get_daily_report()
        trends = self.get_performance_trends(days=7)
        
        return {
            "z3_metrics": {
                "today": today_report,
                "trends": trends,
                "summary": {
                    "total_operations_today": today_report["operations"],
                    "avg_response_time_ms": today_report["avg_execution_time_ms"],
                    "success_rate": today_report["success_rate"],
                    "trend": trends.get("trend_direction", "unknown")
                }
            }
        }


def get_analytics_z3_connector():
    """Get global analytics Z3 connector."""
    return AnalyticsZ3Connector()


if __name__ == "__main__":
    print("Analytics Z3 Connector initialized")
