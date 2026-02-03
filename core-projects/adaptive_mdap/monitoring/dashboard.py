"""Dashboard utilities for Adaptive MDAP monitoring."""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Dashboard
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from adaptive_mdap.utils.logger import get_logger
from adaptive_mdap.utils.metrics import get_metrics

logger = get_logger("monitoring.dashboard")


@dataclass
class DashboardPanel:
    """A single dashboard panel."""
    title: str
    panel_type: str  # metric, chart, table, stat
    data: Dict[str, Any]
    position: int = 0


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    refresh_interval_seconds: int = 30
    panels: List[DashboardPanel] = field(default_factory=list)
    theme: str = "light"


class DashboardGenerator:
    """Generate monitoring dashboards."""
    
    def __init__(self):
        self.metrics = get_metrics()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary dashboard."""
        all_metrics = self.metrics.get_all_metrics()
        
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_classifications": all_metrics.get("counters", {}).get("classification_success", 0) +
                                         all_metrics.get("counters", {}).get("classification_failure", 0),
                "successful_classifications": all_metrics.get("counters", {}).get("classification_success", 0),
                "failed_classifications": all_metrics.get("counters", {}).get("classification_failure", 0),
                "total_allocations": sum(
                    v for k, v in all_metrics.get("counters", {}).items()
                    if k.startswith("allocation_")
                ),
                "total_executions": sum(
                    v for k, v in all_metrics.get("counters", {}).items()
                    if k.startswith("execution_")
                ),
            },
            "performance": {
                "classification_latency_ms": all_metrics.get("timers", {}).get("classification_latency_ms", {}),
                "allocation_latency_ms": all_metrics.get("timers", {}).get("allocation_latency_ms", {}),
            },
        }
    
    def generate_execution_metrics(self) -> Dict[str, Any]:
        """Generate execution metrics dashboard."""
        all_metrics = self.metrics.get_all_metrics()
        timers = all_metrics.get("timers", {})
        counters = all_metrics.get("counters", {})
        
        # Group execution metrics by strategy
        strategies = ["direct", "mdap_light", "mdap_medium", "maker_full", "maker_ultra"]
        execution_data = {}
        
        for strategy in strategies:
            latency_timer = timers.get(f"execution_{strategy}_latency_ms", {})
            success_count = counters.get(f"execution_{strategy}_success", 0)
            failure_count = counters.get(f"execution_{strategy}_failure", 0)
            
            total = success_count + failure_count
            success_rate = success_count / total if total > 0 else 0
            
            execution_data[strategy] = {
                "latency_ms": latency_timer,
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": success_rate,
                "total_executions": total,
            }
        
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "strategies": execution_data,
        }
    
    def generate_cost_dashboard(self) -> Dict[str, Any]:
        """Generate cost metrics dashboard."""
        all_metrics = self.metrics.get_all_metrics()
        histograms = all_metrics.get("histograms", {})
        
        # Calculate estimated costs from execution histograms
        cost_data = {}
        for key, hist in histograms.items():
            if "cost" in key:
                strategy = key.replace("_cost", "")
                cost_data[strategy] = {
                    "mean_cost": hist.get("mean", 0),
                    "max_cost": hist.get("max", 0),
                    "p95_cost": hist.get("p95", 0),
                    "count": hist.get("count", 0),
                }
        
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "costs": cost_data,
        }
    
    def generate_allocations_dashboard(self) -> Dict[str, Any]:
        """Generate allocation metrics dashboard."""
        all_metrics = self.metrics.get_all_metrics()
        counters = all_metrics.get("counters", {})
        histograms = all_metrics.get("histograms", {})
        
        # Count allocations by strategy
        allocation_counts = {}
        for key, value in counters.items():
            if key.startswith("allocation_"):
                strategy = key.replace("allocation_", "")
                allocation_counts[strategy] = value
        
        complexity_dist = histograms.get("complexity_score", {})
        
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "allocation_counts": allocation_counts,
            "complexity_distribution": complexity_dist,
        }
    
    def generate_full_dashboard(self) -> Dict[str, Any]:
        """Generate complete dashboard with all sections."""
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": self.generate_summary()["summary"],
            "execution": self.generate_execution_metrics()["strategies"],
            "costs": self.generate_cost_dashboard()["costs"],
            "allocations": self.generate_allocations_dashboard()["allocation_counts"],
            "complexity_distribution": self.generate_allocations_dashboard()["complexity_distribution"],
        }
    
    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        return self.metrics.export_prometheus()


# Global dashboard generator
_dashboard_generator = DashboardGenerator()


def get_dashboard() -> DashboardGenerator:
    """Get the global dashboard generator."""
    return _dashboard_generator


def get_summary() -> Dict[str, Any]:
    """Get summary dashboard."""
    return get_dashboard().generate_summary()


def get_full_dashboard() -> Dict[str, Any]:
    """Get full dashboard."""
    return get_dashboard().generate_full_dashboard()


def get_prometheus_metrics() -> str:
    """Get Prometheus-format metrics."""
    return get_dashboard().export_prometheus()
