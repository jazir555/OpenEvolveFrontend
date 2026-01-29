"""
Analytics System for Unified MCP Gateway.

This module tracks and analyzes tool usage, performance metrics,
and provides insights for gateway optimization.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field

from .models import ToolCallResult

logger = logging.getLogger(__name__)


@dataclass
class ToolMetrics:
    """Metrics for a single tool."""
    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    last_called: Optional[datetime] = None
    first_called: Optional[datetime] = None
    server_used: Dict[str, int] = field(default_factory=dict)  # server -> count
    error_types: Dict[str, int] = field(default_factory=dict)  # error -> count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0,
            "total_execution_time": self.total_execution_time,
            "avg_execution_time": self.avg_execution_time,
            "last_called": self.last_called.isoformat() if self.last_called else None,
            "first_called": self.first_called.isoformat() if self.first_called else None,
            "servers_used": self.server_used.copy(),
            "error_types": self.error_types.copy(),
        }


@dataclass
class ServerMetrics:
    """Metrics for a single server."""
    server_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    circuit_breaker_trips: int = 0
    last_failure: Optional[datetime] = None
    uptime_percentage: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server_name": self.server_name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0,
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "uptime_percentage": self.uptime_percentage,
        }


class MCPGatewayAnalytics:
    """
    Track and analyze tool usage.

    Features:
    - Tool call tracking
    - Performance metrics
    - Error analysis
    - Usage reports
    - Popular tools identification
    """

    def __init__(self, retention_days: int = 30):
        """
        Initialize analytics system.

        Args:
            retention_days: Number of days to retain analytics data
        """
        self.retention_days = retention_days
        self.retention_cutoff = timedelta(days=retention_days)

        # Tool metrics
        self.tool_metrics: Dict[str, ToolMetrics] = {}

        # Server metrics
        self.server_metrics: Dict[str, ServerMetrics] = {}

        # Time-series data (for trends)
        self.calls_over_time: Dict[datetime, int] = defaultdict(int)
        self.failures_over_time: Dict[datetime, int] = defaultdict(int)

        logger.info(f"MCPGatewayAnalytics initialized (retention: {retention_days} days)")

    async def track_tool_call(self, result: ToolCallResult):
        """
        Track a tool call result.

        Args:
            result: ToolCallResult to track
        """
        key = f"{result.namespace}/{result.tool_name}"

        # Get or create tool metrics
        if key not in self.tool_metrics:
            self.tool_metrics[key] = ToolMetrics(tool_name=key)

        metrics = self.tool_metrics[key]

        # Update metrics
        metrics.total_calls += 1
        metrics.total_execution_time += result.execution_time
        metrics.avg_execution_time = metrics.total_execution_time / metrics.total_calls

        if result.success:
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1
            # Track error type
            if result.error:
                error_type = result.error.split(":")[0]  # Get first part of error
                metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1

        # Update timestamps
        if metrics.first_called is None:
            metrics.first_called = result.timestamp
        metrics.last_called = result.timestamp

        # Track server usage
        if result.server_name:
            metrics.server_used[result.server_name] = metrics.server_used.get(result.server_name, 0) + 1

        # Track server metrics
        if result.server_name:
            if result.server_name not in self.server_metrics:
                self.server_metrics[result.server_name] = ServerMetrics(server_name=result.server_name)

            server_metrics = self.server_metrics[result.server_name]
            server_metrics.total_calls += 1

            if result.success:
                server_metrics.successful_calls += 1
            else:
                server_metrics.failed_calls += 1
                server_metrics.last_failure = result.timestamp

        # Track time-series (bucket by hour)
        time_bucket = result.timestamp.replace(minute=0, second=0, microsecond=0)
        self.calls_over_time[time_bucket] += 1

        if not result.success:
            self.failures_over_time[time_bucket] += 1

        logger.debug(f"Tracked tool call: {key} (success={result.success})")

    def get_popular_tools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most frequently used tools.

        Args:
            limit: Maximum number of tools to return

        Returns:
            List of tools with usage stats
        """
        # Sort by total calls
        sorted_tools = sorted(
            self.tool_metrics.values(),
            key=lambda m: m.total_calls,
            reverse=True,
        )

        return [tool.to_dict() for tool in sorted_tools[:limit]]

    def get_slowest_tools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get tools with highest average execution time.

        Args:
            limit: Maximum number of tools to return

        Returns:
            List of tools with execution times
        """
        # Filter tools with at least 5 calls
        qualified_tools = [
            tool for tool in self.tool_metrics.values()
            if tool.total_calls >= 5
        ]

        # Sort by average execution time
        sorted_tools = sorted(
            qualified_tools,
            key=lambda m: m.avg_execution_time,
            reverse=True,
        )

        return [tool.to_dict() for tool in sorted_tools[:limit]]

    def get_least_reliable_tools(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get tools with lowest success rate.

        Args:
            limit: Maximum number of tools to return

        Returns:
            List of tools with success rates
        """
        # Filter tools with at least 10 calls
        qualified_tools = [
            tool for tool in self.tool_metrics.values()
            if tool.total_calls >= 10
        ]

        # Sort by success rate (ascending)
        sorted_tools = sorted(
            qualified_tools,
            key=lambda m: m.successful_calls / m.total_calls if m.total_calls > 0 else 0.0,
        )

        return [tool.to_dict() for tool in sorted_tools[:limit]]

    def get_tool_success_rate(self, tool_name: str, namespace: Optional[str] = None) -> float:
        """
        Get success rate for a specific tool.

        Args:
            tool_name: Name of the tool
            namespace: Optional namespace

        Returns:
            Success rate (0.0 to 1.0)
        """
        key = f"{namespace}/{tool_name}" if namespace else tool_name

        if key not in self.tool_metrics:
            return 0.0

        metrics = self.tool_metrics[key]
        if metrics.total_calls == 0:
            return 0.0

        return metrics.successful_calls / metrics.total_calls

    def get_server_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all servers.

        Returns:
            Dict mapping server names to stats
        """
        return {
            name: metrics.to_dict()
            for name, metrics in self.server_metrics.items()
        }

    def get_usage_trends(
        self,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get usage trends over time.

        Args:
            hours: Number of hours to look back

        Returns:
            Usage trend data
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Filter data points within time range
        calls = {
            time_bucket: count
            for time_bucket, count in self.calls_over_time.items()
            if time_bucket >= cutoff
        }

        failures = {
            time_bucket: count
            for time_bucket, count in self.failures_over_time.items()
            if time_bucket >= cutoff
        }

        # Calculate stats
        total_calls = sum(calls.values())
        total_failures = sum(failures.values())

        return {
            "hours_analyzed": hours,
            "total_calls": total_calls,
            "total_failures": total_failures,
            "overall_success_rate": (total_calls - total_failures) / total_calls if total_calls > 0 else 0.0,
            "calls_over_time": sorted(calls.items()),
            "failures_over_time": sorted(failures.items()),
            "peak_hour": max(calls.items(), key=lambda x: x[1])[0] if calls else None,
        }

    def generate_usage_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive usage analytics report.

        Returns:
            Complete analytics report
        """
        # Calculate overall stats
        total_calls = sum(m.total_calls for m in self.tool_metrics.values())
        total_successful = sum(m.successful_calls for m in self.tool_metrics.values())
        total_failed = sum(m.failed_calls for m in self.tool_metrics.values())
        total_execution_time = sum(m.total_execution_time for m in self.tool_metrics.values())

        return {
            "summary": {
                "total_tools_tracked": len(self.tool_metrics),
                "total_calls": total_calls,
                "total_successful": total_successful,
                "total_failed": total_failed,
                "overall_success_rate": total_successful / total_calls if total_calls > 0 else 0.0,
                "total_execution_time": total_execution_time,
                "avg_execution_time": total_execution_time / total_calls if total_calls > 0 else 0.0,
                "servers_tracked": len(self.server_metrics),
            },
            "popular_tools": self.get_popular_tools(limit=10),
            "slowest_tools": self.get_slowest_tools(limit=5),
            "least_reliable_tools": self.get_least_reliable_tools(limit=5),
            "server_stats": self.get_server_stats(),
            "usage_trends": self.get_usage_trends(hours=24),
            "generated_at": datetime.utcnow().isoformat(),
        }

    def cleanup_old_data(self):
        """
        Remove analytics data older than retention period.

        Should be called periodically (e.g., daily).
        """
        cutoff = datetime.utcnow() - self.retention_cutoff

        # Clean up time-series data
        old_calls = [
            time_bucket
            for time_bucket in self.calls_over_time
            if time_bucket < cutoff
        ]
        for time_bucket in old_calls:
            del self.calls_over_time[time_bucket]

        old_failures = [
            time_bucket
            for time_bucket in self.failures_over_time
            if time_bucket < cutoff
        ]
        for time_bucket in old_failures:
            del self.failures_over_time[time_bucket]

        logger.info(f"Cleaned up {len(old_calls)} old call records and {len(old_failures)} old failure records")

    def reset_metrics(self, tool_name: Optional[str] = None):
        """
        Reset metrics for a tool or all tools.

        Args:
            tool_name: Optional tool name to reset (resets all if None)
        """
        if tool_name:
            if tool_name in self.tool_metrics:
                del self.tool_metrics[tool_name]
                logger.info(f"Reset metrics for tool: {tool_name}")
        else:
            self.tool_metrics.clear()
            self.calls_over_time.clear()
            self.failures_over_time.clear()
            logger.info("Reset all metrics")
