"""
Data models for the Unified MCP Gateway.

This module defines the core data structures used throughout the gateway system,
including tool definitions, server configurations, and routing results.
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Models
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum


class ServerStatus(Enum):
    """Status of an MCP server connection."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    ERROR = "error"


class ToolCategory(Enum):
    """Categories for organizing tools."""
    KNOWLEDGE = "knowledge"
    EVOLUTION = "evolution"
    ORCHESTRATION = "orchestration"
    LEARNING = "learning"
    ANALYSIS = "analysis"
    DECOMPOSITION = "decomposition"
    VERIFICATION = "verification"
    UTILITIES = "utilities"


@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""
    name: str
    description: str
    namespace: str
    server_name: str
    parameters: Dict[str, Any]  # JSON schema for parameters
    category: ToolCategory = ToolCategory.UTILITIES
    version: str = "1.0.0"
    deprecated: bool = False
    deprecation_replacement: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "namespace": self.namespace,
            "server_name": self.server_name,
            "parameters": self.parameters,
            "category": self.category.value,
            "version": self.version,
            "deprecated": self.deprecated,
            "deprecation_replacement": self.deprecation_replacement,
            "tags": self.tags,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolDefinition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            namespace=data["namespace"],
            server_name=data["server_name"],
            parameters=data["parameters"],
            category=ToolCategory(data.get("category", "utilities")),
            version=data.get("version", "1.0.0"),
            deprecated=data.get("deprecated", False),
            deprecation_replacement=data.get("deprecation_replacement"),
            tags=data.get("tags", []),
            examples=data.get("examples", []),
        )


@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    name: str
    url: str
    timeout: int
    namespace: str
    description: str
    enabled: bool = True
    health_check_interval: int = 60
    status: ServerStatus = ServerStatus.OFFLINE
    last_health_check: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "url": self.url,
            "timeout": self.timeout,
            "namespace": self.namespace,
            "description": self.description,
            "enabled": self.enabled,
            "health_check_interval": self.health_check_interval,
            "status": self.status.value,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
        }


@dataclass
class ToolCallResult:
    """Result of a tool call."""
    success: bool
    tool_name: str
    namespace: str
    server_name: str
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "namespace": self.namespace,
            "server_name": self.server_name,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RouteDestination:
    """Destination for routing a tool call."""
    server_name: str
    server_url: str
    namespace: str
    tool_name: str
    fallback_servers: List[str] = field(default_factory=list)
    priority: int = 0  # Higher = more preferred

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server_name": self.server_name,
            "server_url": self.server_url,
            "namespace": self.namespace,
            "tool_name": self.tool_name,
            "fallback_servers": self.fallback_servers,
            "priority": self.priority,
        }


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker for a server."""
    server_name: str
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    is_open: bool = False
    last_success_time: Optional[datetime] = None

    def reset(self):
        """Reset the circuit breaker."""
        self.failure_count = 0
        self.is_open = False
        self.last_failure_time = None

    def record_failure(self):
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

    def record_success(self):
        """Record a success."""
        self.failure_count = 0
        self.is_open = False
        self.last_success_time = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server_name": self.server_name,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "is_open": self.is_open,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
        }


@dataclass
class GatewayConfig:
    """Configuration for the entire gateway."""
    host: str
    port: int
    log_level: str
    max_workers: int
    request_timeout: int
    enable_cors: bool

    # Tool registry settings
    categorization_enabled: bool
    versioning_enabled: bool
    deprecation_grace_period: int
    cache_ttl: int

    # Routing settings
    load_balancing: str
    circuit_breaker_threshold: int
    circuit_breaker_timeout: int
    fallback_enabled: bool
    max_retries: int
    retry_delay: int

    # Monitoring settings
    metrics_enabled: bool
    log_tool_calls: bool
    alert_on_failures: bool
    analytics_retention_days: int
    performance_tracking: bool

    # Cache settings
    cache_enabled: bool
    cache_backend: str
    cache_ttl: int
    cache_max_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gateway": {
                "host": self.host,
                "port": self.port,
                "log_level": self.log_level,
                "max_workers": self.max_workers,
                "request_timeout": self.request_timeout,
                "enable_cors": self.enable_cors,
            },
            "tool_registry": {
                "categorization_enabled": self.categorization_enabled,
                "versioning_enabled": self.versioning_enabled,
                "deprecation_grace_period": self.deprecation_grace_period,
                "cache_ttl": self.cache_ttl,
            },
            "routing": {
                "load_balancing": self.load_balancing,
                "circuit_breaker_threshold": self.circuit_breaker_threshold,
                "circuit_breaker_timeout": self.circuit_breaker_timeout,
                "fallback_enabled": self.fallback_enabled,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
            },
            "monitoring": {
                "metrics_enabled": self.metrics_enabled,
                "log_tool_calls": self.log_tool_calls,
                "alert_on_failures": self.alert_on_failures,
                "analytics_retention_days": self.analytics_retention_days,
                "performance_tracking": self.performance_tracking,
            },
            "cache": {
                "enabled": self.cache_enabled,
                "backend": self.cache_backend,
                "ttl": self.cache_ttl,
                "max_size": self.cache_max_size,
            },
        }
