"""
Canonical Schemas for Shared Data Models

This module defines the canonical data models used across all reliability components.
These models provide a standardized interface for results, validation, and status reporting.

Following the Anti-Corruption Layer (ACL) pattern, these schemas serve as the canonical
data models that all adapters must normalize to/from.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


# =============================================================================
# Base Models
# =============================================================================

class ResultStatus(str, Enum):
    """Status of a result"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class BaseResult:
    """
    Base result class for all reliability operations.

    Provides a common structure for all operation results with success status,
    timing information, correlation tracking, and error reporting.
    """
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    correlation_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        from dataclasses import asdict
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


# =============================================================================
# Validation Result Models
# =============================================================================

@dataclass
class ValidationResult:
    """
    Result of validation operation.

    Tracks whether validation passed, the output, any failures or warnings,
    and whether remediation was applied.
    """
    is_valid: bool
    output: Optional[Any] = None
    original_output: Optional[Any] = None
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    remediated: bool = False
    remediation_strategy: Optional[str] = None
    validation_time_ms: float = 0.0
    validator_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def has_failures(self) -> bool:
        """Check if validation failed"""
        return len(self.failures) > 0

    def has_warnings(self) -> bool:
        """Check if validation produced warnings"""
        return len(self.warnings) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        from dataclasses import asdict
        return asdict(self)

    def get_summary(self) -> str:
        """Get a human-readable summary"""
        if self.is_valid:
            msg = "Validation passed"
            if self.remediated:
                msg += f" (remediated with {self.remediation_strategy})"
            return msg
        else:
            return f"Validation failed: {', '.join(self.failures[:3])}"


@dataclass
class VoteValidationResult:
    """
    Result of vote validation.

    Specialized validation result for MDAP vote validation with tracking
    of the original vote before any remediation.
    """
    is_valid: bool
    vote: Any
    original_vote: Any
    failures: List[str] = field(default_factory=list)
    remediated: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def has_failures(self) -> bool:
        """Check if validation failed"""
        return len(self.failures) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        from dataclasses import asdict
        result = asdict(self)
        # Handle non-serializable vote objects
        if not isinstance(result['vote'], (str, int, float, bool, list, dict, type(None))):
            result['vote'] = str(result['vote'])
        if not isinstance(result['original_vote'], (str, int, float, bool, list, dict, type(None))):
            result['original_vote'] = str(result['original_vote'])
        return result


# =============================================================================
# Generation Result Models
# =============================================================================

@dataclass
class GenerationResult(BaseResult):
    """
    Result from LMQL/guarded generation.

    Tracks generation output, which reliability layers were used,
    constraint violations, validation failures, retry counts, and performance metrics.
    """
    output: Optional[str] = None
    prompt: Optional[str] = None
    layers_used: List[str] = field(default_factory=list)
    layers_failed: List[str] = field(default_factory=list)
    constraint_violations: List[str] = field(default_factory=list)
    validation_failures: List[str] = field(default_factory=list)
    retry_count: int = 0
    total_latency_ms: float = 0.0
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    fallback_used: bool = False

    def has_violations(self) -> bool:
        """Check if constraint violations occurred"""
        return len(self.constraint_violations) > 0

    def has_validation_failures(self) -> bool:
        """Check if validation failures occurred"""
        return len(self.validation_failures) > 0

    def has_failures(self) -> bool:
        """Check if any type of failure occurred"""
        return self.has_violations() or self.has_validation_failures()

    def get_layers_summary(self) -> str:
        """Get a summary of layers used"""
        if not self.layers_used:
            return "No layers used"
        return f"Layers: {', '.join(self.layers_used)}"

    def get_failure_summary(self) -> str:
        """Get a summary of all failures"""
        failures = []
        if self.constraint_violations:
            failures.append(f"Constraints: {', '.join(self.constraint_violations[:3])}")
        if self.validation_failures:
            failures.append(f"Validation: {', '.join(self.validation_failures[:3])}")
        if self.layers_failed:
            failures.append(f"Failed Layers: {', '.join(self.layers_failed)}")
        return '; '.join(failures) if failures else "No failures"


# =============================================================================
# ROMA-Specific Models
# =============================================================================

@dataclass
class RomaDecompositionResult(BaseResult):
    """
    Result from ROMA decomposition.

    Tracks decomposition results including tree structure, constraints,
    validation results, and performance metrics.
    """
    result: Optional[Dict[str, Any]] = None
    task: Optional[str] = None
    depth_reached: int = 0
    total_nodes: int = 0
    max_depth: int = 0
    execution_mode: str = "recursive"
    constraint_violations: List[str] = field(default_factory=list)
    validation_failures: List[Dict[str, Any]] = field(default_factory=list)
    remediation_applied: List[str] = field(default_factory=list)
    layers_used: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def has_violations(self) -> bool:
        """Check if constraint violations occurred"""
        return len(self.constraint_violations) > 0

    def has_validation_failures(self) -> bool:
        """Check if validation failures occurred"""
        return len(self.validation_failures) > 0

    def get_complexity_metric(self) -> float:
        """Calculate complexity metric (nodes * depth)"""
        return self.total_nodes * self.depth_reached if self.depth_reached > 0 else 0

    def get_summary(self) -> str:
        """Get a human-readable summary"""
        if self.success:
            return (f"Decomposition complete: {self.total_nodes} nodes, "
                   f"depth {self.depth_reached}/{self.max_depth}, "
                   f"{self.execution_time_ms:.2f}ms")
        else:
            return f"Decomposition failed: {self.error}"


@dataclass
class RomaAnalysisResult(BaseResult):
    """
    Result from ROMA analysis.

    Tracks analysis results including complexity scoring and
    estimation of decomposition parameters.
    """
    result: Optional[Dict[str, Any]] = None
    task: Optional[str] = None
    complexity_score: float = 0.0
    estimated_depth: int = 0
    estimated_nodes: int = 0
    analysis_time_ms: float = 0.0

    def get_complexity_level(self) -> str:
        """Get human-readable complexity level"""
        if self.complexity_score < 0.3:
            return "Low"
        elif self.complexity_score < 0.7:
            return "Medium"
        else:
            return "High"


# =============================================================================
# MDAP-Specific Models
# =============================================================================

@dataclass
class MDAPSolveResult(BaseResult):
    """
    Result from MDAP solve operation.

    Tracks voting results, winner selection, confidence scores,
    and any red flags raised during voting.
    """
    result: Optional[Dict[str, Any]] = None
    task: Optional[str] = None
    winner: Optional[Any] = None
    votes: Dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0
    red_flags: List[str] = field(default_factory=list)
    attempts: int = 0
    duration_seconds: float = 0.0

    @property
    def total_votes(self) -> int:
        """Total number of votes"""
        return sum(self.votes.values())

    @property
    def has_red_flags(self) -> bool:
        """Check if red flags were raised"""
        return len(self.red_flags) > 0

    @property
    def is_confident(self) -> bool:
        """Check if result is confident (confidence > threshold)"""
        return self.confidence >= 0.7

    def get_winner_name(self) -> Optional[str]:
        """Get the name of the winner"""
        if self.winner is None:
            return None
        # Handle different winner types
        if isinstance(self.winner, dict):
            return self.winner.get('name', str(self.winner))
        if hasattr(self.winner, 'name'):
            return self.winner.name
        return str(self.winner)

    def get_vote_summary(self) -> str:
        """Get a summary of voting results"""
        if not self.votes:
            return "No votes recorded"
        return ', '.join(f"{k}: {v}" for k, v in sorted(
            self.votes.items(), key=lambda x: x[1], reverse=True
        ))


@dataclass
class MDAPStatistics:
    """
    Statistics for MDAP operations.

    Tracks aggregate statistics for MDAP solve operations including
    success rates, validation statistics, and layer usage.
    """
    total_solves: int = 0
    successful_solves: int = 0
    failed_solves: int = 0
    total_votes_validated: int = 0
    valid_votes: int = 0
    remediated_votes: int = 0
    rejected_votes: int = 0
    core_integration_used: int = 0
    mcp_fallback_used: int = 0
    avg_latency_ms: float = 0.0

    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_solves == 0:
            return 0.0
        return self.successful_solves / self.total_solves

    def get_vote_validation_rate(self) -> float:
        """Calculate vote validation success rate"""
        if self.total_votes_validated == 0:
            return 0.0
        return self.valid_votes / self.total_votes_validated

    def get_remediation_rate(self) -> float:
        """Calculate vote remediation rate"""
        if self.total_votes_validated == 0:
            return 0.0
        return self.remediated_votes / self.total_votes_validated

    def get_summary(self) -> str:
        """Get statistics summary"""
        return (
            f"MDAP Stats: {self.successful_solves}/{self.total_solves} solves "
            f"({self.get_success_rate():.1%}), "
            f"{self.valid_votes}/{self.total_votes_validated} votes valid "
            f"({self.get_vote_validation_rate():.1%})"
        )


# =============================================================================
# Constraint Models
# =============================================================================

class ConstraintType(str, Enum):
    """Types of constraints"""
    REGEX = "regex"
    LENGTH = "length"
    FROM_LIST = "from_list"
    JSON_SCHEMA = "json_schema"
    CUSTOM = "custom"
    NUMERICAL = "numerical"


@dataclass
class Constraint:
    """
    Constraint definition.

    Defines a single constraint with its type, target field, and parameters.
    """
    type: ConstraintType
    field: str
    value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        from dataclasses import asdict
        result = asdict(self)
        # Convert Enum to string
        result['type'] = self.type.value
        return result

    def is_enabled(self) -> bool:
        """Check if constraint is enabled"""
        return self.enabled

    def get_description(self) -> str:
        """Get constraint description"""
        if self.description:
            return self.description
        return f"{self.type.value} constraint on {self.field}"


# =============================================================================
# Layer Status Models
# =============================================================================

@dataclass
class LayerStatus:
    """
    Status of a reliability layer.

    Tracks availability, health, performance metrics, and usage statistics
    for a single reliability layer.
    """
    name: str
    available: bool
    enabled: bool
    healthy: bool = True
    version: Optional[str] = None
    last_error: Optional[str] = None
    request_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    last_check: Optional[str] = None

    def get_failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.request_count == 0:
            return 0.0
        return self.failure_count / self.request_count

    def is_operational(self) -> bool:
        """Check if layer is operational (available and healthy)"""
        return self.available and self.healthy

    def get_status_summary(self) -> str:
        """Get status summary"""
        status = "OPERATIONAL" if self.is_operational() else "DOWN"
        return f"{self.name}: {status} ({self.request_count} requests, {self.get_failure_rate():.1%} failures)"


@dataclass
class SystemHealth:
    """
    Overall system health status.

    Aggregates health status from all reliability components
    and provides system-wide health metrics.
    """
    healthy: bool
    bridge_healthy: bool
    components: Dict[str, LayerStatus] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_unhealthy_components(self) -> List[str]:
        """Get list of unhealthy components"""
        return [
            name for name, status in self.components.items()
            if not status.healthy or not status.available
        ]

    def get_healthy_components(self) -> List[str]:
        """Get list of healthy components"""
        return [
            name for name, status in self.components.items()
            if status.healthy and status.available
        ]

    def get_disabled_components(self) -> List[str]:
        """Get list of disabled components"""
        return [
            name for name, status in self.components.items()
            if not status.enabled
        ]

    def get_summary(self) -> str:
        """Get health summary"""
        unhealthy = self.get_unhealthy_components()
        if not unhealthy:
            return f"All systems operational ({len(self.components)} components)"
        return f"System degraded: {len(unhealthy)} components unhealthy"


# =============================================================================
# Statistics Models
# =============================================================================

@dataclass
class BridgeStatistics:
    """
    Statistics for unified bridge.

    Tracks aggregate statistics for all bridge operations including
    success rates, layer usage, retry distribution, and latency.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    layers: Dict[str, Dict[str, int]] = field(default_factory=dict)
    retry_distribution: Dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0

    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def get_most_used_layer(self) -> Optional[str]:
        """Get the most used layer"""
        if not self.layers:
            return None
        return max(self.layers.items(), key=lambda x: x[1].get('success', 0))[0]

    def get_layer_stats(self, layer_name: str) -> Dict[str, int]:
        """Get statistics for a specific layer"""
        return self.layers.get(layer_name, {'success': 0, 'failure': 0})

    def get_summary(self) -> str:
        """Get statistics summary"""
        return (
            f"Bridge: {self.successful_requests}/{self.total_requests} requests "
            f"({self.get_success_rate():.1%}), "
            f"{self.avg_latency_ms:.2f}ms avg latency"
        )


@dataclass
class AdapterStatistics:
    """
    Base statistics for adapters.

    Provides basic statistics tracking for adapter implementations.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    last_request_time: Optional[str] = None

    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def update_success(self, latency_ms: float) -> None:
        """Record a successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.last_request_time = datetime.utcnow().isoformat()
        # Update rolling average
        self.avg_latency_ms = (
            (self.avg_latency_ms * (self.total_requests - 1) + latency_ms) /
            self.total_requests
        )

    def update_failure(self) -> None:
        """Record a failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_request_time = datetime.utcnow().isoformat()


# =============================================================================
# Configuration Models
# =============================================================================

@dataclass
class ValidationConfig:
    """
    Configuration for validation.

    Defines how validation should be performed including
    which validators to use and failure handling strategy.
    """
    enabled: bool = True
    validators: List[str] = field(default_factory=list)
    on_fail: str = "reask"  # Options: "reask", "remediate", "reject"
    max_retries: int = 3
    timeout_seconds: int = 30

    def is_enabled(self) -> bool:
        """Check if validation is enabled"""
        return self.enabled

    def has_validators(self) -> bool:
        """Check if validators are configured"""
        return len(self.validators) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        from dataclasses import asdict
        return asdict(self)


@dataclass
class ConstraintConfig:
    """
    Configuration for constraints.

    Defines constraints to be applied during generation
    and generation parameters.
    """
    enabled: bool = True
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    decoding: str = "argmax"
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def is_enabled(self) -> bool:
        """Check if constraints are enabled"""
        return self.enabled

    def has_constraints(self) -> bool:
        """Check if constraints are defined"""
        return len(self.constraints) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        from dataclasses import asdict
        return asdict(self)


# =============================================================================
# Helper Functions
# =============================================================================

def create_success_result(
    output: Any,
    **kwargs
) -> GenerationResult:
    """
    Create a successful generation result.

    Args:
        output: The generated output
        **kwargs: Additional result fields

    Returns:
        GenerationResult configured for success
    """
    return GenerationResult(
        success=True,
        output=output,
        **kwargs
    )


def create_error_result(
    error: str,
    **kwargs
) -> GenerationResult:
    """
    Create an error result.

    Args:
        error: Error message
        **kwargs: Additional result fields

    Returns:
        GenerationResult configured for failure
    """
    return GenerationResult(
        success=False,
        error=error,
        **kwargs
    )


def create_validation_result(
    is_valid: bool,
    output: Any = None,
    **kwargs
) -> ValidationResult:
    """
    Create a validation result.

    Args:
        is_valid: Whether validation passed
        output: The validated output
        **kwargs: Additional result fields

    Returns:
        ValidationResult configured appropriately
    """
    return ValidationResult(
        is_valid=is_valid,
        output=output,
        **kwargs
    )


def create_vote_validation_result(
    is_valid: bool,
    vote: Any,
    original_vote: Any,
    **kwargs
) -> VoteValidationResult:
    """
    Create a vote validation result.

    Args:
        is_valid: Whether validation passed
        vote: The (possibly remediated) vote
        original_vote: The original vote before remediation
        **kwargs: Additional result fields

    Returns:
        VoteValidationResult configured appropriately
    """
    return VoteValidationResult(
        is_valid=is_valid,
        vote=vote,
        original_vote=original_vote,
        **kwargs
    )


def create_layer_status(
    name: str,
    available: bool = True,
    enabled: bool = True,
    **kwargs
) -> LayerStatus:
    """
    Create a layer status object.

    Args:
        name: Layer name
        available: Whether layer is available
        enabled: Whether layer is enabled
        **kwargs: Additional status fields

    Returns:
        LayerStatus configured appropriately
    """
    return LayerStatus(
        name=name,
        available=available,
        enabled=enabled,
        last_check=datetime.utcnow().isoformat(),
        **kwargs
    )


def merge_statistics(*stats: AdapterStatistics) -> AdapterStatistics:
    """
    Merge multiple adapter statistics into one.

    Args:
        *stats: Variable number of AdapterStatistics to merge

    Returns:
        AdapterStatistics with merged values
    """
    merged = AdapterStatistics()
    for stat in stats:
        merged.total_requests += stat.total_requests
        merged.successful_requests += stat.successful_requests
        merged.failed_requests += stat.failed_requests
        # Use weighted average for latency
        if stat.total_requests > 0:
            merged.avg_latency_ms = (
                (merged.avg_latency_ms * (merged.total_requests - stat.total_requests) +
                 stat.avg_latency_ms * stat.total_requests) /
                merged.total_requests
            )
    return merged


# =============================================================================
# Serialization Utilities
# =============================================================================

def result_to_dict(result: Union[BaseResult, ValidationResult, LayerStatus]) -> Dict[str, Any]:
    """
    Convert a result object to a dictionary.

    Args:
        result: Result object to convert

    Returns:
        Dictionary representation of the result
    """
    if hasattr(result, 'to_dict'):
        return result.to_dict()
    # Fallback for basic dataclasses
    from dataclasses import asdict
    return asdict(result)


def result_to_json(result: Union[BaseResult, ValidationResult, LayerStatus]) -> str:
    """
    Convert a result object to a JSON string.

    Args:
        result: Result object to convert

    Returns:
        JSON string representation of the result
    """
    return json.dumps(result_to_dict(result))


def dict_to_validation_result(data: Dict[str, Any]) -> ValidationResult:
    """
    Convert a dictionary to ValidationResult.

    Args:
        data: Dictionary to convert

    Returns:
        ValidationResult instance
    """
    return ValidationResult(**data)


def dict_to_generation_result(data: Dict[str, Any]) -> GenerationResult:
    """
    Convert a dictionary to GenerationResult.

    Args:
        data: Dictionary to convert

    Returns:
        GenerationResult instance
    """
    return GenerationResult(**data)
