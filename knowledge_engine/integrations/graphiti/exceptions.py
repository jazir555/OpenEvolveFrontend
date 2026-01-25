"""
Custom exceptions for Graphiti integration.

Following CLAUDE.md: Handle failure gracefully with specific exceptions
that enable proper error handling and monitoring.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class GraphitiIntegrationError(Exception):
    """Base exception for all Graphiti integration errors."""

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize exception with context for structured logging.

        Args:
            message: Error message
            correlation_id: Request correlation ID for tracing
            details: Additional error context
        """
        self.message = message
        self.correlation_id = correlation_id
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "correlation_id": self.correlation_id,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class ConfigurationError(GraphitiIntegrationError):
    """Raised when configuration is invalid or missing required values."""

    def __init__(
        self,
        message: str,
        missing_keys: Optional[list[str]] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize configuration error.

        Args:
            message: Error message
            missing_keys: List of missing configuration keys
            correlation_id: Request correlation ID
        """
        details = {"missing_keys": missing_keys} if missing_keys else {}
        super().__init__(message, correlation_id, details)
        self.missing_keys = missing_keys or []


class ConnectionError(GraphitiIntegrationError):
    """Raised when connection to Graphiti database fails."""

    def __init__(
        self,
        message: str,
        uri: Optional[str] = None,
        provider: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize connection error.

        Args:
            message: Error message
            uri: Database URI (sanitized in logs)
            provider: Graph provider (neo4j, falkordb, etc.)
            correlation_id: Request correlation ID
        """
        details = {
            "provider": provider,
            "uri_present": uri is not None,
        }
        super().__init__(message, correlation_id, details)
        self.uri = uri
        self.provider = provider


class ContradictionError(GraphitiIntegrationError):
    """Raised when contradiction detection finds critical issues."""

    def __init__(
        self,
        message: str,
        contradictions: Optional[list[Dict[str, Any]]] = None,
        entity_name: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize contradiction error.

        Args:
            message: Error message
            contradictions: List of detected contradictions
            entity_name: Entity with contradictions
            correlation_id: Request correlation ID
        """
        details = {
            "contradiction_count": len(contradictions) if contradictions else 0,
            "entity_name": entity_name,
        }
        super().__init__(message, correlation_id, details)
        self.contradictions = contradictions or []
        self.entity_name = entity_name


class InvalidTimestampError(GraphitiIntegrationError):
    """Raised when timestamp operations fail validation."""

    def __init__(
        self,
        message: str,
        timestamp: Optional[datetime] = None,
        reason: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize timestamp error.

        Args:
            message: Error message
            timestamp: Invalid timestamp
            reason: Why timestamp is invalid
            correlation_id: Request correlation ID
        """
        details = {
            "timestamp": timestamp.isoformat() if timestamp else None,
            "reason": reason,
        }
        super().__init__(message, correlation_id, details)
        self.timestamp = timestamp
        self.reason = reason


class EpisodeProcessingError(GraphitiIntegrationError):
    """Raised when episode ingestion or processing fails."""

    def __init__(
        self,
        message: str,
        episode_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize episode processing error.

        Args:
            message: Error message
            episode_id: Failed episode ID
            artifact_id: Source artifact ID
            correlation_id: Request correlation ID
        """
        details = {
            "episode_id": episode_id,
            "artifact_id": artifact_id,
        }
        super().__init__(message, correlation_id, details)
        self.episode_id = episode_id
        self.artifact_id = artifact_id


class IncrementalUpdateError(GraphitiIntegrationError):
    """Raised when incremental graph update fails."""

    def __init__(
        self,
        message: str,
        update_type: Optional[str] = None,
        affected_entities: Optional[list[str]] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Initialize incremental update error.

        Args:
            message: Error message
            update_type: Type of update (merge, invalidate, etc.)
            affected_entities: List of affected entity UUIDs
            correlation_id: Request correlation ID
        """
        details = {
            "update_type": update_type,
            "affected_count": len(affected_entities) if affected_entities else 0,
        }
        super().__init__(message, correlation_id, details)
        self.update_type = update_type
        self.affected_entities = affected_entities or []
