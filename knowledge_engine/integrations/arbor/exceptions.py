"""
Exceptions for Arbor Integration

Following CLAUDE.md principles:
- ZERO TRUST: Clear error messages for debugging
- STRUCTURED LOGGING: Exceptions include context
"""

from typing import Optional, Dict, Any


class ArborError(Exception):
    """Base exception for Arbor integration."""
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.cause = cause
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None
        }


class ArborConnectionError(ArborError):
    """Raised when connection to Arbor server fails."""
    
    def __init__(
        self,
        ws_url: str,
        message: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        msg = message or f"Failed to connect to Arbor at {ws_url}"
        super().__init__(
            msg,
            context={"ws_url": ws_url},
            cause=cause
        )
        self.ws_url = ws_url


class ArborNotConnectedError(ArborError):
    """Raised when attempting operation without connection."""
    
    def __init__(self, operation: str):
        super().__init__(
            f"Cannot perform '{operation}': not connected to Arbor server",
            context={"operation": operation}
        )


class ArborQueryError(ArborError):
    """Raised when Arbor query fails."""
    
    def __init__(
        self,
        query: str,
        error_code: Optional[int] = None,
        message: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        msg = message or f"Query failed: {query}"
        super().__init__(
            msg,
            context={
                "query": query,
                "error_code": error_code
            },
            cause=cause
        )
        self.query = query
        self.error_code = error_code


class ArborTimeoutError(ArborError):
    """Raised when operation times out."""
    
    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds}s",
            context={
                "operation": operation,
                "timeout_seconds": timeout_seconds
            },
            cause=cause
        )


class ArborSchemaError(ArborError):
    """Raised when graph schema conversion fails."""
    
    def __init__(
        self,
        node_type: str,
        message: Optional[str] = None,
        data: Optional[Dict] = None
    ):
        msg = message or f"Schema conversion failed for type: {node_type}"
        super().__init__(
            msg,
            context={
                "node_type": node_type,
                "data": data
            }
        )


class ArborSyncError(ArborError):
    """Raised when graph synchronization fails."""
    
    def __init__(
        self,
        sync_type: str,
        message: Optional[str] = None,
        nodes_affected: Optional[int] = None
    ):
        msg = message or f"Sync failed: {sync_type}"
        context = {"sync_type": sync_type}
        if nodes_affected is not None:
            context["nodes_affected"] = nodes_affected
        super().__init__(msg, context=context)


class ArborMCPError(ArborError):
    """Raised when MCP tool execution fails."""
    
    def __init__(
        self,
        tool_name: str,
        message: Optional[str] = None,
        params: Optional[Dict] = None,
        cause: Optional[Exception] = None
    ):
        msg = message or f"MCP tool '{tool_name}' execution failed"
        context = {"tool_name": tool_name}
        if params:
            context["params"] = params
        super().__init__(msg, context=context, cause=cause)
        self.tool_name = tool_name
