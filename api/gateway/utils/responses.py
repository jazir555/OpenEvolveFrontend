"""
Response formatting utilities
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Responses
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

from typing import Any, Optional, List, Dict, TypeVar, Generic
from pydantic import BaseModel, Field
from datetime import datetime
from models.schemas import PaginatedResponse


T = TypeVar("T")


class SuccessResponse(BaseModel, Generic[T]):
    """Standard success response wrapper"""

    success: bool = True
    message: Optional[str] = None
    data: T
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedData(PaginatedResponse, Generic[T]):
    """Paginated response with data"""

    items: List[T]

    def __init__(self, items: List[T], total: int, limit: int = 20, offset: int = 0):
        super().__init__(
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + limit < total,
        )
        self.items = items


def success(
    data: Any,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a success response"""
    response = {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if message:
        response["message"] = message
    return response


def paginated(
    items: List[Any],
    total: int,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    """Create a paginated response"""
    return {
        "success": True,
        "data": {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


def created(
    data: Any,
    message: str = "Resource created successfully",
) -> Dict[str, Any]:
    """Create a resource created response"""
    return success(data=data, message=message)


def updated(
    data: Any,
    message: str = "Resource updated successfully",
) -> Dict[str, Any]:
    """Create a resource updated response"""
    return success(data=data, message=message)


def deleted(
    message: str = "Resource deleted successfully",
) -> Dict[str, Any]:
    """Create a resource deleted response"""
    return success(data=None, message=message)


def error(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create an error response"""
    error_data = {
        "code": code,
        "message": message,
    }
    if details:
        error_data["details"] = details

    return {
        "success": False,
        "error": error_data,
        "timestamp": datetime.utcnow().isoformat(),
    }


