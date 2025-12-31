"""
BubbleLabs Security Layer

This module provides comprehensive security hardening for BubbleLabs integration including:
- Authentication and authorization middleware
- Input validation (UUID, whitelists, ranges)
- SSRF protection (URL whitelisting)
- CSRF protection
- Rate limiting basics
- Role-based access control (RBAC)

Author: OpenEvolve Team
Date: 2025-12-29
"""

import uuid
import re
import hashlib
import secrets
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# SSRF Protection: Allowed URL patterns (whitelist)
# BUG FIX: Allow API paths (e.g., /v1, /v1/messages) for legitimate API endpoints
ALLOWED_URL_PATTERNS = [
    r'^https?://localhost(:\d+)?(/.*)?$',  # Allow optional paths
    r'^https?://127\.0\.0\.1(:\d+)?(/.*)?$',  # Allow optional paths
    r'^https?://api\.openai\.com(/.*)?$',  # Allow /v1 and other paths
    r'^https?://api\.anthropic\.com(/.*)?$',  # Allow /v1/messages and other paths
    r'^https?://generativelanguage\.googleapis\.com(/.*)?$',  # Allow paths
    # AWS Bedrock pattern (already correct)
    r'^https://[a-z0-9-]*\.amazonaws\.com(/.*)?$',  # AWS Bedrock
]

# Workflow type whitelist
ALLOWED_WORKFLOW_TYPES = {
    'evolution',
    'adversarial',
    'sovereign',
    'sovereign_decomposition',
    'bubblelabs_openevolve'
}

# Action whitelist for workflow control
ALLOWED_WORKFLOW_ACTIONS = {
    'start',
    'pause',
    'resume',
    'stop',
    'cancel',
    'restart'
}

# Role definitions
class UserRole(Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    GUEST = "guest"


# =============================================================================
# INPUT VALIDATION
# =============================================================================

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_uuid(instance_id: str, param_name: str = "instance_id") -> str:
    """
    Validate that a string is a valid UUID.

    Args:
        instance_id: The string to validate
        param_name: Parameter name for error messages

    Returns:
        The validated UUID string

    Raises:
        ValidationError: If validation fails
    """
    if not instance_id or not isinstance(instance_id, str):
        raise ValidationError(f"{param_name} must be a non-empty string")

    try:
        # Validate UUID format
        uuid.UUID(instance_id)
        return instance_id
    except ValueError:
        raise ValidationError(f"{param_name} must be a valid UUID format: {instance_id}")


def validate_workflow_type(workflow_type: str) -> str:
    """
    Validate workflow_type against whitelist.

    Args:
        workflow_type: The workflow type to validate

    Returns:
        The validated workflow type

    Raises:
        ValidationError: If validation fails
    """
    if not workflow_type or not isinstance(workflow_type, str):
        raise ValidationError("workflow_type must be a non-empty string")

    workflow_type = workflow_type.strip().lower()

    if workflow_type not in ALLOWED_WORKFLOW_TYPES:
        raise ValidationError(
            f"Invalid workflow_type '{workflow_type}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_WORKFLOW_TYPES))}"
        )

    return workflow_type


def validate_workflow_action(action: str) -> str:
    """
    Validate action parameter for workflow control.

    Args:
        action: The action to validate

    Returns:
        The validated action

    Raises:
        ValidationError: If validation fails
    """
    if not action or not isinstance(action, str):
        raise ValidationError("action must be a non-empty string")

    action = action.strip().lower()

    if action not in ALLOWED_WORKFLOW_ACTIONS:
        raise ValidationError(
            f"Invalid action '{action}'. "
            f"Allowed actions: {', '.join(sorted(ALLOWED_WORKFLOW_ACTIONS))}"
        )

    return action


def validate_url(url: str, param_name: str = "url") -> str:
    """
    Validate URL against SSRF whitelist.

    Args:
        url: The URL to validate
        param_name: Parameter name for error messages

    Returns:
        The validated URL

    Raises:
        ValidationError: If validation fails
    """
    if not url or not isinstance(url, str):
        raise ValidationError(f"{param_name} must be a non-empty string")

    url = url.strip()

    # Check against whitelist patterns
    for pattern in ALLOWED_URL_PATTERNS:
        if re.match(pattern, url, re.IGNORECASE):
            return url

    # Check if it's a relative URL (allowed for local paths)
    if url.startswith('/') or url.startswith('./'):
        return url

    raise ValidationError(
        f"{param_name} '{url}' is not in the allowed URL whitelist. "
        f"URLs must match one of: {', '.join(ALLOWED_URL_PATTERNS[:3])}..."
    )


def validate_range(
    value: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    param_name: str = "value"
) -> float:
    """
    Validate a numeric value is within range.

    Args:
        value: The value to validate
        min_value: Minimum allowed value (None for no minimum)
        max_value: Maximum allowed value (None for no maximum)
        param_name: Parameter name for error messages

    Returns:
        The validated numeric value

    Raises:
        ValidationError: If validation fails
    """
    try:
        num_value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{param_name} must be a numeric value")

    if min_value is not None and num_value < min_value:
        raise ValidationError(f"{param_name} must be >= {min_value}, got {num_value}")

    if max_value is not None and num_value > max_value:
        raise ValidationError(f"{param_name} must be <= {max_value}, got {num_value}")

    return num_value


def validate_string_length(
    value: str,
    max_length: int,
    param_name: str = "value",
    min_length: int = 0
) -> str:
    """
    Validate string length.

    Args:
        value: The string to validate
        max_length: Maximum allowed length
        param_name: Parameter name for error messages
        min_length: Minimum allowed length (default: 0)

    Returns:
        The validated string

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"{param_name} must be a string")

    if len(value) < min_length:
        raise ValidationError(f"{param_name} must be at least {min_length} characters")

    if len(value) > max_length:
        raise ValidationError(f"{param_name} must be at most {max_length} characters")

    return value


# =============================================================================
# AUTHENTICATION & AUTHORIZATION
# =============================================================================

@dataclass(frozen=True)
class SecurityContext:
    """
    Security context for a request.

    CONCURRENCY FIX (Issue #10): Made immutable with frozen=True to prevent
    external modification. This ensures thread-safety when returning SecurityContext
    objects from validation methods, as callers cannot mutate the returned object.
    """
    user_id: Optional[str] = None
    role: UserRole = UserRole.GUEST
    session_id: Optional[str] = None
    authenticated: bool = False
    permissions: frozenset = frozenset()  # Use frozenset for immutability

    def __post_init__(self):
        # Convert regular set to frozenset for immutability
        if self.permissions and not isinstance(self.permissions, frozenset):
            object.__setattr__(self, 'permissions', frozenset(self.permissions))


class AuthenticationManager:
    """
    Manages authentication for BubbleLabs operations.

    In production, this would integrate with a proper auth system.
    For now, provides a framework with token-based auth.

    MEMORY LEAK FIXES:
    - Leak #2: Sessions now expire after 24 hours
    - Leak #6: API keys have max_size limit with LRU eviction
    """

    # MEMORY LEAK FIX (Leak #2): Session TTL configuration
    SESSION_TTL_SECONDS = 24 * 3600  # 24 hours
    MAX_SESSIONS = 1000  # Maximum sessions to store

    # MEMORY LEAK FIX (Leak #6): API key limits
    MAX_API_KEYS = 1000  # Maximum API keys to store

    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}  # MEMORY LEAK FIX: Now tracks created_at
        self.sessions: Dict[str, Dict[str, Any]] = {}  # MEMORY LEAK FIX: Now tracks created_at
        self.lock = threading.Lock()

        # Generate a default admin API key for development
        default_key = self._generate_api_key()
        self.api_keys[default_key] = {
            "context": SecurityContext(
                user_id="admin",
                role=UserRole.ADMIN,
                authenticated=True,
                permissions={"*"}  # All permissions
            ),
            "created_at": time.time(),
            "last_used": time.time(),
            "is_admin": True  # Admin keys are exempt from cleanup
        }
        logger.info(f"Generated default admin API key: {default_key}")

    def _generate_api_key(self) -> str:
        """Generate a secure random API key."""
        return f"bl_{secrets.token_urlsafe(32)}"

    def _create_session(self, user_id: str, role: UserRole, permissions: Set[str] = None) -> str:
        """
        Create a new session with timestamp for expiration tracking.

        MEMORY LEAK FIX (Leak #2): Sessions now have creation timestamps for TTL.
        """
        session_id = secrets.token_urlsafe(32)
        with self.lock:
            # Enforce max sessions limit (MEMORY LEAK FIX #2)
            if len(self.sessions) >= self.MAX_SESSIONS:
                # Remove oldest non-admin session
                oldest_session = min(
                    [(sid, data["created_at"]) for sid, data in self.sessions.items()
                     if data.get("context", SecurityContext()).role != UserRole.ADMIN],
                    key=lambda x: x[1],
                    default=None
                )
                if oldest_session:
                    del self.sessions[oldest_session[0]]
                    logger.warning(f"Session limit reached, removed oldest session: {oldest_session[0]}")

            self.sessions[session_id] = {
                "context": SecurityContext(
                    user_id=user_id,
                    role=role,
                    session_id=session_id,
                    authenticated=True,
                    permissions=permissions or set()
                ),
                "created_at": time.time(),
                "last_used": time.time()
            }

        return session_id

    def clean_expired_sessions(self) -> int:
        """
        Remove expired sessions based on TTL.

        MEMORY LEAK FIX (Leak #2): Proactive cleanup of expired sessions.
        Should be called periodically (e.g., every hour).

        Returns:
            Number of sessions removed
        """
        now = time.time()
        removed = 0

        with self.lock:
            expired_sessions = [
                session_id for session_id, data in self.sessions.items()
                if now - data["created_at"] > self.SESSION_TTL_SECONDS
            ]

            for session_id in expired_sessions:
                del self.sessions[session_id]
                removed += 1

        if removed > 0:
            logger.info(f"Cleaned {removed} expired sessions (TTL: {self.SESSION_TTL_SECONDS}s)")

        return removed

    def clean_unused_api_keys(self) -> int:
        """
        Remove unused API keys (non-admin, not used recently).

        MEMORY LEAK FIX (Leak #6): Cleanup of stale API keys.
        Should be called periodically (e.g., every hour).

        Returns:
            Number of API keys removed
        """
        now = time.time()
        UNUSED_THRESHOLD = 7 * 24 * 3600  # 7 days
        removed = 0

        with self.lock:
            unused_keys = [
                key for key, data in self.api_keys.items()
                if not data.get("is_admin", False) and
                (now - data.get("last_used", data["created_at"]) > UNUSED_THRESHOLD)
            ]

            for key in unused_keys:
                del self.api_keys[key]
                removed += 1

        if removed > 0:
            logger.info(f"Cleaned {removed} unused API keys (unused for > {UNUSED_THRESHOLD}s)")

        return removed

    def validate_api_key(self, api_key: str) -> Optional[SecurityContext]:
        """
        Validate an API key and return security context.

        CONCURRENCY FIX (Issue #10): Returns immutable SecurityContext (frozen dataclass).
        This prevents TOCTOU (Time-Of-Check-Time-Of-Use) vulnerabilities by ensuring
        the returned context cannot be mutated by the caller.

        MEMORY LEAK FIX (Leak #6): Updates last_used timestamp.

        Args:
            api_key: The API key to validate

        Returns:
            Immutable SecurityContext if valid, None otherwise
        """
        if not api_key:
            return None

        with self.lock:
            key_data = self.api_keys.get(api_key)
            if key_data:
                # Update last_used timestamp (MEMORY LEAK FIX #6)
                key_data["last_used"] = time.time()
                context = key_data.get("context")
                # Return context directly (already immutable due to frozen=True)
                return context
            return None

    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """
        Validate a session ID and return security context.

        CONCURRENCY FIX (Issue #10): Returns immutable SecurityContext (frozen dataclass).
        This prevents TOCTOU vulnerabilities by ensuring the returned context
        cannot be mutated by the caller.

        MEMORY LEAK FIX (Leak #2): Checks session TTL and updates last_used.

        Args:
            session_id: The session ID to validate

        Returns:
            Immutable SecurityContext if valid, None otherwise
        """
        if not session_id:
            return None

        with self.lock:
            session_data = self.sessions.get(session_id)
            if session_data:
                # Check if session has expired (MEMORY LEAK FIX #2)
                now = time.time()
                if now - session_data["created_at"] > self.SESSION_TTL_SECONDS:
                    # Session expired, remove it
                    del self.sessions[session_id]
                    logger.debug(f"Session {session_id} expired and removed")
                    return None

                # Update last_used timestamp
                session_data["last_used"] = now
                return session_data.get("context")
            return None

    def check_permission(
        self,
        context: SecurityContext,
        required_permission: str
    ) -> bool:
        """
        Check if a security context has a required permission.

        Args:
            context: The security context
            required_permission: The permission to check

        Returns:
            True if permission granted, False otherwise
        """
        if not context or not context.authenticated:
            return False

        # Admin has all permissions
        if context.role == UserRole.ADMIN:
            return True

        # Check for wildcard permission
        if "*" in context.permissions:
            return True

        # Check for specific permission
        return required_permission in context.permissions


# =============================================================================
# CSRF PROTECTION
# =============================================================================

class CSRFProtection:
    """
    CSRF protection using token validation.

    For state-changing operations, a valid CSRF token must be provided.

    MEMORY LEAK FIX (Leak #3): Proactive cleanup of expired tokens.
    """

    # MEMORY LEAK FIX (Leak #3): Token TTL configuration
    TOKEN_TTL_SECONDS = 3600  # 1 hour
    MAX_TOKENS = 10000  # Maximum tokens to store

    def __init__(self):
        self.tokens: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def generate_token(self, session_id: str) -> str:
        """
        Generate a CSRF token for a session.

        MEMORY LEAK FIX (Leak #3): Enforces max_tokens limit.

        Args:
            session_id: The session ID

        Returns:
            CSRF token
        """
        token = secrets.token_urlsafe(32)

        with self.lock:
            # Enforce max tokens limit (MEMORY LEAK FIX #3)
            if len(self.tokens) >= self.MAX_TOKENS:
                # Remove oldest token
                oldest_token = min(
                    self.tokens.items(),
                    key=lambda x: x[1]["created_at"]
                )
                del self.tokens[oldest_token[0]]
                logger.warning(f"Token limit reached, removed oldest token: {oldest_token[0]}")

            self.tokens[token] = {
                "session_id": session_id,
                "created_at": time.time()
            }

        return token

    def validate_token(self, token: str, session_id: str) -> bool:
        """
        Validate a CSRF token.

        CONCURRENCY FIX (Issue #12): Uses .pop() instead of del for expired token cleanup.
        This is more robust when multiple threads validate the same expired token concurrently.
        While del would raise KeyError if token was already deleted, .pop() silently handles it.

        MEMORY LEAK FIX (Leak #3): Lazy cleanup of expired tokens during validation.

        Args:
            token: The CSRF token to validate
            session_id: The session ID

        Returns:
            True if valid, False otherwise
        """
        if not token or not session_id:
            return False

        with self.lock:
            token_data = self.tokens.get(token)

            if not token_data:
                return False

            # Check session match
            if token_data["session_id"] != session_id:
                return False

            # Check token age (1 hour expiry)
            if time.time() - token_data["created_at"] > self.TOKEN_TTL_SECONDS:
                # CONCURRENCY FIX (Issue #12): Use .pop() instead of del
                # If multiple threads validate the same expired token concurrently,
                # del would raise KeyError on the second thread, but .pop() handles it gracefully
                self.tokens.pop(token, None)  # Returns None if already deleted
                return False

            return True

    def cleanup_expired_tokens(self) -> int:
        """
        Remove expired tokens proactively.

        MEMORY LEAK FIX (Leak #3): Proactive cleanup method.
        Should be called periodically (e.g., every 30 minutes).

        Returns:
            Number of tokens removed
        """
        now = time.time()
        removed = 0

        with self.lock:
            expired_tokens = [
                token for token, data in self.tokens.items()
                if now - data["created_at"] > self.TOKEN_TTL_SECONDS
            ]

            for token in expired_tokens:
                del self.tokens[token]
                removed += 1

        if removed > 0:
            logger.info(f"Cleaned {removed} expired CSRF tokens (TTL: {self.TOKEN_TTL_SECONDS}s)")

        return removed

    def invalidate_token(self, token: str):
        """Invalidate a CSRF token."""
        with self.lock:
            self.tokens.pop(token, None)


# =============================================================================
# RATE LIMITING
# =============================================================================

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests: int = 100
    window_seconds: int = 60
    burst_size: int = 10


class RateLimiter:
    """
    Simple rate limiter using token bucket algorithm.

    Limits request rate per user/session.

    CONCURRENCY FIX (Issue #11): Made buckets private (_buckets) and added
    read-only accessor to prevent external modification. All bucket access
    is now controlled through proper locking.

    MEMORY LEAK FIX (Leak #4): Bounded bucket size with LRU eviction and cleanup.
    """

    # MEMORY LEAK FIX (Leak #4): Bucket limits
    MAX_BUCKETS = 10000  # Maximum number of buckets to store
    BUCKET_INACTIVE_SECONDS = 3600  # 1 hour

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        # CONCURRENCY FIX (Issue #11): Made buckets private
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def get_bucket_info(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get read-only information about a rate limit bucket.

        CONCURRENCY FIX (Issue #11): Provides read-only access to bucket state
        without exposing the internal mutable dictionary. Returns a copy to
        prevent external modification.

        Args:
            identifier: The bucket identifier

        Returns:
            Dictionary with bucket info or None if bucket doesn't exist
        """
        with self.lock:
            if identifier in self._buckets:
                # Return a copy to prevent external modification
                bucket = self._buckets[identifier]
                return {
                    "tokens": bucket["tokens"],
                    "last_update": bucket["last_update"],
                    "max_requests": self.config.max_requests,
                    "window_seconds": self.config.window_seconds
                }
            return None

    def check_rate_limit(
        self,
        identifier: str,
        tokens: int = 1
    ) -> tuple[bool, Optional[int]]:
        """
        Check if a request is within rate limits.

        CONCURRENCY FIX (Issue #11): All bucket access is protected by lock,
        and buckets dictionary is now private (_buckets) to prevent external
        modification that could bypass rate limiting logic.

        MEMORY LEAK FIX (Leak #4): Enforces max_buckets limit with LRU eviction.

        Args:
            identifier: Unique identifier (user_id, session_id, etc.)
            tokens: Number of tokens to consume (default: 1)

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        now = time.time()

        with self.lock:
            # Enforce max buckets limit (MEMORY LEAK FIX #4)
            if identifier not in self._buckets:
                if len(self._buckets) >= self.MAX_BUCKETS:
                    # Remove oldest bucket (LRU eviction)
                    oldest_bucket = min(
                        self._buckets.items(),
                        key=lambda x: x[1]["last_update"]
                    )
                    del self._buckets[oldest_bucket[0]]
                    logger.warning(f"Bucket limit reached, removed oldest bucket: {oldest_bucket[0]}")

                self._buckets[identifier] = {
                    "tokens": self.config.max_requests - 1,
                    "last_update": now
                }
                return True, None

            bucket = self._buckets[identifier]

            # Refill tokens based on time passed
            time_passed = now - bucket["last_update"]
            refill = int(time_passed * self.config.max_requests / self.config.window_seconds)

            bucket["tokens"] = min(
                self.config.max_requests,
                bucket["tokens"] + refill
            )
            bucket["last_update"] = now

            # Check if enough tokens
            if bucket["tokens"] >= tokens:
                bucket["tokens"] -= tokens
                return True, None
            else:
                # Calculate retry after
                # BUG FIX: Ensure at least 1 second retry time when exhausted
                retry_after = max(1, int(
                    (tokens - bucket["tokens"]) *
                    self.config.window_seconds / self.config.max_requests
                ))
                return False, retry_after

    def cleanup_inactive_buckets(self) -> int:
        """
        Remove inactive buckets proactively.

        MEMORY LEAK FIX (Leak #4): Proactive cleanup of inactive buckets.
        Should be called periodically (e.g., every hour).

        Returns:
            Number of buckets removed
        """
        now = time.time()
        removed = 0

        with self.lock:
            inactive_buckets = [
                identifier for identifier, data in self.buckets.items()
                if now - data["last_update"] > self.BUCKET_INACTIVE_SECONDS
            ]

            for identifier in inactive_buckets:
                del self.buckets[identifier]
                removed += 1

        if removed > 0:
            logger.info(f"Cleaned {removed} inactive rate limiter buckets (inactive > {self.BUCKET_INACTIVE_SECONDS}s)")

        return removed


# =============================================================================
# SECURITY DECORATORS
# =============================================================================

# Global security components
auth_manager = AuthenticationManager()
csrf_protection = CSRFProtection()
rate_limiter = RateLimiter()


def require_auth(
    permissions: Optional[Set[str]] = None
):
    """
    Decorator to require authentication for a function.

    Args:
        permissions: Optional set of required permissions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check for API key in kwargs
            api_key = kwargs.pop('api_key', None)
            context = kwargs.pop('security_context', None)

            if not context:
                if api_key:
                    context = auth_manager.validate_api_key(api_key)
                else:
                    # Allow unauthenticated access in development
                    # In production, this should raise an exception
                    context = SecurityContext(role=UserRole.GUEST)

            if not context or not context.authenticated:
                logger.warning(f"Unauthorized access attempt to {func.__name__}")
                return {
                    "success": False,
                    "error": "Authentication required",
                    "message": "Please provide valid API credentials"
                }

            # Check permissions if required
            if permissions:
                for perm in permissions:
                    if not auth_manager.check_permission(context, perm):
                        logger.warning(
                            f"Permission denied: {context.user_id} "
                            f"attempted to access {func.__name__} "
                            f"requiring {perm}"
                        )
                        return {
                            "success": False,
                            "error": "Permission denied",
                            "message": f"Permission '{perm}' required"
                        }

            # Add context to kwargs
            kwargs['security_context'] = context

            return func(*args, **kwargs)

        return wrapper
    return decorator


def require_csrf(func):
    """
    Decorator to require CSRF token for state-changing operations.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        csrf_token = kwargs.pop('csrf_token', None)
        session_id = kwargs.pop('session_id', None)

        if not csrf_token or not session_id:
            logger.warning(f"CSRF token missing for {func.__name__}")
            return {
                "success": False,
                "error": "CSRF token required",
                "message": "State-changing operations require CSRF protection"
            }

        if not csrf_protection.validate_token(csrf_token, session_id):
            logger.warning(f"Invalid CSRF token for {func.__name__}")
            return {
                "success": False,
                "error": "Invalid CSRF token",
                "message": "CSRF token validation failed"
            }

        return func(*args, **kwargs)

    return wrapper


def validate_input(**validators):
    """
    Decorator to validate function inputs.

    Args:
        **validators: Mapping of parameter names to validator functions

    Example:
        @validate_input(
            instance_id=validate_uuid,
            action=validate_workflow_action
        )
        def control_workflow(instance_id, action):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate parameters
            for param_name, validator in validators.items():
                if param_name in kwargs:
                    try:
                        kwargs[param_name] = validator(kwargs[param_name])
                    except ValidationError as e:
                        logger.warning(
                            f"Input validation failed for {param_name} "
                            f"in {func.__name__}: {e}"
                        )
                        return {
                            "success": False,
                            "error": "Invalid input",
                            "message": str(e)
                        }

            return func(*args, **kwargs)

        return wrapper
    return decorator


# =============================================================================
# INITIALIZATION
# =============================================================================

# Export security components
__all__ = [
    # Validation functions
    'validate_uuid',
    'validate_workflow_type',
    'validate_workflow_action',
    'validate_url',
    'validate_range',
    'validate_string_length',

    # Security classes
    'SecurityContext',
    'UserRole',
    'AuthenticationManager',
    'CSRFProtection',
    'RateLimiter',
    'ValidationError',

    # Decorators
    'require_auth',
    'require_csrf',
    'validate_input',

    # Global instances
    'auth_manager',
    'csrf_protection',
    'rate_limiter',

    # Configuration
    'ALLOWED_URL_PATTERNS',
    'ALLOWED_WORKFLOW_TYPES',
    'ALLOWED_WORKFLOW_ACTIONS'
]


if __name__ == "__main__":
    # Test security components
    print("BubbleLabs Security Layer")
    print(f"Default admin API key: {list(auth_manager.api_keys.keys())[0]}")

    # Test validation
    try:
        validate_uuid("not-a-uuid")
    except ValidationError as e:
        print(f"UUID validation working: {e}")

    try:
        validate_workflow_type("invalid-type")
    except ValidationError as e:
        print(f"Workflow type validation working: {e}")
