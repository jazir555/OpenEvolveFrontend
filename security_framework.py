"""
OpenEvolve Security Framework - Defense in Depth Implementation

This module provides comprehensive security features for all workflow files.
Author: Security Implementation Team
Version: 1.0.0
"""

import functools
import hashlib
import json
import logging
import os
import secrets
import time
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
import asyncio
from collections import defaultdict
import jwt

# FastAPI security
from fastapi import HTTPException, Request, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================

class SecurityConfig:
    """Centralized security configuration"""
    
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
    JWT_ALGORITHM = 'HS256'
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
    RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv('RATE_LIMIT_REQUESTS_PER_MINUTE', '100'))
    RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    AUDIT_LOG_ENABLED = os.getenv('AUDIT_LOG_ENABLED', 'true').lower() == 'true'


# ============================================================================
# PERMISSIONS & ROLES
# ============================================================================

class Permission(Enum):
    """System permissions"""
    WORKFLOW_CREATE = "workflow:create"
    WORKFLOW_READ = "workflow:read"
    WORKFLOW_UPDATE = "workflow:update"
    WORKFLOW_DELETE = "workflow:delete"
    WORKFLOW_EXECUTE = "workflow:execute"
    TEAM_CREATE = "team:create"
    TEAM_READ = "team:read"
    TEAM_UPDATE = "team:update"
    TEAM_DELETE = "team:delete"
    GAUNTLET_CREATE = "gauntlet:create"
    GAUNTLET_READ = "gauntlet:read"
    GAUNTLET_UPDATE = "gauntlet:update"
    GAUNTLET_DELETE = "gauntlet:delete"
    GAUNTLET_EXECUTE = "gauntlet:execute"
    KNOWLEDGE_CREATE = "knowledge:create"
    KNOWLEDGE_READ = "knowledge:read"
    KNOWLEDGE_UPDATE = "knowledge:update"
    KNOWLEDGE_DELETE = "knowledge:delete"
    API_ACCESS = "api:access"
    API_ADMIN = "api:admin"
    SYSTEM_ADMIN = "system:admin"
    AUDIT_READ = "audit:read"
    USER_MANAGE = "user:manage"


class Role:
    """Role definitions with permissions"""
    ADMIN = {"name": "admin", "permissions": [p.value for p in Permission]}
    WORKFLOW_MANAGER = {
        "name": "workflow_manager",
        "permissions": [
            Permission.WORKFLOW_CREATE.value, Permission.WORKFLOW_READ.value,
            Permission.WORKFLOW_UPDATE.value, Permission.WORKFLOW_DELETE.value,
            Permission.WORKFLOW_EXECUTE.value, Permission.TEAM_READ.value,
            Permission.GAUNTLET_READ.value, Permission.GAUNTLET_EXECUTE.value,
            Permission.KNOWLEDGE_CREATE.value, Permission.KNOWLEDGE_READ.value,
            Permission.KNOWLEDGE_UPDATE.value, Permission.API_ACCESS.value,
        ]
    }
    ANALYST = {
        "name": "analyst",
        "permissions": [
            Permission.WORKFLOW_READ.value, Permission.WORKFLOW_EXECUTE.value,
            Permission.TEAM_READ.value, Permission.GAUNTLET_READ.value,
            Permission.GAUNTLET_EXECUTE.value, Permission.KNOWLEDGE_READ.value,
            Permission.API_ACCESS.value,
        ]
    }
    VIEWER = {
        "name": "viewer",
        "permissions": [
            Permission.WORKFLOW_READ.value, Permission.TEAM_READ.value,
            Permission.GAUNTLET_READ.value, Permission.KNOWLEDGE_READ.value,
        ]
    }


@dataclass
class UserContext:
    """User context for authentication/authorization"""
    user_id: str
    username: str
    email: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    is_active: bool = True
    is_superuser: bool = False
    api_key_id: Optional[str] = None
    
    def has_permission(self, permission: Union[str, Permission]) -> bool:
        if self.is_superuser:
            return True
        perm_value = permission.value if isinstance(permission, Permission) else permission
        if perm_value in self.permissions:
            return True
        for role_name in self.roles:
            role = getattr(Role, role_name.upper(), None)
            if role and perm_value in role.get("permissions", []):
                return True
        return False


# ============================================================================
# JWT MANAGEMENT
# ============================================================================

class JWTManager:
    """JWT Token Management"""
    
    def __init__(self):
        self.secret_key = SecurityConfig.JWT_SECRET_KEY
        self.algorithm = SecurityConfig.JWT_ALGORITHM
        self.access_token_expire = timedelta(minutes=SecurityConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    def create_access_token(self, user_context: UserContext, expires_delta: Optional[timedelta] = None) -> str:
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + self.access_token_expire
        payload = {
            "sub": user_context.user_id,
            "username": user_context.username,
            "email": user_context.email,
            "roles": user_context.roles,
            "permissions": user_context.permissions,
            "is_superuser": user_context.is_superuser,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def get_user_context(self, token: str) -> Optional[UserContext]:
        payload = self.decode_token(token)
        if not payload:
            return None
        return UserContext(
            user_id=payload.get("sub", ""),
            username=payload.get("username", ""),
            email=payload.get("email", ""),
            roles=payload.get("roles", []),
            permissions=payload.get("permissions", []),
            is_superuser=payload.get("is_superuser", False)
        )


_jwt_manager = None

def get_jwt_manager() -> JWTManager:
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Rate limiter with token bucket algorithm"""
    
    def __init__(self, requests_per_minute: int = 100, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self._buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"tokens": burst_size, "last_update": time.time()})
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        async with self._lock:
            now = time.time()
            bucket = self._buckets[key]
            time_passed = now - bucket["last_update"]
            tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
            bucket["tokens"] = min(self.burst_size, bucket["tokens"] + tokens_to_add)
            bucket["last_update"] = now
            
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True, {"limit": self.requests_per_minute, "remaining": int(bucket["tokens"]), "reset": int(now + 60)}
            else:
                return False, {"limit": self.requests_per_minute, "remaining": 0, "reset": int(bucket["last_update"] + 60)}


_rate_limiter = None

def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(SecurityConfig.RATE_LIMIT_REQUESTS_PER_MINUTE, 10)
    return _rate_limiter


# ============================================================================
# INPUT VALIDATION
# ============================================================================

class ValidationError(Exception):
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation error for field '{field}': {message}")


class InputValidator:
    MAX_LENGTHS = {'title': 200, 'description': 10000, 'name': 100, 'email': 255, 'url': 2048, 'id': 100}
    
    @classmethod
    def validate_string(cls, value: Any, field_name: str, min_length: int = 1, max_length: int = 1000) -> str:
        if value is None:
            raise ValidationError(field_name, "Value cannot be None", value)
        if not isinstance(value, str):
            value = str(value)
        if len(value) < min_length:
            raise ValidationError(field_name, f"Must be at least {min_length} characters", value)
        if len(value) > max_length:
            raise ValidationError(field_name, f"Must be no more than {max_length} characters", value)
        return value
    
    @classmethod
    def validate_email(cls, email: str, field_name: str = "email") -> str:
        email = cls.validate_string(email, field_name, max_length=cls.MAX_LENGTHS['email'])
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise ValidationError(field_name, "Invalid email format", email)
        return email.lower()
    
    @classmethod
    def validate_id(cls, id_value: str, field_name: str = "id") -> str:
        return cls.validate_string(id_value, field_name, min_length=1, max_length=cls.MAX_LENGTHS['id'])
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        filename = os.path.basename(filename)
        filename = filename.replace('\x00', '')
        filename = re.sub(r'\.{2,}', '', filename)
        filename = re.sub(r'[^a-zA-Z0-9._\-]', '_', filename)
        return filename


# ============================================================================
# AUDIT LOGGING
# ============================================================================

@dataclass
class AuditLogEntry:
    timestamp: datetime
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    success: bool
    ip_address: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    def __init__(self):
        self.enabled = SecurityConfig.AUDIT_LOG_ENABLED
        self._logs: List[AuditLogEntry] = []
        self._lock = asyncio.Lock()
    
    async def log(self, entry: AuditLogEntry):
        if not self.enabled:
            return
        async with self._lock:
            self._logs.append(entry)
            if entry.success:
                logger.info(f"AUDIT: {entry.action} {entry.resource_type}/{entry.resource_id} by {entry.user_id}")
            else:
                logger.warning(f"AUDIT FAIL: {entry.action} {entry.resource_type}/{entry.resource_id} by {entry.user_id}")
    
    async def log_auth_attempt(self, user_id: str, success: bool, ip_address: str = None, details: Dict = None):
        await self.log(AuditLogEntry(
            timestamp=datetime.utcnow(), user_id=user_id, action="AUTHENTICATE",
            resource_type="auth", resource_id=user_id, success=success,
            ip_address=ip_address, details=details or {}
        ))


_audit_logger = None

def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# ============================================================================
# SECURITY MIDDLEWARE
# ============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not SecurityConfig.RATE_LIMIT_ENABLED:
            return await call_next(request)
        client_id = request.headers.get("X-API-Key") or request.client.host
        allowed, headers = await get_rate_limiter().is_allowed(client_id)
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(headers["limit"])
        response.headers["X-RateLimit-Remaining"] = str(headers["remaining"])
        return response


# ============================================================================
# AUTHENTICATION DEPENDENCIES
# ============================================================================

security_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security_scheme),
    api_key: str = Security(api_key_scheme)
) -> Optional[UserContext]:
    jwt_manager = get_jwt_manager()
    if credentials and credentials.scheme == "Bearer":
        user_context = jwt_manager.get_user_context(credentials.credentials)
        if user_context:
            return user_context
    if api_key and api_key.startswith("sk-"):
        return UserContext(user_id="api_user", username="api_user", email="api@localhost",
                          roles=["viewer"], permissions=[Permission.API_ACCESS.value], api_key_id=api_key[:8])
    return None


async def require_auth(current_user: UserContext = Depends(get_current_user)) -> UserContext:
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated",
                          headers={"WWW-Authenticate": "Bearer"})
    return current_user


# ============================================================================
# USER ROLE ENUM
# ============================================================================

class UserRole(Enum):
    """User roles for role-based access control"""
    ADMIN = "admin"
    WORKFLOW_MANAGER = "workflow_manager"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


# ============================================================================
# DECORATORS
# ============================================================================

def authenticated(required: bool = True):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if required and current_user is None:
                raise HTTPException(status_code=401, detail="Authentication required")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def authorized(permission: Permission):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if current_user is None:
                raise HTTPException(status_code=401, detail="Authentication required")
            if not current_user.has_permission(permission):
                raise HTTPException(status_code=403, detail=f"Permission required: {permission.value}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_permission(permission: str):
    """Decorator to require specific permission for endpoint access.
    
    Works with both sync and async functions.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if current_user is None:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, 
                                  detail="Authentication required")
            if not current_user.has_permission(permission):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
                                  detail=f"Permission required: {permission}")
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if current_user is None:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, 
                                  detail="Authentication required")
            if not current_user.has_permission(permission):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
                                  detail=f"Permission required: {permission}")
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def require_role(role: UserRole):
    """Decorator to require specific role for endpoint access.
    
    Works with both sync and async functions.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if current_user is None:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, 
                                  detail="Authentication required")
            if role.value not in current_user.roles and not current_user.is_superuser:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
                                  detail=f"Role required: {role.value}")
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if current_user is None:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, 
                                  detail="Authentication required")
            if role.value not in current_user.roles and not current_user.is_superuser:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
                                  detail=f"Role required: {role.value}")
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# ============================================================================
# PERMISSION CHECKING FUNCTIONS
# ============================================================================

def check_permission(user: UserContext, permission: str) -> bool:
    """Check if user has specific permission.
    
    Args:
        user: The user context to check
        permission: The permission string to check for
        
    Returns:
        True if user has the permission, False otherwise
    """
    if user is None:
        return False
    return user.has_permission(permission)


def get_current_user_context() -> Optional[UserContext]:
    """Get current user context from request context.
    
    Note: This is a placeholder that returns None. In a real implementation,
    this would extract the user context from the current request context.
    For FastAPI endpoints, use the `get_current_user` dependency instead.
    
    Returns:
        The current user context or None if not available
    """
    # This is a placeholder - in practice, user context should be passed
    # through the request or use FastAPI's dependency injection
    return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_secure_id(prefix: str = "") -> str:
    return f"{prefix}{secrets.token_urlsafe(16)}"


def hash_sensitive_data(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
    """Mask sensitive data for display"""
    if len(data) <= visible_chars * 2:
        return "*" * len(data)
    return data[:visible_chars] + "*" * (len(data) - visible_chars * 2) + data[-visible_chars:]
