"""
OpenEvolve Security Framework - Defense in Depth Implementation

This module provides comprehensive security features for all workflow files.
Author: Security Implementation Team
Version: 2.0.0 - Production Ready
"""

import functools
import hashlib
import json
import logging
import os
import secrets
import time
import re
import sqlite3
import ssl
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
import asyncio
from collections import defaultdict
import jwt
from contextlib import contextmanager

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
    AUDIT_LOG_DB_PATH = os.getenv('AUDIT_LOG_DB_PATH', 'audit_logs.db')
    API_KEY_DB_PATH = os.getenv('API_KEY_DB_PATH', 'sovereign_decomposition.db')
    
    # TLS Configuration
    TLS_ENABLED = os.getenv('TLS_ENABLED', 'false').lower() == 'true'
    TLS_CERT_PATH = os.getenv('TLS_CERT_PATH', 'cert.pem')
    TLS_KEY_PATH = os.getenv('TLS_KEY_PATH', 'key.pem')
    TLS_MIN_VERSION = ssl.TLSVersion.TLSv1_2
    
    # Production Security
    ENFORCE_SECURE_COOKIES = os.getenv('ENFORCE_SECURE_COOKIES', 'false').lower() == 'true'
    SESSION_TIMEOUT_MINUTES = int(os.getenv('SESSION_TIMEOUT_MINUTES', '60'))


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
    
    @classmethod
    def sanitize_sql(cls, value: str) -> str:
        """Sanitize string for SQL - basic SQL injection prevention"""
        if not isinstance(value, str):
            return str(value)
        # Remove null bytes
        value = value.replace('\x00', '')
        # Escape single quotes (though parameterized queries should be used)
        value = value.replace("'", "''")
        return value


# ============================================================================
# API KEY DATABASE MODEL
# ============================================================================

class APIKeyStatus(Enum):
    """Status of API keys"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class APIKeyRecord:
    """API key record from database"""
    id: str
    key_hash: str
    key_prefix: str
    name: str
    user_id: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    permissions: List[str] = field(default_factory=list)
    
    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> 'APIKeyRecord':
        """Create from database row"""
        return cls(
            id=row['id'],
            key_hash=row['key_hash'],
            key_prefix=row['key_prefix'],
            name=row['name'],
            user_id=row['user_id'],
            created_at=datetime.fromisoformat(row['created_at']),
            expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
            last_used=datetime.fromisoformat(row['last_used']) if row['last_used'] else None,
            usage_count=row['usage_count'],
            status=APIKeyStatus(row['status']),
            permissions=json.loads(row['permissions']) if row['permissions'] else []
        )


class APIKeyDatabase:
    """Database-backed API key storage"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or SecurityConfig.API_KEY_DB_PATH
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema for API keys."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL UNIQUE,
                    key_prefix TEXT NOT NULL,
                    name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    last_used TEXT,
                    usage_count INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    permissions TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status)
            """)
    
    def get_key_by_hash(self, key_hash: str) -> Optional[APIKeyRecord]:
        """Get API key by its hash."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM api_keys WHERE key_hash = ?",
                (key_hash,)
            )
            row = cursor.fetchone()
            return APIKeyRecord.from_db_row(row) if row else None
    
    def update_last_used(self, key_id: str):
        """Update last used timestamp and increment usage count."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE api_keys 
                SET last_used = ?, usage_count = usage_count + 1
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), key_id))
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE api_keys SET status = ? WHERE id = ?",
                (APIKeyStatus.REVOKED.value, key_id)
            )
            return cursor.rowcount > 0


# Global API key database instance
_api_key_db = None

def get_api_key_database() -> APIKeyDatabase:
    global _api_key_db
    if _api_key_db is None:
        _api_key_db = APIKeyDatabase()
    return _api_key_db


# ============================================================================
# AUDIT LOGGING - WITH DATABASE PERSISTENCE
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'success': self.success,
            'ip_address': self.ip_address,
            'details': json.dumps(self.details)
        }
    
    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> 'AuditLogEntry':
        return cls(
            timestamp=datetime.fromisoformat(row['timestamp']),
            user_id=row['user_id'],
            action=row['action'],
            resource_type=row['resource_type'],
            resource_id=row['resource_id'],
            success=bool(row['success']),
            ip_address=row['ip_address'],
            details=json.loads(row['details']) if row['details'] else {}
        )


class AuditLogger:
    """Production-grade audit logger with database persistence"""
    
    def __init__(self, db_path: str = None):
        self.enabled = SecurityConfig.AUDIT_LOG_ENABLED
        self.db_path = db_path or SecurityConfig.AUDIT_LOG_DB_PATH
        self._lock = asyncio.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize audit log database."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    ip_address TEXT,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Create indexes for efficient querying
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource_type, resource_id)
            """)
            conn.commit()
            logger.info(f"Audit log database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize audit log database: {e}")
            raise
        finally:
            conn.close()
    
    async def log(self, entry: AuditLogEntry):
        """Log an audit entry to the database."""
        if not self.enabled:
            return
        
        async with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO audit_logs 
                    (timestamp, user_id, action, resource_type, resource_id, success, ip_address, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.timestamp.isoformat(),
                    entry.user_id,
                    entry.action,
                    entry.resource_type,
                    entry.resource_id,
                    1 if entry.success else 0,
                    entry.ip_address,
                    json.dumps(entry.details)
                ))
                conn.commit()
                conn.close()
                
                if entry.success:
                    logger.info(f"AUDIT: {entry.action} {entry.resource_type}/{entry.resource_id} by {entry.user_id}")
                else:
                    logger.warning(f"AUDIT FAIL: {entry.action} {entry.resource_type}/{entry.resource_id} by {entry.user_id}")
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")
                # Don't raise - audit logging should not break functionality
    
    async def log_auth_attempt(self, user_id: str, success: bool, ip_address: str = None, details: Dict = None):
        """Log an authentication attempt."""
        await self.log(AuditLogEntry(
            timestamp=datetime.utcnow(), user_id=user_id, action="AUTHENTICATE",
            resource_type="auth", resource_id=user_id, success=success,
            ip_address=ip_address, details=details or {}
        ))
    
    def query_logs(
        self, 
        user_id: Optional[str] = None, 
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Query audit logs with filters."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            query = "SELECT * FROM audit_logs WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            if action:
                query += " AND action = ?"
                params.append(action)
            if resource_type:
                query += " AND resource_type = ?"
                params.append(resource_type)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [AuditLogEntry.from_db_row(row) for row in rows]
        finally:
            conn.close()
    
    def export_logs(self, filepath: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None):
        """Export audit logs to a JSON file."""
        logs = self.query_logs(start_time=start_time, end_time=end_time, limit=10000)
        with open(filepath, 'w') as f:
            json.dump([log.to_dict() for log in logs], f, indent=2)
        logger.info(f"Exported {len(logs)} audit logs to {filepath}")


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
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
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


class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """Redirect HTTP to HTTPS in production"""
    async def dispatch(self, request: Request, call_next):
        if SecurityConfig.ENFORCE_SECURE_COOKIES and request.url.scheme == "http":
            url = request.url.replace(scheme="https")
            raise HTTPException(
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                headers={"Location": str(url)}
            )
        return await call_next(request)


# ============================================================================
# AUTHENTICATION DEPENDENCIES - WITH DATABASE VALIDATION
# ============================================================================

security_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security_scheme),
    api_key: str = Security(api_key_scheme)
) -> Optional[UserContext]:
    """
    Authenticate user via JWT token or API key.
    API keys are validated against the database with expiration/revocation checks.
    """
    jwt_manager = get_jwt_manager()
    
    # Try JWT authentication first
    if credentials and credentials.scheme == "Bearer":
        user_context = jwt_manager.get_user_context(credentials.credentials)
        if user_context:
            return user_context
    
    # Try API key authentication with database validation
    if api_key:
        # Validate against database
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        db = get_api_key_database()
        key_record = db.get_key_by_hash(key_hash)
        
        if key_record:
            # Check if key is active
            if key_record.status != APIKeyStatus.ACTIVE:
                logger.warning(f"API key {key_record.id} is not active (status: {key_record.status.value})")
                return None
            
            # Check expiration
            if key_record.expires_at and key_record.expires_at < datetime.utcnow():
                logger.warning(f"API key {key_record.id} has expired")
                return None
            
            # Update usage statistics
            db.update_last_used(key_record.id)
            
            return UserContext(
                user_id=key_record.user_id,
                username=f"api_user_{key_record.key_prefix}",
                email="",
                roles=["api_user"],
                permissions=key_record.permissions or [Permission.API_ACCESS.value],
                api_key_id=key_record.id
            )
    
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
# TLS/SSL CONFIGURATION
# ============================================================================

def create_ssl_context(
    cert_path: str = None,
    key_path: str = None,
    min_version: ssl.TLSVersion = None
) -> ssl.SSLContext:
    """Create a secure SSL context for HTTPS.
    
    Args:
        cert_path: Path to SSL certificate file
        key_path: Path to SSL private key file
        min_version: Minimum TLS version to accept
        
    Returns:
        Configured SSLContext
        
    Raises:
        FileNotFoundError: If certificate or key files don't exist
        ssl.SSLError: If there's an error loading the certificates
    """
    cert_path = cert_path or SecurityConfig.TLS_CERT_PATH
    key_path = key_path or SecurityConfig.TLS_KEY_PATH
    min_version = min_version or SecurityConfig.TLS_MIN_VERSION
    
    # Verify files exist
    if not os.path.exists(cert_path):
        raise FileNotFoundError(f"SSL certificate not found: {cert_path}")
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"SSL private key not found: {key_path}")
    
    # Create SSL context with secure defaults
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = min_version
    context.load_cert_chain(cert_path, key_path)
    
    # Security hardening
    context.options |= ssl.OP_NO_COMPRESSION  # Disable compression (CRIME attack)
    context.options |= ssl.OP_SINGLE_DH_USE  # Use ephemeral DH keys
    context.options |= ssl.OP_SINGLE_ECDH_USE  # Use ephemeral ECDH keys
    
    # Use secure cipher suites
    context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
    
    logger.info(f"SSL context created with min_version={min_version.name}")
    return context


def get_tls_config() -> Dict[str, Any]:
    """Get TLS configuration for the application.
    
    Returns:
        Dictionary with TLS configuration or None if TLS is disabled
    """
    if not SecurityConfig.TLS_ENABLED:
        return None
    
    try:
        ssl_context = create_ssl_context()
        return {
            'ssl_context': ssl_context,
            'cert_path': SecurityConfig.TLS_CERT_PATH,
            'key_path': SecurityConfig.TLS_KEY_PATH,
            'enabled': True
        }
    except Exception as e:
        logger.error(f"Failed to create TLS configuration: {e}")
        return {'enabled': False, 'error': str(e)}


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


# ============================================================================
# SECURITY INITIALIZATION
# ============================================================================

def initialize_security() -> Dict[str, Any]:
    """Initialize all security components.
    
    Returns:
        Dictionary with initialization status for each component
    """
    status = {
        'jwt': False,
        'audit_log': False,
        'api_key_db': False,
        'rate_limiter': False,
        'tls': False
    }
    
    try:
        # Initialize JWT manager
        get_jwt_manager()
        status['jwt'] = True
        logger.info("JWT manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize JWT manager: {e}")
    
    try:
        # Initialize audit logger
        get_audit_logger()
        status['audit_log'] = True
        logger.info("Audit logger initialized")
    except Exception as e:
        logger.error(f"Failed to initialize audit logger: {e}")
    
    try:
        # Initialize API key database
        get_api_key_database()
        status['api_key_db'] = True
        logger.info("API key database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize API key database: {e}")
    
    try:
        # Initialize rate limiter
        get_rate_limiter()
        status['rate_limiter'] = True
        logger.info("Rate limiter initialized")
    except Exception as e:
        logger.error(f"Failed to initialize rate limiter: {e}")
    
    try:
        # Initialize TLS if enabled
        if SecurityConfig.TLS_ENABLED:
            get_tls_config()
            status['tls'] = True
            logger.info("TLS configuration initialized")
        else:
            status['tls'] = None  # Disabled, not an error
    except Exception as e:
        logger.error(f"Failed to initialize TLS: {e}")
    
    return status
