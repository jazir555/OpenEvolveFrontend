"""
Sovereign-Grade Problem Decomposition System - Authentication and Authorization System
Implements role-based access controls for workflow management and sensitive operations.
"""
from __future__ import annotations


import hashlib
import os
import secrets
import jwt
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import logging
from contextlib import contextmanager
import sqlite3
import json

from sovereign_data_models import generate_id
from sovereign_persistence import SovereignDatabase

logger = logging.getLogger(__name__)


class Role(Enum):
    """User roles in the system"""
    ADMIN = "admin"
    WORKFLOW_MANAGER = "workflow_manager"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Permission(Enum):
    """Specific permissions in the system"""
    # Problem management permissions
    CREATE_PROBLEM = "create_problem"
    READ_PROBLEM = "read_problem"
    UPDATE_PROBLEM = "update_problem"
    DELETE_PROBLEM = "delete_problem"
    
    # Decomposition plan permissions
    CREATE_PLAN = "create_plan"
    READ_PLAN = "read_plan"
    UPDATE_PLAN = "update_plan"
    DELETE_PLAN = "delete_plan"
    
    # Solution management permissions
    CREATE_SOLUTION = "create_solution"
    READ_SOLUTION = "read_solution"
    UPDATE_SOLUTION = "update_solution"
    DELETE_SOLUTION = "delete_solution"
    
    # Team management permissions
    MANAGE_TEAMS = "manage_teams"
    RUN_GAUNTLETS = "run_gauntlets"
    
    # System administration permissions
    ADMIN_ACCESS = "admin_access"
    MANAGE_USERS = "manage_users"


class User:
    """User model for authentication system"""
    
    def __init__(
        self, 
        user_id: str, 
        username: str, 
        email: str, 
        password_hash: str,
        roles: List[Role],
        permissions: Optional[List[Permission]] = None,
        created_at: datetime = None,
        last_login: Optional[datetime] = None,
        is_active: bool = True
    ):
        self.id = user_id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.roles = roles if roles else []
        self.permissions = permissions if permissions else []
        self.created_at = created_at or datetime.now()
        self.last_login = last_login
        self.is_active = is_active


class AuditLog:
    """Audit log entry model"""
    
    def __init__(
        self,
        log_id: str,
        user_id: str,
        operation: str,
        resource: str,
        resource_id: str,
        timestamp: datetime,
        success: bool,
        details: Dict[str, Any]
    ):
        self.id = log_id
        self.user_id = user_id
        self.operation = operation
        self.resource = resource
        self.resource_id = resource_id
        self.timestamp = timestamp
        self.success = success
        self.details = details
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'operation': self.operation,
            'resource': self.resource,
            'resource_id': self.resource_id,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'details': self.details
        }


class AuthenticationSystem:
    """Authentication system with JWT token management"""
    
    def __init__(self, secret_key: Optional[str] = None, db_path: str = "sovereign_decomposition.db"):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.db_path = db_path
        self.db = SovereignDatabase(db_path)
        self._initialize_auth_tables()
    
    def _initialize_auth_tables(self):
        """Initialize authentication-related database tables."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    roles TEXT NOT NULL,
                    permissions TEXT,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    is_active INTEGER DEFAULT 1
                )
            """)
            
            # API keys table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    api_key TEXT UNIQUE NOT NULL,
                    name TEXT,
                    permissions TEXT,
                    created_at TEXT NOT NULL,
                    last_used TEXT,
                    expires_at TEXT,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Audit logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    resource_id TEXT,
                    timestamp TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    details TEXT NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_username 
                ON users(username)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_user_id 
                ON api_keys(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp 
                ON audit_logs(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_logs_user 
                ON audit_logs(user_id)
            """)
    
    def hash_password(self, password: str) -> str:
        """Hash a password using a salt."""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return f"{salt}:{pwd_hash.hex()}"
    
    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, stored_pwd_hash = stored_hash.split(':')
            pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
            return pwd_hash.hex() == stored_pwd_hash
        except ValueError:
            return False
    
    def create_user(
        self, 
        username: str, 
        email: str, 
        password: str, 
        roles: List[Role],
        permissions: Optional[List[Permission]] = None
    ) -> User:
        """Create a new user account."""
        user_id = generate_id("user")
        password_hash = self.hash_password(password)
        created_at = datetime.now()
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles,
            permissions=permissions or [],
            created_at=created_at
        )
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (id, username, email, password_hash, roles, permissions, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user.id, user.username, user.email, user.password_hash,
                json.dumps([r.value for r in user.roles]),
                json.dumps([p.value for p in user.permissions]) if user.permissions else None,
                user.created_at.isoformat(), user.is_active
            ))
        
        # Log user creation
        self.log_audit(user.id, "CREATE_USER", "user", user.id, True, {
            "username": username,
            "email": email,
            "roles": [r.value for r in roles],
            "permissions": [p.value for p in (permissions or [])]
        })
        
        return user
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user and return user object if successful."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM users WHERE username = ? AND is_active = 1
            """, (username,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            data = dict(row)
            
            # Verify password
            if not self.verify_password(password, data['password_hash']):
                # Log failed authentication
                self.log_audit("unknown", "AUTHENTICATE", "user", data['id'], False, {
                    "username": username,
                    "result": "invalid_password"
                })
                return None
            
            # Update last login
            cursor.execute("""
                UPDATE users SET last_login = ? WHERE id = ?
            """, (datetime.now().isoformat(), data['id']))
            
            # Create user object
            user = User(
                user_id=data['id'],
                username=data['username'],
                email=data['email'],
                password_hash=data['password_hash'],
                roles=[Role(r) for r in json.loads(data['roles'])],
                permissions=[Permission(p) for p in json.loads(data['permissions'])] if data['permissions'] else [],
                created_at=datetime.fromisoformat(data['created_at']),
                last_login=datetime.fromisoformat(data['last_login']) if data['last_login'] else None,
                is_active=bool(data['is_active'])
            )
            
            # Log successful authentication
            self.log_audit(user.id, "AUTHENTICATE", "user", user.id, True, {
                "username": username
            })
            
            return user
    
    def generate_jwt_token(self, user_id: str, expires_in: int = 3600) -> str:
        """Generate a JWT token for the authenticated user."""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return user_id if valid."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload.get('user_id')
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Retrieve a user by ID."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM users WHERE id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            data = dict(row)
            return User(
                user_id=data['id'],
                username=data['username'],
                email=data['email'],
                password_hash=data['password_hash'],
                roles=[Role(r) for r in json.loads(data['roles'])],
                permissions=[Permission(p) for p in json.loads(data['permissions'])] if data['permissions'] else [],
                created_at=datetime.fromisoformat(data['created_at']),
                last_login=datetime.fromisoformat(data['last_login']) if data['last_login'] else None,
                is_active=bool(data['is_active'])
            )


class AuthorizationSystem:
    """Authorization system for role-based access control"""
    
    def __init__(self, auth_system: AuthenticationSystem):
        self.auth_system = auth_system
        self.role_permissions = self._initialize_role_permissions()
    
    def _initialize_role_permissions(self) -> Dict[Role, List[Permission]]:
        """Initialize default permissions for each role."""
        return {
            Role.ADMIN: list(Permission),  # Admin has all permissions
            Role.WORKFLOW_MANAGER: [
                Permission.CREATE_PROBLEM,
                Permission.READ_PROBLEM,
                Permission.UPDATE_PROBLEM,
                Permission.CREATE_PLAN,
                Permission.READ_PLAN,
                Permission.UPDATE_PLAN,
                Permission.CREATE_SOLUTION,
                Permission.READ_SOLUTION,
                Permission.UPDATE_SOLUTION,
                Permission.MANAGE_TEAMS,
                Permission.RUN_GAUNTLETS
            ],
            Role.ANALYST: [
                Permission.CREATE_PROBLEM,
                Permission.READ_PROBLEM,
                Permission.READ_PLAN,
                Permission.CREATE_SOLUTION,
                Permission.READ_SOLUTION,
                Permission.RUN_GAUNTLETS
            ],
            Role.VIEWER: [
                Permission.READ_PROBLEM,
                Permission.READ_PLAN,
                Permission.READ_SOLUTION
            ]
        }
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission."""
        # Check direct permissions
        if permission in user.permissions:
            return True
        
        # Check role-based permissions
        for role in user.roles:
            if permission in self.role_permissions.get(role, []):
                return True
        
        return False
    
    def check_resource_access(self, user: User, resource_type: str, resource_id: str, operation: str) -> bool:
        """Check if user has access to a specific resource."""
        permission_map = {
            ('problem', 'create'): Permission.CREATE_PROBLEM,
            ('problem', 'read'): Permission.READ_PROBLEM,
            ('problem', 'update'): Permission.UPDATE_PROBLEM,
            ('problem', 'delete'): Permission.DELETE_PROBLEM,
            ('plan', 'create'): Permission.CREATE_PLAN,
            ('plan', 'read'): Permission.READ_PLAN,
            ('plan', 'update'): Permission.UPDATE_PLAN,
            ('plan', 'delete'): Permission.DELETE_PLAN,
            ('solution', 'create'): Permission.CREATE_SOLUTION,
            ('solution', 'read'): Permission.READ_SOLUTION,
            ('solution', 'update'): Permission.UPDATE_SOLUTION,
            ('solution', 'delete'): Permission.DELETE_SOLUTION,
        }
        
        permission = permission_map.get((resource_type, operation))
        if not permission:
            return False
        
        return self.check_permission(user, permission)


class APIKeyManager:
    """API key management system"""
    
    def __init__(self, auth_system: AuthenticationSystem):
        self.auth_system = auth_system
        self.db = auth_system.db
    
    def create_api_key(self, user_id: str, name: str, permissions: Optional[List[Permission]] = None) -> str:
        """Create a new API key for a user."""
        api_key_id = generate_id("apikey")
        api_key = f"sk-{secrets.token_urlsafe(32)}"
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_keys (id, user_id, api_key, name, permissions, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                api_key_id, user_id, api_key, name,
                json.dumps([p.value for p in permissions]) if permissions else None,
                datetime.now().isoformat(), True
            ))
        
        # Log API key creation
        self.auth_system.log_audit(user_id, "CREATE_API_KEY", "api_key", api_key_id, True, {
            "api_key_id": api_key_id,
            "name": name,
            "permissions": [p.value for p in (permissions or [])]
        })
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key and return user info."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM api_keys 
                WHERE api_key = ? AND is_active = 1
            """, (api_key,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            data = dict(row)
            
            # Update last used timestamp
            cursor.execute("""
                UPDATE api_keys SET last_used = ? WHERE id = ?
            """, (datetime.now().isoformat(), data['id']))
            
            permissions = json.loads(data['permissions']) if data['permissions'] else None
            if permissions:
                permissions = [Permission(p) for p in permissions]
            
            return {
                'user_id': data['user_id'],
                'api_key_id': data['id'],
                'name': data['name'],
                'permissions': permissions,
                'expires_at': datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None
            }
    
    def revoke_api_key(self, api_key_id: str, user_id: str) -> bool:
        """Revoke an API key."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE api_keys 
                SET is_active = 0 
                WHERE id = ? AND user_id = ?
            """, (api_key_id, user_id))
            
            success = cursor.rowcount > 0
            
            if success:
                # Log API key revocation
                self.auth_system.log_audit(user_id, "REVOKE_API_KEY", "api_key", api_key_id, True, {
                    "api_key_id": api_key_id
                })
            
            return success


class AuditSystem:
    """Audit logging system for tracking system activities"""
    
    def __init__(self, auth_system: AuthenticationSystem):
        self.auth_system = auth_system
        self.db = auth_system.db
    
    def log_audit(
        self, 
        user_id: str, 
        operation: str, 
        resource: str, 
        resource_id: str, 
        success: bool, 
        details: Dict[str, Any]
    ):
        """Log an audit event."""
        log_id = generate_id("audit")
        
        audit_log = AuditLog(
            log_id=log_id,
            user_id=user_id,
            operation=operation,
            resource=resource,
            resource_id=resource_id,
            timestamp=datetime.now(),
            success=success,
            details=details
        )
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (id, user_id, operation, resource, resource_id, timestamp, success, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_log.id, audit_log.user_id, audit_log.operation,
                audit_log.resource, audit_log.resource_id, audit_log.timestamp.isoformat(),
                audit_log.success, json.dumps(audit_log.details)
            ))
    
    def get_audit_logs(
        self, 
        user_id: Optional[str] = None, 
        operation: Optional[str] = None, 
        resource: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Retrieve audit logs with optional filters."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM audit_logs 
                WHERE 1=1
            """
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if operation:
                query += " AND operation = ?"
                params.append(operation)
            
            if resource:
                query += " AND resource = ?"
                params.append(resource)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            logs = []
            for row in cursor.fetchall():
                data = dict(row)
                logs.append(AuditLog(
                    log_id=data['id'],
                    user_id=data['user_id'],
                    operation=data['operation'],
                    resource=data['resource'],
                    resource_id=data['resource_id'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    success=bool(data['success']),
                    details=json.loads(data['details'])
                ))
            
            return logs


class SecureAPI:
    """Secure API wrapper for protecting endpoints"""
    
    def __init__(self, auth_system: AuthenticationSystem, authz_system: AuthorizationSystem):
        self.auth_system = auth_system
        self.authz_system = authz_system
        self.api_key_manager = APIKeyManager(auth_system)
        self.audit_system = AuditSystem(auth_system)
    
    def authenticate_request(self, token: Optional[str] = None, api_key: Optional[str] = None) -> Optional[User]:
        """Authenticate a request using JWT token or API key."""
        user = None
        
        if token:
            # JWT token authentication
            user_id = self.auth_system.verify_jwt_token(token)
            if user_id:
                user = self.auth_system.get_user_by_id(user_id)
        elif api_key:
            # API key authentication
            api_key_info = self.api_key_manager.verify_api_key(api_key)
            if api_key_info:
                user = self.auth_system.get_user_by_id(api_key_info['user_id'])
        
        return user
    
    def authorize_request(self, user: User, resource_type: str, resource_id: str, operation: str) -> bool:
        """Check if user is authorized for the requested operation."""
        return self.authz_system.check_resource_access(user, resource_type, resource_id, operation)
    
    def log_request(self, user_id: str, operation: str, resource: str, resource_id: str, success: bool, details: dict):
        """Log the API request for audit purposes."""
        self.audit_system.log_audit(user_id, operation, resource, resource_id, success, details)
    
    def validate_input(self, data: Any, schema: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate input data against a schema."""
        errors = []
        
        # Basic validation based on provided schema
        if 'required' in schema:
            for field in schema['required']:
                if field not in data or data[field] is None or data[field] == "":
                    errors.append(f"Missing required field: {field}")
        
        # Additional validation can be implemented here based on field types, lengths, patterns, etc.
        if 'fields' in schema:
            for field, field_spec in schema['fields'].items():
                if field in data:
                    value = data[field]
                    
                    # Type validation
                    if 'type' in field_spec and not isinstance(value, field_spec['type']):
                        errors.append(f"Field {field} must be of type {field_spec['type'].__name__}")
                    
                    # Length validation
                    if 'min_length' in field_spec and len(str(value)) < field_spec['min_length']:
                        errors.append(f"Field {field} must be at least {field_spec['min_length']} characters")
                    
                    if 'max_length' in field_spec and len(str(value)) > field_spec['max_length']:
                        errors.append(f"Field {field} must be no more than {field_spec['max_length']} characters")
        
        return errors


# Global instances for use throughout the application
_auth_system = None
_authz_system = None
_secure_api = None


def get_auth_system() -> AuthenticationSystem:
    """Get the authentication system instance."""
    global _auth_system
    if _auth_system is None:
        _auth_system = AuthenticationSystem()
    return _auth_system


def get_authz_system() -> AuthorizationSystem:
    """Get the authorization system instance."""
    global _authz_system
    if _authz_system is None:
        _auth_system = get_auth_system()  # Initialize auth system first
        _authz_system = AuthorizationSystem(_auth_system)
    return _authz_system


def get_secure_api() -> SecureAPI:
    """Get the secure API wrapper instance."""
    global _secure_api
    if _secure_api is None:
        _auth_system = get_auth_system()
        _authz_system = get_authz_system()
        _secure_api = SecureAPI(_auth_system, _authz_system)
    return _secure_api


# Example usage
if __name__ == "__main__":
    # Initialize systems
    auth = get_auth_system()
    authz = get_authz_system()
    secure_api = get_secure_api()
    
    # Create an admin user
    admin_password = os.environ.get('ADMIN_PASSWORD')
    if not admin_password:
        raise ValueError("ADMIN_PASSWORD environment variable must be set")
    
    admin_user = auth.create_user(
        username="admin",
        email="admin@example.com",
        password=admin_password,
        roles=[Role.ADMIN]
    )
    
    print(f"Created admin user: {admin_user.username}")
    
    # Authenticate as admin
    authenticated_user = auth.authenticate("admin", admin_password)
    if authenticated_user:
        print(f"Successfully authenticated user: {authenticated_user.username}")
        
        # Generate JWT token
        token = auth.generate_jwt_token(authenticated_user.id)
        print(f"Generated JWT token: {token[:20]}...")
        
        # Check permission
        has_permission = authz.check_permission(authenticated_user, Permission.MANAGE_USERS)
        print(f"Admin has MANAGE_USERS permission: {has_permission}")
    
    # Create an API key
    api_key = secure_api.api_key_manager.create_api_key(
        admin_user.id,
        "test-key",
        [Permission.CREATE_PROBLEM, Permission.READ_PROBLEM]
    )
    print(f"Created API key: {api_key[:20]}...")