"""
Production-Ready RBAC (Role-Based Access Control) System

This module provides a comprehensive, persistent RBAC system with:
- Persistent storage via sovereign_persistence.py
- Multiple authentication backend integration options
- Permission checking decorators and middleware
- User authentication and authorization
- Configurable and persistent role definitions
- Comprehensive error handling with specific exceptions
- Type hints throughout
- Production-ready logging
- Streamlit UI integration

Author: Enhanced from basic stub
Version: 2.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
)

import streamlit as st

# Optional imports for authentication backends
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# Import sovereign persistence if available
try:
    from sovereign_persistence import SovereignDatabase
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False

# ============================================================================
# EXCEPTIONS
# ============================================================================

class RBACError(Exception):
    """Base exception for RBAC errors."""
    pass


class AuthenticationError(RBACError):
    """Authentication failed."""
    pass


class AuthorizationError(RBACError):
    """Authorization failed (insufficient permissions)."""
    pass


class UserNotFoundError(RBACError):
    """User not found in the system."""
    pass


class RoleNotFoundError(RBACError):
    """Role not found in the system."""
    pass


class PermissionNotFoundError(RBACError):
    """Permission not found in the system."""
    pass


class InvalidConfigurationError(RBACError):
    """Invalid RBAC configuration."""
    pass


class PersistenceError(RBACError):
    """Persistence layer error."""
    pass


class BackendNotAvailableError(RBACError):
    """Requested authentication backend is not available."""
    pass


# ============================================================================
# ENUMS AND DATA MODELS
# ============================================================================

class AuthBackend(Enum):
    """Supported authentication backends."""
    NATIVE = "native"  # Built-in database authentication
    JWT = "jwt"  # JWT token-based
    OAUTH = "oauth"  # OAuth2/OIDC
    LDAP = "ldap"  # LDAP/Active Directory
    SAML = "saml"  # SAML SSO
    API_KEY = "api_key"  # API key authentication


class Permission(Enum):
    """System permissions."""
    # User management
    CREATE_USER = "create_user"
    READ_USER = "read_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"

    # Content management
    CREATE_CONTENT = "create_content"
    READ_CONTENT = "read_content"
    UPDATE_CONTENT = "update_content"
    DELETE_CONTENT = "delete_content"
    PUBLISH_CONTENT = "publish_content"

    # Project management
    CREATE_PROJECT = "create_project"
    READ_PROJECT = "read_project"
    UPDATE_PROJECT = "update_project"
    DELETE_PROJECT = "delete_project"
    MANAGE_PROJECT_MEMBERS = "manage_project_members"

    # System administration
    SYSTEM_ADMIN = "system_admin"
    VIEW_LOGS = "view_logs"
    MANAGE_SYSTEM = "manage_system"

    # Analytics and reporting
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_DATA = "export_data"

    # API access
    API_ACCESS = "api_access"
    API_WRITE = "api_write"


@dataclass
class Role:
    """Role definition with permissions."""
    name: str
    description: str
    permissions: Set[str]
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'permissions': list(self.permissions),
            'is_system_role': self.is_system_role,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Role':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            permissions=set(data.get('permissions', [])),
            is_system_role=data.get('is_system_role', False),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.utcnow(),
            metadata=data.get('metadata', {})
        )


@dataclass
class User:
    """User account information."""
    user_id: str
    username: str
    email: str
    full_name: Optional[str] = None
    password_hash: Optional[str] = None  # Only for native auth
    role_names: Set[str] = field(default_factory=set)
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role_names': list(self.role_names),
            'is_active': self.is_active,
            'is_superuser': self.is_superuser,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create from dictionary."""
        return cls(
            user_id=data['user_id'],
            username=data['username'],
            email=data['email'],
            full_name=data.get('full_name'),
            password_hash=data.get('password_hash'),
            role_names=set(data.get('role_names', [])),
            is_active=data.get('is_active', True),
            is_superuser=data.get('is_superuser', False),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.utcnow(),
            last_login=datetime.fromisoformat(data['last_login']) if data.get('last_login') else None,
            metadata=data.get('metadata', {})
        )


@dataclass
class AuditLog:
    """Audit log entry."""
    log_id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: Optional[str]
    success: bool
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'log_id': self.log_id,
            'user_id': self.user_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'success': self.success,
            'timestamp': self.timestamp.isoformat(),
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'details': self.details
        }


# ============================================================================
# RBAC STORAGE LAYER
# ============================================================================

class RBACStorage:
    """
    Storage backend for RBAC data.

    Supports multiple storage backends:
    - SQLite/PostgreSQL via sovereign_persistence
    - File-based JSON storage (fallback)
    - Streamlit session state (development only)
    """

    def __init__(
        self,
        backend: str = "auto",
        db_path: Optional[str] = None,
        file_path: Optional[str] = None,
        use_session_state: bool = False
    ):
        """
        Initialize RBAC storage.

        Args:
            backend: Storage backend ('auto', 'database', 'file', 'session')
            db_path: Database path (for database backend)
            file_path: JSON file path (for file backend)
            use_session_state: Use Streamlit session state (development only)
        """
        self.logger = logging.getLogger(__name__)
        self.backend_type = backend
        self.use_session_state = use_session_state

        # Try to initialize database backend
        if backend == "auto":
            if PERSISTENCE_AVAILABLE:
                self.backend_type = "database"
            elif use_session_state:
                self.backend_type = "session"
            else:
                self.backend_type = "file"

        if self.backend_type == "database" and PERSISTENCE_AVAILABLE:
            self.db = SovereignDatabase(
                backend='sqlite',
                database_path=db_path or 'rbac_system.db'
            )
            self._init_database_tables()
        elif self.backend_type == "session":
            self._init_session_state()
        else:
            self.file_path = file_path or 'rbac_data.json'
            self._load_from_file()

    def _init_database_tables(self):
        """Initialize database tables."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS rbac_users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        full_name TEXT,
                        password_hash TEXT,
                        role_names TEXT NOT NULL,
                        is_active INTEGER DEFAULT 1,
                        is_superuser INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_login TEXT,
                        metadata TEXT
                    )
                """)

                # Roles table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS rbac_roles (
                        name TEXT PRIMARY KEY,
                        description TEXT NOT NULL,
                        permissions TEXT NOT NULL,
                        is_system_role INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT
                    )
                """)

                # Audit logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS rbac_audit_logs (
                        log_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        resource_type TEXT NOT NULL,
                        resource_id TEXT,
                        success INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        ip_address TEXT,
                        user_agent TEXT,
                        details TEXT
                    )
                """)

                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON rbac_users(username)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON rbac_users(email)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON rbac_audit_logs(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON rbac_audit_logs(timestamp)")

                conn.commit()
                self.logger.info("Database tables initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize database tables: {e}")
            raise PersistenceError(f"Database initialization failed: {e}")

    def _init_session_state(self):
        """Initialize Streamlit session state storage."""
        if 'rbac_users' not in st.session_state:
            st.session_state.rbac_users = {}
        if 'rbac_roles' not in st.session_state:
            st.session_state.rbac_roles = {}
        if 'rbac_audit_logs' not in st.session_state:
            st.session_state.rbac_audit_logs = []

    def _load_from_file(self):
        """Load data from JSON file."""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    content = f.read()
                    if content.strip():
                        data = json.loads(content)
                    else:
                        # Empty file, initialize it
                        data = {
                            'users': {},
                            'roles': self._get_default_roles_dict(),
                            'audit_logs': []
                        }
                        self._save_to_file(data)
            else:
                # Initialize new file
                data = {
                    'users': {},
                    'roles': self._get_default_roles_dict(),
                    'audit_logs': []
                }
                self._save_to_file(data)
        except Exception as e:
            self.logger.error(f"Failed to load from file: {e}")
            # Initialize with default data on error
            data = {
                'users': {},
                'roles': self._get_default_roles_dict(),
                'audit_logs': []
            }
            self._save_to_file(data)

    def _save_to_file(self, data: Optional[Dict[str, Any]] = None):
        """Save data to JSON file."""
        try:
            if data is None:
                # Load current data
                data = {
                    'users': {},
                    'roles': self._get_default_roles_dict(),
                    'audit_logs': []
                }

            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save to file: {e}")
            raise PersistenceError(f"File save failed: {e}")

    def _get_default_roles_dict(self) -> Dict[str, Any]:
        """Get default roles as dictionary."""
        return {
            'admin': {
                'name': 'admin',
                'description': 'Full system access',
                'permissions': [p.value for p in Permission],
                'is_system_role': True,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'metadata': {}
            },
            'editor': {
                'name': 'editor',
                'description': 'Can edit and manage content',
                'permissions': [
                    Permission.READ_CONTENT.value,
                    Permission.CREATE_CONTENT.value,
                    Permission.UPDATE_CONTENT.value,
                    Permission.READ_PROJECT.value,
                ],
                'is_system_role': True,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'metadata': {}
            },
            'viewer': {
                'name': 'viewer',
                'description': 'Read-only access',
                'permissions': [
                    Permission.READ_CONTENT.value,
                    Permission.READ_PROJECT.value,
                ],
                'is_system_role': True,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'metadata': {}
            }
        }

    # User operations
    def create_user(self, user: User) -> bool:
        """Create a new user."""
        try:
            if self.backend_type == "database":
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO rbac_users
                        (user_id, username, email, full_name, password_hash, role_names,
                         is_active, is_superuser, created_at, updated_at, last_login, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user.user_id, user.username, user.email, user.full_name,
                        user.password_hash, json.dumps(list(user.role_names)),
                        int(user.is_active), int(user.is_superuser),
                        user.created_at.isoformat(), user.updated_at.isoformat(),
                        user.last_login.isoformat() if user.last_login else None,
                        json.dumps(user.metadata)
                    ))
                    conn.commit()

            elif self.backend_type == "session":
                user_dict = user.to_dict()
                # Include password hash for authentication (it's already hashed)
                user_dict['password_hash'] = user.password_hash
                st.session_state.rbac_users[user.user_id] = user_dict

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                user_dict = user.to_dict()
                # Include password hash for authentication (it's already hashed)
                user_dict['password_hash'] = user.password_hash
                data['users'][user.user_id] = user_dict
                self._save_to_file(data)

            self.logger.info(f"Created user: {user.username}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            return False

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        try:
            if self.backend_type == "database":
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM rbac_users WHERE user_id = ?", (user_id,))
                    row = cursor.fetchone()
                    if row:
                        return self._row_to_user(dict(row))

            elif self.backend_type == "session":
                data = st.session_state.rbac_users.get(user_id)
                if data:
                    return User.from_dict(data)

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                user_data = data['users'].get(user_id)
                if user_data:
                    return User.from_dict(user_data)

            return None

        except Exception as e:
            self.logger.error(f"Failed to get user: {e}")
            return None

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            if self.backend_type == "database":
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM rbac_users WHERE username = ?", (username,))
                    row = cursor.fetchone()
                    if row:
                        return self._row_to_user(dict(row))

            elif self.backend_type == "session":
                for user_data in st.session_state.rbac_users.values():
                    if user_data.get('username') == username:
                        return User.from_dict(user_data)

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                for user_data in data['users'].values():
                    if user_data.get('username') == username:
                        return User.from_dict(user_data)

            return None

        except Exception as e:
            self.logger.error(f"Failed to get user by username: {e}")
            return None

    def _row_to_user(self, row: Dict[str, Any]) -> User:
        """Convert database row to User object."""
        return User(
            user_id=row['user_id'],
            username=row['username'],
            email=row['email'],
            full_name=row.get('full_name'),
            password_hash=row.get('password_hash'),
            role_names=set(json.loads(row['role_names'])),
            is_active=bool(row['is_active']),
            is_superuser=bool(row['is_superuser']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            last_login=datetime.fromisoformat(row['last_login']) if row.get('last_login') else None,
            metadata=json.loads(row['metadata']) if row.get('metadata') else {}
        )

    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user information."""
        try:
            if self.backend_type == "database":
                set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
                values = list(updates.values())
                values.append(user_id)

                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"UPDATE rbac_users SET {set_clause} WHERE user_id = ?", values)
                    conn.commit()
                    return cursor.rowcount > 0

            elif self.backend_type == "session":
                if user_id in st.session_state.rbac_users:
                    st.session_state.rbac_users[user_id].update(updates)
                    return True

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                if user_id in data['users']:
                    data['users'][user_id].update(updates)
                    self._save_to_file(data)
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to update user: {e}")
            return False

    def list_users(self) -> List[User]:
        """List all users."""
        try:
            users = []

            if self.backend_type == "database":
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM rbac_users")
                    for row in cursor.fetchall():
                        users.append(self._row_to_user(dict(row)))

            elif self.backend_type == "session":
                for user_data in st.session_state.rbac_users.values():
                    users.append(User.from_dict(user_data))

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                for user_data in data['users'].values():
                    users.append(User.from_dict(user_data))

            return users

        except Exception as e:
            self.logger.error(f"Failed to list users: {e}")
            return []

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        try:
            if self.backend_type == "database":
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM rbac_users WHERE user_id = ?", (user_id,))
                    conn.commit()
                    return cursor.rowcount > 0

            elif self.backend_type == "session":
                if user_id in st.session_state.rbac_users:
                    del st.session_state.rbac_users[user_id]
                    return True

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                if user_id in data['users']:
                    del data['users'][user_id]
                    self._save_to_file(data)
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to delete user: {e}")
            return False

    # Role operations
    def create_role(self, role: Role) -> bool:
        """Create a new role."""
        try:
            if self.backend_type == "database":
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO rbac_roles
                        (name, description, permissions, is_system_role, created_at, updated_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        role.name, role.description, json.dumps(list(role.permissions)),
                        int(role.is_system_role), role.created_at.isoformat(),
                        role.updated_at.isoformat(), json.dumps(role.metadata)
                    ))
                    conn.commit()

            elif self.backend_type == "session":
                st.session_state.rbac_roles[role.name] = role.to_dict()

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                data['roles'][role.name] = role.to_dict()
                self._save_to_file(data)

            self.logger.info(f"Created role: {role.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create role: {e}")
            return False

    def get_role(self, role_name: str) -> Optional[Role]:
        """Get role by name."""
        try:
            if self.backend_type == "database":
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM rbac_roles WHERE name = ?", (role_name,))
                    row = cursor.fetchone()
                    if row:
                        return self._row_to_role(dict(row))

            elif self.backend_type == "session":
                role_data = st.session_state.rbac_roles.get(role_name)
                if role_data:
                    return Role.from_dict(role_data)

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                role_data = data['roles'].get(role_name)
                if role_data:
                    return Role.from_dict(role_data)

            return None

        except Exception as e:
            self.logger.error(f"Failed to get role: {e}")
            return None

    def _row_to_role(self, row: Dict[str, Any]) -> Role:
        """Convert database row to Role object."""
        return Role(
            name=row['name'],
            description=row['description'],
            permissions=set(json.loads(row['permissions'])),
            is_system_role=bool(row['is_system_role']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            metadata=json.loads(row['metadata']) if row.get('metadata') else {}
        )

    def list_roles(self) -> List[Role]:
        """List all roles."""
        try:
            roles = []

            if self.backend_type == "database":
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM rbac_roles")
                    for row in cursor.fetchall():
                        roles.append(self._row_to_role(dict(row)))

            elif self.backend_type == "session":
                for role_data in st.session_state.rbac_roles.values():
                    roles.append(Role.from_dict(role_data))

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                for role_data in data['roles'].values():
                    roles.append(Role.from_dict(role_data))

            return roles

        except Exception as e:
            self.logger.error(f"Failed to list roles: {e}")
            return []

    def update_role(self, role_name: str, updates: Dict[str, Any]) -> bool:
        """Update role information."""
        try:
            if self.backend_type == "database":
                set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
                values = list(updates.values())
                values.append(role_name)

                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"UPDATE rbac_roles SET {set_clause} WHERE name = ?", values)
                    conn.commit()
                    return cursor.rowcount > 0

            elif self.backend_type == "session":
                if role_name in st.session_state.rbac_roles:
                    st.session_state.rbac_roles[role_name].update(updates)
                    return True

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                if role_name in data['roles']:
                    data['roles'][role_name].update(updates)
                    self._save_to_file(data)
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to update role: {e}")
            return False

    def delete_role(self, role_name: str) -> bool:
        """Delete a role."""
        try:
            # Don't allow deletion of system roles
            role = self.get_role(role_name)
            if role and role.is_system_role:
                raise PermissionError("Cannot delete system roles")

            if self.backend_type == "database":
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM rbac_roles WHERE name = ?", (role_name,))
                    conn.commit()
                    return cursor.rowcount > 0

            elif self.backend_type == "session":
                if role_name in st.session_state.rbac_roles:
                    del st.session_state.rbac_roles[role_name]
                    return True

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                if role_name in data['roles']:
                    del data['roles'][role_name]
                    self._save_to_file(data)
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to delete role: {e}")
            return False

    # Audit log operations
    def create_audit_log(self, log: AuditLog) -> bool:
        """Create an audit log entry."""
        try:
            if self.backend_type == "database":
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO rbac_audit_logs
                        (log_id, user_id, action, resource_type, resource_id, success,
                         timestamp, ip_address, user_agent, details)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        log.log_id, log.user_id, log.action, log.resource_type,
                        log.resource_id, int(log.success), log.timestamp.isoformat(),
                        log.ip_address, log.user_agent, json.dumps(log.details)
                    ))
                    conn.commit()

            elif self.backend_type == "session":
                st.session_state.rbac_audit_logs.append(log.to_dict())

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                data['audit_logs'].append(log.to_dict())
                self._save_to_file(data)

            return True

        except Exception as e:
            self.logger.error(f"Failed to create audit log: {e}")
            return False

    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs with optional filtering."""
        try:
            logs = []

            if self.backend_type == "database":
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    query = "SELECT * FROM rbac_audit_logs WHERE 1=1"
                    params = []

                    if user_id:
                        query += " AND user_id = ?"
                        params.append(user_id)
                    if action:
                        query += " AND action = ?"
                        params.append(action)

                    query += " ORDER BY timestamp DESC LIMIT ?"
                    params.append(limit)

                    cursor.execute(query, params)
                    for row in cursor.fetchall():
                        logs.append(self._row_to_audit_log(dict(row)))

            elif self.backend_type == "session":
                for log_data in st.session_state.rbac_audit_logs:
                    if user_id and log_data.get('user_id') != user_id:
                        continue
                    if action and log_data.get('action') != action:
                        continue
                    logs.append(AuditLog(**log_data))
                logs = logs[:limit]

            else:  # file
                data = json.load(open(self.file_path, 'r'))
                for log_data in data['audit_logs']:
                    if user_id and log_data.get('user_id') != user_id:
                        continue
                    if action and log_data.get('action') != action:
                        continue
                    logs.append(AuditLog(**log_data))
                logs = logs[:limit]

            return logs

        except Exception as e:
            self.logger.error(f"Failed to get audit logs: {e}")
            return []

    def _row_to_audit_log(self, row: Dict[str, Any]) -> AuditLog:
        """Convert database row to AuditLog object."""
        return AuditLog(
            log_id=row['log_id'],
            user_id=row['user_id'],
            action=row['action'],
            resource_type=row['resource_type'],
            resource_id=row.get('resource_id'),
            success=bool(row['success']),
            timestamp=datetime.fromisoformat(row['timestamp']),
            ip_address=row.get('ip_address'),
            user_agent=row.get('user_agent'),
            details=json.loads(row['details']) if row.get('details') else {}
        )


# ============================================================================
# AUTHENTICATION BACKENDS
# ============================================================================

class AuthenticationBackend:
    """Base class for authentication backends."""

    def authenticate(self, username: str, password: str, **kwargs) -> Optional[User]:
        """Authenticate a user.

        Args:
            username: Username or identifier
            password: Password or token
            **kwargs: Additional backend-specific parameters

        Returns:
            Authenticated user or None
        """
        raise NotImplementedError

    def verify_token(self, token: str) -> Optional[User]:
        """Verify an authentication token.

        Args:
            token: Authentication token

        Returns:
            Authenticated user or None
        """
        raise NotImplementedError


class NativeAuthBackend(AuthenticationBackend):
    """Native database authentication."""

    def __init__(self, storage: RBACStorage):
        self.storage = storage
        self.logger = logging.getLogger(__name__)

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}:{pwd_hash.hex()}"

    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, stored_pwd_hash = stored_hash.split(':')
            pwd_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return pwd_hash.hex() == stored_pwd_hash
        except (ValueError, AttributeError):
            return False

    def authenticate(self, username: str, password: str, **kwargs) -> Optional[User]:
        """Authenticate with username and password."""
        user = self.storage.get_user_by_username(username)

        if not user:
            self.logger.warning(f"Authentication failed: User not found: {username}")
            return None

        if not user.is_active:
            self.logger.warning(f"Authentication failed: User inactive: {username}")
            return None

        if not self.verify_password(password, user.password_hash or ""):
            self.logger.warning(f"Authentication failed: Invalid password: {username}")
            return None

        # Update last login
        user.last_login = datetime.utcnow()
        self.storage.update_user(user.user_id, {
            'last_login': user.last_login.isoformat()
        })

        # Return the updated user from storage
        updated_user = self.storage.get_user(user.user_id)

        self.logger.info(f"User authenticated successfully: {username}")
        return updated_user

    def verify_token(self, token: str) -> Optional[User]:
        """Native auth doesn't support token verification."""
        return None


class JWTAuthBackend(AuthenticationBackend):
    """JWT token-based authentication."""

    def __init__(self, storage: RBACStorage, secret_key: Optional[str] = None):
        if not JWT_AVAILABLE:
            raise BackendNotAvailableError("JWT library not available")

        self.storage = storage
        self.secret_key = secret_key or secrets.token_hex(32)
        self.algorithm = 'HS256'
        self.logger = logging.getLogger(__name__)

    def generate_token(self, user: User, expires_in: int = 3600) -> str:
        """Generate a JWT token for a user."""
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[User]:
        """Verify a JWT token and return the user."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            user_id = payload.get('user_id')

            if user_id:
                return self.storage.get_user(user_id)

        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token has expired")
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid JWT token: {e}")

        return None

    def authenticate(self, username: str, password: str, **kwargs) -> Optional[User]:
        """Delegate to native auth for initial authentication."""
        native_backend = NativeAuthBackend(self.storage)
        user = native_backend.authenticate(username, password)

        if user:
            # Generate and return token
            token = self.generate_token(user)
            return user

        return None


class APIKeyAuthBackend(AuthenticationBackend):
    """API key authentication."""

    def __init__(self, storage: RBACStorage):
        self.storage = storage
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        self.logger = logging.getLogger(__name__)

    def generate_api_key(self, user_id: str) -> str:
        """Generate a new API key for a user."""
        api_key = f"sk-{secrets.token_urlsafe(32)}"
        self.api_keys[api_key] = user_id
        return api_key

    def verify_token(self, token: str) -> Optional[User]:
        """Verify an API key."""
        user_id = self.api_keys.get(token)
        if user_id:
            return self.storage.get_user(user_id)
        return None

    def authenticate(self, username: str, password: str, **kwargs) -> Optional[User]:
        """API key auth doesn't use username/password."""
        return None


# ============================================================================
# MAIN RBAC SYSTEM
# ============================================================================

class RBACSystem:
    """
    Main RBAC system orchestrating authentication and authorization.

    Features:
    - Multiple authentication backends
    - Role-based access control
    - Permission checking
    - Audit logging
    - Decorators for easy integration
    """

    def __init__(
        self,
        storage_backend: str = "auto",
        storage_config: Optional[Dict[str, Any]] = None,
        auth_backends: Optional[List[AuthBackend]] = None,
        jwt_secret: Optional[str] = None
    ):
        """
        Initialize RBAC system.

        Args:
            storage_backend: Storage backend type
            storage_config: Storage configuration
            auth_backends: List of enabled auth backends
            jwt_secret: JWT secret key
        """
        self.logger = logging.getLogger(__name__)

        # Initialize storage
        storage_config = storage_config or {}
        self.storage = RBACStorage(
            backend=storage_backend,
            db_path=storage_config.get('db_path'),
            file_path=storage_config.get('file_path'),
            use_session_state=storage_config.get('use_session_state', False)
        )

        # Initialize authentication backends
        self.auth_backends: Dict[AuthBackend, AuthenticationBackend] = {}

        # Always enable native auth
        self.auth_backends[AuthBackend.NATIVE] = NativeAuthBackend(self.storage)

        # Enable JWT if available and requested
        if JWT_AVAILABLE:
            self.auth_backends[AuthBackend.JWT] = JWTAuthBackend(self.storage, jwt_secret)

        # Enable API key auth
        self.auth_backends[AuthBackend.API_KEY] = APIKeyAuthBackend(self.storage)

        # Initialize default roles if they don't exist
        self._init_default_roles()

    def _init_default_roles(self):
        """Initialize default system roles."""
        default_roles = [
            Role(
                name='admin',
                description='Full system access',
                permissions={p.value for p in Permission},
                is_system_role=True
            ),
            Role(
                name='editor',
                description='Can edit and manage content',
                permissions={
                    Permission.READ_CONTENT.value,
                    Permission.CREATE_CONTENT.value,
                    Permission.UPDATE_CONTENT.value,
                    Permission.READ_PROJECT.value,
                },
                is_system_role=True
            ),
            Role(
                name='viewer',
                description='Read-only access',
                permissions={
                    Permission.READ_CONTENT.value,
                    Permission.READ_PROJECT.value,
                },
                is_system_role=True
            )
        ]

        for role in default_roles:
            existing = self.storage.get_role(role.name)
            if not existing:
                self.storage.create_role(role)
                self.logger.info(f"Created default role: {role.name}")

    # User management
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        roles: Optional[List[str]] = None,
        is_superuser: bool = False
    ) -> User:
        """Create a new user.

        Args:
            username: Unique username
            email: User email
            password: Password (will be hashed)
            full_name: Full name
            roles: List of role names to assign
            is_superuser: Superuser flag

        Returns:
            Created user object

        Raises:
            RBACError: If user creation fails
        """
        # Check if user already exists
        existing = self.storage.get_user_by_username(username)
        if existing:
            raise RBACError(f"User already exists: {username}")

        # Hash password
        native_backend = self.auth_backends[AuthBackend.NATIVE]
        password_hash = native_backend.hash_password(password)

        # Create user
        user_id = f"user_{secrets.token_hex(8)}"
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            full_name=full_name,
            password_hash=password_hash,
            role_names=set(roles or ['viewer']),
            is_superuser=is_superuser
        )

        if not self.storage.create_user(user):
            raise RBACError(f"Failed to create user: {username}")

        # Log action
        self.log_audit(
            user_id=user_id,
            action="CREATE_USER",
            resource_type="user",
            resource_id=user_id,
            success=True,
            details={'username': username, 'email': email}
        )

        return user

    def authenticate(
        self,
        username: str,
        password: str,
        backend: AuthBackend = AuthBackend.NATIVE
    ) -> Optional[User]:
        """Authenticate a user.

        Args:
            username: Username or identifier
            password: Password or token
            backend: Authentication backend to use

        Returns:
            Authenticated user or None
        """
        if backend not in self.auth_backends:
            raise BackendNotAvailableError(f"Backend not available: {backend}")

        auth_backend = self.auth_backends[backend]
        user = auth_backend.authenticate(username, password)

        if user:
            # Log successful authentication
            self.log_audit(
                user_id=user.user_id,
                action="AUTHENTICATE",
                resource_type="user",
                resource_id=user.user_id,
                success=True,
                details={'backend': backend.value}
            )
        else:
            # Log failed authentication
            self.log_audit(
                user_id="unknown",
                action="AUTHENTICATE",
                resource_type="user",
                resource_id=username,
                success=False,
                details={'backend': backend.value, 'reason': 'invalid_credentials'}
            )

        return user

    def verify_token(self, token: str) -> Optional[User]:
        """Verify an authentication token.

        Args:
            token: JWT token or API key

        Returns:
            Authenticated user or None
        """
        # Try JWT first
        if AuthBackend.JWT in self.auth_backends:
            user = self.auth_backends[AuthBackend.JWT].verify_token(token)
            if user:
                return user

        # Try API key
        if AuthBackend.API_KEY in self.auth_backends:
            user = self.auth_backends[AuthBackend.API_KEY].verify_token(token)
            if user:
                return user

        return None

    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self.storage.get_user(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        return self.storage.get_user_by_username(username)

    def list_users(self) -> List[User]:
        """List all users."""
        return self.storage.list_users()

    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user information."""
        return self.storage.update_user(user_id, updates)

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        return self.storage.delete_user(user_id)

    # Role management
    def create_role(
        self,
        name: str,
        description: str,
        permissions: List[str],
        is_system_role: bool = False
    ) -> Role:
        """Create a new role.

        Args:
            name: Unique role name
            description: Role description
            permissions: List of permission strings
            is_system_role: System role flag

        Returns:
            Created role object
        """
        role = Role(
            name=name,
            description=description,
            permissions=set(permissions),
            is_system_role=is_system_role
        )

        if not self.storage.create_role(role):
            raise RBACError(f"Failed to create role: {name}")

        # Log action
        self.log_audit(
            user_id="system",
            action="CREATE_ROLE",
            resource_type="role",
            resource_id=name,
            success=True,
            details={'name': name, 'permissions': permissions}
        )

        return role

    def get_role(self, role_name: str) -> Optional[Role]:
        """Get a role by name."""
        return self.storage.get_role(role_name)

    def list_roles(self) -> List[Role]:
        """List all roles."""
        return self.storage.list_roles()

    def update_role(self, role_name: str, updates: Dict[str, Any]) -> bool:
        """Update role information."""
        return self.storage.update_role(role_name, updates)

    def delete_role(self, role_name: str) -> bool:
        """Delete a role."""
        return self.storage.delete_role(role_name)

    # Permission checking
    def has_permission(self, user: User, permission: Union[str, Permission]) -> bool:
        """Check if a user has a specific permission.

        Args:
            user: User object
            permission: Permission to check (string or enum)

        Returns:
            True if user has permission
        """
        # Superusers have all permissions
        if user.is_superuser:
            return True

        permission_str = permission.value if isinstance(permission, Permission) else permission

        # Check user's roles
        for role_name in user.role_names:
            role = self.storage.get_role(role_name)
            if role and permission_str in role.permissions:
                return True

        return False

    def has_any_permission(self, user: User, permissions: List[Union[str, Permission]]) -> bool:
        """Check if user has any of the specified permissions."""
        return any(self.has_permission(user, perm) for perm in permissions)

    def has_all_permissions(self, user: User, permissions: List[Union[str, Permission]]) -> bool:
        """Check if user has all of the specified permissions."""
        return all(self.has_permission(user, perm) for perm in permissions)

    def require_permission(self, permission: Union[str, Permission]):
        """Decorator to require a specific permission.

        Usage:
            @rbac.require_permission(Permission.CREATE_USER)
            def create_user_handler():
                pass
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get current user from context (implementation depends on your framework)
                user = self._get_current_user()

                if not user:
                    raise AuthenticationError("Not authenticated")

                if not self.has_permission(user, permission):
                    raise AuthorizationError(f"Missing permission: {permission}")

                return func(*args, **kwargs)
            return wrapper
        return decorator

    def _get_current_user(self) -> Optional[User]:
        """Get the current authenticated user from context.

        This is a placeholder - implementation depends on your framework.
        For Streamlit, you might use session_state.
        For FastAPI, you'd use dependency injection.
        """
        # Try Streamlit session state
        if hasattr(st, 'session_state') and 'current_user' in st.session_state:
            user_data = st.session_state.current_user
            if isinstance(user_data, dict):
                return User.from_dict(user_data)
            return user_data

        return None

    # Audit logging
    def log_audit(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Create an audit log entry."""
        log = AuditLog(
            log_id=f"audit_{secrets.token_hex(8)}",
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            timestamp=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )
        self.storage.create_audit_log(log)

    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs."""
        return self.storage.get_audit_logs(user_id, action, limit)

    # Token generation
    def generate_jwt_token(self, user: User, expires_in: int = 3600) -> Optional[str]:
        """Generate a JWT token for a user.

        Args:
            user: User object
            expires_in: Expiration time in seconds

        Returns:
            JWT token string or None if JWT not available
        """
        if AuthBackend.JWT in self.auth_backends:
            backend = self.auth_backends[AuthBackend.JWT]
            if isinstance(backend, JWTAuthBackend):
                return backend.generate_token(user, expires_in)
        return None

    def generate_api_key(self, user_id: str) -> Optional[str]:
        """Generate an API key for a user.

        Args:
            user_id: User ID

        Returns:
            API key string or None
        """
        if AuthBackend.API_KEY in self.auth_backends:
            backend = self.auth_backends[AuthBackend.API_KEY]
            if isinstance(backend, APIKeyAuthBackend):
                return backend.generate_api_key(user_id)
        return None


# ============================================================================
# STREAMLIT INTEGRATION
# ============================================================================

class StreamlitRBAC:
    """Streamlit-specific RBAC integration."""

    def __init__(self, rbac: RBACSystem):
        """Initialize Streamlit RBAC integration.

        Args:
            rbac: RBAC system instance
        """
        self.rbac = rbac
        self.logger = logging.getLogger(__name__)

    def login_form(self, key: str = "login_form") -> Optional[User]:
        """Render a login form in Streamlit.

        Args:
            key: Unique key for the form

        Returns:
            Authenticated user if successful, None otherwise
        """
        st.subheader("🔐 Login")

        with st.form(key=key):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                user = self.rbac.authenticate(username, password)

                if user:
                    # Store in session state
                    st.session_state.current_user = user.to_dict()
                    st.session_state.authenticated = True
                    st.success(f"Welcome back, {user.username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        # Check if already authenticated
        if st.session_state.get('authenticated'):
            user_data = st.session_state.get('current_user')
            if user_data:
                return User.from_dict(user_data)

        return None

    def logout(self):
        """Logout the current user."""
        if 'current_user' in st.session_state:
            del st.session_state.current_user
        if 'authenticated' in st.session_state:
            del st.session_state.authenticated
        st.rerun()

    def get_current_user(self) -> Optional[User]:
        """Get the current authenticated user."""
        if st.session_state.get('authenticated'):
            user_data = st.session_state.get('current_user')
            if user_data:
                return User.from_dict(user_data)
        return None

    def require_permission(self, permission: Union[str, Permission]):
        """Streamlit decorator to require permission.

        Usage:
            @st_rbac.require_permission(Permission.MANAGE_USERS)
            def render_admin_panel():
                st.write("Admin panel")
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                user = self.get_current_user()

                if not user:
                    st.error("Please login to access this feature")
                    self.login_form()
                    return None

                if not self.rbac.has_permission(user, permission):
                    st.error(f"You don't have permission: {permission}")
                    return None

                return func(*args, **kwargs)
            return wrapper
        return decorator

    def permission_check(self, permission: Union[str, Permission]) -> bool:
        """Check if current user has permission (for use in if statements).

        Usage:
            if st_rbac.permission_check(Permission.CREATE_USER):
                st.button("Create User")
        """
        user = self.get_current_user()
        if not user:
            return False
        return self.rbac.has_permission(user, permission)

    def render_rbac_ui(self):
        """Render the complete RBAC management UI."""
        user = self.get_current_user()

        if not user:
            self.login_form()
            return

        # Show user info and logout
        st.sidebar.write(f"Logged in as: **{user.username}**")
        if st.sidebar.button("Logout"):
            self.logout()

        # Only show admin panel if user has permission
        if self.rbac.has_permission(user, Permission.MANAGE_ROLES):
            self._render_admin_panel()
        else:
            st.info("You don't have permission to manage RBAC settings")

    def _render_admin_panel(self):
        """Render admin panel for RBAC management."""
        st.header("🔒 RBAC Management")

        tab1, tab2, tab3 = st.tabs(["Users", "Roles", "Audit Logs"])

        with tab1:
            self._render_user_management()

        with tab2:
            self._render_role_management()

        with tab3:
            self._render_audit_logs()

    def _render_user_management(self):
        """Render user management interface."""
        st.subheader("User Management")

        # Add user form
        with st.expander("Add New User", expanded=False):
            with st.form("add_user"):
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                full_name = st.text_input("Full Name")
                roles = st.multiselect(
                    "Roles",
                    [r.name for r in self.rbac.list_roles()],
                    default=['viewer']
                )
                is_superuser = st.checkbox("Superuser")

                if st.form_submit_button("Create User"):
                    try:
                        user = self.rbac.create_user(
                            username=username,
                            email=email,
                            password=password,
                            full_name=full_name or None,
                            roles=roles,
                            is_superuser=is_superuser
                        )
                        st.success(f"User created: {user.username}")
                        st.rerun()
                    except RBACError as e:
                        st.error(f"Failed to create user: {e}")

        # List users
        st.subheader("Current Users")
        users = self.rbac.list_users()

        if users:
            for user in users:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([3, 2, 1])

                    with col1:
                        st.write(f"**{user.username}**")
                        st.caption(user.email)

                    with col2:
                        st.caption(f"Roles: {', '.join(user.role_names)}")
                        if user.is_superuser:
                            st.caption("👑 Superuser")

                    with col3:
                        if st.button("Delete", key=f"delete_{user.user_id}"):
                            if self.rbac.delete_user(user.user_id):
                                st.success(f"Deleted user: {user.username}")
                                st.rerun()
        else:
            st.info("No users found")

    def _render_role_management(self):
        """Render role management interface."""
        st.subheader("Role Management")

        # Add role form
        with st.expander("Create Role", expanded=False):
            with st.form("add_role"):
                name = st.text_input("Role Name")
                description = st.text_input("Description")
                permissions = st.multiselect(
                    "Permissions",
                    [p.value for p in Permission],
                    help="Select permissions for this role"
                )

                if st.form_submit_button("Create Role"):
                    try:
                        role = self.rbac.create_role(
                            name=name,
                            description=description,
                            permissions=permissions
                        )
                        st.success(f"Role created: {role.name}")
                        st.rerun()
                    except RBACError as e:
                        st.error(f"Failed to create role: {e}")

        # List roles
        st.subheader("Current Roles")
        roles = self.rbac.list_roles()

        if roles:
            for role in roles:
                with st.container(border=True):
                    st.write(f"**{role.name}**")
                    st.caption(role.description)

                    if role.is_system_role:
                        st.caption("🔒 System Role")

                    st.write("Permissions:")
                    for perm in role.permissions:
                        st.caption(f"  • {perm}")

                    # Allow editing non-system roles
                    if not role.is_system_role:
                        if st.button("Delete", key=f"delete_role_{role.name}"):
                            if self.rbac.delete_role(role.name):
                                st.success(f"Deleted role: {role.name}")
                                st.rerun()
        else:
            st.info("No roles found")

    def _render_audit_logs(self):
        """Render audit logs interface."""
        st.subheader("Audit Logs")

        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            action_filter = st.selectbox(
                "Filter by Action",
                ["All", "CREATE_USER", "AUTHENTICATE", "DELETE_USER", "CREATE_ROLE"],
                index=0
            )

        # Get logs
        action = None if action_filter == "All" else action_filter
        logs = self.rbac.get_audit_logs(action=action, limit=50)

        if logs:
            for log in logs:
                with st.container(border=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**{log.action}**")
                        st.caption(f"User: {log.user_id}")
                        st.caption(f"Time: {log.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

                    with col2:
                        if log.success:
                            st.success("✓ Success")
                        else:
                            st.error("✗ Failed")

                    if log.details:
                        with st.expander("Details"):
                            st.json(log.details)
        else:
            st.info("No audit logs found")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_rbac_system(
    storage_backend: str = "auto",
    use_database: bool = True,
    database_path: str = "rbac_system.db",
    use_session_state: bool = False
) -> RBACSystem:
    """
    Factory function to create an RBAC system.

    Args:
        storage_backend: Storage backend type
        use_database: Use database storage if available
        database_path: Path to database file
        use_session_state: Use Streamlit session state

    Returns:
        Initialized RBACSystem instance
    """
    storage_config = {
        'db_path': database_path if use_database else None,
        'use_session_state': use_session_state
    }

    return RBACSystem(
        storage_backend=storage_backend,
        storage_config=storage_config
    )


# ============================================================================
# EXAMPLES AND USAGE
# ============================================================================

def example_basic_usage():
    """Example: Basic RBAC usage."""
    # Create RBAC system
    rbac = create_rbac_system(use_database=False, use_session_state=True)

    # Create users
    admin = rbac.create_user(
        username="admin",
        email="admin@example.com",
        password="secure_password",
        roles=["admin"]
    )

    viewer = rbac.create_user(
        username="viewer",
        email="viewer@example.com",
        password="viewer_password",
        roles=["viewer"]
    )

    # Authenticate
    user = rbac.authenticate("admin", "secure_password")

    # Check permissions
    if rbac.has_permission(user, Permission.MANAGE_USERS):
        print("Admin can manage users")

    if rbac.has_permission(viewer, Permission.MANAGE_USERS):
        print("Viewer can manage users")
    else:
        print("Viewer cannot manage users")


def example_streamlit_integration():
    """Example: Streamlit integration."""
    import streamlit as st

    # Create RBAC system
    rbac = create_rbac_system()
    st_rbac = StreamlitRBAC(rbac)

    # Get current user or show login form
    user = st_rbac.get_current_user()

    if not user:
        st_rbac.login_form()
        st.stop()

    # Show different content based on permissions
    st.write(f"Welcome, {user.username}!")

    if st_rbac.permission_check(Permission.MANAGE_USERS):
        st.write("You can manage users")

    if st_rbac.permission_check(Permission.CREATE_CONTENT):
        if st.button("Create Content"):
            st.write("Creating content...")


def example_decorator_usage():
    """Example: Using permission decorators."""
    rbac = create_rbac_system()

    @rbac.require_permission(Permission.MANAGE_USERS)
    def create_new_user(username: str, email: str):
        print(f"Creating user: {username}")
        # User creation logic here

    # This will check permissions before executing
    try:
        create_new_user("john", "john@example.com")
    except AuthorizationError as e:
        print(f"Access denied: {e}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Enhanced RBAC System")
    print("=" * 60)

    # Run examples
    example_basic_usage()
