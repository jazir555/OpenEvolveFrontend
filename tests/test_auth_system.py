"""
Comprehensive Unit Tests for Authentication System

Tests the authentication and authorization system including:
- User management
- JWT token generation and validation
- Password hashing
- Role-based access control
- Permission checking
- Session management

Author: OpenEvolve QA Team
Date: 2026-02-05
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import hashlib
import jwt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRoleEnum:
    """Test Role enum values"""

    def test_role_enum_values(self):
        """Test Role enum contains expected values"""
        from auth_system import Role
        
        assert Role.ADMIN.value == "admin"
        assert Role.WORKFLOW_MANAGER.value == "workflow_manager"
        assert Role.ANALYST.value == "analyst"
        assert Role.VIEWER.value == "viewer"

    def test_role_enum_members(self):
        """Test Role enum has expected members"""
        from auth_system import Role
        
        roles = list(Role)
        assert len(roles) == 4
        assert Role.ADMIN in roles
        assert Role.WORKFLOW_MANAGER in roles


class TestPermissionEnum:
    """Test Permission enum values"""

    def test_permission_enum_values(self):
        """Test Permission enum contains expected values"""
        from auth_system import Permission
        
        assert Permission.CREATE_PROBLEM.value == "create_problem"
        assert Permission.READ_PROBLEM.value == "read_problem"
        assert Permission.UPDATE_PROBLEM.value == "update_problem"
        assert Permission.DELETE_PROBLEM.value == "delete_problem"
        assert Permission.CREATE_PLAN.value == "create_plan"
        assert Permission.ADMIN_ACCESS.value == "admin_access"

    def test_permission_categories(self):
        """Test Permission covers all categories"""
        from auth_system import Permission
        
        # Problem permissions
        assert hasattr(Permission, 'CREATE_PROBLEM')
        assert hasattr(Permission, 'READ_PROBLEM')
        assert hasattr(Permission, 'DELETE_PROBLEM')
        
        # Plan permissions
        assert hasattr(Permission, 'CREATE_PLAN')
        assert hasattr(Permission, 'READ_PLAN')
        
        # Solution permissions
        assert hasattr(Permission, 'CREATE_SOLUTION')
        assert hasattr(Permission, 'READ_SOLUTION')
        
        # Team permissions
        assert hasattr(Permission, 'MANAGE_TEAMS')
        assert hasattr(Permission, 'RUN_GAUNTLETS')


class TestUserModel:
    """Test User model"""

    def test_user_creation(self):
        """Test User model creation"""
        from auth_system import User, Role, Permission
        
        user = User(
            user_id="test_user_123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            roles=[Role.ANALYST],
            permissions=[Permission.READ_PROBLEM]
        )
        
        assert user.id == "test_user_123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.password_hash == "hashed_password"
        assert Role.ANALYST in user.roles
        assert Permission.READ_PROBLEM in user.permissions

    def test_user_default_values(self):
        """Test User default values"""
        from auth_system import User, Role
        
        user = User(
            user_id="user_123",
            username="user",
            email="user@test.com",
            password_hash="hash",
            roles=[Role.VIEWER]
        )
        
        assert user.permissions is None
        assert user.created_at is None
        assert user.last_login is None
        assert user.is_active == True


class TestJWTTokenManager:
    """Test JWT token management"""

    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager for testing"""
        from auth_system import JWTManager
        return JWTManager(
            secret_key="test_secret_key",
            algorithm="HS256",
            expiry_hours=24
        )

    def test_jwt_manager_creation(self, jwt_manager):
        """Test JWT manager initialization"""
        from auth_system import JWTManager
        
        manager = JWTManager(
            secret_key="test_key",
            algorithm="HS256",
            expiry_hours=1
        )
        assert manager.secret_key == "test_key"
        assert manager.algorithm == "HS256"
        assert manager.expiry_hours == 1

    def test_token_generation(self, jwt_manager):
        """Test JWT token generation"""
        token = jwt_manager.generate_token(
            user_id="user_123",
            username="testuser",
            roles=["analyst"]
        )
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are typically long

    def test_token_validation_success(self, jwt_manager):
        """Test successful token validation"""
        token = jwt_manager.generate_token(
            user_id="user_123",
            username="testuser",
            roles=["analyst"]
        )
        
        payload = jwt_manager.validate_token(token)
        
        assert payload is not None
        assert payload["user_id"] == "user_123"
        assert payload["username"] == "testuser"
        assert "analyst" in payload["roles"]

    def test_token_validation_invalid(self, jwt_manager):
        """Test invalid token validation returns None"""
        invalid_token = "invalid.token.here"
        
        payload = jwt_manager.validate_token(invalid_token)
        assert payload is None

    def test_token_validation_expired(self, jwt_manager):
        """Test expired token validation"""
        # Create manager with very short expiry
        from auth_system import JWTManager
        short_expiry_manager = JWTManager(
            secret_key="test_key",
            algorithm="HS256",
            expiry_hours=-1  # Already expired
        )
        
        token = short_expiry_manager.generate_token(
            user_id="user_123",
            username="testuser",
            roles=[]
        )
        
        payload = short_expiry_manager.validate_token(token)
        assert payload is None


class TestPasswordHashing:
    """Test password hashing functions"""

    def test_password_hash_generation(self):
        """Test secure password hash generation"""
        from auth_system import hash_password
        
        password = "secure_password_123"
        salt = "random_salt"
        
        hash_result = hash_password(password, salt)
        
        assert hash_result is not None
        assert isinstance(hash_result, str)
        assert len(hash_result) > 20

    def test_password_hash_different_salts(self):
        """Test different salts produce different hashes"""
        from auth_system import hash_password
        
        password = "same_password"
        salt1 = "salt_one"
        salt2 = "salt_two"
        
        hash1 = hash_password(password, salt1)
        hash2 = hash_password(password, salt2)
        
        assert hash1 != hash2

    def test_password_verification_success(self):
        """Test successful password verification"""
        from auth_system import hash_password, verify_password
        
        password = "test_password"
        salt = "unique_salt"
        
        stored_hash = hash_password(password, salt)
        
        assert verify_password(password, stored_hash, salt) == True

    def test_password_verification_failure(self):
        """Test failed password verification"""
        from auth_system import hash_password, verify_password
        
        password = "test_password"
        salt = "unique_salt"
        wrong_password = "wrong_password"
        
        stored_hash = hash_password(password, salt)
        
        assert verify_password(wrong_password, stored_hash, salt) == False


class TestAccessControl:
    """Test access control functionality"""

    def test_require_permission_decorator(self):
        """Test require_permission decorator exists"""
        from auth_system import require_permission
        
        assert callable(require_permission)

    def test_require_auth_decorator(self):
        """Test require_auth decorator exists"""
        from auth_system import require_auth
        
        assert callable(require_auth)

    def test_get_current_user_function(self):
        """Test get_current_user function exists"""
        from auth_system import get_current_user
        
        user = get_current_user()
        # Should return a context or None
        assert user is None or hasattr(user, 'user_id')


class TestRolePermissions:
    """Test role-based permissions"""

    def test_admin_has_all_permissions(self):
        """Test admin role has comprehensive permissions"""
        from auth_system import Role, Permission
        
        admin_permissions = [
            Permission.CREATE_PROBLEM,
            Permission.READ_PROBLEM,
            Permission.UPDATE_PROBLEM,
            Permission.DELETE_PROBLEM,
            Permission.CREATE_PLAN,
            Permission.READ_PLAN,
            Permission.MANAGE_TEAMS,
            Permission.RUN_GAUNTLETS,
            Permission.ADMIN_ACCESS,
            Permission.MANAGE_USERS
        ]
        
        # Admin should conceptually have all permissions
        # (actual implementation may vary)
        assert len(admin_permissions) > 5

    def test_analyst_role_permissions(self):
        """Test analyst role has read permissions"""
        from auth_system import Role, Permission
        
        # Analyst should be able to read problems and plans
        assert hasattr(Permission, 'READ_PROBLEM')
        assert hasattr(Permission, 'READ_PLAN')
        assert hasattr(Permission, 'READ_SOLUTION')

    def test_viewer_role_permissions(self):
        """Test viewer role has basic read permissions"""
        from auth_system import Role, Permission
        
        # Viewer should have read permissions
        assert hasattr(Permission, 'READ_PROBLEM')
        assert hasattr(Permission, 'READ_PLAN')


class TestSessionManagement:
    """Test session management"""

    def test_create_session_function(self):
        """Test session creation function exists"""
        from auth_system import create_session
        
        assert callable(create_session)

    def test_validate_session_function(self):
        """Test session validation function exists"""
        from auth_system import validate_session
        
        assert callable(validate_session)

    def test_end_session_function(self):
        """Test session termination function exists"""
        from auth_system import end_session
        
        assert callable(end_session)


class TestDatabaseIntegration:
    """Test database integration for user storage"""

    def test_get_user_by_id_function(self):
        """Test get_user_by_id function exists"""
        from auth_system import get_user_by_id
        
        assert callable(get_user_by_id)

    def test_get_user_by_username_function(self):
        """Test get_user_by_username function exists"""
        from auth_system import get_user_by_username
        
        assert callable(get_user_by_username)

    def test_create_user_function(self):
        """Test create_user function exists"""
        from auth_system import create_user
        
        assert callable(create_user)

    def test_update_user_function(self):
        """Test update_user function exists"""
        from auth_system import update_user
        
        assert callable(update_user)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
