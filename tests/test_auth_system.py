"""
Comprehensive Unit Tests for Authentication System

Tests the authentication and authorization system.

Author: OpenEvolve QA Team
Date: 2026-02-05
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

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
        
        # permissions defaults to empty list, not None
        assert user.permissions == []
        assert user.created_at is not None
        assert user.last_login is None
        assert user.is_active == True


class TestAuditLog:
    """Test AuditLog model"""

    def test_audit_log_creation(self):
        """Test AuditLog model creation"""
        from auth_system import AuditLog
        from datetime import datetime
        
        log = AuditLog(
            log_id="log_001",
            user_id="user_123",
            operation="create",
            resource="problem",
            resource_id="prob_001",
            timestamp=datetime.now(),
            success=True,
            details={"key": "value"}
        )
        
        assert log.id == "log_001"
        assert log.user_id == "user_123"
        assert log.success == True

    def test_audit_log_to_dict(self):
        """Test AuditLog to_dict conversion"""
        from auth_system import AuditLog
        from datetime import datetime
        
        log = AuditLog(
            log_id="log_002",
            user_id="user_456",
            operation="read",
            resource="plan",
            resource_id="plan_001",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            success=True,
            details={}
        )
        
        result = log.to_dict()
        
        assert isinstance(result, dict)
        assert result["id"] == "log_002"
        assert result["success"] == True


class TestAuthenticationSystem:
    """Test AuthenticationSystem class"""

    @pytest.fixture
    def auth_system(self, tmp_path):
        """Create authentication system for testing"""
        from auth_system import AuthenticationSystem
        
        db_path = str(tmp_path / "test_auth.db")
        return AuthenticationSystem(secret_key="test_secret_key", db_path=db_path)

    def test_auth_system_creation(self, auth_system):
        """Test AuthenticationSystem initialization"""
        from auth_system import AuthenticationSystem
        
        system = AuthenticationSystem(
            secret_key="test_key",
            db_path=":memory:"
        )
        assert system.secret_key == "test_key"
        assert system.db_path == ":memory:"

    def test_auth_system_has_db(self, auth_system):
        """Test AuthenticationSystem has database"""
        assert auth_system.db is not None

    def test_auth_system_initialize_tables(self, auth_system):
        """Test table initialization"""
        # Should not raise exception
        auth_system._initialize_auth_tables()
        
        # Verify tables exist
        with auth_system.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "users" in tables
            assert "api_keys" in tables
            assert "audit_logs" in tables


class TestGenerateID:
    """Test ID generation"""

    def test_generate_id_function_exists(self):
        """Test generate_id function exists"""
        from auth_system import generate_id
        assert callable(generate_id)

    def test_generate_id_returns_string(self):
        """Test generate_id returns string"""
        from auth_system import generate_id
        
        id_value = generate_id()
        
        assert isinstance(id_value, str)
        assert len(id_value) > 0


class TestAuthSystemExports:
    """Test what is exported from auth_system"""

    def test_auth_system_module_exports(self):
        """Check what functions/classes are available"""
        import auth_system
        
        # Check for expected exports
        assert hasattr(auth_system, 'Role')
        assert hasattr(auth_system, 'Permission')
        assert hasattr(auth_system, 'User')
        assert hasattr(auth_system, 'AuditLog')
        assert hasattr(auth_system, 'AuthenticationSystem')
        assert hasattr(auth_system, 'generate_id')

    def test_auth_system_has_authenticate_method(self):
        """Test AuthenticationSystem has authenticate method"""
        from auth_system import AuthenticationSystem
        assert hasattr(AuthenticationSystem, 'authenticate')
        assert callable(AuthenticationSystem.authenticate)

    def test_auth_system_has_create_user_method(self):
        """Test AuthenticationSystem has create_user method"""
        from auth_system import AuthenticationSystem
        assert hasattr(AuthenticationSystem, 'create_user')
        assert callable(AuthenticationSystem.create_user)

    def test_auth_system_has_get_user_method(self):
        """Test AuthenticationSystem has get_user_by_id method (actual implementation)"""
        from auth_system import AuthenticationSystem
        assert hasattr(AuthenticationSystem, 'get_user_by_id')
        assert callable(AuthenticationSystem.get_user_by_id)

    def test_auth_system_has_validate_token_method(self):
        """Test AuthenticationSystem has verify_jwt_token method (actual implementation)"""
        from auth_system import AuthenticationSystem
        assert hasattr(AuthenticationSystem, 'verify_jwt_token')
        assert callable(AuthenticationSystem.verify_jwt_token)

    def test_auth_system_has_create_api_key_method(self):
        """Test APIKeyManager has create_api_key method (actual implementation)"""
        from auth_system import APIKeyManager
        assert hasattr(APIKeyManager, 'create_api_key')
        assert callable(APIKeyManager.create_api_key)

    def test_auth_system_has_log_operation_method(self):
        """Test AuditSystem has log_audit method (actual implementation)"""
        from auth_system import AuditSystem
        assert hasattr(AuditSystem, 'log_audit')
        assert callable(AuditSystem.log_audit)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
