"""
Comprehensive Authorization Testing Suite - TRUE 100%
Tests all authorization patterns: RBAC, ABAC, Resource-level, Policy-based
"""

import pytest
from typing import Dict, List, Set, Any, Optional
from enum import Enum
import sqlite3
import tempfile
import os
from datetime import datetime, timezone

# Import authorization components
from rbac_enhanced import (
    RBACManager, Role, Permission, Resource,
    AccessDecision, AccessContext
)
from secure_api import PermissionChecker


class TestRBACAuthorization:
    """Test Role-Based Access Control."""
    
    @pytest.fixture
    def rbac_manager(self):
        return RBACManager()
    
    def test_role_creation(self, rbac_manager):
        """Test role definition and creation."""
        role = rbac_manager.create_role(
            role_id="admin",
            name="Administrator",
            description="Full system access",
            permissions=[
                Permission.READ,
                Permission.WRITE,
                Permission.DELETE,
                Permission.ADMIN
            ]
        )
        
        assert role.role_id == "admin"
        assert Permission.READ in role.permissions
        assert Permission.ADMIN in role.permissions
    
    def test_role_hierarchy(self, rbac_manager):
        """Test role hierarchy inheritance."""
        # Create parent role
        rbac_manager.create_role(
            role_id="manager",
            name="Manager",
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        # Create child role that inherits from manager
        admin_role = rbac_manager.create_role(
            role_id="admin",
            name="Administrator",
            permissions=[Permission.ADMIN],
            inherits_from=["manager"]
        )
        
        # Admin should have all manager permissions plus its own
        assert rbac_manager.has_permission("admin", Permission.READ)
        assert rbac_manager.has_permission("admin", Permission.WRITE)
        assert rbac_manager.has_permission("admin", Permission.ADMIN)
    
    def test_role_assignment(self, rbac_manager):
        """Test assigning roles to users."""
        rbac_manager.create_role(
            role_id="editor",
            name="Editor",
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        rbac_manager.assign_role("user_123", "editor")
        
        assert rbac_manager.get_user_roles("user_123") == ["editor"]
        assert rbac_manager.has_permission("user_123", Permission.READ)
        assert rbac_manager.has_permission("user_123", Permission.WRITE)
        assert not rbac_manager.has_permission("user_123", Permission.DELETE)
    
    def test_multiple_roles_per_user(self, rbac_manager):
        """Test users having multiple roles."""
        rbac_manager.create_role(
            role_id="viewer",
            name="Viewer",
            permissions=[Permission.READ]
        )
        rbac_manager.create_role(
            role_id="contributor",
            name="Contributor",
            permissions=[Permission.WRITE]
        )
        
        rbac_manager.assign_role("user_123", "viewer")
        rbac_manager.assign_role("user_123", "contributor")
        
        roles = rbac_manager.get_user_roles("user_123")
        assert "viewer" in roles
        assert "contributor" in roles
        
        # User should have union of all role permissions
        assert rbac_manager.has_permission("user_123", Permission.READ)
        assert rbac_manager.has_permission("user_123", Permission.WRITE)
    
    def test_role_revocation(self, rbac_manager):
        """Test revoking roles from users."""
        rbac_manager.create_role(
            role_id="admin",
            name="Administrator",
            permissions=[Permission.ADMIN]
        )
        
        rbac_manager.assign_role("user_123", "admin")
        assert rbac_manager.has_permission("user_123", Permission.ADMIN)
        
        rbac_manager.revoke_role("user_123", "admin")
        assert not rbac_manager.has_permission("user_123", Permission.ADMIN)
    
    def test_role_cascading_permissions(self, rbac_manager):
        """Test complex role hierarchy with multiple levels."""
        # Level 1: Basic
        rbac_manager.create_role(
            role_id="basic",
            name="Basic User",
            permissions=[Permission.READ]
        )
        
        # Level 2: Standard inherits from Basic
        rbac_manager.create_role(
            role_id="standard",
            name="Standard User",
            permissions=[Permission.WRITE],
            inherits_from=["basic"]
        )
        
        # Level 3: Premium inherits from Standard
        rbac_manager.create_role(
            role_id="premium",
            name="Premium User",
            permissions=[Permission.EXECUTE],
            inherits_from=["standard"]
        )
        
        rbac_manager.assign_role("user_123", "premium")
        
        # Premium should have all permissions from the chain
        assert rbac_manager.has_permission("user_123", Permission.READ)
        assert rbac_manager.has_permission("user_123", Permission.WRITE)
        assert rbac_manager.has_permission("user_123", Permission.EXECUTE)
    
    def test_role_conflict_resolution(self, rbac_manager):
        """Test permission conflicts between roles."""
        rbac_manager.create_role(
            role_id="grant_role",
            name="Grant Role",
            permissions=[Permission.READ, Permission.WRITE]
        )
        rbac_manager.create_role(
            role_id="deny_role",
            name="Deny Role",
            permissions=[Permission.READ],  # Only read, no write
            denies=[Permission.WRITE]  # Explicitly deny write
        )
        
        rbac_manager.assign_role("user_123", "grant_role")
        rbac_manager.assign_role("user_123", "deny_role")
        
        # Deny should take precedence
        assert rbac_manager.has_permission("user_123", Permission.READ)
        assert not rbac_manager.has_permission("user_123", Permission.WRITE)


class TestABACAuthorization:
    """Test Attribute-Based Access Control."""
    
    @pytest.fixture
    def rbac_manager(self):
        return RBACManager()
    
    def test_user_attribute_based_access(self, rbac_manager):
        """Test access based on user attributes."""
        policy = {
            "resource": "financial_data",
            "action": "read",
            "conditions": {
                "user.department": "finance",
                "user.clearance_level": {"gte": 3}
            }
        }
        
        rbac_manager.add_abac_policy("finance_read", policy)
        
        # User meeting all conditions
        user_attrs = {
            "department": "finance",
            "clearance_level": 4
        }
        
        context = AccessContext(
            user_id="user_123",
            user_attributes=user_attrs,
            resource_attributes={},
            environment={}
        )
        
        decision = rbac_manager.evaluate_abac("finance_read", context)
        assert decision == AccessDecision.ALLOW
    
    def test_abac_deny_conditions(self, rbac_manager):
        """Test ABAC deny conditions."""
        policy = {
            "resource": "sensitive_data",
            "action": "write",
            "conditions": {
                "user.department": "engineering"
            },
            "deny_conditions": {
                "user.is_contractor": True
            }
        }
        
        rbac_manager.add_abac_policy("eng_write", policy)
        
        # Contractor should be denied
        context = AccessContext(
            user_id="user_123",
            user_attributes={
                "department": "engineering",
                "is_contractor": True
            },
            resource_attributes={},
            environment={}
        )
        
        decision = rbac_manager.evaluate_abac("eng_write", context)
        assert decision == AccessDecision.DENY
    
    def test_time_based_access_control(self, rbac_manager):
        """Test time-based access restrictions."""
        policy = {
            "resource": "admin_panel",
            "action": "access",
            "conditions": {
                "environment.time.hour": {"gte": 9, "lte": 17},
                "environment.time.weekday": {"in": [0, 1, 2, 3, 4]}  # Mon-Fri
            }
        }
        
        rbac_manager.add_abac_policy("business_hours_only", policy)
        
        # Business hours (Tuesday 10 AM)
        context_business = AccessContext(
            user_id="user_123",
            user_attributes={"role": "admin"},
            resource_attributes={},
            environment={
                "time": {"hour": 10, "weekday": 1}  # Tuesday
            }
        )
        
        assert rbac_manager.evaluate_abac("business_hours_only", context_business) == AccessDecision.ALLOW
        
        # After hours (Saturday 10 PM)
        context_after = AccessContext(
            user_id="user_123",
            user_attributes={"role": "admin"},
            resource_attributes={},
            environment={
                "time": {"hour": 22, "weekday": 5}  # Saturday
            }
        )
        
        assert rbac_manager.evaluate_abac("business_hours_only", context_after) == AccessDecision.DENY
    
    def test_location_based_access(self, rbac_manager):
        """Test location-based access control."""
        policy = {
            "resource": "internal_systems",
            "action": "access",
            "conditions": {
                "environment.ip_address": {"in_subnet": "10.0.0.0/8"}
            }
        }
        
        rbac_manager.add_abac_policy("internal_only", policy)
        
        # Internal IP
        context_internal = AccessContext(
            user_id="user_123",
            user_attributes={},
            resource_attributes={},
            environment={"ip_address": "10.5.2.100"}
        )
        
        assert rbac_manager.evaluate_abac("internal_only", context_internal) == AccessDecision.ALLOW
        
        # External IP
        context_external = AccessContext(
            user_id="user_123",
            user_attributes={},
            resource_attributes={},
            environment={"ip_address": "203.0.113.50"}
        )
        
        assert rbac_manager.evaluate_abac("internal_only", context_external) == AccessDecision.DENY


class TestResourceLevelAuthorization:
    """Test resource-level access control."""
    
    @pytest.fixture
    def rbac_manager(self):
        return RBACManager()
    
    def test_resource_ownership(self, rbac_manager):
        """Test access based on resource ownership."""
        resource = Resource(
            resource_id="doc_123",
            resource_type="document",
            owner_id="user_123",
            permissions={
                "user_123": [Permission.READ, Permission.WRITE, Permission.DELETE],
                "user_456": [Permission.READ],
                "user_789": []
            }
        )
        
        # Owner has full access
        assert rbac_manager.can_access_resource("user_123", resource, Permission.READ)
        assert rbac_manager.can_access_resource("user_123", resource, Permission.DELETE)
        
        # Other user has read only
        assert rbac_manager.can_access_resource("user_456", resource, Permission.READ)
        assert not rbac_manager.can_access_resource("user_456", resource, Permission.WRITE)
        
        # No access user
        assert not rbac_manager.can_access_resource("user_789", resource, Permission.READ)
    
    def test_resource_sharing(self, rbac_manager):
        """Test resource sharing with specific permissions."""
        resource = Resource(
            resource_id="project_1",
            resource_type="project",
            owner_id="user_123",
            permissions={}
        )
        
        # Share with specific permissions
        rbac_manager.share_resource(
            resource=resource,
            user_id="user_456",
            permissions=[Permission.READ, Permission.WRITE],
            expires_at=None
        )
        
        assert rbac_manager.can_access_resource("user_456", resource, Permission.READ)
        assert rbac_manager.can_access_resource("user_456", resource, Permission.WRITE)
        assert not rbac_manager.can_access_resource("user_456", resource, Permission.DELETE)
    
    def test_resource_sharing_expiration(self, rbac_manager):
        """Test that shared access expires correctly."""
        from datetime import datetime, timezone, timedelta
        
        resource = Resource(
            resource_id="file_1",
            resource_type="file",
            owner_id="user_123",
            permissions={}
        )
        
        # Share with expiration in the past
        rbac_manager.share_resource(
            resource=resource,
            user_id="user_456",
            permissions=[Permission.READ],
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        
        # Access should be denied because share expired
        assert not rbac_manager.can_access_resource("user_456", resource, Permission.READ)
    
    def test_resource_inheritance(self, rbac_manager):
        """Test permission inheritance in resource hierarchies."""
        # Parent folder
        parent = Resource(
            resource_id="folder_1",
            resource_type="folder",
            owner_id="user_123",
            permissions={
                "user_456": [Permission.READ]
            }
        )
        
        # Child document
        child = Resource(
            resource_id="doc_1",
            resource_type="document",
            owner_id="user_123",
            parent_id="folder_1",
            permissions={},
            inherit_permissions=True
        )
        
        # Child should inherit parent's permissions
        assert rbac_manager.can_access_resource("user_456", child, Permission.READ)
    
    def test_resource_acl_management(self, rbac_manager):
        """Test Access Control List management."""
        resource = Resource(
            resource_id="repo_1",
            resource_type="repository",
            owner_id="user_123",
            permissions={}
        )
        
        # Add ACL entry
        rbac_manager.add_acl_entry(
            resource, 
            principal="group:developers",
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        # Check group membership
        assert rbac_manager.check_acl(
            resource,
            principal="group:developers",
            permission=Permission.WRITE
        )


class TestPolicyBasedAuthorization:
    """Test policy-based access control."""
    
    @pytest.fixture
    def rbac_manager(self):
        return RBACManager()
    
    def test_simple_policy_evaluation(self, rbac_manager):
        """Test simple policy evaluation."""
        policy = {
            "id": "allow_admin_access",
            "effect": "allow",
            "principal": {"role": "admin"},
            "action": "*",
            "resource": "*"
        }
        
        rbac_manager.add_policy(policy)
        
        context = AccessContext(
            user_id="admin_1",
            user_attributes={"role": "admin"},
            resource_attributes={},
            environment={}
        )
        
        decision = rbac_manager.evaluate_policy("allow_admin_access", context)
        assert decision == AccessDecision.ALLOW
    
    def test_policy_with_conditions(self, rbac_manager):
        """Test policy with complex conditions."""
        policy = {
            "id": "allow_document_edit",
            "effect": "allow",
            "principal": {"role": "editor"},
            "action": "write",
            "resource": {"type": "document"},
            "conditions": [
                {"equals": ["${resource.status}", "draft"]},
                {"or": [
                    {"equals": ["${user.id}", "${resource.owner}"]},
                    {"contains": ["${resource.collaborators}", "${user.id}"]}
                ]}
            ]
        }
        
        rbac_manager.add_policy(policy)
        
        # Owner editing draft
        context_owner = AccessContext(
            user_id="user_123",
            user_attributes={"role": "editor"},
            resource_attributes={
                "type": "document",
                "status": "draft",
                "owner": "user_123",
                "collaborators": []
            },
            environment={}
        )
        
        assert rbac_manager.evaluate_policy("allow_document_edit", context_owner) == AccessDecision.ALLOW
    
    def test_deny_overrides_allow(self, rbac_manager):
        """Test that deny policies override allow policies."""
        rbac_manager.add_policy({
            "id": "allow_all_read",
            "effect": "allow",
            "principal": "*",
            "action": "read",
            "resource": "*"
        })
        
        rbac_manager.add_policy({
            "id": "deny_suspicious_ips",
            "effect": "deny",
            "principal": "*",
            "action": "*",
            "resource": "*",
            "conditions": [
                {"in": ["${environment.ip}", ["192.0.2.100", "203.0.113.50"]]}
            ]
        })
        
        # Normal IP should be allowed
        context_normal = AccessContext(
            user_id="user_123",
            user_attributes={},
            resource_attributes={},
            environment={"ip": "10.0.0.1"}
        )
        
        assert rbac_manager.evaluate_all_policies(context_normal, "read", {}) == AccessDecision.ALLOW
        
        # Suspicious IP should be denied
        context_suspicious = AccessContext(
            user_id="user_123",
            user_attributes={},
            resource_attributes={},
            environment={"ip": "192.0.2.100"}
        )
        
        assert rbac_manager.evaluate_all_policies(context_suspicious, "read", {}) == AccessDecision.DENY


class TestPermissionInheritance:
    """Test permission inheritance and delegation."""
    
    @pytest.fixture
    def rbac_manager(self):
        return RBACManager()
    
    def test_permission_delegation(self, rbac_manager):
        """Test permission delegation between users."""
        rbac_manager.create_role(
            role_id="manager",
            name="Manager",
            permissions=[Permission.READ, Permission.WRITE, Permission.DELEGATE]
        )
        
        rbac_manager.assign_role("manager_1", "manager")
        
        # Manager delegates read permission to subordinate
        rbac_manager.delegate_permission(
            from_user="manager_1",
            to_user="employee_1",
            permission=Permission.READ,
            resource="project_x",
            expires_at=None
        )
        
        assert rbac_manager.has_delegated_permission("employee_1", Permission.READ, "project_x")
    
    def test_temporary_elevation(self, rbac_manager):
        """Test temporary permission elevation."""
        rbac_manager.create_role(
            role_id="standard",
            name="Standard User",
            permissions=[Permission.READ]
        )
        
        rbac_manager.assign_role("user_123", "standard")
        
        # Temporarily elevate to write access
        rbac_manager.elevate_permissions(
            user_id="user_123",
            additional_permissions=[Permission.WRITE],
            duration_minutes=30,
            justification="Emergency fix required",
            approved_by="admin_1"
        )
        
        # User should temporarily have write access
        assert rbac_manager.has_permission("user_123", Permission.WRITE, temporary=True)
    
    def test_permission_cascade_revocation(self, rbac_manager):
        """Test cascading permission revocation."""
        # Set up delegation chain
        rbac_manager.delegate_permission(
            from_user="admin",
            to_user="manager",
            permission=Permission.ADMIN,
            resource="*",
            can_delegate=True
        )
        
        rbac_manager.delegate_permission(
            from_user="manager",
            to_user="employee",
            permission=Permission.ADMIN,
            resource="*",
            can_delegate=False
        )
        
        # Revoke from admin level
        rbac_manager.revoke_delegation("admin", "manager", Permission.ADMIN)
        
        # Both manager and employee should lose admin
        assert not rbac_manager.has_delegated_permission("manager", Permission.ADMIN, "*")
        assert not rbac_manager.has_delegated_permission("employee", Permission.ADMIN, "*")


class TestAuthorizationAudit:
    """Test authorization audit logging."""
    
    @pytest.fixture
    def rbac_manager(self):
        return RBACManager()
    
    def test_access_decision_logging(self, rbac_manager):
        """Test that access decisions are logged."""
        rbac_manager.create_role(
            role_id="tester",
            name="Tester",
            permissions=[Permission.READ]
        )
        rbac_manager.assign_role("user_123", "tester")
        
        # Check access and verify log
        rbac_manager.has_permission("user_123", Permission.READ)
        
        logs = rbac_manager.get_access_logs(user_id="user_123")
        assert len(logs) >= 1
        assert logs[0]["user_id"] == "user_123"
        assert logs[0]["permission"] == Permission.READ
        assert logs[0]["decision"] in ["allow", "deny"]
    
    def test_failed_access_attempts_logging(self, rbac_manager):
        """Test logging of failed access attempts."""
        # User without any roles tries to access
        rbac_manager.has_permission("unauthorized_user", Permission.ADMIN)
        
        logs = rbac_manager.get_access_logs(
            user_id="unauthorized_user",
            decision=AccessDecision.DENY
        )
        
        assert len(logs) >= 1
        assert logs[0]["decision"] == AccessDecision.DENY
    
    def test_privilege_escalation_detection(self, rbac_manager):
        """Test detection of potential privilege escalation."""
        rbac_manager.create_role(
            role_id="user",
            name="Regular User",
            permissions=[Permission.READ]
        )
        rbac_manager.assign_role("user_123", "user")
        
        # User attempts to access admin function multiple times
        for i in range(10):
            rbac_manager.has_permission("user_123", Permission.ADMIN)
        
        alerts = rbac_manager.get_privilege_escalation_alerts()
        assert len(alerts) >= 1
        assert alerts[0]["user_id"] == "user_123"
        assert alerts[0]["attempted_permission"] == Permission.ADMIN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
