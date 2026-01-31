"""
Multi-Tenant Architecture

Supports multiple isolated tenants (organizations) on a single platform:
- Tenant isolation
- Resource quotas
- Custom branding per tenant
- Tenant-specific configurations
- Cross-tenant sharing (optional)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Tenant:
    """Represents a tenant (organization)."""
    tenant_id: str
    name: str
    slug: str  # URL-friendly identifier
    description: str = ""
    status: str = "active"  # active, suspended, deleted
    created_at: datetime = field(default_factory=datetime.utcnow)
    settings: Dict[str, Any] = field(default_factory=dict)
    quotas: Dict[str, int] = field(default_factory=dict)
    features: Set[str] = field(default_factory=set)
    branding: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    owner_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "settings": self.settings,
            "quotas": self.quotas,
            "features": list(self.features),
            "branding": self.branding,
            "metadata": self.metadata,
            "owner_id": self.owner_id
        }


@dataclass
class TenantUser:
    """User within a tenant."""
    user_id: str
    tenant_id: str
    global_user_id: str  # Reference to global user
    roles: List[str] = field(default_factory=list)
    permissions: Set[str] = field(default_factory=set)
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_active: Optional[datetime] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "global_user_id": self.global_user_id,
            "roles": self.roles,
            "permissions": list(self.permissions),
            "joined_at": self.joined_at.isoformat(),
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "is_active": self.is_active
        }


@dataclass
class ResourceUsage:
    """Resource usage for a tenant."""
    tenant_id: str
    knowledge_items: int = 0
    storage_bytes: int = 0
    api_calls: int = 0
    users: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "knowledge_items": self.knowledge_items,
            "storage_bytes": self.storage_bytes,
            "api_calls": self.api_calls,
            "users": self.users,
            "last_updated": self.last_updated.isoformat()
        }


class TenantManager:
    """Manages tenants and their isolation."""
    
    # Default quotas for new tenants
    DEFAULT_QUOTAS = {
        "max_knowledge_items": 10000,
        "max_storage_mb": 1000,
        "max_users": 100,
        "max_api_calls_per_day": 10000
    }
    
    # Available features
    AVAILABLE_FEATURES = {
        "basic": {"search", "storage", "sharing"},
        "professional": {"search", "storage", "sharing", "analytics", "workflows"},
        "enterprise": {"search", "storage", "sharing", "analytics", "workflows", 
                      "ml", "distributed", "advanced_security"}
    }
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}  # tenant_id -> Tenant
        self.tenants_by_slug: Dict[str, str] = {}  # slug -> tenant_id
        self.tenant_users: Dict[str, Dict[str, TenantUser]] = defaultdict(dict)  # tenant_id -> {user_id -> TenantUser}
        self.user_tenants: Dict[str, Set[str]] = defaultdict(set)  # global_user_id -> {tenant_ids}
        self.resource_usage: Dict[str, ResourceUsage] = {}
        self.default_quotas = self.DEFAULT_QUOTAS.copy()
        
    def create_tenant(
        self,
        name: str,
        slug: str,
        description: str = "",
        owner_id: Optional[str] = None,
        plan: str = "basic"
    ) -> Tenant:
        """
        Create a new tenant.
        
        Args:
            name: Tenant name
            slug: URL-friendly identifier
            description: Optional description
            owner_id: ID of owner user
            plan: Subscription plan (basic, professional, enterprise)
        """
        # Check slug uniqueness
        if slug in self.tenants_by_slug:
            raise ValueError(f"Tenant with slug '{slug}' already exists")
        
        tenant_id = str(uuid.uuid4())
        
        # Get features for plan
        features = self.AVAILABLE_FEATURES.get(plan, self.AVAILABLE_FEATURES["basic"])
        
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            slug=slug,
            description=description,
            quotas=self.default_quotas.copy(),
            features=features.copy(),
            owner_id=owner_id,
            settings={
                "plan": plan,
                "created_by": owner_id
            }
        )
        
        self.tenants[tenant_id] = tenant
        self.tenants_by_slug[slug] = tenant_id
        self.resource_usage[tenant_id] = ResourceUsage(tenant_id=tenant_id)
        
        logger.info(f"Created tenant: {name} ({tenant_id}) with plan {plan}")
        
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)
    
    def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        tenant_id = self.tenants_by_slug.get(slug)
        if tenant_id:
            return self.tenants.get(tenant_id)
        return None
    
    def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        quotas: Optional[Dict[str, int]] = None
    ) -> Optional[Tenant]:
        """Update tenant settings."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None
        
        if name:
            tenant.name = name
        if description:
            tenant.description = description
        if settings:
            tenant.settings.update(settings)
        if quotas:
            tenant.quotas.update(quotas)
        
        return tenant
    
    def suspend_tenant(self, tenant_id: str, reason: str = "") -> bool:
        """Suspend a tenant."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        tenant.status = "suspended"
        tenant.metadata["suspended_at"] = datetime.utcnow().isoformat()
        tenant.metadata["suspension_reason"] = reason
        
        logger.warning(f"Tenant {tenant_id} suspended: {reason}")
        return True
    
    def activate_tenant(self, tenant_id: str) -> bool:
        """Activate a suspended tenant."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        tenant.status = "active"
        tenant.metadata.pop("suspended_at", None)
        tenant.metadata.pop("suspension_reason", None)
        
        logger.info(f"Tenant {tenant_id} activated")
        return True
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Soft-delete a tenant."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        tenant.status = "deleted"
        tenant.metadata["deleted_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Tenant {tenant_id} deleted")
        return True
    
    def add_user_to_tenant(
        self,
        tenant_id: str,
        global_user_id: str,
        roles: Optional[List[str]] = None
    ) -> Optional[TenantUser]:
        """Add a user to a tenant."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None
        
        # Check if tenant is active
        if tenant.status != "active":
            raise ValueError(f"Tenant {tenant_id} is not active")
        
        # Check user quota
        current_users = len(self.tenant_users[tenant_id])
        if current_users >= tenant.quotas.get("max_users", 100):
            raise ValueError(f"User quota exceeded for tenant {tenant_id}")
        
        # Create tenant user
        tenant_user = TenantUser(
            user_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            global_user_id=global_user_id,
            roles=roles or ["member"]
        )
        
        self.tenant_users[tenant_id][tenant_user.user_id] = tenant_user
        self.user_tenants[global_user_id].add(tenant_id)
        
        # Update usage
        self.resource_usage[tenant_id].users += 1
        
        logger.info(f"Added user {global_user_id} to tenant {tenant_id}")
        
        return tenant_user
    
    def remove_user_from_tenant(
        self,
        tenant_id: str,
        user_id: str
    ) -> bool:
        """Remove a user from a tenant."""
        tenant_user = self.tenant_users[tenant_id].pop(user_id, None)
        if not tenant_user:
            return False
        
        self.user_tenants[tenant_user.global_user_id].discard(tenant_id)
        
        # Update usage
        self.resource_usage[tenant_id].users = max(
            0, 
            self.resource_usage[tenant_id].users - 1
        )
        
        return True
    
    def get_tenant_users(self, tenant_id: str) -> List[TenantUser]:
        """Get all users in a tenant."""
        return list(self.tenant_users.get(tenant_id, {}).values())
    
    def get_user_tenants(self, global_user_id: str) -> List[Tenant]:
        """Get all tenants a user belongs to."""
        tenant_ids = self.user_tenants.get(global_user_id, set())
        return [self.tenants[tid] for tid in tenant_ids if tid in self.tenants]
    
    def check_permission(
        self,
        tenant_id: str,
        user_id: str,
        permission: str
    ) -> bool:
        """Check if a user has a permission in a tenant."""
        tenant_user = self.tenant_users.get(tenant_id, {}).get(user_id)
        if not tenant_user:
            return False
        
        if not tenant_user.is_active:
            return False
        
        return permission in tenant_user.permissions or "admin" in tenant_user.roles
    
    def get_resource_usage(self, tenant_id: str) -> Optional[ResourceUsage]:
        """Get resource usage for a tenant."""
        return self.resource_usage.get(tenant_id)
    
    def update_resource_usage(
        self,
        tenant_id: str,
        knowledge_items: Optional[int] = None,
        storage_bytes: Optional[int] = None,
        api_calls: Optional[int] = None
    ):
        """Update resource usage for a tenant."""
        usage = self.resource_usage.get(tenant_id)
        if not usage:
            return
        
        if knowledge_items is not None:
            usage.knowledge_items = knowledge_items
        if storage_bytes is not None:
            usage.storage_bytes = storage_bytes
        if api_calls is not None:
            usage.api_calls += api_calls
        
        usage.last_updated = datetime.utcnow()
    
    def check_quota(
        self,
        tenant_id: str,
        resource_type: str,
        requested_amount: int = 1
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a quota allows additional resource usage.
        
        Returns:
            (allowed, details)
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False, {"error": "Tenant not found"}
        
        usage = self.resource_usage.get(tenant_id)
        if not usage:
            return False, {"error": "Usage data not found"}
        
        quota_key = f"max_{resource_type}"
        quota = tenant.quotas.get(quota_key)
        
        if quota is None:
            return True, {"quota": "unlimited"}
        
        if resource_type == "knowledge_items":
            current = usage.knowledge_items
        elif resource_type == "storage_mb":
            current = usage.storage_bytes / (1024 * 1024)
        elif resource_type == "users":
            current = usage.users
        else:
            current = 0
        
        remaining = quota - current
        allowed = remaining >= requested_amount
        
        return allowed, {
            "quota": quota,
            "used": current,
            "remaining": remaining,
            "requested": requested_amount
        }
    
    def upgrade_plan(self, tenant_id: str, new_plan: str) -> bool:
        """Upgrade tenant to a new plan."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        if new_plan not in self.AVAILABLE_FEATURES:
            return False
        
        # Update plan and features
        tenant.settings["plan"] = new_plan
        tenant.features = self.AVAILABLE_FEATURES[new_plan].copy()
        
        # Update quotas based on plan
        plan_quotas = {
            "basic": self.DEFAULT_QUOTAS,
            "professional": {
                "max_knowledge_items": 50000,
                "max_storage_mb": 5000,
                "max_users": 500,
                "max_api_calls_per_day": 100000
            },
            "enterprise": {
                "max_knowledge_items": -1,  # Unlimited
                "max_storage_mb": -1,
                "max_users": -1,
                "max_api_calls_per_day": -1
            }
        }
        
        tenant.quotas.update(plan_quotas.get(new_plan, {}))
        
        logger.info(f"Tenant {tenant_id} upgraded to {new_plan}")
        return True
    
    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get statistics for a tenant."""
        tenant = self.tenants.get(tenant_id)
        usage = self.resource_usage.get(tenant_id)
        
        if not tenant or not usage:
            return {}
        
        return {
            "tenant": tenant.to_dict(),
            "usage": usage.to_dict(),
            "user_count": len(self.tenant_users.get(tenant_id, {})),
            "quota_usage": {
                "knowledge_items": {
                    "used": usage.knowledge_items,
                    "quota": tenant.quotas.get("max_knowledge_items", -1),
                    "percentage": (usage.knowledge_items / tenant.quotas["max_knowledge_items"] * 100) 
                                  if tenant.quotas.get("max_knowledge_items", -1) > 0 else 0
                },
                "storage": {
                    "used_mb": usage.storage_bytes / (1024 * 1024),
                    "quota_mb": tenant.quotas.get("max_storage_mb", -1),
                    "percentage": ((usage.storage_bytes / (1024 * 1024)) / tenant.quotas["max_storage_mb"] * 100)
                                  if tenant.quotas.get("max_storage_mb", -1) > 0 else 0
                }
            }
        }
    
    def list_tenants(
        self,
        status: Optional[str] = None,
        plan: Optional[str] = None
    ) -> List[Tenant]:
        """List tenants with optional filtering."""
        tenants = list(self.tenants.values())
        
        if status:
            tenants = [t for t in tenants if t.status == status]
        
        if plan:
            tenants = [t for t in tenants if t.settings.get("plan") == plan]
        
        return tenants


class TenantContext:
    """Context manager for tenant-scoped operations."""
    
    def __init__(self, tenant_manager: TenantManager, tenant_id: str):
        self.tenant_manager = tenant_manager
        self.tenant_id = tenant_id
        self.tenant = tenant_manager.get_tenant(tenant_id)
    
    def __enter__(self):
        if not self.tenant:
            raise ValueError(f"Tenant {self.tenant_id} not found")
        if self.tenant.status != "active":
            raise ValueError(f"Tenant {self.tenant_id} is not active")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def check_feature(self, feature: str) -> bool:
        """Check if tenant has access to a feature."""
        return feature in self.tenant.features
    
    def check_quota(self, resource_type: str, amount: int = 1) -> bool:
        """Check quota within context."""
        allowed, _ = self.tenant_manager.check_quota(
            self.tenant_id, resource_type, amount
        )
        return allowed


__all__ = [
    "TenantManager",
    "Tenant",
    "TenantUser",
    "ResourceUsage",
    "TenantContext"
]
