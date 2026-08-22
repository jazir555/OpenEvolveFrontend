"""
Resource Multi-Tenancy and Quota Management.

Provides:
  * QuotaManager      - per-tenant, per-resource-type quota enforcement
  * MultiTenantManager - tenant registry, capabilities and isolation of
                         allocations across tenants

Keeps the flat engines/other import style (no relative imports, no __init__).
"""
from __future__ import annotations

import threading
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from resource_pool_manager import (
    ResourceType,
    ResourceRequest,
    ResourceManagementError,
)

logger = logging.getLogger(__name__)


@dataclass
class TenantCapabilities:
    """Maximum resource request a tenant may make in a single request."""
    max_cpu_cores: float = float("inf")
    max_memory_mb: float = float("inf")
    max_gpu_units: float = float("inf")
    max_storage_gb: float = float("inf")
    max_network_mbps: float = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_cpu_cores": self.max_cpu_cores,
            "max_memory_mb": self.max_memory_mb,
            "max_gpu_units": self.max_gpu_units,
            "max_storage_gb": self.max_storage_gb,
            "max_network_mbps": self.max_network_mbps,
        }


# Default per-tenant quotas (cumulative allocated amount per resource type).
DEFAULT_QUOTAS = {
    ResourceType.CPU: 32.0,
    ResourceType.MEMORY: 131072.0,
    ResourceType.GPU: 8.0,
    ResourceType.STORAGE: 4096.0,
    ResourceType.NETWORK: 10240.0,
    ResourceType.CUSTOM: float("inf"),
}


class QuotaManager:
    """
    Enforces per-tenant quotas across resource types.

    A quota is the maximum *allocated* amount of a resource type a tenant may
    hold at any one time. The manager tracks live usage by observing
    allocations recorded via ``record_allocation`` / ``record_release`` and
    answers ``check_quota`` before new allocations are granted.
    """

    def __init__(self, default_quotas: Optional[Dict[ResourceType, float]] = None):
        self.default_quotas = dict(default_quotas or DEFAULT_QUOTAS)
        self.tenant_quotas: Dict[str, Dict[ResourceType, float]] = {}
        self.tenant_capabilities: Dict[str, TenantCapabilities] = {}
        self.usage: Dict[str, Dict[ResourceType, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._lock = threading.RLock()

    # -- configuration -----------------------------------------------------
    def set_tenant_quota(self, tenant_id: str, resource_type: ResourceType,
                         limit: float) -> None:
        with self._lock:
            self.tenant_quotas.setdefault(tenant_id, {})[resource_type] = limit

    def set_tenant_capabilities(self, tenant_id: str,
                                caps: TenantCapabilities) -> None:
        with self._lock:
            self.tenant_capabilities[tenant_id] = caps

    def get_quota(self, tenant_id: str, resource_type: ResourceType) -> float:
        with self._lock:
            return self.tenant_quotas.get(tenant_id, {}).get(
                resource_type, self.default_quotas.get(resource_type, float("inf"))
            )

    def get_tenant_capabilities(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            caps = self.tenant_capabilities.get(tenant_id)
            if caps is None:
                # Default: no per-request caps (quotas still apply).
                caps = TenantCapabilities()
            return caps.to_dict()

    # -- usage tracking ----------------------------------------------------
    def record_allocation(self, tenant_id: str, resource_type: ResourceType,
                          amount: float) -> None:
        with self._lock:
            self.usage[tenant_id][resource_type] += amount

    def record_release(self, tenant_id: str, resource_type: ResourceType,
                       amount: float) -> None:
        with self._lock:
            current = self.usage[tenant_id].get(resource_type, 0.0)
            self.usage[tenant_id][resource_type] = max(0.0, current - amount)

    def get_usage(self, tenant_id: str) -> Dict[str, float]:
        with self._lock:
            return {
                rt.value: self.usage.get(tenant_id, {}).get(rt, 0.0)
                for rt in ResourceType
            }

    # -- enforcement -------------------------------------------------------
    def check_quota(self, tenant_id: str, request: ResourceRequest) -> bool:
        """Return True if the request would stay within every tenant quota."""
        with self._lock:
            for rt in ResourceType:
                demand = request.demand_for(rt)
                if demand <= 0:
                    continue
                limit = self.get_quota(tenant_id, rt)
                if limit == float("inf"):
                    continue
                used = self.usage.get(tenant_id, {}).get(rt, 0.0)
                if used + demand > limit + 1e-9:
                    logger.info(
                        "Quota exceeded for tenant %s on %s (used=%s limit=%s demand=%s)",
                        tenant_id, rt.value, used, limit, demand,
                    )
                    return False
            return True

    def check_quota_multi(self, tenant_id: str,
                          demands: Dict[ResourceType, float]) -> bool:
        with self._lock:
            for rt, demand in demands.items():
                if demand <= 0:
                    continue
                limit = self.get_quota(tenant_id, rt)
                if limit == float("inf"):
                    continue
                used = self.usage.get(tenant_id, {}).get(rt, 0.0)
                if used + demand > limit + 1e-9:
                    return False
            return True


class MultiTenantManager:
    """
    Manages tenant identity, capabilities and resource isolation.

    Allocations are tagged with a ``tenant_id``; this manager guarantees that
    accounting and capability lookups are scoped per tenant and provides an
    isolation boundary so that one tenant cannot observe or consume another
    tenant's reserved capacity.
    """

    def __init__(self, quota_manager: Optional[QuotaManager] = None):
        self.quota_manager = quota_manager or QuotaManager()
        self.tenants: Dict[str, Dict[str, Any]] = {}
        self.tenant_allocations: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.RLock()

    def register_tenant(self, tenant_id: str,
                        name: Optional[str] = None,
                        capabilities: Optional[TenantCapabilities] = None,
                        quotas: Optional[Dict[ResourceType, float]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            if tenant_id in self.tenants:
                raise ResourceManagementError(
                    f"Tenant {tenant_id} already registered"
                )
            self.tenants[tenant_id] = {
                "tenant_id": tenant_id,
                "name": name or tenant_id,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
                "active": True,
            }
            if capabilities is not None:
                self.quota_manager.set_tenant_capabilities(tenant_id, capabilities)
            if quotas:
                for rt, limit in quotas.items():
                    self.quota_manager.set_tenant_quota(tenant_id, rt, limit)
            logger.info("Registered tenant %s", tenant_id)

    def deactivate_tenant(self, tenant_id: str) -> None:
        with self._lock:
            if tenant_id in self.tenants:
                self.tenants[tenant_id]["active"] = False

    def is_active(self, tenant_id: str) -> bool:
        t = self.tenants.get(tenant_id)
        return bool(t and t.get("active"))

    def ensure_isolation(self, tenant_id: str) -> None:
        """Raise if the tenant is unknown or inactive."""
        if not self.is_active(tenant_id):
            raise ResourceManagementError(
                f"Tenant {tenant_id} is not active / unknown"
            )

    def associate_allocation(self, tenant_id: str, allocation_id: str) -> None:
        with self._lock:
            self.tenant_allocations[tenant_id].append(allocation_id)

    def get_tenant_allocations(self, tenant_id: str) -> List[str]:
        return list(self.tenant_allocations.get(tenant_id, []))

    def get_tenant_summary(self, tenant_id: str) -> Dict[str, Any]:
        return {
            "tenant": self.tenants.get(tenant_id),
            "quotas": {
                rt.value: self.quota_manager.get_quota(tenant_id, rt)
                for rt in ResourceType
            },
            "usage": self.quota_manager.get_usage(tenant_id),
            "allocation_count": len(self.get_tenant_allocations(tenant_id)),
        }

    def list_tenants(self) -> List[Dict[str, Any]]:
        return [t for t in self.tenants.values()]
