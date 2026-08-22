"""
Resource Pool Manager - Resource abstraction, pooling and allocation.

Implements the resource abstraction described in
docs/architecture/RESOURCE_MANAGEMENT_SPEC.md:

  * ResourceType / ResourceRequest / ResourceAllocation / ResourcePool models
  * AllocationStrategy hierarchy (bin-packing, fair-share, priority/preemption)
  * ResourcePoolManager: pool lifecycle, request/validate/allocate/release,
    tenant-aware allocation backed by a QuotaManager and ReservationManager.

This module is intentionally dependency-free (stdlib only) and is wired to the
other resource modules (resource_tenancy, resource_scheduler, resource_billing)
through lazy, flat imports so there are no circular-import issues and so the
existing engines/other/resource_manager.py ResourceManager used by
workflow_engine.py is untouched.
"""
from __future__ import annotations

import threading
import uuid
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"
    CUSTOM = "custom"


# Unit labels used for accounting / billing display.
RESOURCE_UNITS = {
    ResourceType.CPU: "cores",
    ResourceType.MEMORY: "mb",
    ResourceType.GPU: "units",
    ResourceType.STORAGE: "gb",
    ResourceType.NETWORK: "mbps",
    ResourceType.CUSTOM: "units",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ResourceManagementError(Exception):
    """Base error for the resource management layer."""


class ResourceRequestValidationError(ResourceManagementError):
    """Raised when a resource request fails validation."""


class QuotaExceededError(ResourceManagementError):
    """Raised when a request exceeds the tenant's quota."""


class ResourceNotAvailableError(ResourceManagementError):
    """Raised when no suitable pool / reservation can satisfy a request."""


class ReservationConflictError(ResourceManagementError):
    """Raised when a reservation conflicts with an existing one."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class ResourceRequest:
    """Resource request specification (multi-dimensional)."""
    cpu_cores: float = 0.0
    memory_mb: float = 0.0
    gpu_units: float = 0.0
    storage_gb: float = 0.0
    network_bandwidth_mbps: float = 0.0
    custom_resources: Dict[str, float] = field(default_factory=dict)

    # Primary allocation dimension used by single-type pools.
    resource_type: ResourceType = ResourceType.CPU
    required_amount: float = 0.0

    # Optional routing hints.
    tenant_id: Optional[str] = None
    workload_id: Optional[str] = None
    priority: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_units": self.gpu_units,
            "storage_gb": self.storage_gb,
            "network_bandwidth_mbps": self.network_bandwidth_mbps,
            "custom_resources": dict(self.custom_resources),
            "resource_type": self.resource_type.value,
            "required_amount": self.required_amount,
            "tenant_id": self.tenant_id,
            "workload_id": self.workload_id,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceRequest":
        rt = data.get("resource_type", ResourceType.CPU.value)
        if isinstance(rt, str):
            rt = ResourceType(rt)
        return cls(
            cpu_cores=float(data.get("cpu_cores", 0.0)),
            memory_mb=float(data.get("memory_mb", 0.0)),
            gpu_units=float(data.get("gpu_units", 0.0)),
            storage_gb=float(data.get("storage_gb", 0.0)),
            network_bandwidth_mbps=float(data.get("network_bandwidth_mbps", 0.0)),
            custom_resources=dict(data.get("custom_resources", {}) or {}),
            resource_type=rt,
            required_amount=float(data.get("required_amount", 0.0)),
            tenant_id=data.get("tenant_id"),
            workload_id=data.get("workload_id"),
            priority=data.get("priority", "normal"),
        )

    def demand_for(self, resource_type: ResourceType) -> float:
        """Return the requested amount for a given resource type."""
        mapping = {
            ResourceType.CPU: self.cpu_cores,
            ResourceType.MEMORY: self.memory_mb,
            ResourceType.GPU: self.gpu_units,
            ResourceType.STORAGE: self.storage_gb,
            ResourceType.NETWORK: self.network_bandwidth_mbps,
        }
        return float(mapping.get(resource_type, 0.0))

    def has_demand(self, resource_type: ResourceType) -> bool:
        return self.demand_for(resource_type) > 0.0


@dataclass
class ResourceAllocation:
    """An active resource allocation."""
    resource_id: str
    resource_type: ResourceType
    allocated_amount: float
    allocated_to: str
    allocated_at: datetime
    pool_id: str = ""
    tenant_id: Optional[str] = None
    workload_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    status: str = "active"  # active | reserved | released
    request: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type.value,
            "allocated_amount": self.allocated_amount,
            "allocated_to": self.allocated_to,
            "allocated_at": self.allocated_at.isoformat(),
            "pool_id": self.pool_id,
            "tenant_id": self.tenant_id,
            "workload_id": self.workload_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status,
            "request": self.request,
            "metadata": dict(self.metadata),
        }


@dataclass
class ResourcePool:
    """A logical pool of a single resource type."""
    pool_id: str
    name: str
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    allocated_capacity: float = 0.0
    reserved_capacity: float = 0.0
    utilization: float = 0.0
    tenants: List[str] = field(default_factory=list)
    policies: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provider_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "name": self.name,
            "resource_type": self.resource_type.value,
            "total_capacity": self.total_capacity,
            "available_capacity": self.available_capacity,
            "allocated_capacity": self.allocated_capacity,
            "reserved_capacity": self.reserved_capacity,
            "utilization": round(self.utilization, 4),
            "tenants": list(self.tenants),
            "policies": dict(self.policies),
            "metadata": dict(self.metadata),
            "provider_id": self.provider_id,
        }


# ---------------------------------------------------------------------------
# Allocation strategies
# ---------------------------------------------------------------------------
PRIORITY_WEIGHTS = {"critical": 3, "high": 2, "normal": 1, "low": 0}


class AllocationStrategy:
    """Base class for resource allocation strategies."""

    def can_allocate_to_pool(self, request: ResourceRequest, pool: ResourcePool) -> bool:
        if pool.resource_type != request.resource_type:
            return False
        return pool.available_capacity >= max(request.required_amount, 0.0)

    def allocate_to_pool(self, request: ResourceRequest, pool: ResourcePool,
                          tenant_id: str) -> ResourceAllocation:
        amount = request.required_amount
        allocation = ResourceAllocation(
            resource_id=f"alloc_{uuid.uuid4().hex[:8]}",
            resource_type=pool.resource_type,
            allocated_amount=amount,
            allocated_to=tenant_id,
            allocated_at=datetime.utcnow(),
            pool_id=pool.pool_id,
            tenant_id=tenant_id,
            workload_id=request.workload_id,
            request=request.to_dict(),
        )
        pool.available_capacity -= amount
        pool.allocated_capacity += amount
        pool.utilization = (pool.allocated_capacity / pool.total_capacity
                            if pool.total_capacity > 0 else 0.0)
        if tenant_id and tenant_id not in pool.tenants:
            pool.tenants.append(tenant_id)
        return allocation

    def select(self, request: ResourceRequest,
               pools: List[ResourcePool], tenant_id: str) -> ResourceAllocation:
        raise NotImplementedError


class BinPackingAllocation(AllocationStrategy):
    """Pack resources tightly by allocating to the least-utilized pool first."""

    def select(self, request: ResourceRequest,
               pools: List[ResourcePool], tenant_id: str) -> ResourceAllocation:
        candidates = [p for p in pools if self.can_allocate_to_pool(request, p)]
        candidates.sort(key=lambda p: p.utilization)
        if not candidates:
            return None
        return self.allocate_to_pool(request, candidates[0], tenant_id)


class FairShareAllocation(AllocationStrategy):
    """Allocate within a tenant's fair share of available resources."""

    def __init__(self, tenant_shares: Optional[Dict[str, float]] = None):
        self.tenant_shares = tenant_shares or {}

    def select(self, request: ResourceRequest,
               pools: List[ResourcePool], tenant_id: str) -> ResourceAllocation:
        share = self.tenant_shares.get(tenant_id, 0.1)
        total_available = sum(
            p.available_capacity for p in pools
            if p.resource_type == request.resource_type
        )
        fair_share = total_available * share
        needed = min(request.required_amount, fair_share)
        if needed <= 0:
            return None
        adjusted = ResourceRequest(**{
            **request.to_dict(),
            "required_amount": needed,
        })
        candidates = [p for p in pools if self.can_allocate_to_pool(adjusted, p)]
        candidates.sort(key=lambda p: p.utilization)
        if not candidates:
            return None
        return self.allocate_to_pool(adjusted, candidates[0], tenant_id)


class PriorityBasedAllocation(AllocationStrategy):
    """Priority-aware allocation with optional preemption of lower priorities."""

    def __init__(self, preemption_enabled: bool = False):
        self.preemption_enabled = preemption_enabled

    def _priority(self, value: str) -> int:
        return PRIORITY_WEIGHTS.get(value, 1)

    def select(self, request: ResourceRequest,
               pools: List[ResourcePool], tenant_id: str) -> ResourceAllocation:
        candidates = [p for p in pools if p.resource_type == request.resource_type]
        candidates.sort(key=lambda p: p.utilization)
        req_priority = self._priority(request.priority)
        for pool in candidates:
            if self.can_allocate_to_pool(request, pool):
                return self.allocate_to_pool(request, pool, tenant_id)
            if self.preemption_enabled:
                freed = self._try_preempt(pool, request, req_priority, tenant_id)
                if freed:
                    return self.allocate_to_pool(request, pool, tenant_id)
        return None

    def _try_preempt(self, pool: ResourcePool, request: ResourceRequest,
                     req_priority: int, tenant_id: str) -> bool:
        # Preemption requires access to the owning manager's allocations.
        manager = getattr(self, "_manager", None)
        if manager is None:
            return False
        freed = 0.0
        for alloc in list(manager.allocations.values()):
            if alloc.pool_id != pool.pool_id or alloc.status != "active":
                continue
            alloc_priority = self._priority(
                (alloc.request or {}).get("priority", "normal")
            )
            if alloc_priority < req_priority:
                freed += alloc.allocated_amount
        return freed >= request.required_amount


# ---------------------------------------------------------------------------
# Resource provider interface (pluggable capacity sources)
# ---------------------------------------------------------------------------
class ResourceProvider:
    """Abstract interface for a resource provider."""

    def get_available_resources(self) -> Dict[ResourceType, float]:
        raise NotImplementedError

    def get_resource_health(self) -> Dict[ResourceType, str]:
        return {rt: "healthy" for rt in ResourceType}


class StaticResourceProvider(ResourceProvider):
    """A provider backed by a fixed capacity dictionary (useful for tests/local)."""

    def __init__(self, capacities: Dict[ResourceType, float]):
        self.capacities = dict(capacities)

    def get_available_resources(self) -> Dict[ResourceType, float]:
        return dict(self.capacities)


# ---------------------------------------------------------------------------
# Resource Pool Manager
# ---------------------------------------------------------------------------
class ResourcePoolManager:
    """
    Manages resource pools, allocations and the request lifecycle.

    The manager is the central entry point for allocating and releasing
    resources. It validates requests against tenant capabilities, enforces
    per-tenant quotas (delegated to QuotaManager) and can reserve capacity
    ahead of time (delegated to ReservationManager).
    """

    def __init__(self, quota_manager: Any = None,
                 reservation_manager: Any = None):
        self.providers: Dict[str, ResourceProvider] = {}
        self.pools: Dict[str, ResourcePool] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        self._quota_manager = quota_manager
        self._reservation_manager = reservation_manager
        self._strategies: Dict[str, AllocationStrategy] = {
            "bin_packing": BinPackingAllocation(),
            "fair_share": FairShareAllocation(),
            "priority": PriorityBasedAllocation(),
        }
        self._lock = threading.RLock()
        logger.info("ResourcePoolManager initialized")

    # -- lazy collaborators ------------------------------------------------
    def _quota(self):
        if self._quota_manager is None:
            from resource_tenancy import QuotaManager
            self._quota_manager = QuotaManager()
        return self._quota_manager

    def _reservations(self):
        if self._reservation_manager is None:
            from resource_scheduler import ReservationManager
            self._reservation_manager = ReservationManager(self)
        return self._reservation_manager

    # -- providers ---------------------------------------------------------
    def register_provider(self, provider_id: str, provider: ResourceProvider):
        with self._lock:
            self.providers[provider_id] = provider

    def create_pool(self, pool_id: str, name: str,
                    resource_type: ResourceType, total_capacity: float,
                    provider_id: Optional[str] = None,
                    policies: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> ResourcePool:
        with self._lock:
            if pool_id in self.pools:
                raise ResourceManagementError(f"Pool {pool_id} already exists")
            pool = ResourcePool(
                pool_id=pool_id,
                name=name,
                resource_type=resource_type,
                total_capacity=total_capacity,
                available_capacity=total_capacity,
                allocated_capacity=0.0,
                reserved_capacity=0.0,
                utilization=0.0,
                tenants=[],
                policies=policies or {},
                metadata=metadata or {},
                provider_id=provider_id,
            )
            self.pools[pool_id] = pool
            logger.info(f"Created resource pool {pool_id} ({resource_type.value})")
            return pool

    def create_pool_from_provider(self, pool_id: str, name: str,
                                  resource_type: ResourceType,
                                  provider_id: str) -> ResourcePool:
        if provider_id not in self.providers:
            raise ResourceManagementError(f"Unknown provider {provider_id}")
        available = self.providers[provider_id].get_available_resources()
        capacity = available.get(resource_type, 0.0)
        return self.create_pool(pool_id, name, resource_type, capacity,
                                provider_id=provider_id)

    # -- request lifecycle -------------------------------------------------
    def validate_request(self, tenant_id: str, request: ResourceRequest) -> bool:
        # Validate against tenant capabilities (delegated to tenant manager
        # if it is installed on the quota manager).
        quota = self._quota()
        tenant_caps = getattr(quota, "get_tenant_capabilities", None)
        if tenant_caps is not None:
            caps = tenant_caps(tenant_id)
            if caps is None:
                return False
            if request.cpu_cores > (caps.get("max_cpu_cores", float("inf"))):
                return False
            if request.memory_mb > (caps.get("max_memory_mb", float("inf"))):
                return False
            if request.gpu_units > (caps.get("max_gpu_units", float("inf"))):
                return False
            if request.storage_gb > (caps.get("max_storage_gb", float("inf"))):
                return False
        if request.required_amount <= 0:
            return False
        if request.resource_type != ResourceType.CUSTOM and request.required_amount <= 0:
            # A non-custom pool allocation requires a positive required amount.
            return True
        return True

    def find_suitable_pool(self, request: ResourceRequest) -> Optional[ResourcePool]:
        candidates = [
            p for p in self.pools.values()
            if p.resource_type == request.resource_type
            and p.available_capacity >= request.required_amount
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.utilization)
        return candidates[0]

    def request_resources(self, tenant_id: str, request: ResourceRequest,
                          priority: Optional[str] = None,
                          strategy: str = "bin_packing") -> ResourceAllocation:
        """Validate, quota-check, and allocate resources for a tenant."""
        if priority:
            request.priority = priority
        request.tenant_id = tenant_id

        if not self.validate_request(tenant_id, request):
            raise ResourceRequestValidationError("Invalid resource request")

        if not self._quota().check_quota(tenant_id, request):
            raise QuotaExceededError(
                f"Resource request exceeds quota for tenant {tenant_id}"
            )

        strat = self._strategies.get(strategy, self._strategies["bin_packing"])
        if isinstance(strat, PriorityBasedAllocation):
            strat._manager = self

        with self._lock:
            allocation = strat.select(request, list(self.pools.values()), tenant_id)
            if allocation is None:
                raise ResourceNotAvailableError(
                    "No suitable resource pool available"
                )
            self.allocations[allocation.resource_id] = allocation
            # Accounting hook: notify billing of an allocation.
            self._notify_accounting(allocation)
            logger.info(
                "Allocated %s of %s to tenant %s (alloc %s)",
                allocation.allocated_amount, allocation.resource_type.value,
                tenant_id, allocation.resource_id,
            )
            return allocation

    def release_resources(self, allocation_id: str) -> None:
        with self._lock:
            if allocation_id not in self.allocations:
                raise ResourceManagementError(
                    f"Allocation {allocation_id} not found"
                )
            allocation = self.allocations[allocation_id]
            pool = self.pools.get(allocation.pool_id)
            if pool is not None:
                pool.available_capacity += allocation.allocated_amount
                pool.allocated_capacity -= allocation.allocated_amount
                pool.utilization = (pool.allocated_capacity / pool.total_capacity
                                    if pool.total_capacity > 0 else 0.0)
            allocation.status = "released"
            allocation.expires_at = datetime.utcnow()
            # Accounting hook: notify billing of release.
            self._notify_accounting_release(allocation)
            del self.allocations[allocation_id]
            logger.info("Released allocation %s", allocation_id)

    def _notify_accounting(self, allocation: ResourceAllocation) -> None:
        try:
            from resource_billing import get_accounting_manager
            acct = get_accounting_manager()
            acct.record_allocation(allocation)
        except Exception as exc:  # pragma: no cover - accounting is best-effort
            logger.debug("Accounting notification skipped: %s", exc)

    def _notify_accounting_release(self, allocation: ResourceAllocation) -> None:
        try:
            from resource_billing import get_accounting_manager
            acct = get_accounting_manager()
            acct.record_release(allocation)
        except Exception as exc:  # pragma: no cover
            logger.debug("Accounting release notification skipped: %s", exc)

    # -- introspection -----------------------------------------------------
    def get_allocation(self, allocation_id: str) -> Optional[ResourceAllocation]:
        return self.allocations.get(allocation_id)

    def get_tenant_allocations(self, tenant_id: str) -> List[ResourceAllocation]:
        return [a for a in self.allocations.values() if a.tenant_id == tenant_id]

    def get_pool_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                pool_id: pool.to_dict() for pool_id, pool in self.pools.items()
            }

    def get_allocations(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [a.to_dict() for a in self.allocations.values()]

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total = defaultdict(float)
            for pool in self.pools.values():
                total[pool.resource_type.value] += pool.total_capacity
            return {
                "pools": len(self.pools),
                "active_allocations": len(self.allocations),
                "total_capacity_by_type": dict(total),
            }


# Module-level default manager (lazy singleton) ----------------------------
_default_pool_manager: Optional[ResourcePoolManager] = None


def get_pool_manager() -> ResourcePoolManager:
    global _default_pool_manager
    if _default_pool_manager is None:
        _default_pool_manager = ResourcePoolManager()
    return _default_pool_manager
