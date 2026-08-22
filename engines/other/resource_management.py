"""
Resource Management Layer - facade that composes the full epic.

Wires together: ResourcePoolManager, QuotaManager, MultiTenantManager,
ReservationManager, WorkloadScheduler, PlacementEngine, AutoScaler,
ResourceOptimizationEngine, AccountingManager and BillingManager into a single
cohesive, ready-to-use API.

This keeps the flat engines/other import style (no relative imports, no
__init__.py) and does not modify the existing resource_pool.py /
resource_manager.py modules used by workflow_engine.py, so nothing is broken.

    from resource_management import ResourceManagementLayer
    rml = ResourceManagementLayer()
    rml.register_tenant("acme")
    rml.create_pool("cpu-pool", "CPU", 64.0)
    alloc = rml.request_resources("acme", request)
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from resource_pool_manager import (
    ResourcePoolManager,
    ResourceType,
    ResourceRequest,
    ResourceAllocation,
    ResourceManagementError,
    ResourceNotAvailableError,
)
from resource_tenancy import (
    MultiTenantManager,
    QuotaManager,
    TenantCapabilities,
)
from resource_scheduler import (
    ReservationManager,
    WorkloadScheduler,
    PlacementEngine,
    ComputeNode,
    AutoScaler,
    ResourceOptimizationEngine,
    ScalingDecision,
)
from resource_billing import (
    AccountingManager,
    BillingManager,
)

logger = logging.getLogger(__name__)


class ResourceManagementLayer:
    """
    Top-level coordinator for the resource management epic.

    Owns the lifecycle of every sub-manager and exposes consolidated
    operations: tenant registration, pool creation, quota-aware allocation,
    time reservations, workload scheduling, elastic optimization and billing.
    """

    def __init__(self, autoscaler: Optional[AutoScaler] = None,
                 optimization_history_window: int = 20):
        # Accounting + billing share a ledger.
        self.accounting_manager = AccountingManager()
        self.billing_manager = BillingManager(
            accounting_manager=self.accounting_manager
        )

        # Tenancy + quotas.
        self.quota_manager = QuotaManager()
        self.tenant_manager = MultiTenantManager(
            quota_manager=self.quota_manager
        )

        # Pools + reservations.
        self.pool_manager = ResourcePoolManager(
            quota_manager=self.quota_manager,
        )
        self.reservation_manager = ReservationManager(self.pool_manager)

        # Scheduling + placement.
        self.placement_engine = PlacementEngine()
        self.scheduler = WorkloadScheduler(
            pool_manager=self.pool_manager,
            placement_engine=self.placement_engine,
        )

        # Elastic optimization.
        self.autoscaler = autoscaler or AutoScaler()
        self.optimization_engine = ResourceOptimizationEngine(
            autoscaler=self.autoscaler,
            history_window=optimization_history_window,
        )

        logger.info("ResourceManagementLayer initialized")

    # -- tenancy ----------------------------------------------------------
    def register_tenant(self, tenant_id: str, name: Optional[str] = None,
                        capabilities: Optional[TenantCapabilities] = None,
                        quotas: Optional[Dict[ResourceType, float]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        self.tenant_manager.register_tenant(
            tenant_id, name=name, capabilities=capabilities,
            quotas=quotas, metadata=metadata,
        )

    def list_tenants(self) -> List[Dict[str, Any]]:
        return self.tenant_manager.list_tenants()

    # -- pools ------------------------------------------------------------
    def create_pool(self, pool_id: str, name: str,
                    resource_type: ResourceType, total_capacity: float,
                    policies: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pool = self.pool_manager.create_pool(
            pool_id, name, resource_type, total_capacity,
            policies=policies, metadata=metadata,
        )
        return pool.to_dict()

    def create_node(self, node_id: str,
                    capacity: Dict[ResourceType, float],
                    labels: Optional[Dict[str, str]] = None,
                    cost_per_hour: float = 0.0,
                    reliability: float = 1.0,
                    performance_score: float = 1.0) -> None:
        node = ComputeNode(
            node_id=node_id, capacity=capacity,
            labels=labels or {}, cost_per_hour=cost_per_hour,
            reliability=reliability, performance_score=performance_score,
        )
        self.scheduler.register_node(node)

    # -- allocation -------------------------------------------------------
    def request_resources(self, tenant_id: str,
                          request: ResourceRequest,
                          priority: Optional[str] = None,
                          strategy: str = "bin_packing"
                          ) -> Dict[str, Any]:
        self.tenant_manager.ensure_isolation(tenant_id)
        allocation = self.pool_manager.request_resources(
            tenant_id, request, priority=priority, strategy=strategy
        )
        self.tenant_manager.associate_allocation(tenant_id, allocation.resource_id)
        return allocation.to_dict()

    def release(self, allocation_id: str) -> None:
        self.pool_manager.release_resources(allocation_id)

    # -- reservations -----------------------------------------------------
    def reserve_resources(self, tenant_id: str, pool_id: str, amount: float,
                          start_time: datetime, duration: timedelta,
                          resource_type: Optional[ResourceType] = None,
                          workload_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
        self.tenant_manager.ensure_isolation(tenant_id)
        return self.reservation_manager.create_reservation(
            tenant_id, pool_id, amount, start_time, duration,
            resource_type=resource_type, workload_id=workload_id,
            metadata=metadata,
        )

    def cancel_reservation(self, reservation_id: str) -> Dict[str, Any]:
        return self.reservation_manager.cancel_reservation(reservation_id)

    def activate_reservation(self, reservation_id: str) -> Dict[str, Any]:
        allocation = self.reservation_manager.activate_reservation(reservation_id)
        self.tenant_manager.associate_allocation(
            allocation.tenant_id, allocation.resource_id
        )
        return allocation.to_dict()

    # -- scheduling -------------------------------------------------------
    def schedule_workload(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        self.tenant_manager.ensure_isolation(spec["tenant_id"])
        return self.scheduler.schedule_workload(spec)

    def drain_workload_queue(self) -> List[Dict[str, Any]]:
        return self.scheduler.drain_queue()

    # -- billing ----------------------------------------------------------
    def topup(self, tenant_id: str, amount: float) -> float:
        return self.billing_manager.topup(tenant_id, amount)

    def invoice(self, tenant_id: str, apply_balance: bool = True
                ) -> Dict[str, Any]:
        return self.billing_manager.generate_invoice(tenant_id, apply_balance)

    def get_statement(self, tenant_id: str) -> Dict[str, Any]:
        return self.billing_manager.get_statement(tenant_id)

    # -- elastic optimization --------------------------------------------
    def record_demand(self, pool_id: str, demand: float) -> None:
        self.optimization_engine.record_demand(pool_id, demand)

    def optimize(self, pool_id: str) -> Dict[str, Any]:
        pool = self.pool_manager.pools.get(pool_id)
        if pool is None:
            raise ResourceManagementError(f"Unknown pool {pool_id}")
        return self.optimization_engine.optimize(pool)

    def autoscale(self, pool_id: str, forecast_demand: Optional[float] = None
                  ) -> Dict[str, Any]:
        pool = self.pool_manager.pools.get(pool_id)
        if pool is None:
            raise ResourceManagementError(f"Unknown pool {pool_id}")
        if forecast_demand is None:
            forecast_demand = self.optimization_engine.predict_demand(pool_id)
        decision = self.autoscaler.evaluate(pool, forecast_demand)
        self.autoscaler.apply(pool, decision)
        return decision.to_dict()

    # -- introspection ----------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "pools": self.pool_manager.get_pool_stats(),
            "allocations": self.pool_manager.get_allocations(),
            "reservations": self.reservation_manager.list_reservations(),
            "tenants": [t["tenant_id"] for t in self.tenant_manager.list_tenants()],
            "summary": self.pool_manager.get_summary(),
        }


# Module-level singleton ---------------------------------------------------
_default_layer: Optional["ResourceManagementLayer"] = None


def get_resource_management_layer() -> "ResourceManagementLayer":
    global _default_layer
    if _default_layer is None:
        _default_layer = ResourceManagementLayer()
    return _default_layer


# Re-export common names so callers can `from resource_management import X`.
__all__ = [
    "ResourceManagementLayer",
    "get_resource_management_layer",
    "ResourcePoolManager",
    "ResourceType",
    "ResourceRequest",
    "ResourceAllocation",
    "MultiTenantManager",
    "QuotaManager",
    "ReservationManager",
    "WorkloadScheduler",
    "PlacementEngine",
    "ComputeNode",
    "AutoScaler",
    "ResourceOptimizationEngine",
    "AccountingManager",
    "BillingManager",
    "ScalingDecision",
    "TenantCapabilities",
]
