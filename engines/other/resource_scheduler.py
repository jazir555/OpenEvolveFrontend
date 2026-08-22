"""
Elastic resource scheduling, placement and reservation.

Implements the resource-scheduling chapter of
docs/architecture/RESOURCE_MANAGEMENT_SPEC.md:

  * ReservationManager   - time-windowed capacity reservations
  * PlacementEngine      - scores candidate nodes for a workload
  * WorkloadScheduler    - validates, places and allocates workloads
  * AutoScaler           - demand-based scale up/down decisions
  * ResourceOptimizationEngine - analyzes usage, predicts demand and
                                 recommends optimizations

Flat engines/other module: no relative imports, no __init__.
"""
from __future__ import annotations

import heapq
import threading
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict

from resource_pool_manager import (
    ResourceType,
    ResourceRequest,
    ResourcePool,
    ResourceAllocation,
    ResourceManagementError,
    ResourceNotAvailableError,
    ReservationConflictError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compute nodes & placement
# ---------------------------------------------------------------------------
@dataclass
class ComputeNode:
    """A schedulable node with capacity and soft attributes."""
    node_id: str
    capacity: Dict[ResourceType, float] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    cost_per_hour: float = 0.0
    reliability: float = 1.0
    performance_score: float = 1.0

    def capacity_for(self, rt: ResourceType) -> float:
        return float(self.capacity.get(rt, 0.0))


@dataclass
class PlacementResult:
    node_id: str
    score: float
    resource_fit: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "score": round(self.score, 4),
            "resource_fit": round(self.resource_fit, 4),
        }


class PlacementEngine:
    """Scores and ranks nodes for a placement decision."""

    # Composite score weights. Resource availability is king.
    SCORE_WEIGHTS = {
        "resources": 0.40,
        "performance": 0.25,
        "reliability": 0.15,
        "cost": 0.10,
        "affinity": 0.10,
    }

    def __init__(self, score_weights: Optional[Dict[str, float]] = None):
        self.score_weights = dict(score_weights or self.SCORE_WEIGHTS)

    def filter_nodes(self, nodes: List[ComputeNode],
                     constraints: Optional[Dict[str, Any]] = None) -> List[ComputeNode]:
        if not constraints:
            return list(nodes)
        result = []
        for node in nodes:
            ok = True
            for key, value in constraints.items():
                if key == "label":
                    for lk, lv in value.items():
                        if node.labels.get(lk) != lv:
                            ok = False
                elif key == "min_reliability":
                    if node.reliability < value:
                        ok = False
                elif key == "max_cost_per_hour":
                    if node.cost_per_hour > value:
                        ok = False
            if ok:
                result.append(node)
        return result

    def score_node(self, node: ComputeNode,
                   request: ResourceRequest) -> float:
        res = self._score_resources(node, request)
        perf = min(max(node.performance_score, 0.0), 1.0)
        rel = min(max(node.reliability, 0.0), 1.0)
        cost = self._score_cost(node)
        aff = self._score_affinity(node, request)
        w = self.score_weights
        return (
            res * w.get("resources", 0.4)
            + perf * w.get("performance", 0.25)
            + rel * w.get("reliability", 0.15)
            + cost * w.get("cost", 0.10)
            + aff * w.get("affinity", 0.10)
        )

    def _score_resources(self, node: ComputeNode,
                         request: ResourceRequest) -> float:
        rt = request.resource_type
        need = request.required_amount
        have = node.capacity_for(rt)
        if have <= 0:
            return 0.0
        # Higher score when the node has ample spare capacity relative to need.
        return min(have / max(need, 1e-9), 2.0) / 2.0

    def _score_cost(self, node: ComputeNode) -> float:
        # Lower cost -> higher score (normalized against a soft ceiling).
        ceiling = 10.0
        return max(0.0, 1.0 - node.cost_per_hour / ceiling)

    def _score_affinity(self, node: ComputeNode,
                        request: ResourceRequest) -> float:
        # Honor a preferred label if the request carries one.
        preferred = (request.metadata or {}).get("preferred_label")
        if not preferred:
            return 0.5
        for lk, lv in node.labels.items():
            if f"{lk}={lv}" == preferred or lv == preferred:
                return 1.0
        return 0.2

    def find_placement(self, request: ResourceRequest,
                       nodes: List[ComputeNode],
                       constraints: Optional[Dict[str, Any]] = None
                       ) -> Optional[PlacementResult]:
        if not nodes:
            return None
        candidates = self.filter_nodes(nodes, constraints)
        if not candidates:
            return None
        scored = []
        for node in candidates:
            score = self.score_node(node, request)
            fit = self._score_resources(node, request)
            scored.append((score, PlacementResult(node.node_id, score, fit)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]


# ---------------------------------------------------------------------------
# Reservation manager
# ---------------------------------------------------------------------------
@dataclass
class Reservation:
    reservation_id: str
    tenant_id: str
    pool_id: str
    resource_type: ResourceType
    amount: float
    start_time: datetime
    end_time: datetime
    status: str = "confirmed"
    workload_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "tenant_id": self.tenant_id,
            "pool_id": self.pool_id,
            "resource_type": self.resource_type.value,
            "amount": self.amount,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "status": self.status,
            "workload_id": self.workload_id,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


class ReservationManager:
    """Manages time-windowed capacity reservations for resource pools."""

    def __init__(self, pool_manager: Optional[Any] = None):
        self.pool_manager = pool_manager
        self.reservations: Dict[str, Reservation] = {}
        self._lock = threading.RLock()
        logger.info("ReservationManager initialized")

    def _get_pool(self, pool_id: str) -> Optional[ResourcePool]:
        if self.pool_manager is None:
            return None
        return self.pool_manager.pools.get(pool_id)

    def check_conflict(self, pool_id: str, amount: float,
                       start_time: datetime, end_time: datetime) -> bool:
        """True if a new reservation would over-commit the pool in [start,end]."""
        pool = self._get_pool(pool_id)
        if pool is None:
            return False
        # Existing reserved amount active during the window.
        overlap_reserved = 0.0
        for res in self.reservations.values():
            if res.pool_id != pool_id or res.status != "confirmed":
                continue
            if res.start_time < end_time and res.end_time > start_time:
                overlap_reserved += res.amount
        # Capacity available for *new* reservation = available - currently reserved
        available_for_reservation = pool.available_capacity - pool.reserved_capacity
        return (overlap_reserved + amount) > (pool.total_capacity + 1e-9) or \
            amount > available_for_reservation + 1e-9

    def create_reservation(self, tenant_id: str, pool_id: str,
                           amount: float, start_time: datetime,
                           duration: timedelta, resource_type: Optional[ResourceType] = None,
                           workload_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None
                           ) -> Dict[str, Any]:
        pool = self._get_pool(pool_id)
        if pool is None:
            raise ResourceManagementError(f"Unknown pool {pool_id}")
        rt = resource_type or pool.resource_type
        end_time = start_time + duration

        with self._lock:
            if self.check_conflict(pool_id, amount, start_time, end_time):
                alts = self.find_alternative_times(
                    pool_id, amount, start_time, duration
                )
                return {
                    "success": False,
                    "conflict": True,
                    "alternatives": alts,
                }
            reservation = Reservation(
                reservation_id=f"res_{uuid.uuid4().hex[:8]}",
                tenant_id=tenant_id,
                pool_id=pool_id,
                resource_type=rt,
                amount=amount,
                start_time=start_time,
                end_time=end_time,
                status="confirmed",
                workload_id=workload_id,
                metadata=metadata or {},
            )
            self.reservations[reservation.reservation_id] = reservation
            pool.reserved_capacity += amount
            pool.available_capacity -= amount
            logger.info(
                "Reservation %s created for tenant %s on pool %s",
                reservation.reservation_id, tenant_id, pool_id,
            )
            return {
                "success": True,
                "reservation_id": reservation.reservation_id,
                "start_time": reservation.start_time.isoformat(),
                "end_time": reservation.end_time.isoformat(),
            }

    def find_alternative_times(self, pool_id: str, amount: float,
                               preferred_start: datetime,
                               duration: timedelta) -> List[Dict[str, Any]]:
        alternatives = []
        for offset in [-30, -15, 15, 30, 60]:
            start = preferred_start + timedelta(minutes=offset)
            end = start + duration
            if not self.check_conflict(pool_id, amount, start, end):
                alternatives.append({
                    "start_time": start.isoformat(),
                    "end_time": end.isoformat(),
                    "offset_minutes": offset,
                })
            if len(alternatives) >= 5:
                break
        return alternatives

    def cancel_reservation(self, reservation_id: str) -> Dict[str, Any]:
        with self._lock:
            res = self.reservations.get(reservation_id)
            if res is None:
                raise ResourceManagementError(
                    f"Reservation {reservation_id} not found"
                )
            pool = self._get_pool(res.pool_id)
            if pool is not None and res.status == "confirmed":
                pool.reserved_capacity = max(
                    0.0, pool.reserved_capacity - res.amount
                )
                pool.available_capacity += res.amount
            res.status = "cancelled"
            res.metadata["cancelled_at"] = datetime.utcnow().isoformat()
            logger.info("Reservation %s cancelled", reservation_id)
            return {"success": True, "reservation_id": reservation_id}

    def activate_reservation(self, reservation_id: str,
                             pool_manager: Optional[Any] = None) -> ResourceAllocation:
        """Convert a reservation into an active allocation."""
        with self._lock:
            res = self.reservations.get(reservation_id)
            if res is None:
                raise ResourceManagementError(
                    f"Reservation {reservation_id} not found"
                )
            if res.status != "confirmed":
                raise ReservationConflictError(
                    f"Reservation {reservation_id} is {res.status}"
                )
            pm = pool_manager or self.pool_manager
            if pm is None:
                raise ResourceManagementError("No pool manager available")
            request = ResourceRequest(
                resource_type=res.resource_type,
                required_amount=res.amount,
                tenant_id=res.tenant_id,
                workload_id=res.workload_id,
                priority="high",
            )
            allocation = pm.request_resources(
                res.tenant_id, request, priority="high"
            )
            pool = self._get_pool(res.pool_id)
            if pool is not None:
                # Reserve -> allocated: free the reserved bookkeeping.
                pool.reserved_capacity = max(
                    0.0, pool.reserved_capacity - res.amount
                )
            res.status = "activated"
            res.metadata["allocation_id"] = allocation.resource_id
            return allocation

    def list_reservations(self, tenant_id: Optional[str] = None
                           ) -> List[Dict[str, Any]]:
        with self._lock:
            items = [r for r in self.reservations.values()]
            if tenant_id is not None:
                items = [r for r in items if r.tenant_id == tenant_id]
            return [r.to_dict() for r in items]


# ---------------------------------------------------------------------------
# Workload scheduler
# ---------------------------------------------------------------------------
@dataclass(order=True)
class _QueueItem:
    priority: int
    enqueue_order: int
    workload_id: str
    spec: Dict[str, Any] = field(compare=False)


class WorkloadScheduler:
    """Validates, places and allocates workloads across nodes/pools."""

    def __init__(self, pool_manager: ResourcePoolManager,
                 placement_engine: Optional[PlacementEngine] = None,
                 nodes: Optional[List[ComputeNode]] = None):
        self.pool_manager = pool_manager
        self.placement_engine = placement_engine or PlacementEngine()
        self.nodes: List[ComputeNode] = nodes or []
        self._queue: List[_QueueItem] = []
        self._queue_index: Dict[str, _QueueItem] = {}
        self._order = 0
        self._lock = threading.RLock()
        self.scheduling_results: Dict[str, Dict[str, Any]] = {}

    def register_node(self, node: ComputeNode) -> None:
        with self._lock:
            self.nodes.append(node)

    def _build_request(self, spec: Dict[str, Any]) -> ResourceRequest:
        resources = spec.get("resources", {})
        request = ResourceRequest(
            cpu_cores=float(resources.get("cpu_cores", 0.0)),
            memory_mb=float(resources.get("memory_mb", 0.0)),
            gpu_units=float(resources.get("gpu_units", 0.0)),
            storage_gb=float(resources.get("storage_gb", 0.0)),
            network_bandwidth_mbps=float(resources.get("network_bandwidth_mbps", 0.0)),
            tenant_id=spec.get("tenant_id"),
            workload_id=spec.get("workload_id") or spec.get("id"),
            priority=spec.get("priority", "normal"),
            metadata=spec.get("metadata", {}),
        )
        # Derive the primary pool allocation from the dominant resource.
        rt, amount = self._dominant_resource(spec)
        request.resource_type = rt
        request.required_amount = amount
        return request

    @staticmethod
    def _dominant_resource(spec: Dict[str, Any]) -> (ResourceType, float):
        resources = spec.get("resources", {})
        mapping = {
            ResourceType.CPU: float(resources.get("cpu_cores", 0.0)),
            ResourceType.MEMORY: float(resources.get("memory_mb", 0.0)),
            ResourceType.GPU: float(resources.get("gpu_units", 0.0)),
            ResourceType.STORAGE: float(resources.get("storage_gb", 0.0)),
            ResourceType.NETWORK: float(resources.get("network_bandwidth_mbps", 0.0)),
        }
        best_rt, best_amt = ResourceType.CPU, 0.0
        for rt, amt in mapping.items():
            if amt > best_amt:
                best_rt, best_amt = rt, amt
        return best_rt, best_amt

    def validate_workload(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        resources = spec.get("resources", {})
        if float(resources.get("cpu_cores", 0.0)) <= 0:
            errors.append("CPU cores must be positive")
        if float(resources.get("memory_mb", 0.0)) <= 0:
            errors.append("Memory must be positive")
        if "tenant_id" not in spec:
            errors.append("tenant_id is required")
        return {"valid": len(errors) == 0, "errors": errors}

    def schedule_workload(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        validation = self.validate_workload(spec)
        if not validation["valid"]:
            raise ResourceManagementError(
                "Invalid workload: " + "; ".join(validation["errors"])
            )

        request = self._build_request(spec)
        tenant_id = spec["tenant_id"]

        # Check overall resource availability.
        available = self.pool_manager.find_suitable_pool(request) is not None
        if not available:
            return self._enqueue(spec, reason="no_suitable_pool")

        # Find optimal placement among nodes (best-effort; does not block alloc).
        placement = None
        if self.nodes:
            placement = self.placement_engine.find_placement(
                request, self.nodes, spec.get("constraints")
            )
            if placement is None:
                return self._enqueue(spec, reason="no_suitable_placement")

        allocation = self.pool_manager.request_resources(tenant_id, request)
        result = {
            "status": "scheduled",
            "allocation_id": allocation.resource_id,
            "pool_id": allocation.pool_id,
            "placement_node": placement.node_id if placement else None,
            "placement_score": placement.score if placement else None,
        }
        self.scheduling_results[spec.get("workload_id") or spec.get("id")] = result
        return result

    def _enqueue(self, spec: Dict[str, Any], reason: str) -> Dict[str, Any]:
        wid = spec.get("workload_id") or spec.get("id")
        priority = {"critical": 3, "high": 2, "normal": 1, "low": 0}.get(
            spec.get("priority", "normal"), 1
        )
        with self._lock:
            if wid in self._queue_index:
                return {
                    "status": "queued",
                    "queue_position": len(self._queue),
                    "reason": "already_queued",
                }
            item = _QueueItem(
                priority=priority, enqueue_order=self._order,
                workload_id=wid, spec=spec,
            )
            self._order += 1
            heapq.heappush(self._queue, item)
            self._queue_index[wid] = item
        return {
            "status": "queued",
            "workload_id": wid,
            "reason": reason,
            "queue_position": len(self._queue),
        }

    def drain_queue(self) -> List[Dict[str, Any]]:
        """Attempt to schedule every queued workload (single synchronous pass)."""
        results = []
        with self._lock:
            pending = []
            while self._queue:
                pending.append(heapq.heappop(self._queue))
            self._queue_index.clear()
        for item in pending:
            try:
                result = self.schedule_workload(item.spec)
            except ResourceManagementError as exc:
                result = {"status": "failed", "error": str(exc)}
            wid = item.spec.get("workload_id") or item.spec.get("id")
            self.scheduling_results[wid] = result
            results.append(result)
        return results

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.scheduling_results)


# ---------------------------------------------------------------------------
# Auto-scaler & optimization engine
# ---------------------------------------------------------------------------
@dataclass
class ScalingDecision:
    direction: str  # scale_up | scale_down | none
    current_capacity: float
    target_capacity: float
    recommended_change: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "current_capacity": self.current_capacity,
            "target_capacity": self.target_capacity,
            "recommended_change": self.recommended_change,
            "reason": self.reason,
        }


class AutoScaler:
    """
    Demand-based auto-scaler.

    Given a pool's current utilization and a demand forecast (the expected
    amount of resources required over the next interval), decide whether to
    scale the pool's total capacity up or down, honoring configurable
    thresholds and min/max bounds.
    """

    def __init__(self, target_utilization: float = 0.7,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3,
                 min_capacity: float = 0.0,
                 max_capacity: Optional[float] = None,
                 scale_step_factor: float = 0.25):
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.scale_step_factor = scale_step_factor

    def evaluate(self, pool: ResourcePool,
                 forecast_demand: float) -> ScalingDecision:
        total = pool.total_capacity
        # Use the greater of current allocated and forecast to size capacity.
        required = max(pool.allocated_capacity, forecast_demand)
        target = required / max(self.target_utilization, 1e-6)

        current_util = (pool.allocated_capacity / total) if total > 0 else 1.0

        if current_util >= self.scale_up_threshold or target > total:
            change = max(target - total, total * self.scale_step_factor)
            new_cap = total + change
            if self.max_capacity is not None:
                new_cap = min(new_cap, self.max_capacity)
            return ScalingDecision(
                "scale_up", total, new_cap, new_cap - total,
                "Utilization above scale-up threshold or forecast exceeds capacity",
            )

        if current_util <= self.scale_down_threshold and target < total:
            change = total * self.scale_step_factor
            new_cap = max(total - change, self.min_capacity, target)
            return ScalingDecision(
                "scale_down", total, new_cap, new_cap - total,
                "Utilization below scale-down threshold with spare capacity",
            )

        return ScalingDecision("none", total, total, 0.0,
                               "Within target utilization band")

    def apply(self, pool: ResourcePool, decision: ScalingDecision) -> None:
        """Apply a scaling decision to a pool (adjusts total/available)."""
        if decision.direction == "none" or decision.recommended_change == 0:
            return
        delta = decision.recommended_change
        pool.total_capacity += delta
        pool.available_capacity += delta
        pool.utilization = (pool.allocated_capacity / pool.total_capacity
                            if pool.total_capacity > 0 else 0.0)


class ResourceOptimizationEngine:
    """
    Analyzes usage history, predicts demand and recommends optimizations.

    The engine keeps a rolling history of per-pool demand samples and uses a
    simple moving-average forecast to drive the AutoScaler.
    """

    def __init__(self, autoscaler: Optional[AutoScaler] = None,
                 history_window: int = 20):
        self.autoscaler = autoscaler or AutoScaler()
        self.history_window = history_window
        self._demand_history: Dict[str, List[float]] = defaultdict(list)
        self.recommendations: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def record_demand(self, pool_id: str, demand: float) -> None:
        with self._lock:
            hist = self._demand_history[pool_id]
            hist.append(demand)
            if len(hist) > self.history_window:
                hist.pop(0)

    def predict_demand(self, pool_id: str) -> float:
        with self._lock:
            hist = self._demand_history.get(pool_id, [])
        if not hist:
            return 0.0
        return sum(hist) / len(hist)

    def optimize(self, pool: ResourcePool) -> Dict[str, Any]:
        forecast = self.predict_demand(pool.pool_id)
        decision = self.autoscaler.evaluate(pool, forecast)
        recommendation = {
            "pool_id": pool.pool_id,
            "forecast_demand": round(forecast, 4),
            "decision": decision.to_dict(),
            "generated_at": datetime.utcnow().isoformat(),
        }
        with self._lock:
            self.recommendations.append(recommendation)
        return recommendation

    def apply_recommendations(self, pool_manager: ResourcePoolManager
                              ) -> List[Dict[str, Any]]:
        applied = []
        with self._lock:
            recs = list(self.recommendations)
            self.recommendations.clear()
        for rec in recs:
            pool = pool_manager.pools.get(rec["pool_id"])
            if pool is None:
                continue
            decision = ScalingDecision(**{
                k: (v if k != "direction" else v)
                for k, v in rec["decision"].items()
            })
            self.autoscaler.apply(pool, decision)
            applied.append({"pool_id": pool.pool_id, "applied": decision.to_dict()})
        return applied
