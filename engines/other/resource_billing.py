"""
Resource Billing & Accounting.

Provides:
  * AccountingManager - records live allocations/releases, maintains per-tenant
                         consumption and feeds quota usage tracking
  * BillingManager    - meters usage into billable line items, prices them via a
                         configurable rate card and produces per-tenant invoices
                         with account balances

Flat engines/other module: no relative imports, no __init__.
"""
from __future__ import annotations

import threading
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

from resource_pool_manager import (
    ResourceType,
    ResourceAllocation,
    ResourceManagementError,
)

logger = logging.getLogger(__name__)


# Default rate card: cost per unit per hour for each resource type.
DEFAULT_RATE_CARD: Dict[ResourceType, float] = {
    ResourceType.CPU: 0.05,       # $/core-hour
    ResourceType.MEMORY: 0.01,    # $/MB-hour
    ResourceType.GPU: 2.0,       # $/unit-hour
    ResourceType.STORAGE: 0.001,  # $/GB-hour
    ResourceType.NETWORK: 0.0005, # $/Mbps-hour
    ResourceType.CUSTOM: 0.0,
}


@dataclass
class UsageEvent:
    """A metered usage interval for a single allocation."""
    tenant_id: str
    resource_type: ResourceType
    amount: float
    start: datetime
    end: Optional[datetime] = None
    allocation_id: Optional[str] = None
    workload_id: Optional[str] = None
    cost: float = 0.0
    billed: bool = False

    def duration_hours(self) -> float:
        if self.end is None:
            return 0.0
        return (self.end - self.start).total_seconds() / 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "resource_type": self.resource_type.value,
            "amount": self.amount,
            "start": self.start.isoformat(),
            "end": self.end.isoformat() if self.end else None,
            "allocation_id": self.allocation_id,
            "workload_id": self.workload_id,
            "duration_hours": round(self.duration_hours(), 6),
            "cost": round(self.cost, 6),
            "billed": self.billed,
        }


class AccountingManager:
    """
    Central ledger of resource consumption.

    ResourcePoolManager notifies this manager on every allocation/release so
    that (a) tenant quota usage is reported and (b) usage is metered for
    downstream billing. A module-level singleton is used so all pool managers
    share the same ledger.
    """

    def __init__(self, quota_manager: Any = None):
        self.quota_manager = quota_manager
        self._active: Dict[str, UsageEvent] = {}
        self.events: List[UsageEvent] = []
        self._lock = threading.RLock()
        logger.info("AccountingManager initialized")

    # -- pool-manager hooks ----------------------------------------------
    def record_allocation(self, allocation: ResourceAllocation) -> None:
        tenant = allocation.tenant_id or allocation.allocated_to
        rt = allocation.resource_type
        amount = allocation.allocated_amount
        with self._lock:
            event = UsageEvent(
                tenant_id=tenant,
                resource_type=rt,
                amount=amount,
                start=datetime.utcnow(),
                allocation_id=allocation.resource_id,
                workload_id=allocation.workload_id,
            )
            self._active[allocation.resource_id] = event
            self.events.append(event)
        if self.quota_manager is not None and tenant:
            try:
                self.quota_manager.record_allocation(tenant, rt, amount)
            except Exception as exc:  # pragma: no cover
                logger.debug("Quota usage update failed: %s", exc)

    def record_release(self, allocation: ResourceAllocation) -> None:
        tenant = allocation.tenant_id or allocation.allocated_to
        rt = allocation.resource_type
        amount = allocation.allocated_amount
        with self._lock:
            event = self._active.pop(allocation.resource_id, None)
            if event is not None and event.end is None:
                event.end = datetime.utcnow()
        if self.quota_manager is not None and tenant:
            try:
                self.quota_manager.record_release(tenant, rt, amount)
            except Exception as exc:  # pragma: no cover
                logger.debug("Quota release failed: %s", exc)

    # -- queries ----------------------------------------------------------
    def tenant_consumption(self, tenant_id: str) -> Dict[str, float]:
        totals: Dict[str, float] = defaultdict(float)
        with self._lock:
            for ev in self.events:
                if ev.tenant_id != tenant_id:
                    continue
                dur = ev.duration_hours()
                totals[ev.resource_type.value] += ev.amount * dur
        return dict(totals)

    def get_active_usage(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._active.values()]

    def get_events(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            evs = self.events
            if tenant_id is not None:
                evs = [e for e in evs if e.tenant_id == tenant_id]
            return [e.to_dict() for e in evs]


class BillingManager:
    """
    Turns metered usage into priced, billable line items and invoices.

    On each release (or explicit ``meter`` call) the manager computes the cost
    of a usage interval using the rate card and accumulates it against the
    tenant's account. Invoices finalize unbilled usage and may be paid against
    a prepaid balance.
    """

    def __init__(self, rate_card: Optional[Dict[ResourceType, float]] = None,
                 accounting_manager: Optional[AccountingManager] = None):
        self.rate_card = dict(rate_card or DEFAULT_RATE_CARD)
        self.accounting = accounting_manager or AccountingManager()
        self.unbilled: Dict[str, float] = defaultdict(float)
        self.invoices: Dict[str, Dict[str, Any]] = {}
        self.balances: Dict[str, float] = defaultdict(float)
        self._invoice_counter = 0
        self._lock = threading.RLock()

    def set_rate(self, resource_type: ResourceType, price_per_hour: float) -> None:
        self.rate_card[resource_type] = price_per_hour

    def price_interval(self, resource_type: ResourceType, amount: float,
                       duration_hours: float) -> float:
        rate = self.rate_card.get(resource_type, 0.0)
        return rate * amount * max(duration_hours, 0.0)

    def meter(self, tenant_id: str, resource_type: ResourceType, amount: float,
              duration_hours: float) -> float:
        """Meter an arbitrary usage interval and accrue it to the tenant."""
        cost = self.price_interval(resource_type, amount, duration_hours)
        with self._lock:
            self.unbilled[tenant_id] += cost
        return cost

    def meter_allocation(self, allocation: ResourceAllocation,
                         end: Optional[datetime] = None) -> float:
        """Meter a single allocation from start to (end or now)."""
        end = end or datetime.utcnow()
        start = allocation.allocated_at
        duration = (end - start).total_seconds() / 3600.0
        tenant = allocation.tenant_id or allocation.allocated_to
        return self.meter(tenant, allocation.resource_type,
                          allocation.allocated_amount, duration)

    def topup(self, tenant_id: str, amount: float) -> float:
        with self._lock:
            self.balances[tenant_id] += amount
            return self.balances[tenant_id]

    def get_balance(self, tenant_id: str) -> float:
        return self.balances.get(tenant_id, 0.0)

    def get_unbilled(self, tenant_id: str) -> float:
        return self.unbilled.get(tenant_id, 0.0)

    def generate_invoice(self, tenant_id: str,
                         apply_balance: bool = True) -> Dict[str, Any]:
        with self._lock:
            amount_due = self.unbilled.get(tenant_id, 0.0)
            self._invoice_counter += 1
            invoice_id = f"inv_{tenant_id}_{self._invoice_counter:06d}"
            paid_from_balance = 0.0
            if apply_balance and amount_due > 0:
                available = self.balances[tenant_id]
                paid_from_balance = min(available, amount_due)
                self.balances[tenant_id] -= paid_from_balance
            remaining = max(0.0, amount_due - paid_from_balance)
            invoice = {
                "invoice_id": invoice_id,
                "tenant_id": tenant_id,
                "issued_at": datetime.utcnow().isoformat(),
                "amount_due": round(amount_due, 4),
                "paid_from_balance": round(paid_from_balance, 4),
                "outstanding": round(remaining, 4),
                "status": "paid" if remaining <= 1e-9 else "open",
            }
            self.invoices[invoice_id] = invoice
            self.unbilled[tenant_id] = 0.0
        logger.info("Invoice %s for tenant %s: %s", invoice_id, tenant_id, invoice)
        return invoice

    def get_invoices(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self.invoices.values())
            if tenant_id is not None:
                items = [i for i in items if i["tenant_id"] == tenant_id]
            return items

    def get_statement(self, tenant_id: str) -> Dict[str, Any]:
        return {
            "tenant_id": tenant_id,
            "balance": self.get_balance(tenant_id),
            "unbilled": self.get_unbilled(tenant_id),
            "total_billed": sum(
                i["amount_due"] for i in self.get_invoices(tenant_id)
            ),
            "recent_invoices": self.get_invoices(tenant_id)[-5:],
        }


# Module-level shared singletons -------------------------------------------
_default_accounting: Optional[AccountingManager] = None
_default_billing: Optional[BillingManager] = None


def get_accounting_manager() -> AccountingManager:
    global _default_accounting
    if _default_accounting is None:
        _default_accounting = AccountingManager()
    return _default_accounting


def get_billing_manager() -> BillingManager:
    global _default_billing
    if _default_billing is None:
        _default_billing = BillingManager(
            accounting_manager=get_accounting_manager()
        )
    return _default_billing
