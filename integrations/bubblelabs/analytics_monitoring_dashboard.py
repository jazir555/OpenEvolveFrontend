"""
Analytics and monitoring dashboard (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the full dashboard streams metrics from the analytics
service into live charts. This stub keeps a real background sampling loop (so
start/stop semantics are genuine) but samples only the deterministic values
already present on each :class:`~.workflow_structures.WorkflowState`. No random
data and no fabricated telemetry.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from ._stub_support import STUB
except ImportError:
    from _stub_support import STUB
try:
    from .openevolve_bubblelabs_api import OpenEvolveBubbleLabsIntegration
except ImportError:
    from openevolve_bubblelabs_api import OpenEvolveBubbleLabsIntegration
try:
    from .ui_shim import ui as st
except ImportError:
    from ui_shim import ui as st

logger = logging.getLogger(__name__)

__all__ = ["STUB", "AnalyticsManager", "AnalyticsMonitoringDashboard"]


def _utcnow() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


class AnalyticsManager:
    """
    Minimal in-memory analytics sink.

    Attributes:
        events: Recorded ``(name, payload)`` analytics events, oldest first.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def record_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Record one analytics event.

        Args:
            name: Event name.
            payload: Optional structured event data.

        Returns:
            The stored event mapping.
        """
        event = {"name": name, "payload": dict(payload or {}), "timestamp": _utcnow().isoformat()}
        self.events.append(event)
        return event

    def summarize(self) -> Dict[str, Any]:
        """
        Summarise recorded events.

        Returns:
            Mapping with the total event count and per-name counts.
        """
        counts: Dict[str, int] = {}
        for event in self.events:
            counts[event["name"]] = counts.get(event["name"], 0) + 1
        return {"total_events": len(self.events), "by_name": counts}


class AnalyticsMonitoringDashboard:
    """
    Poll workflow instances and retain a rolling metrics history.

    Args:
        integration: Optional integration to observe. A fresh one is created
            when omitted.

    Attributes:
        integration: The workflow integration being observed.
        analytics_manager: Sink for analytics events.
        is_monitoring: Whether the sampling loop is active.
        metrics_history: Collected metric samples, oldest first.
    """

    #: Cap on retained samples, so long runs cannot grow without bound.
    MAX_HISTORY: int = 10_000

    def __init__(self, integration: Optional[OpenEvolveBubbleLabsIntegration] = None) -> None:
        self.integration = integration or OpenEvolveBubbleLabsIntegration()
        self.analytics_manager = AnalyticsManager()
        self.is_monitoring: bool = False
        self.metrics_history: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # -- sampling -------------------------------------------------------------

    def collect_metrics(self) -> Dict[str, Any]:
        """
        Take one metrics sample across all workflow instances.

        Only values already recorded on the workflow states are reported.

        Returns:
            The sample mapping that was appended to :attr:`metrics_history`.
        """
        instances = self.integration.list_workflow_instances()
        sample: Dict[str, Any] = {
            "timestamp": _utcnow().isoformat(),
            "instance_count": len(instances),
            "by_status": {},
            "instances": instances,
        }
        for instance in instances:
            status = str(instance.get("status", "unknown"))
            sample["by_status"][status] = sample["by_status"].get(status, 0) + 1

        self.metrics_history.append(sample)
        if len(self.metrics_history) > self.MAX_HISTORY:
            del self.metrics_history[: len(self.metrics_history) - self.MAX_HISTORY]
        return sample

    def _monitoring_loop(self, interval: float) -> None:
        """
        Sample metrics immediately, then once per ``interval`` until stopped.

        Args:
            interval: Seconds between samples.
        """
        while self.is_monitoring and not self._stop_event.is_set():
            try:
                self.collect_metrics()
            except Exception:  # pragma: no cover - a sampling error must not kill the thread
                logger.exception("Metrics collection failed")
            # Returns True as soon as stop is requested, so shutdown is prompt.
            if self._stop_event.wait(interval):
                break

    def start_real_time_monitoring(self, interval: float = 1.0) -> bool:
        """
        Start the background sampling loop.

        Args:
            interval: Seconds between samples.

        Returns:
            ``True`` if a loop is now running.
        """
        if self.is_monitoring:
            return True

        self._stop_event.clear()
        self.is_monitoring = True
        self._thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            name="bubblelabs-analytics-monitor",
            daemon=True,
        )
        self._thread.start()
        self.analytics_manager.record_event("monitoring_started", {"interval": interval})
        return True

    def stop_real_time_monitoring(self) -> bool:
        """
        Stop the background sampling loop and join its thread.

        Returns:
            ``True`` once monitoring has stopped.
        """
        self.is_monitoring = False
        self._stop_event.set()

        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

        self.analytics_manager.record_event("monitoring_stopped")
        return True

    # -- rendering ------------------------------------------------------------

    def render_dashboard(self) -> None:
        """Render the dashboard through the headless UI."""
        st.subheader("BubbleLabs analytics")
        st.metric("Samples", len(self.metrics_history))
        st.metric("Monitoring", "on" if self.is_monitoring else "off")
        st.write(self.analytics_manager.summarize())
