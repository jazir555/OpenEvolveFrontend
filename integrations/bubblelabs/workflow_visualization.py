"""
Workflow visualization (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the full ``OpenEvolveVisualizer`` renders live panels in
the ``bubble-studio`` UI. This stub renders through the headless
:mod:`.ui_shim`, so its methods run without error in tests and CI while
recording what *would* have been drawn. Swap in the real module to get live
rendering.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from ._stub_support import STUB
except ImportError:
    from _stub_support import STUB
try:
    from .ui_shim import ui as st
except ImportError:
    from ui_shim import ui as st
try:
    from .workflow_structures import WorkflowState
except ImportError:
    from workflow_structures import WorkflowState

__all__ = ["STUB", "OpenEvolveVisualizer"]


class OpenEvolveVisualizer:
    """
    Headless renderer for workflow state.

    Every ``render_*`` method is side-effect-free apart from appending to the
    headless UI's call log, so callers and tests can invoke them freely.

    Attributes:
        rendered: Ordered log of ``(method_name, workflow_id)`` render calls.
    """

    #: Status string -> glyph. Matches the strings asserted by the test suite.
    _STATUS_ICONS: Dict[str, str] = {
        "created": "🆕",
        "pending": "⏳",
        "queued": "⏳",
        "running": "🏃",
        "paused": "⏸️",
        "completed": "[OK]",
        "failed": "[FAIL]",
        "cancelled": "🚫",
    }

    def __init__(self) -> None:
        self.rendered: List[tuple] = []

    def _get_status_icon(self, status: str) -> str:
        """
        Map a status string to a display glyph.

        Args:
            status: Workflow status string.

        Returns:
            The glyph for ``status``, or ``"❓"`` when it is unknown.
        """
        return self._STATUS_ICONS.get(str(status).lower(), "❓")

    def render_workflow_status_pane(self, workflow_state: WorkflowState) -> None:
        """
        Render the high-level status pane for a workflow.

        Args:
            workflow_state: The workflow whose status should be shown.
        """
        icon = self._get_status_icon(getattr(workflow_state, "status", "unknown"))
        st.subheader(f"{icon} {getattr(workflow_state, 'workflow_type', 'workflow')}")
        st.progress(float(getattr(workflow_state, "progress", 0.0) or 0.0))
        st.write(getattr(workflow_state, "problem_statement", ""))
        self.rendered.append(("status_pane", getattr(workflow_state, "workflow_id", None)))

    def render_execution_metrics(self, workflow_state: WorkflowState) -> None:
        """
        Render numeric execution metrics for a workflow.

        Args:
            workflow_state: The workflow whose metrics should be shown.
        """
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best fitness", getattr(workflow_state, "best_fitness", 0.0))
        with col2:
            st.metric("Avg fitness", getattr(workflow_state, "avg_fitness", 0.0))
        with col3:
            st.metric("Diversity", getattr(workflow_state, "diversity", 0.0))
        st.metric("Population", getattr(workflow_state, "population_size", 0))
        st.metric("Execution time (s)", getattr(workflow_state, "execution_time", 0.0))
        self.rendered.append(("execution_metrics", getattr(workflow_state, "workflow_id", None)))

    def render_workflow_graph(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        """
        Record a workflow-graph render request.

        Args:
            nodes: Graph node dictionaries.
            edges: Graph edge dictionaries.
        """
        st.write(f"graph: {len(nodes)} nodes / {len(edges)} edges")
        self.rendered.append(("workflow_graph", (len(nodes), len(edges))))
