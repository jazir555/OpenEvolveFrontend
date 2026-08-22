"""
Workflow lifecycle controller (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the full controller renders lifecycle controls in the UI
and drives the API. This stub owns an :class:`OpenEvolveBubbleLabsIntegration`
and forwards lifecycle actions to it, rendering through the headless
:mod:`.ui_shim` so the control surface stays importable and testable.
"""

from __future__ import annotations

import logging
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
    from .ui_shim import ui
except ImportError:
    from ui_shim import ui

logger = logging.getLogger(__name__)

__all__ = ["STUB", "WorkflowLifecycleController"]


class WorkflowLifecycleController:
    """
    Thin controller over :class:`OpenEvolveBubbleLabsIntegration`.

    Args:
        integration: Optional integration to control. A fresh one is created
            when omitted.

    Attributes:
        integration: The workflow integration this controller drives.
    """

    def __init__(self, integration: Optional[OpenEvolveBubbleLabsIntegration] = None) -> None:
        self.integration = integration or OpenEvolveBubbleLabsIntegration()

    def create_new_workflow_definition(
        self,
        name: str,
        description: str = "",
        workflow_type: str = "evolution",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a workflow definition via the integration.

        Args:
            name: Definition name.
            description: Definition description.
            workflow_type: Workflow family.
            parameters: Default parameters.

        Returns:
            The new definition's identifier.
        """
        return self.integration.create_workflow_definition(
            name=name,
            description=description,
            workflow_type=workflow_type,
            parameters=parameters or {},
        )

    def list_workflow_definitions(self) -> List[Dict[str, Any]]:
        """Return all workflow definitions known to the integration."""
        return self.integration.list_workflow_definitions()

    def start_workflow(self, instance_id: str) -> Dict[str, Any]:
        """Start a workflow instance and return the API result mapping."""
        return self.integration.start_workflow_instance(instance_id)

    def pause_workflow(self, instance_id: str) -> Dict[str, Any]:
        """Pause a workflow instance and return the API result mapping."""
        return self.integration.pause_workflow_instance(instance_id)

    def resume_workflow(self, instance_id: str) -> Dict[str, Any]:
        """Resume a workflow instance and return the API result mapping."""
        return self.integration.resume_workflow_instance(instance_id)

    def stop_workflow(self, instance_id: str) -> Dict[str, Any]:
        """Stop a workflow instance and return the API result mapping."""
        return self.integration.stop_workflow_instance(instance_id)

    def render_lifecycle_controls(self, instance_id: str) -> None:
        """
        Render lifecycle buttons for an instance through the headless UI.

        Args:
            instance_id: Instance the controls act on.
        """
        status = self.integration.get_workflow_instance_status(instance_id)
        ui.subheader(f"Lifecycle: {instance_id}")
        ui.write(status)
        start, pause, stop = ui.columns(3)
        with start:
            ui.button("Start")
        with pause:
            ui.button("Pause")
        with stop:
            ui.button("Stop")
