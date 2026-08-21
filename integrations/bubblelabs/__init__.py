"""
BubbleLabs integration for OpenEvolve.

Workflow visualization and control library: workflow definitions and instances,
lifecycle control, headless status rendering, MCP tooling, plugin system,
analytics and validation helpers.

The public entry point is :class:`~integrations.bubblelabs.adapter.BubbleLabsAdapter`,
registered as the ``bubblelabs`` integration in
:mod:`integrations.registry`.

Importing this package is deliberately cheap: submodules with heavy optional
third-party requirements (``mcp``, ``fastapi``, ``crewai``, ``plotly``, ...) are
*not* imported here. Import them explicitly, e.g.::

    from integrations.bubblelabs.bubblelabs_mcp_tools import ...

Stub modules
------------
Several sibling modules this package was written against live elsewhere in the
repo and are only reachable via the legacy flat ``sys.path`` layout. They are
satisfied here by thin, well-typed stubs (see :mod:`._stub_support` for the
policy). Stub modules set ``STUB = True`` and are listed in :data:`STUB_MODULES`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List

logger = logging.getLogger(__name__)

__version__ = "1.0.0"

#: Modules in this package that are thin stubs standing in for absent siblings.
STUB_MODULES: List[str] = [
    "analytics_monitoring_dashboard",
    "bubblelabs_gauntlet_bubbles",
    "bubblelabs_integration",
    "bubblelabs_leanaide_integration",
    "bubblelabs_security",
    "bubblelabs_ui_component",
    "crewai_integration_layer",
    "leanaide_client",
    "openevolve_bubblelabs_api",
    "parameter_sync_manager",
    "ui_shim",
    "unified_math_service",
    "workflow_lifecycle_controller",
    "workflow_structures",
    "workflow_visualization",
    "z3_cav_nlp_integration",
]

# Lightweight, dependency-free core re-exports.
from .openevolve_bubblelabs_api import (  # noqa: E402
    OpenEvolveBubbleLabsIntegration,
    WorkflowMetrics,
)
from .ui_shim import ui  # noqa: E402
from .workflow_structures import (  # noqa: E402
    WorkflowStage,
    WorkflowState,
    WorkflowStatus,
)
from .workflow_visualization import OpenEvolveVisualizer  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from .adapter import BubbleLabsAdapter

__all__ = [
    "__version__",
    "STUB_MODULES",
    "BubbleLabsAdapter",
    "OpenEvolveBubbleLabsIntegration",
    "OpenEvolveVisualizer",
    "WorkflowMetrics",
    "WorkflowStage",
    "WorkflowState",
    "WorkflowStatus",
    "ui",
]


def __getattr__(name: str) -> Any:
    """
    Lazily expose :class:`BubbleLabsAdapter`.

    The adapter pulls in ``numpy`` through the OpenEvolve visualization base
    interface, so it is imported on first access rather than at package import
    time. This keeps ``import integrations.bubblelabs`` working even in a
    minimal environment.

    Args:
        name: Attribute being looked up.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If ``name`` is not a public attribute of this package.
    """
    if name == "BubbleLabsAdapter":
        from .adapter import BubbleLabsAdapter

        return BubbleLabsAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
