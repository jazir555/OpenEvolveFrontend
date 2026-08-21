"""
BubbleLabs workflow UI component (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the full component renders the workflow builder in the
``bubble-studio`` IDE. This stub renders through the headless :mod:`.ui_shim` so
modules that extend it (see :mod:`.bubblelabs_evolution_ui_patch`) stay
importable and testable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ._stub_support import STUB
from .ui_shim import ui as st

logger = logging.getLogger(__name__)

__all__ = ["STUB", "BubbleLabsWorkflowUI"]


class BubbleLabsWorkflowUI:
    """
    Headless stand-in for the BubbleLabs workflow builder UI.

    Args:
        title: Panel title.

    Attributes:
        title: Panel title.
        nodes: Nodes currently loaded into the builder.
        edges: Edges currently loaded into the builder.
        extensions: Names of extensions registered via :meth:`register_extension`.
    """

    def __init__(self, title: str = "BubbleLabs Workflow") -> None:
        self.title = title
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.extensions: List[str] = []

    def load_workflow(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        """
        Replace the builder's current graph.

        Args:
            nodes: Node dictionaries.
            edges: Edge dictionaries.
        """
        self.nodes = list(nodes)
        self.edges = list(edges)

    def register_extension(self, name: str, renderer: Optional[Any] = None) -> None:
        """
        Register a UI extension panel.

        Args:
            name: Extension name.
            renderer: Optional callable that renders the extension.
        """
        self.extensions.append(name)
        logger.debug("Registered BubbleLabs UI extension: %s", name)

    def render(self) -> None:
        """Render the workflow panel through the headless UI."""
        st.header(self.title)
        st.write(f"{len(self.nodes)} nodes / {len(self.edges)} edges")
        for name in self.extensions:
            st.write(f"extension: {name}")
