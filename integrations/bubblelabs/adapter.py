"""
BubbleLabs Adapter for OpenEvolve

Wraps the BubbleLabs workflow visualization/control library in this package to
implement the OpenEvolve :class:`VisualizationInterface`, so BubbleLab workflows
can be registered, inspected and rendered through the standard
``IntegrationRegistry`` surface.

BubbleLabs is an in-process library rather than a remote service, so this adapter
holds no network client. Graph-analytics capabilities that genuinely require an
ML stack (embeddings, clustering) degrade gracefully to ``None`` instead of
returning fabricated data, per the plugin architecture's "no mocks" rule.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from integrations.base.visualization_interface import VisualizationInterface

try:
    from .bubblelabs_gauntlet_bubbles import create_bubble_edge
except ImportError:
    from bubblelabs_gauntlet_bubbles import create_bubble_edge
try:
    from .openevolve_bubblelabs_api import OpenEvolveBubbleLabsIntegration
except ImportError:
    from openevolve_bubblelabs_api import OpenEvolveBubbleLabsIntegration
try:
    from .workflow_structures import WorkflowState
except ImportError:
    from workflow_structures import WorkflowState
try:
    from .workflow_visualization import OpenEvolveVisualizer
except ImportError:
    from workflow_visualization import OpenEvolveVisualizer

logger = logging.getLogger(__name__)

__all__ = ["BubbleLabsAdapter"]


class BubbleLabsAdapter(VisualizationInterface):
    """
    Adapter exposing BubbleLabs workflow visualization and control to OpenEvolve.

    Supports parameterless instantiation so the registry can discover
    capabilities before configuration is available.

    Attributes:
        config: Normalized (snake_case) configuration supplied to
            :meth:`initialize`.
        is_initialized: Whether :meth:`initialize` completed successfully.
        studio_url: Base URL of a ``bubble-studio`` instance, when configured.
        export_dir: Directory used for exported workflow-graph artefacts.
        integration: In-process workflow definition/instance registry.
        visualizer: Headless renderer for workflow state.
    """

    #: Layouts the BubbleLab studio canvas understands.
    SUPPORTED_LAYOUTS = ("force_directed", "circular", "hierarchical", "dagre")

    def __init__(self) -> None:
        """Initialize the adapter without touching configuration or I/O."""
        self.config: Optional[Dict[str, Any]] = None
        self.is_initialized = False
        self.studio_url: Optional[str] = None
        self.export_dir: Optional[str] = None
        self.integration = OpenEvolveBubbleLabsIntegration()
        self.visualizer = OpenEvolveVisualizer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the adapter.

        Args:
            config: Configuration dictionary with keys:
                - studio_url: Base URL of a bubble-studio instance (optional;
                  also read from ``BUBBLELAB_STUDIO_URL``).
                - export_dir: Directory for exported graph artefacts
                  (default: ``<repo>/data/bubblelabs_exports``).
                - default_layout: Default canvas layout.

        Returns:
            True if initialization was successful.
        """
        try:
            self.config = dict(config or {})
            self.studio_url = self.config.get("studio_url") or os.environ.get("BUBBLELAB_STUDIO_URL")

            export_dir = self.config.get("export_dir") or os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data",
                "bubblelabs_exports",
            )
            os.makedirs(export_dir, exist_ok=True)
            self.export_dir = export_dir

            self.is_initialized = True
            logger.info(
                "BubbleLabs adapter initialized (studio_url=%s, export_dir=%s)",
                self.studio_url or "<not configured>",
                self.export_dir,
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize BubbleLabs adapter: {e}")
            self.is_initialized = False
            return False

    async def shutdown(self) -> bool:
        """
        Cleanly shutdown the adapter.

        Returns:
            True if shutdown successful.
        """
        if not self.is_initialized:
            return True

        try:
            self.integration.workflow_instances.clear()
            self.integration.workflow_definitions.clear()
            self.is_initialized = False
            logger.info("BubbleLabs adapter shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            return False

    async def validate(self) -> Dict[str, Any]:
        """
        Validate the BubbleLabs integration.

        Returns:
            Dictionary with ``valid``, ``version``, ``capabilities``, ``checks``
            and ``errors`` keys.
        """
        export_writable = bool(self.export_dir) and os.access(self.export_dir or "", os.W_OK)
        checks = {
            "initialized": self.is_initialized,
            "studio_configured": bool(self.studio_url),
            "export_dir_writable": export_writable,
        }

        errors: List[str] = []
        if not self.is_initialized:
            errors.append("adapter not initialized")
        if self.is_initialized and not export_writable:
            errors.append(f"export directory not writable: {self.export_dir}")

        return {
            "valid": self.is_initialized and export_writable,
            "version": self._get_version(),
            "capabilities": self._get_capabilities(),
            "checks": checks,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # VisualizationInterface
    # ------------------------------------------------------------------

    async def visualize_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        layout: str = "force_directed",
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Render a graph as a BubbleLab workflow canvas document.

        Args:
            nodes: Node dictionaries with ``id`` and attributes.
            edges: Edge dictionaries with ``source``, ``target`` and attributes.
            layout: Canvas layout algorithm.
            output_path: Optional explicit path for the exported document.

        Returns:
            Path to the exported canvas document, a studio deep link when a
            ``studio_url`` is configured, or ``None`` on failure.
        """
        if not self.is_initialized:
            logger.error("BubbleLabs adapter not initialized")
            return None

        if layout not in self.SUPPORTED_LAYOUTS:
            logger.warning("Unsupported layout %r, falling back to force_directed", layout)
            layout = "force_directed"

        try:
            document = {
                "format": "bubblelab.canvas/v1",
                "layout": layout,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "nodes": [self._normalize_node(node, index) for index, node in enumerate(nodes)],
                "edges": [self._normalize_edge(edge) for edge in edges],
            }

            target = output_path or os.path.join(
                self.export_dir or ".",
                f"canvas-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json",
            )
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2, default=str)

            logger.info("BubbleLabs canvas exported to %s", target)

            if self.studio_url:
                return f"{self.studio_url.rstrip('/')}/canvas?source={os.path.basename(target)}"
            return target

        except Exception as e:
            logger.error(f"Failed to export BubbleLabs canvas: {e}")
            return None

    async def compute_embeddings(
        self,
        nodes: List[Dict[str, Any]],
        method: str = "umap",
        n_components: int = 2,
    ) -> Optional[np.ndarray]:
        """
        Compute node embeddings.

        BubbleLabs has no embedding backend of its own; use the pygraphistry
        integration for embeddings rather than receiving fabricated vectors.

        Args:
            nodes: Node dictionaries with features.
            method: Embedding method.
            n_components: Output dimensionality.

        Returns:
            Always ``None`` - capability intentionally not provided here.
        """
        logger.warning(
            "BubbleLabs does not provide embeddings (requested method=%r); "
            "use the pygraphistry integration instead",
            method,
        )
        return None

    async def cluster_nodes(
        self,
        embeddings: np.ndarray,
        method: str = "dbscan",
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        """
        Cluster nodes from embeddings.

        Args:
            embeddings: Node embeddings.
            method: Clustering method.
            **kwargs: Additional clustering parameters.

        Returns:
            Always ``None`` - capability intentionally not provided here.
        """
        logger.warning(
            "BubbleLabs does not provide clustering (requested method=%r); "
            "use the pygraphistry integration instead",
            method,
        )
        return None

    async def create_interactive_dashboard(
        self,
        data: Dict[str, Any],
        dashboard_type: str = "graph",
    ) -> Optional[str]:
        """
        Create an interactive BubbleLab studio dashboard.

        Args:
            data: Data dictionary with ``nodes``/``edges`` keys.
            dashboard_type: Dashboard flavour (``graph``, ``workflow``, ...).

        Returns:
            Deep link into the configured studio, a local export path when no
            studio is configured, or ``None`` on failure.
        """
        if not self.is_initialized:
            logger.error("BubbleLabs adapter not initialized")
            return None

        exported = await self.visualize_graph(
            nodes=list(data.get("nodes", [])),
            edges=list(data.get("edges", [])),
            layout=str(data.get("layout", "force_directed")),
        )
        if exported is None:
            return None

        if self.studio_url:
            return f"{self.studio_url.rstrip('/')}/dashboard/{dashboard_type}"
        return exported

    # ------------------------------------------------------------------
    # BubbleLabs workflow control
    # ------------------------------------------------------------------

    def create_workflow_definition(
        self,
        name: str,
        description: str = "",
        workflow_type: str = "evolution",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a BubbleLab workflow definition.

        Args:
            name: Definition name.
            description: Definition description.
            workflow_type: Workflow family.
            parameters: Default parameters for instances.

        Returns:
            The new definition's identifier.
        """
        return self.integration.create_workflow_definition(
            name=name,
            description=description,
            workflow_type=workflow_type,
            parameters=parameters or {},
        )

    def create_workflow_instance(
        self,
        definition_id: str,
        instance_name: str = "",
        inputs: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Instantiate a registered workflow definition.

        Args:
            definition_id: Definition to instantiate.
            instance_name: Friendly instance name.
            inputs: Run inputs.
            parameters: Per-instance parameter overrides.

        Returns:
            The new instance's identifier.
        """
        return self.integration.create_workflow_instance(
            definition_id=definition_id,
            instance_name=instance_name,
            inputs=inputs,
            parameters=parameters,
        )

    def get_workflow_status(self, instance_id: str) -> Dict[str, Any]:
        """
        Return the status of a workflow instance.

        Args:
            instance_id: Instance identifier.

        Returns:
            The instance's status mapping.
        """
        return self.integration.get_workflow_instance_status(instance_id)

    def list_workflows(self) -> List[Dict[str, Any]]:
        """Return status mappings for every known workflow instance."""
        return self.integration.list_workflow_instances()

    def render_workflow_status(self, instance_id: str) -> bool:
        """
        Render a workflow's status panes through the headless visualizer.

        Args:
            instance_id: Instance identifier.

        Returns:
            ``True`` if the instance existed and was rendered.
        """
        state: Optional[WorkflowState] = self.integration.get_workflow_instance(instance_id)
        if state is None:
            logger.warning("Unknown BubbleLabs workflow instance: %s", instance_id)
            return False

        self.visualizer.render_workflow_status_pane(state)
        self.visualizer.render_execution_metrics(state)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_node(node: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Coerce an arbitrary node dict into BubbleLab canvas node shape.

        Args:
            node: Source node dictionary.
            index: Positional index, used to synthesise a missing ``id``.

        Returns:
            Node dictionary with guaranteed ``id``, ``type`` and ``label`` keys.
        """
        node_id = str(node.get("id", f"node-{index}"))
        return {
            "id": node_id,
            "type": str(node.get("type", node.get("node_type", "bubble"))),
            "label": str(node.get("label", node.get("name", node_id))),
            "data": {k: v for k, v in node.items() if k not in {"id", "type", "node_type", "label", "name"}},
        }

    @staticmethod
    def _normalize_edge(edge: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coerce an arbitrary edge dict into BubbleLab canvas edge shape.

        Args:
            edge: Source edge dictionary, needing ``source`` and ``target``.

        Returns:
            Edge dictionary produced by :func:`create_bubble_edge`.
        """
        return create_bubble_edge(
            source_id=str(edge.get("source", "")),
            target_id=str(edge.get("target", "")),
            edge_type=str(edge.get("type", "default")),
            label=edge.get("label"),
        )

    def _get_capabilities(self) -> List[str]:
        """Get the list of capabilities this adapter provides."""
        capabilities = [
            "workflow_definitions",
            "workflow_instances",
            "workflow_lifecycle",
            "canvas_export",
            "status_rendering",
        ]
        if self.studio_url:
            capabilities.append("studio_deep_links")
        return capabilities

    def _get_version(self) -> str:
        """Get the BubbleLabs integration package version."""
        try:
            try:
                from . import __version__
            except ImportError:
                import __version__

            return __version__
        except Exception:  # pragma: no cover - version is best-effort
            return "unknown"
