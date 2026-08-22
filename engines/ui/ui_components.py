"""
ui_components - Flat re-export module for the documented UI Components API.

The Decomposition Workflow UI render functions are implemented across the
flat engines/ layout (see docs/Architecture/UI_COMPONENTS_DOCUMENTATION.md).
This module re-exports the documented ``ui_components`` public API so that:

    from ui_components import render_dependency_graph

works via the flat sys.path layout without importing internal modules.

This file is intentionally FLAT: no ``__init__.py`` anywhere in engines/ and
no relative imports. Sibling modules are imported through the flat sys.path.
"""
from __future__ import annotations


import importlib.util
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_source(module_name: str, rel_path: str):
    """Load a sibling module directly from disk under a private name.

    Avoids any name collision with this module's own name (``ui_components``)
    when the engines/ flat directories are all on sys.path.
    """
    path = os.path.join(_THIS_DIR, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Real implementing modules (flat layout).
_real = _load_source(
    "_ui_components_real",
    os.path.join("..", "other", "ui_components.py"),
)
_additional = _load_source(
    "_ui_components_additional",
    "ui_components_additional.py",
)

# batch_operations lives in engines/orchestration (flat import).
from batch_operations import render_batch_operations_ui  # noqa: E402


# --- Documented ui_components API ------------------------------------------
render_dependency_graph = _real.render_dependency_graph
render_dependency_graph_controls = _real.render_dependency_graph_controls
render_analytics_dashboard = _real.render_analytics_dashboard
render_knowledge_base_interface = _real.render_knowledge_base_interface
render_auto_approval_config = _real.render_auto_approval_config
render_enhanced_monitoring = _real.render_enhanced_monitoring
render_workflow_templates = _real.render_workflow_templates
render_workflow_orchestrator = _additional.render_workflow_orchestrator
render_realtime_monitoring = _additional.render_realtime_monitoring


__all__ = [
    "render_dependency_graph",
    "render_dependency_graph_controls",
    "render_analytics_dashboard",
    "render_knowledge_base_interface",
    "render_auto_approval_config",
    "render_enhanced_monitoring",
    "render_workflow_templates",
    "render_batch_operations_ui",
    "render_workflow_orchestrator",
    "render_realtime_monitoring",
]
