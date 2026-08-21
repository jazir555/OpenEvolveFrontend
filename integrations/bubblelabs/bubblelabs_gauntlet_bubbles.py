"""
Gauntlet bubble helpers (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the full module builds the OpenEvolve "gauntlet" bubble
set. Only the graph-construction helper used inside this package is provided;
edge construction is pure data shaping, so it is implemented for real rather
than stubbed out.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

try:
    from ._stub_support import STUB
except ImportError:
    from _stub_support import STUB

__all__ = ["STUB", "create_bubble_edge"]


def create_bubble_edge(
    source_id: str,
    target_id: str,
    edge_type: str = "default",
    label: Optional[str] = None,
    **attributes: Any,
) -> Dict[str, Any]:
    """
    Build an edge dictionary connecting two bubbles.

    Args:
        source_id: Identifier of the upstream bubble.
        target_id: Identifier of the downstream bubble.
        edge_type: Edge kind, e.g. ``"default"`` or ``"conditional"``.
        label: Optional human-readable edge label.
        **attributes: Extra edge attributes merged into the result.

    Returns:
        Mapping with ``id``, ``source``, ``target``, ``type`` and any extra
        attributes supplied by the caller.
    """
    edge: Dict[str, Any] = {
        "id": f"edge-{uuid.uuid4()}",
        "source": source_id,
        "target": target_id,
        "type": edge_type,
    }
    if label is not None:
        edge["label"] = label
    edge.update(attributes)
    return edge
