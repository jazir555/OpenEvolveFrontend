"""
Local seed nodes for the OpenEvolve gRPC NodeRegistry.

The production adapter is meant to load ``bubblelabs_nodes.NodeRegistry`` from the
monorepo root. That package pulls in the full ``openevolve`` stack (slow, and only
registers a single node in this checkout), which makes it unsuitable for an offline
unit test. This module provides a lightweight, dependency-free registry that mirrors
the ``bubblelabs_nodes`` node contract (``DISPLAY_NAME``/``DESCRIPTION``/``CATEGORY``/
``VERSION``/``get_parameter_schema``/``execute``) so the gRPC server always has real
nodes to serve.

``NodeAdapter`` uses this by default and can be pointed at the real registry with
``OPENEVOLVE_USE_REAL_NODES=1``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Type


class LocalNode:
    """Minimal duck-typed equivalent of ``bubblelabs_nodes.BubbleLabsNode``."""

    DISPLAY_NAME: str = "Local Node"
    DESCRIPTION: str = "Base local node"
    ICON: str = "default-node"
    CATEGORY: str = "general"
    VERSION: str = "1.0.0"

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    # --- metadata --------------------------------------------------------
    def get_parameter_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        return []

    # --- execution -------------------------------------------------------
    def execute(self, inputs: Dict[str, Any], context: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def execute_safe(self, inputs: Dict[str, Any], context: Any) -> Dict[str, Any]:
        errors = self.validate_inputs(inputs)
        if errors:
            raise ValueError("; ".join(errors))
        return self.execute(inputs, context)


class EchoNode(LocalNode):
    DISPLAY_NAME = "Echo"
    DESCRIPTION = "Returns its inputs unchanged; useful for connectivity checks."
    ICON = "echo"
    CATEGORY = "utility"
    VERSION = "1.0.0"

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prefix": {"type": "string", "default": ""},
            },
        }

    def execute(self, inputs: Dict[str, Any], context: Any) -> Dict[str, Any]:
        prefix = self.config.get("prefix", "")
        return {"echo": inputs, "prefix": prefix}


class DecompositionNode(LocalNode):
    DISPLAY_NAME = "Problem Decomposition"
    DESCRIPTION = "Splits a problem statement into candidate subproblems."
    ICON = "decomposition"
    CATEGORY = "analysis"
    VERSION = "1.0.0"

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_subproblems": {"type": "integer", "default": 5, "minimum": 1},
            },
            "required": ["problem_statement"],
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        if not inputs.get("problem_statement"):
            errors.append("problem_statement is required")
        return errors

    def execute(self, inputs: Dict[str, Any], context: Any) -> Dict[str, Any]:
        statement = str(inputs.get("problem_statement", ""))
        max_sub = int(self.config.get("max_subproblems", 5))
        # Deterministic, dependency-free decomposition: split on sentence
        # boundaries / conjunctions, then trim to the configured budget.
        parts = [p.strip() for p in re.split(r"[.;\n]|\band\b", statement) if p.strip()]
        subproblems = parts[:max_sub] if parts else [statement]

        results = []
        total = len(subproblems)
        for i, text in enumerate(subproblems):
            # Report progress so streaming callers see intermediate updates.
            if hasattr(context, "update_progress"):
                context.update_progress(
                    int((i / total) * 100) if total else 0,
                    f"Decomposing subproblem {i + 1}/{total}",
                )
            results.append({"id": f"sub_{i}", "description": text})

        return {
            "problem_statement": statement,
            "subproblems": results,
            "count": len(results),
        }


class TextStatsNode(LocalNode):
    DISPLAY_NAME = "Text Statistics"
    DESCRIPTION = "Computes simple statistics (chars, words, lines) for text."
    ICON = "analytics"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": ["text"],
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        if "text" not in inputs:
            return ["text is required"]
        return []

    def execute(self, inputs: Dict[str, Any], context: Any) -> Dict[str, Any]:
        text = str(inputs.get("text", ""))
        return {
            "characters": len(text),
            "words": len(text.split()),
            "lines": len(text.splitlines()) or (1 if text else 0),
        }


# Registry key -> node class. Keys are chosen to line up with NodeType enum
# entries in nodes.proto (e.g. "decomposition" -> NODE_TYPE_DECOMPOSITION).
SEED_NODES: Dict[str, Type[LocalNode]] = {
    "echo": EchoNode,
    "decomposition": DecompositionNode,
    "semantic_search": TextStatsNode,
}


class NodeRegistry:
    """In-memory registry mirroring ``bubblelabs_nodes.NodeRegistry``."""

    _nodes: Dict[str, Type[LocalNode]] = dict(SEED_NODES)

    @classmethod
    def register(cls, node_type: str, node_class: Type[LocalNode]) -> None:
        cls._nodes[node_type] = node_class

    @classmethod
    def get(cls, node_type: str, config: Dict[str, Any] | None = None) -> LocalNode:
        node_class = cls._nodes.get(node_type)
        if not node_class:
            available = ", ".join(sorted(cls._nodes)) or "(none)"
            raise ValueError(
                f"Unknown node type: {node_type}. Available types: {available}"
            )
        return node_class(config or {})

    @classmethod
    def list_nodes(cls) -> Dict[str, Type[LocalNode]]:
        return cls._nodes.copy()
