"""
Unified Knowledge Graph Module

Real, dependency-light knowledge-graph building and merging. Documented as the
unified graph representation used across the knowledge engine. Provides a local
graph (no Neo4j/Graphiti required) with merge semantics, plus an optional,
clearly-marked integration point for ``integrations.graphiti_integration``.

Author: OpenEvolve Team
Date: 2026-08
"""
from __future__ import annotations


import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KGNode:
    id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label, "properties": self.properties}


@dataclass
class KGEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target,
                "relation": self.relation, "weight": self.weight,
                "properties": self.properties}


class UnifiedKG:
    """
    Local knowledge graph with add / merge / query and serialization.

    Merge semantics: when two nodes share the same id their properties are
    shallow-merged (existing wins on conflict); edges are de-duplicated by
    (source, relation, target).
    """
    def __init__(self, name: str = "unified"):
        self.name = name
        self.nodes: Dict[str, KGNode] = {}
        self.edges: Dict[tuple, KGEdge] = {}

    def add_node(self, label: str, properties: Optional[Dict[str, Any]] = None,
                 node_id: Optional[str] = None) -> str:
        nid = node_id or str(uuid.uuid4())
        if nid in self.nodes:
            self.nodes[nid].properties.update(properties or {})
        else:
            self.nodes[nid] = KGNode(id=nid, label=label, properties=properties or {})
        return nid

    def add_edge(self, source: str, target: str, relation: str,
                 weight: float = 1.0, properties: Optional[Dict[str, Any]] = None) -> None:
        key = (source, relation, target)
        if key in self.edges:
            self.edges[key].weight = max(self.edges[key].weight, weight)
            self.edges[key].properties.update(properties or {})
        else:
            self.edges[key] = KGEdge(source, target, relation, weight, properties or {})

    def build_from_extractions(self, entities: List[Any], relations: List[Any]) -> None:
        """Populate the graph from ExtractedEntity / ExtractedRelation objects."""
        id_map: Dict[str, str] = {}
        for e in entities:
            nid = self.add_node(getattr(e, "name", str(e)),
                                {"type": getattr(e, "entity_type", "concept"),
                                 "confidence": getattr(e, "confidence", 1.0)})
            id_map[getattr(e, "name", str(e))] = nid
        for r in relations:
            s = id_map.get(getattr(r, "subject", ""), getattr(r, "subject", ""))
            o = id_map.get(getattr(r, "obj", ""), getattr(r, "obj", ""))
            if s and o:
                self.add_edge(s, o, getattr(r, "predicate", "rel"),
                              getattr(r, "confidence", 1.0))

    def merge(self, other: "UnifiedKG") -> None:
        for node in other.nodes.values():
            self.add_node(node.label, dict(node.properties), node_id=node.id)
        for edge in other.edges.values():
            self.add_edge(edge.source, edge.target, edge.relation,
                          edge.weight, dict(edge.properties))

    def neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for e in self.edges.values():
            if e.source == node_id:
                out.append({"direction": "out", **e.to_dict()})
            elif e.target == node_id:
                out.append({"direction": "in", **e.to_dict()})
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedKG":
        g = cls(name=data.get("name", "unified"))
        for n in data.get("nodes", []):
            g.nodes[n["id"]] = KGNode(n["id"], n["label"], n.get("properties", {}))
        for e in data.get("edges", []):
            g.edges[(e["source"], e["relation"], e["target"])] = KGEdge(
                e["source"], e["target"], e["relation"], e.get("weight", 1.0),
                e.get("properties", {}))
        return g

    def use_graphiti(self, host: str = "localhost") -> Any:
        """
        Optional Graphiti integration. Clearly-marked optional import; raises a
        helpful error if the integration is not installed (no hard dependency).
        """
        try:  # pragma: no cover - optional integration
            from integrations.graphiti_integration import GraphitiIntegration
            return GraphitiIntegration(host=host)
        except Exception as exc:
            raise RuntimeError(
                "integrations.graphiti_integration is not available; "
                "UnifiedKG works standalone without it."
            ) from exc
