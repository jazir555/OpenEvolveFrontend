"""
Knowledge Graph query backends.

Provides a pluggable :class:`GraphBackend` abstraction. The default, fully
offline implementation is :class:`InMemoryNetworkXBackend`, which stores the
graph as a NetworkX ``MultiDiGraph``. Remote backends (Neo4j / Memgraph /
SPARQL) are *optional* and degrade gracefully when their drivers are absent
or unreachable.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Dict, List, Optional, Tuple, Iterable
from dataclasses import dataclass, field
import logging

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:  # pragma: no cover
    NX_AVAILABLE = False
    nx = None

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:  # pragma: no cover
    NP_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


@dataclass
class BackendNode:
    id: str
    labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendEdge:
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    key: int = 0


class GraphBackend:
    """Abstract graph query backend."""

    name = "abstract"

    def is_available(self) -> bool:
        raise NotImplementedError

    def add_node(self, node_id: str, labels: Optional[List[str]] = None,
                 properties: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def add_edge(self, source: str, target: str, edge_type: str,
                 properties: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def load_triples(self, triples: Iterable[Tuple[str, str, str, Dict[str, Any]]]) -> None:
        """Load (subject_id, predicate, object_id, properties) triples."""
        for subj, pred, obj, props in triples:
            self.add_node(subj)
            self.add_node(obj)
            self.add_edge(subj, obj, pred, props or {})

    def node_count(self) -> int:
        raise NotImplementedError

    def edge_count(self) -> int:
        raise NotImplementedError

    def shortest_paths(self, source: str, target: str,
                       max_depth: int = 10) -> List[List[str]]:
        raise NotImplementedError

    def as_networkx(self):
        raise NotImplementedError

    def run_ast(self, ast, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute an already-parsed AST against this backend."""
        raise NotImplementedError


class InMemoryNetworkXBackend(GraphBackend):
    """Default offline backend backed by NetworkX."""

    name = "memory"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        if not NX_AVAILABLE:
            raise RuntimeError("networkx is required for the in-memory backend")
        self._graph = nx.MultiDiGraph()
        # index: label -> set of node ids
        self._label_index: Dict[str, set] = {}

    def is_available(self) -> bool:
        return NX_AVAILABLE

    # -- mutation ---------------------------------------------------------- #
    def add_node(self, node_id: str, labels: Optional[List[str]] = None,
                 properties: Optional[Dict[str, Any]] = None) -> None:
        labels = labels or []
        props = dict(properties or {})
        if node_id not in self._graph:
            self._graph.add_node(node_id, labels=list(labels), properties=props)
        else:
            data = self._graph.nodes[node_id]
            merged = list(data.get("labels", []))
            for l in labels:
                if l not in merged:
                    merged.append(l)
            data["labels"] = merged
            data.setdefault("properties", {}).update(props)
        for l in labels:
            self._label_index.setdefault(l, set()).add(node_id)

    def add_edge(self, source: str, target: str, edge_type: str,
                 properties: Optional[Dict[str, Any]] = None) -> None:
        self.add_node(source)
        self.add_node(target)
        self._graph.add_edge(source, target, type=edge_type,
                             properties=dict(properties or {}))

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def as_networkx(self):
        return self._graph

    # -- traversal helpers ------------------------------------------------- #
    def _node_matches(self, node_id: str, labels: List[str],
                      properties: Dict[str, Any]) -> bool:
        data = self._graph.nodes[node_id]
        if labels:
            node_labels = set(data.get("labels", []))
            if not set(labels).issubset(node_labels):
                return False
        if properties:
            node_props = data.get("properties", {})
            for k, v in properties.items():
                if isinstance(v, dict) and "name" in v and isinstance(v["name"], str):
                    # Parameter placeholder rendered as dict; compare against params elsewhere
                    continue
                if node_props.get(k) != _resolve_literal(v):
                    return False
        return True

    def match_nodes(self, labels: List[str], properties: Dict[str, Any]) -> List[str]:
        if labels:
            candidate_sets = [self._label_index.get(l, set()) for l in labels]
            if not candidate_sets:
                return []
            candidates = set.intersection(*candidate_sets) if candidate_sets else set()
        else:
            candidates = set(self._graph.nodes())
        return [n for n in candidates if self._node_matches(n, labels, properties)]

    def expand(self, source: str, rel) -> List[Tuple[str, Dict[str, Any]]]:
        """Return (target_node, edge_data) for edges matching rel from source."""
        results = []
        direction = getattr(rel, "direction", "out")
        types = set(getattr(rel, "types", []) or [])
        edge_props = getattr(rel, "properties", {}) or {}
        if direction in ("out", "both"):
            for _, tgt, key, data in self._graph.out_edges(source, keys=True, data=True):
                if self._edge_matches(data, types, edge_props):
                    results.append((tgt, data))
        if direction in ("in", "both"):
            for pred, _, key, data in self._graph.in_edges(source, keys=True, data=True):
                if self._edge_matches(data, types, edge_props):
                    results.append((pred, data))
        return results

    @staticmethod
    def _edge_matches(data: Dict[str, Any], types: set,
                      edge_props: Dict[str, Any]) -> bool:
        if types and data.get("type") not in types:
            return False
        if edge_props:
            eprops = data.get("properties", {})
            for k, v in edge_props.items():
                if eprops.get(k) != _resolve_literal(v):
                    return False
        return True

    def shortest_paths(self, source: str, target: str,
                       max_depth: int = 10) -> List[List[str]]:
        try:
            paths = list(nx.all_shortest_paths(
                self._graph, source, target, cutoff=max_depth))
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        except Exception:
            return []


def _resolve_literal(value: Any) -> Any:
    """Resolve a Parameter placeholder (rendered as dict) to a marker."""
    if isinstance(value, dict) and "name" in value and "kind" in value:
        return value  # leave parameter placeholders; comparisons handled later
    return value


class RemoteGraphBackend(GraphBackend):
    """Base for remote backends; degrades gracefully."""

    name = "remote"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.uri = self.config.get("uri")
        self.driver = None
        self._connect()

    def _connect(self):
        raise NotImplementedError

    def is_available(self) -> bool:
        return self.driver is not None

    def add_node(self, node_id, labels=None, properties=None):
        if not self.is_available():
            raise RuntimeError(f"{self.name} backend unavailable")


class Neo4jBackend(RemoteGraphBackend):
    name = "neo4j"

    def _connect(self):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.uri or "bolt://localhost:7687",
                auth=(self.config.get("user", "neo4j"),
                      self.config.get("password", "neo4j")),
            )
        except Exception as e:
            logger.debug(f"Neo4j backend unavailable: {e}")
            self.driver = None


class MemgraphBackend(RemoteGraphBackend):
    name = "memgraph"

    def _connect(self):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.uri or "bolt://localhost:7687",
                auth=(self.config.get("user", "memgraph"),
                      self.config.get("password", "memgraph")),
            )
        except Exception as e:
            logger.debug(f"Memgraph backend unavailable: {e}")
            self.driver = None


class SparqlBackend(RemoteGraphBackend):
    name = "sparql"

    def _connect(self):
        try:
            import requests  # noqa: F401
            self.driver = object()  # endpoint-only; presence marks availability
            self.endpoint = self.uri or self.config.get("endpoint")
        except Exception as e:
            logger.debug(f"SPARQL backend unavailable: {e}")
            self.driver = None


def create_backend(name: str, config: Optional[Dict[str, Any]] = None) -> GraphBackend:
    """Factory returning the requested backend, falling back to memory."""
    config = config or {}
    if name == "memory":
        return InMemoryNetworkXBackend(config)
    try:
        if name == "neo4j":
            return Neo4jBackend(config)
        if name == "memgraph":
            return MemgraphBackend(config)
        if name == "sparql":
            return SparqlBackend(config)
    except Exception as e:
        logger.warning(f"Failed to create backend {name}: {e}; using memory")
    return InMemoryNetworkXBackend(config)


__all__ = [
    "BackendNode", "BackendEdge", "GraphBackend", "InMemoryNetworkXBackend",
    "RemoteGraphBackend", "Neo4jBackend", "MemgraphBackend", "SparqlBackend",
    "create_backend",
]
