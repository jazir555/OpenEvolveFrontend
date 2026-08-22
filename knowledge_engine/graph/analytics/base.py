"""
Knowledge Graph Analytics Engine - shared base and request model.

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

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging

import networkx as nx

from ..query.backend import GraphBackend

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsRequest:
    """A request to run an analytics algorithm."""
    type: str
    algorithm: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    subgraph_filter: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalyticsRequest":
        return cls(
            type=data.get("type", "centrality"),
            algorithm=data.get("algorithm"),
            parameters=data.get("parameters", {}) or {},
            source_node=data.get("source_node") or data.get("source"),
            target_node=data.get("target_node") or data.get("target"),
            subgraph_filter=data.get("subgraph_filter"),
            request_id=data.get("request_id"),
        )


def build_nx_graph(backend: GraphBackend, directed: bool = True,
                   subgraph_filter: Optional[Dict[str, Any]] = None) -> nx.Graph:
    """Materialize a NetworkX graph (with weights) from a backend."""
    if subgraph_filter and "node_ids" in subgraph_filter:
        allowed = set(subgraph_filter["node_ids"])
    else:
        allowed = None
    label_filter = (set(subgraph_filter.get("node_labels", []))
                    if subgraph_filter else None)
    edge_filter = (set(subgraph_filter.get("edge_types", []))
                   if subgraph_filter else None)

    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()

    graph = backend.as_networkx()
    for nid, data in graph.nodes(data=True):
        labels = set(data.get("labels", []))
        if allowed is not None and nid not in allowed:
            continue
        if label_filter and not label_filter.issubset(labels):
            continue
        g.add_node(nid, **data)
    for s, t, edata in graph.edges(data=True):
        if s not in g or t not in g:
            continue
        if edge_filter and edata.get("type") not in edge_filter:
            continue
        weight = float(edata.get("properties", {}).get("weight", 1.0))
        g.add_edge(s, t, weight=weight, **edata)
    return g


class AnalyticsError(Exception):
    pass


class BaseAnalyzer:
    """Common helpers for analytics sub-analyzers."""

    def __init__(self, backend: Optional[GraphBackend] = None):
        self.backend = backend

    def _graph(self, request: AnalyticsRequest, directed: bool = True):
        if self.backend is None:
            raise AnalyticsError("No graph backend configured for analytics")
        return build_nx_graph(self.backend, directed=directed,
                              subgraph_filter=request.subgraph_filter)

    @staticmethod
    def _score_dict(scores) -> Dict[str, float]:
        return {str(k): float(v) for k, v in scores.items()}


__all__ = [
    "AnalyticsRequest", "build_nx_graph", "AnalyticsError", "BaseAnalyzer",
]
