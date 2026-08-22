"""
Community detection: Louvain, Label Propagation, Walktrap, Greedy Modularity.

Copyright 2026 OpenEvolve
Licensed under the Apache License, Version 2.0 (the "License").
"""

import time
from typing import Any, Dict, List

import networkx as nx

from .base import BaseAnalyzer, AnalyticsRequest, AnalyticsError


class CommunityDetector(BaseAnalyzer):
    """Detect communities/clusters in the knowledge graph."""

    ALGORITHMS = ("louvain", "label_propagation", "walktrap", "modularity")

    def detect(self, request: AnalyticsRequest) -> Dict[str, Any]:
        algorithm = (request.algorithm or "louvain").lower()
        if algorithm not in self.ALGORITHMS:
            raise AnalyticsError(f"Unknown community algorithm: {algorithm}")
        g = self._graph(request, directed=False)
        params = request.parameters
        start = time.time()

        if algorithm == "louvain":
            communities, modularity = self._louvain(g, params)
        elif algorithm == "label_propagation":
            communities, modularity = self._label_propagation(g, params)
        elif algorithm == "walktrap":
            communities, modularity = self._walktrap(g, params)
        else:
            communities, modularity = self._greedy_modularity(g)

        elapsed = (time.time() - start) * 1000
        return {
            "algorithm": algorithm,
            "communities": communities,
            "modularity": modularity,
            "parameters": params,
            "execution_time_ms": elapsed,
            "community_count": len(communities),
        }

    def _louvain(self, g, params):
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(
            g, weight="weight",
            resolution=params.get("resolution", 1.0),
            max_level=params.get("max_iterations", 10))
        return self._format(comms), nx.community.modularity(g, comms)

    def _label_propagation(self, g, params):
        from networkx.algorithms.community import label_propagation_communities
        comms = list(label_propagation_communities(g))
        return self._format(comms), nx.community.modularity(g, comms)

    def _walktrap(self, g, params):
        # Walktrap is a hierarchical method; use greedy modularity as the
        # offline stand-in (real walktrap needs the igraph lib optionally).
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = list(greedy_modularity_communities(g))
            return self._format(comms), nx.community.modularity(g, comms)
        except Exception:
            return self._louvain(g, params)

    def _greedy_modularity(self, g):
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(g))
        return self._format(comms), nx.community.modularity(g, comms)

    @staticmethod
    def _format(communities) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for i, c in enumerate(communities):
            out[f"community_{i}"] = [str(n) for n in c]
        return out


__all__ = ["CommunityDetector"]
