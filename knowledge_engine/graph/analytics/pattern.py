"""
Pattern mining: frequent subgraph patterns, motifs and anomalies.

Copyright 2026 OpenEvolve
Licensed under the Apache License, Version 2.0 (the "License").
"""

import time
from typing import Any, Dict, List

import networkx as nx

from .base import BaseAnalyzer, AnalyticsRequest, AnalyticsError


class PatternMiner(BaseAnalyzer):
    """Mine recurring structural patterns from the knowledge graph."""

    ALGORITHMS = ("triangles", "motifs", "frequent_edges", "anomalies", "degree_distribution")

    def mine(self, request: AnalyticsRequest) -> Dict[str, Any]:
        algorithm = (request.algorithm or "triangles").lower()
        if algorithm not in self.ALGORITHMS:
            raise AnalyticsError(f"Unknown pattern type: {algorithm}")
        g = self._graph(request, directed=True)
        params = request.parameters
        start = time.time()

        if algorithm == "triangles":
            result = self._triangles(g)
        elif algorithm == "motifs":
            result = self._motifs(g)
        elif algorithm == "frequent_edges":
            result = self._frequent_edges(g, params)
        elif algorithm == "anomalies":
            result = self._anomalies(g, params)
        elif algorithm == "degree_distribution":
            result = self._degree_distribution(g)
        else:  # pragma: no cover
            raise AnalyticsError(f"Unhandled algorithm {algorithm}")

        elapsed = (time.time() - start) * 1000
        return {
            "algorithm": algorithm,
            "results": result,
            "parameters": params,
            "execution_time_ms": elapsed,
        }

    def _triangles(self, g):
        ug = g.to_undirected()
        return {"triangle_count": sum(nx.triangles(ug).values()) // 3,
                "triangles_per_node": {str(k): int(v)
                                       for k, v in nx.triangles(ug).items()}}

    def _motifs(self, g):
        if g.is_directed():
            census = nx.algorithms.triads.triadic_census(g)
        else:
            census = nx.triadic_census(g.to_directed())
        return {"motif_census": {str(k): int(v) for k, v in census.items()}}

    def _frequent_edges(self, g, params):
        top = params.get("top_k", 10)
        counts: Dict[str, int] = {}
        for _, _, data in g.edges(data=True):
            key = str(data.get("type", "UNKNOWN"))
            counts[key] = counts.get(key, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top]
        return {"frequent_edge_types": [{"type": k, "count": v} for k, v in ranked]}

    def _anomalies(self, g, params):
        z = params.get("z_score", 2.0)
        degrees = dict(g.degree())
        if not degrees:
            return {"anomalous_nodes": []}
        vals = list(degrees.values())
        mean = sum(vals) / len(vals)
        std = (sum((d - mean) ** 2 for d in vals) / len(vals)) ** 0.5 or 1.0
        anomalies = [str(n) for n, d in degrees.items()
                     if abs(d - mean) > z * std]
        return {"anomalous_nodes": anomalies, "mean_degree": mean, "std_degree": std}

    def _degree_distribution(self, g):
        from collections import Counter
        dist = Counter(dict(g.degree()).values())
        return {"degree_distribution": {str(k): int(v) for k, v in dist.items()}}


__all__ = ["PatternMiner"]
