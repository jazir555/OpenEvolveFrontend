"""
Centrality analysis: PageRank, Betweenness, Closeness, Eigenvector, Degree.

Copyright 2026 OpenEvolve
Licensed under the Apache License, Version 2.0 (the "License").
"""

import time
from typing import Any, Dict
from dataclasses import dataclass

import networkx as nx

from .base import BaseAnalyzer, AnalyticsRequest, AnalyticsError


class CentralityAnalyzer(BaseAnalyzer):
    """Compute node centrality scores on the knowledge graph."""

    ALGORITHMS = ("pagerank", "betweenness", "closeness",
                  "eigenvector", "degree", "katz")

    def analyze(self, request: AnalyticsRequest) -> Dict[str, Any]:
        algorithm = (request.algorithm or "pagerank").lower()
        if algorithm not in self.ALGORITHMS:
            raise AnalyticsError(f"Unknown centrality algorithm: {algorithm}")
        g = self._graph(request, directed=(algorithm in ("pagerank", "eigenvector", "katz")))
        params = request.parameters
        start = time.time()

        if algorithm == "pagerank":
            scores = self._pagerank(g, params)
        elif algorithm == "betweenness":
            scores = self._betweenness(g, params)
        elif algorithm == "closeness":
            scores = self._closeness(g, params)
        elif algorithm == "eigenvector":
            scores = self._eigenvector(g, params)
        elif algorithm == "degree":
            scores = self._degree(g)
        elif algorithm == "katz":
            scores = self._katz(g, params)
        else:  # pragma: no cover
            raise AnalyticsError(f"Unhandled algorithm {algorithm}")

        elapsed = (time.time() - start) * 1000
        return {
            "algorithm": algorithm,
            "results": self._score_dict(scores),
            "parameters": params,
            "execution_time_ms": elapsed,
            "node_count": g.number_of_nodes(),
        }

    def _pagerank(self, g, params):
        return nx.pagerank(g, alpha=params.get("damping_factor", 0.85),
                           max_iter=params.get("max_iterations", 100),
                           tol=params.get("tolerance", 1e-6))

    def _betweenness(self, g, params):
        return nx.betweenness_centrality(
            g, normalized=params.get("normalized", True),
            endpoints=params.get("endpoints", False),
            weight=params.get("weight", "weight"))

    def _closeness(self, g, params):
        return nx.closeness_centrality(g, distance=params.get("distance", "weight"))

    def _eigenvector(self, g, params):
        return nx.eigenvector_centrality(g, max_iter=params.get("max_iterations", 100),
                                         tol=params.get("tolerance", 1e-6))

    def _degree(self, g):
        return nx.degree_centrality(g)

    def _katz(self, g, params):
        return nx.katz_centrality(g, alpha=params.get("alpha", 0.1),
                                  beta=params.get("beta", 1.0),
                                  max_iter=params.get("max_iterations", 1000),
                                  tol=params.get("tolerance", 1e-6))

    @staticmethod
    def _score_dict(scores) -> Dict[str, float]:
        return {str(k): float(v) for k, v in scores.items()}


__all__ = ["CentralityAnalyzer"]
