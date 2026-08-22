"""
Path analysis: shortest, all, k-shortest, A*, Dijkstra.

Copyright 2026 OpenEvolve
Licensed under the Apache License, Version 2.0 (the "License").
"""

import time
from typing import Any, Dict, List

import networkx as nx

from .base import BaseAnalyzer, AnalyticsRequest, AnalyticsError


class PathAnalyzer(BaseAnalyzer):
    """Find paths between nodes in the knowledge graph."""

    ALGORITHMS = ("shortest_path", "all_shortest_paths", "all_paths",
                  "k_shortest_paths", "astar", "dijkstra")

    def analyze(self, request: AnalyticsRequest) -> Dict[str, Any]:
        algorithm = (request.algorithm or "shortest_path").lower()
        if algorithm not in self.ALGORITHMS:
            raise AnalyticsError(f"Unknown path algorithm: {algorithm}")
        if not request.source_node or not request.target_node:
            raise AnalyticsError("source_node and target_node are required")
        g = self._graph(request, directed=True)
        params = request.parameters
        start = time.time()

        weight = params.get("weight_property", "weight")
        if algorithm == "shortest_path":
            paths = self._shortest(g, request.source_node, request.target_node, weight)
        elif algorithm == "all_shortest_paths":
            paths = self._all_shortest(g, request.source_node, request.target_node, weight)
        elif algorithm == "all_paths":
            paths = self._all_paths(g, request.source_node, request.target_node, params)
        elif algorithm == "k_shortest_paths":
            paths = self._k_shortest(g, request.source_node, request.target_node,
                                    params, weight)
        elif algorithm == "astar":
            paths = self._astar(g, request.source_node, request.target_node, params, weight)
        elif algorithm == "dijkstra":
            paths = self._dijkstra(g, request.source_node, request.target_node, weight)
        else:  # pragma: no cover
            raise AnalyticsError(f"Unhandled algorithm {algorithm}")

        elapsed = (time.time() - start) * 1000
        costs = [self._path_cost(g, p, weight) for p in paths]
        return {
            "algorithm": algorithm,
            "paths": [{"nodes": p, "cost": c} for p, c in zip(paths, costs)],
            "path_count": len(paths),
            "parameters": params,
            "execution_time_ms": elapsed,
        }

    def _shortest(self, g, s, t, weight):
        try:
            return [list(nx.shortest_path(g, s, t, weight=weight))]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _all_shortest(self, g, s, t, weight):
        try:
            return [list(p) for p in nx.all_shortest_paths(g, s, t, weight=weight)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _all_paths(self, g, s, t, params):
        cutoff = params.get("max_depth", 10)
        try:
            return [list(p) for p in nx.all_simple_paths(
                g, s, t, cutoff=cutoff)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _k_shortest(self, g, s, t, params, weight):
        k = params.get("k", 3)
        try:
            paths = list(nx.shortest_simple_paths(g, s, t, weight=weight))
            return [list(p) for p in paths[:k]]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _astar(self, g, s, t, params, weight):
        heuristic = None
        if params.get("heuristic") == "haversine":
            heuristic = self._haversine
        try:
            return [list(nx.astar_path(g, s, t, heuristic=heuristic, weight=weight))]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _dijkstra(self, g, s, t, weight):
        try:
            return [list(nx.dijkstra_path(g, s, t, weight=weight))]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    @staticmethod
    def _path_cost(g, path, weight):
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            try:
                total += g[u][v].get(weight, 1.0)
            except KeyError:
                total += 1.0
        return total

    @staticmethod
    def _haversine(a, b):
        na = a if isinstance(a, str) else a
        return 0.0


__all__ = ["PathAnalyzer"]
