"""
Spatial graph analysis: geospatial proximity and clustering.

Copyright 2026 OpenEvolve
Licensed under the Apache License, Version 2.0 (the "License").
"""

import time
import math
from typing import Any, Dict, List

import networkx as nx

from .base import BaseAnalyzer, AnalyticsRequest, AnalyticsError


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371.0  # km
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class SpatialAnalyzer(BaseAnalyzer):
    """Analyze geographic relationships between graph nodes."""

    ALGORITHMS = ("proximity", "nearest_neighbors", "spatial_clusters")

    def analyze(self, request: AnalyticsRequest) -> Dict[str, Any]:
        algorithm = (request.algorithm or "proximity").lower()
        if algorithm not in self.ALGORITHMS:
            raise AnalyticsError(f"Unknown spatial algorithm: {algorithm}")
        g = self._graph(request, directed=False)
        params = request.parameters
        start = time.time()

        coords = self._coords(g)
        if algorithm == "proximity":
            result = self._proximity(g, coords, params)
        elif algorithm == "nearest_neighbors":
            result = self._nearest(g, coords, params)
        elif algorithm == "spatial_clusters":
            result = self._clusters(g, coords, params)
        else:  # pragma: no cover
            raise AnalyticsError(f"Unhandled algorithm {algorithm}")

        elapsed = (time.time() - start) * 1000
        return {
            "algorithm": algorithm,
            "results": result,
            "parameters": params,
            "execution_time_ms": elapsed,
        }

    def _coords(self, g):
        coords = {}
        for n, data in g.nodes(data=True):
            props = data.get("properties", {})
            lat = props.get("lat") or props.get("latitude")
            lon = props.get("lon") or props.get("longitude")
            if lat is not None and lon is not None:
                try:
                    coords[n] = (float(lat), float(lon))
                except (TypeError, ValueError):
                    pass
        return coords

    def _proximity(self, g, coords, params):
        threshold = params.get("threshold_km", 50.0)
        pairs = []
        items = list(coords.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                n1, c1 = items[i]
                n2, c2 = items[j]
                d = _haversine(c1[0], c1[1], c2[0], c2[1])
                if d <= threshold:
                    pairs.append({"a": str(n1), "b": str(n2), "distance_km": round(d, 3)})
        return {"nearby_pairs": pairs, "threshold_km": threshold}

    def _nearest(self, g, coords, params):
        k = params.get("k", 3)
        out = {}
        for n, c in coords.items():
            dists = sorted(
                ((_haversine(c[0], c[1], oc[0], oc[1]), str(on))
                 for on, oc in coords.items() if on != n))[:k]
            out[str(n)] = [{"node": d[1], "distance_km": round(d[0], 3)} for d in dists]
        return {"nearest_neighbors": out}

    def _clusters(self, g, coords, params):
        threshold = params.get("threshold_km", 50.0)
        # Greedy distance-threshold clustering.
        nodes = list(coords.items())
        assigned: Dict[str, int] = {}
        clusters: List[List[str]] = []
        for n, c in nodes:
            placed = False
            for ci, members in enumerate(clusters):
                rep = coords[members[0]]
                if _haversine(c[0], c[1], rep[0], rep[1]) <= threshold:
                    members.append(str(n))
                    placed = True
                    break
            if not placed:
                clusters.append([str(n)])
        return {"spatial_clusters": {f"cluster_{i}": m for i, m in enumerate(clusters)},
                "threshold_km": threshold}


__all__ = ["SpatialAnalyzer"]
