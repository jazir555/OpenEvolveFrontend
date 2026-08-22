"""
ML integration: embeddings, similarity and clustering on graph data.

Copyright 2026 OpenEvolve
Licensed under the Apache License, Version 2.0 (the "License").
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import networkx as nx

from .base import BaseAnalyzer, AnalyticsRequest, AnalyticsError


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class MLIntegrator(BaseAnalyzer):
    """Integrate machine-learning routines over graph embeddings."""

    ALGORITHMS = ("cosine_similarity", "node_embeddings", "cluster", "feature_matrix")

    def process(self, request: AnalyticsRequest) -> Dict[str, Any]:
        algorithm = (request.algorithm or "cosine_similarity").lower()
        if algorithm not in self.ALGORITHMS:
            raise AnalyticsError(f"Unknown ML algorithm: {algorithm}")
        g = self._graph(request, directed=False)
        params = request.parameters
        start = time.time()

        emb = self._embeddings(g)
        if algorithm == "cosine_similarity":
            result = self._cosine(emb, params)
        elif algorithm == "node_embeddings":
            result = {"embeddings": {str(k): [float(x) for x in v]
                                     for k, v in emb.items()}}
        elif algorithm == "cluster":
            result = self._cluster(emb, params)
        elif algorithm == "feature_matrix":
            result = self._feature_matrix(g, emb)
        else:  # pragma: no cover
            raise AnalyticsError(f"Unhandled algorithm {algorithm}")

        elapsed = (time.time() - start) * 1000
        return {
            "algorithm": algorithm,
            "results": result,
            "parameters": params,
            "execution_time_ms": elapsed,
        }

    def _embeddings(self, g) -> Dict[str, np.ndarray]:
        emb: Dict[str, np.ndarray] = {}
        for n, data in g.nodes(data=True):
            vec = data.get("properties", {}).get("embedding")
            if isinstance(vec, (list, tuple)):
                emb[n] = np.array(vec, dtype=float)
        return emb

    def _cosine(self, emb, params):
        if not emb:
            return {"pairs": []}
        nodes = list(emb.keys())
        threshold = params.get("threshold", 0.0)
        pairs = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                sim = _cosine(emb[nodes[i]], emb[nodes[j]])
                if sim >= threshold:
                    pairs.append({"a": str(nodes[i]), "b": str(nodes[j]),
                                  "similarity": round(sim, 4)})
        return {"pairs": pairs}

    def _cluster(self, emb, params):
        if not emb:
            return {"clusters": {}}
        k = params.get("k", 3)
        nodes = list(emb.keys())
        matrix = np.array([emb[n] for n in nodes], dtype=float)
        labels = self._kmeans(matrix, k)
        clusters: Dict[int, List[str]] = {}
        for n, lab in zip(nodes, labels):
            clusters.setdefault(int(lab), []).append(str(n))
        return {"clusters": clusters}

    @staticmethod
    def _kmeans(matrix: np.ndarray, k: int, iters: int = 50):
        n = matrix.shape[0]
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=k, replace=False)
        centers = matrix[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            dists = np.linalg.norm(matrix[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = matrix[mask].mean(axis=0)
        return labels

    def _feature_matrix(self, g, emb):
        dim = next(iter(emb.values())).shape[0] if emb else 0
        nodes = list(g.nodes())
        matrix = np.array([emb.get(n, np.zeros(dim)) for n in nodes], dtype=float)
        return {"node_count": len(nodes), "feature_dim": dim,
                "matrix_shape": list(matrix.shape)}


__all__ = ["MLIntegrator"]
