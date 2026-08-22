"""
Temporal graph analysis: time-series, evolution and burst detection.

Copyright 2026 OpenEvolve
Licensed under the Apache License, Version 2.0 (the "License").
"""

import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import networkx as nx

from .base import BaseAnalyzer, AnalyticsRequest, AnalyticsError


def _parse_ts(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


class TemporalAnalyzer(BaseAnalyzer):
    """Analyze how the graph evolves over time."""

    ALGORITHMS = ("timeline", "evolution", "bursts", "activity_window")

    def analyze(self, request: AnalyticsRequest) -> Dict[str, Any]:
        algorithm = (request.algorithm or "timeline").lower()
        if algorithm not in self.ALGORITHMS:
            raise AnalyticsError(f"Unknown temporal algorithm: {algorithm}")
        g = self._graph(request, directed=True)
        params = request.parameters
        start = time.time()

        edges = self._collect_timestamps(g)
        if algorithm == "timeline":
            result = self._timeline(edges, params)
        elif algorithm == "evolution":
            result = self._evolution(g, edges, params)
        elif algorithm == "bursts":
            result = self._bursts(edges, params)
        elif algorithm == "activity_window":
            result = self._activity_window(edges, params)
        else:  # pragma: no cover
            raise AnalyticsError(f"Unhandled algorithm {algorithm}")

        elapsed = (time.time() - start) * 1000
        return {
            "algorithm": algorithm,
            "results": result,
            "parameters": params,
            "execution_time_ms": elapsed,
        }

    def _collect_timestamps(self, g):
        out = []
        for s, t, data in g.edges(data=True):
            ts = _parse_ts(data.get("properties", {}).get("timestamp")) \
                or _parse_ts(data.get("timestamp"))
            out.append((ts or time.time(), s, t))
        return out

    def _timeline(self, edges, params):
        buckets = {}
        for ts, s, t in edges:
            key = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            buckets.setdefault(key, {"edges": 0, "nodes": set()})
            buckets[key]["edges"] += 1
            buckets[key]["nodes"].add(s)
            buckets[key]["nodes"].add(t)
        return {"timeline": {k: {"edges": v["edges"], "active_nodes": len(v["nodes"])}
                             for k, v in sorted(buckets.items())}}

    def _evolution(self, g, edges, params):
        window = params.get("window_days", 30)
        secs = window * 86400
        if not edges:
            return {"stages": []}
        edges.sort(key=lambda x: x[0])
        t0 = edges[0][0]
        stages = []
        cur = []
        start_t = t0
        for ts, s, t in edges:
            if ts - start_t > secs:
                stages.append(self._stage(start_t, cur))
                start_t = ts
                cur = []
            cur.append((s, t))
        if cur:
            stages.append(self._stage(start_t, cur))
        return {"stages": stages}

    @staticmethod
    def _stage(start_t, edges):
        return {
            "start": datetime.fromtimestamp(start_t, tz=timezone.utc).isoformat(),
            "edge_count": len(edges),
            "unique_nodes": len({n for e in edges for n in e}),
        }

    def _bursts(self, edges, params):
        threshold = params.get("burst_threshold", 2.0)
        daily = {}
        for ts, s, t in edges:
            key = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            daily[key] = daily.get(key, 0) + 1
        if not daily:
            return {"bursts": []}
        vals = list(daily.values())
        mean = sum(vals) / len(vals)
        std = (sum((d - mean) ** 2 for d in vals) / len(vals)) ** 0.5 or 1.0
        bursts = [k for k, v in daily.items() if v > mean + threshold * std]
        return {"bursts": bursts, "mean_daily": mean}

    def _activity_window(self, edges, params):
        since = params.get("since")
        until = params.get("until")
        since_ts = _parse_ts(since) if since else None
        until_ts = _parse_ts(until) if until else None
        count = 0
        for ts, s, t in edges:
            if since_ts and ts < since_ts:
                continue
            if until_ts and ts > until_ts:
                continue
            count += 1
        return {"activity_count": count}


__all__ = ["TemporalAnalyzer"]
