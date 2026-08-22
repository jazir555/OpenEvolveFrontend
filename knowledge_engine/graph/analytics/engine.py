"""
Unified Analytics Engine that dispatches to the specialized analyzers.

Copyright 2026 OpenEvolve
Licensed under the Apache License, Version 2.0 (the "License").
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..query.backend import GraphBackend
from .base import AnalyticsRequest, AnalyticsError
from .centrality import CentralityAnalyzer
from .community import CommunityDetector
from .path import PathAnalyzer
from .pattern import PatternMiner
from .temporal import TemporalAnalyzer
from .spatial import SpatialAnalyzer
from .ml import MLIntegrator

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Dispatch analytics requests to the appropriate analyzer."""

    def __init__(self, backend: Optional[GraphBackend] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.centrality_analyzer = CentralityAnalyzer(backend)
        self.community_detector = CommunityDetector(backend)
        self.path_analyzer = PathAnalyzer(backend)
        self.pattern_miner = PatternMiner(backend)
        self.temporal_analyzer = TemporalAnalyzer(backend)
        self.spatial_analyzer = SpatialAnalyzer(backend)
        self.ml_integrator = MLIntegrator(backend)
        self._monitor_hooks: List[callable] = []

    def set_backend(self, backend: GraphBackend) -> None:
        for a in (self.centrality_analyzer, self.community_detector,
                  self.path_analyzer, self.pattern_miner, self.temporal_analyzer,
                  self.spatial_analyzer, self.ml_integrator):
            a.backend = backend

    def add_monitor_hook(self, hook: callable) -> None:
        self._monitor_hooks.append(hook)

    # -- dispatch ---------------------------------------------------------- #
    def run(self, request: Any) -> Dict[str, Any]:
        req = request if isinstance(request, AnalyticsRequest) else AnalyticsRequest.from_dict(request)
        start = 0.0
        try:
            if req.type == "centrality":
                result = self.centrality_analyzer.analyze(req)
            elif req.type == "community":
                result = self.community_detector.detect(req)
            elif req.type == "path":
                result = self.path_analyzer.analyze(req)
            elif req.type == "pattern":
                result = self.pattern_miner.mine(req)
            elif req.type == "temporal":
                result = self.temporal_analyzer.analyze(req)
            elif req.type == "spatial":
                result = self.spatial_analyzer.analyze(req)
            elif req.type == "ml":
                result = self.ml_integrator.process(req)
            else:
                raise AnalyticsError(f"Unknown analytics type: {req.type}")
            return self._wrap(req, result, None)
        except Exception as e:
            logger.error(f"Analytics {req.type}/{req.algorithm} failed: {e}")
            return self._wrap(req, None, str(e))

    async def run_async(self, request: Any) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, request)

    def run_sync(self, request: Any) -> Dict[str, Any]:
        return self.run(request)

    def run_multiple(self, requests: List[Any]) -> List[Dict[str, Any]]:
        results = []
        for r in requests:
            res = self.run(r)
            res["request_id"] = (r.request_id if hasattr(r, "request_id") else
                                 (r.get("request_id") if isinstance(r, dict) else None))
            results.append(res)
        return results

    async def run_multiple_async(self, requests: List[Any]) -> List[Dict[str, Any]]:
        tasks = [self.run_async(r) for r in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def _wrap(self, req: AnalyticsRequest, result: Optional[Dict[str, Any]],
              error: Optional[str]) -> Dict[str, Any]:
        if error:
            out = {"type": req.type, "algorithm": req.algorithm, "success": False,
                   "error": error}
        else:
            out = {"type": req.type, "algorithm": req.algorithm, "success": True,
                   "result": result}
        out["request_id"] = req.request_id
        for hook in self._monitor_hooks:
            try:
                hook(req, out)
            except Exception:
                pass
        return out


__all__ = ["AnalyticsEngine"]
