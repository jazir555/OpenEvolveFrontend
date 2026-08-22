"""
Unified multi-query-language Knowledge Graph Query & Analytics Engine.

This is the top-level facade that wires the query parser/optimizer/planner/
executor/cache with the analytics engine against a single pluggable backend.
Everything runs fully offline on the in-memory NetworkX backend; remote
backends (Neo4j/Memgraph/SPARQL) are optional and degrade gracefully.

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

import asyncio
import logging
from typing import Any, Dict, List, Optional, Iterable, Tuple, Union

from .query.parser import QueryParser, QueryBuilder, CypherAst, ast_to_cypher
from .query.optimizer import (
    QueryOptimizer, ExecutionPlanner, ExecutionPlan, PlanOptimizer, BackendSelector,
)
from .query.cache import ResultCache, StatisticsCollector
from .query.languages import MultiLanguageTranslator
from .query.backend import (
    GraphBackend, InMemoryNetworkXBackend, create_backend,
)
from .query.executor import (
    QueryExecutionEngine, CypherCompiler, GraphTraverser, EngineConfig,
)
from .analytics import AnalyticsEngine, AnalyticsRequest

logger = logging.getLogger(__name__)


class UnifiedKGQueryAnalyticsEngine:
    """End-to-end KG query + analytics engine (one backend, one API)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 backend: Optional[GraphBackend] = None):
        self.config = config or {}
        backend_cfg = self.config.get("backend", {})
        backend_name = backend_cfg.get("name", "memory")
        if backend is None:
            self.backend = create_backend(backend_name, backend_cfg)
        else:
            self.backend = backend

        engine_config = EngineConfig(
            parser_config=self.config.get("parser", {}),
            optimizer_config=self.config.get("optimizer", {}),
            planner_config=self.config.get("planner", {}),
            cache_config=self.config.get("cache", {}),
            stats_config=self.config.get("stats", {}),
            backend=backend_name,
            backend_config=backend_cfg,
        )
        self.analytics = AnalyticsEngine(self.backend, self.config.get("analytics", {}))
        self.execution_engine = QueryExecutionEngine(
            engine_config, backend=self.backend, analytics=self.analytics)
        # Keep the analytics engine reference so CALL graph.* works inside queries.
        self.query_parser = QueryParser(self.config.get("parser", {}))
        self.query_optimizer = QueryOptimizer(self.config.get("optimizer", {}))
        self.execution_planner = ExecutionPlanner(self.config.get("planner", {}))
        self.result_cache = self.execution_engine.result_cache
        self.statistics_collector = self.execution_engine.statistics_collector
        self.translator = self.execution_engine.translator

    # -- graph population -------------------------------------------------- #
    def add_node(self, node_id: str, labels: Optional[List[str]] = None,
                 properties: Optional[Dict[str, Any]] = None) -> None:
        self.backend.add_node(node_id, labels, properties)

    def add_edge(self, source: str, target: str, edge_type: str,
                 properties: Optional[Dict[str, Any]] = None) -> None:
        self.backend.add_edge(source, target, edge_type, properties)

    def load_triples(self, triples: Iterable[Tuple[str, str, str, Dict[str, Any]]]) -> None:
        self.backend.load_triples(triples)

    def load_from_unified_kg(self, kg) -> None:
        """Load nodes/edges from a UnifiedKnowledgeGraph-like object."""
        for nid, data in getattr(kg, "nodes", {}).items():
            labels = list(getattr(data, "labels", []) or [])
            props = getattr(data, "properties", None)
            self.backend.add_node(nid, labels,
                                  props if isinstance(props, dict) else {})
        for edge in getattr(kg, "edges", []):
            self.backend.add_edge(getattr(edge, "source"),
                                  getattr(edge, "target"),
                                  getattr(edge, "type", "RELATED_TO"),
                                  getattr(edge, "properties", {}) or {})

    # -- query API --------------------------------------------------------- #
    def query(self, query: str, parameters: Optional[Dict[str, Any]] = None,
              language: str = "cypher",
              options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.execution_engine.execute_query(query, parameters, options, language)

    async def query_async(self, query: str,
                          parameters: Optional[Dict[str, Any]] = None,
                          language: str = "cypher",
                          options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self.execution_engine.execute_query_async(
            query, parameters, options, language)

    def explain(self, query: str, language: str = "cypher") -> Dict[str, Any]:
        """Return the optimized plan + compiled query without executing."""
        if language.lower() not in ("cypher", "opencypher"):
            query, _ = self.translator.translate(language, query)
        parsed = self.query_parser.parse(query)
        optimized = self.query_optimizer.optimize(parsed)
        plan: ExecutionPlan = self.execution_planner.create_plan(optimized)
        return {
            "query_type": parsed.query_type,
            "complexity": parsed.complexity,
            "compiled_cypher": ast_to_cypher(plan.optimized_ast),
            "backend": plan.backend,
            "estimated_cost": plan.estimated_cost,
            "index_hints": plan.index_hints,
            "rules_applied": plan.rules_applied,
        }

    # -- analytics API ----------------------------------------------------- #
    def analytics_run(self, request: Union[Dict[str, Any], AnalyticsRequest]) -> Dict[str, Any]:
        return self.analytics.run(request)

    async def analytics_run_async(self, request) -> Dict[str, Any]:
        return await self.analytics.run_async(request)

    def analytics_run_multiple(self, requests: List[Any]) -> List[Dict[str, Any]]:
        return self.analytics.run_multiple(requests)

    # -- monitoring -------------------------------------------------------- #
    def cache_stats(self) -> Dict[str, Any]:
        return self.result_cache.stats()

    def query_stats(self) -> Dict[str, Any]:
        return self.statistics_collector.summary()

    def graph_stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.name,
            "node_count": self.backend.node_count(),
            "edge_count": self.backend.edge_count(),
        }

    def add_monitor_hook(self, hook: callable) -> None:
        self.execution_engine.add_monitor_hook(hook)
        self.analytics.add_monitor_hook(lambda req, out: hook("analytics", out))

    def close(self) -> None:
        driver = getattr(self.backend, "driver", None)
        if driver is not None and hasattr(driver, "close"):
            try:
                driver.close()
            except Exception:
                pass


__all__ = [
    "UnifiedKGQueryAnalyticsEngine",
]
