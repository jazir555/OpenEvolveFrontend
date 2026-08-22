"""
Knowledge Graph Query Execution Engine.

Wires together the parser, optimizer, planner, cache and statistics, and
dispatches execution to a chosen backend. For the in-memory backend the
:class:`GraphTraverser` interprets the parsed AST directly; for remote
backends a :class:`CypherCompiler` renders the AST back to parameterized
Cypher (or SPARQL) and executes it through the driver with graceful
degradation to the in-memory backend on failure.

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

import time
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .parser import (
    CypherAst, MatchClause, WhereClause, ReturnClause, WithClause, OrderByClause,
    LimitClause, SkipClause, CallClause, NodePattern, RelationshipPattern,
    ExprNode, Literal, Parameter, Variable, FunctionCall, UnaryExpr, BinaryExpr,
    ast_to_cypher, QueryParseError, QueryValidationError, QueryParser,
)
from .optimizer import QueryOptimizer, ExecutionPlanner, ExecutionPlan
from .cache import ResultCache, StatisticsCollector
from .backend import (
    GraphBackend, InMemoryNetworkXBackend, create_backend, BackendEdge,
)
from .languages import MultiLanguageTranslator, TranslationError
from .languages import MultiLanguageTranslator, TranslationError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cypher compiler (AST -> backend query string)
# --------------------------------------------------------------------------- #
class CypherCompiler:
    """Renders a parsed :class:`CypherAst` to a backend-specific query string."""

    def compile(self, ast: CypherAst, backend: str = "memory") -> str:
        query = ast_to_cypher(ast)
        if backend == "sparql":
            return self._to_sparql(ast)
        return query

    def compile_with_params(self, ast: CypherAst, backend: str = "memory") -> tuple:
        return self.compile(ast, backend), ast.parameters

    @staticmethod
    def _to_sparql(ast: CypherAst) -> str:
        # Best-effort translation of a simple MATCH/RETURN to SPARQL.
        match = ast.get_clauses(MatchClause)
        ret = ast.get_clauses(ReturnClause)
        if not match:
            return f"# unsupported cypher for sparql\n{ast_to_cypher(ast)}"
        triples = []
        for m in match:
            for p in m.patterns:
                els = p.elements
                for i in range(0, len(els) - 1, 2):
                    node = els[i]
                    rel = els[i + 1] if i + 1 < len(els) else None
                    tgt = els[i + 2] if i + 2 < len(els) else None
                    s = f"?{node.variable}" if node.variable else "?s"
                    pred = rel.types[0] if rel and rel.types else "relatedTo"
                    o = f"?{tgt.variable}" if tgt and tgt.variable else "?o"
                    triples.append(f"{s} <{pred}> {o} .")
        ret_vars = []
        for rc in ret:
            for item in rc.items:
                if isinstance(item.expression, Variable):
                    ret_vars.append(f"?{item.expression.name}")
        select = " ".join(ret_vars) if ret_vars else "*"
        return f"SELECT {select} WHERE {{ {' '.join(triples)} }}"


# --------------------------------------------------------------------------- #
# Graph traverser (in-memory AST interpreter)
# --------------------------------------------------------------------------- #
_AGGREGATES = {"count", "collect", "sum", "avg", "min", "max"}


class GraphTraverser:
    """Interprets a :class:`CypherAst` against an in-memory backend."""

    def __init__(self, backend: GraphBackend):
        self.backend = backend

    # -- top-level --------------------------------------------------------- #
    def execute(self, ast: CypherAst, parameters: Dict[str, Any],
                analytics=None) -> List[Dict[str, Any]]:
        parameters = parameters or {}
        bindings: List[Dict[str, Any]] = [{}]
        results: Optional[List[Dict[str, Any]]] = None
        for clause in ast.clauses:
            if isinstance(clause, MatchClause):
                bindings = self._match(bindings, clause, parameters)
            elif isinstance(clause, WhereClause):
                if clause.expression:
                    bindings = [b for b in bindings
                                if self._truthy(self._eval(clause.expression, b, parameters))]
            elif isinstance(clause, WithClause):
                bindings = self._with(bindings, clause, parameters)
            elif isinstance(clause, CallClause):
                bindings, results = self._call(bindings, clause, parameters, analytics)
            elif isinstance(clause, ReturnClause):
                results = self._return(bindings, clause, parameters)
            elif isinstance(clause, OrderByClause):
                results = self._order_by(results or [], clause, parameters)
            elif isinstance(clause, LimitClause):
                results = (results or [])[:self._int(self._eval(clause.count, {}, parameters))]
            elif isinstance(clause, SkipClause):
                results = (results or [])[self._int(self._eval(clause.count, {}, parameters)):]
        if results is None:
            # Procedure-only query: return bound variables.
            results = [self._materialize(bindings[0], parameters)] if bindings else []
        return results

    # -- matching ---------------------------------------------------------- #
    def _match(self, bindings: List[Dict[str, Any]], clause: MatchClause,
               params: Dict[str, Any]) -> List[Dict[str, Any]]:
        new_bindings: List[Dict[str, Any]] = []
        for binding in bindings:
            per_pattern: List[List[Dict[str, Any]]] = []
            for pattern in clause.patterns:
                per_pattern.append(self._match_pattern(binding, pattern, params))
            combined = [dict(binding)]
            for pr in per_pattern:
                if not pr:
                    combined = []
                    break
                nxt = []
                for base in combined:
                    for ext in pr:
                        merged = dict(base)
                        merged.update(ext)
                        nxt.append(merged)
                combined = nxt
            if clause.optional:
                if not combined:
                    new_bindings.append(dict(binding))
                else:
                    new_bindings.extend(combined)
            else:
                new_bindings.extend(combined)
        return new_bindings

    def _match_pattern(self, binding: Dict[str, Any], pattern, params) -> List[Dict[str, Any]]:
        els = pattern.elements
        first = els[0]
        if not isinstance(first, NodePattern):
            return [dict(binding)]
        seeds = self.backend.match_nodes(
            self._resolve_labels(first.labels),
            self._resolve_props(first.properties, params))
        out: List[Dict[str, Any]] = []
        for nid in seeds:
            base = dict(binding)
            base[first.variable] = nid
            out.extend(self._expand(base, first.variable, els[1:], params))
        return out

    def _expand(self, binding: Dict[str, Any], src_var: str, rest, params) -> List[Dict[str, Any]]:
        if not rest:
            return [binding]
        rel = rest[0]
        node = rest[1] if len(rest) > 1 else None
        if not isinstance(rel, RelationshipPattern):
            return [binding]
        min_hops = rel.min_hops or 1
        max_hops = rel.max_hops or 1
        src_id = binding.get(src_var)
        if src_id is None:
            return []
        paths = self._enumerate_paths(src_id, rel, params, min_hops, max_hops)
        results: List[Dict[str, Any]] = []
        for node_seq, edge_seq in paths:
            terminal = node_seq[-1]
            nb = dict(binding)
            if node and (not self.backend._node_matches(
                    terminal, self._resolve_labels(node.labels),
                    self._resolve_props(node.properties, params))):
                continue
            if node:
                nb[node.variable] = terminal
            if rel.variable:
                nb[rel.variable] = [self._edge_repr(e) for e in edge_seq]
            if len(rest) > 2:
                results.extend(self._expand(nb, node.variable, rest[2:], params))
            else:
                results.append(nb)
        return results

    def _enumerate_paths(self, src, rel, params, min_hops, max_hops):
        rel_props = self._resolve_props(rel.properties, params)
        results = []
        stack = [(src, [src], [], 0)]
        while stack:
            cur, nodes, edges, depth = stack.pop()
            if min_hops <= depth <= max_hops and depth > 0:
                results.append((list(nodes), list(edges)))
            if depth < max_hops:
                for tgt, edata in self.backend.expand(cur, rel):
                    if tgt not in nodes and self.backend._edge_matches(
                            edata, set(rel.types), rel_props):
                        stack.append((tgt, nodes + [tgt], edges + [edata], depth + 1))
        return results

    # -- WITH / aggregation ----------------------------------------------- #
    def _with(self, bindings: List[Dict[str, Any]], clause: WithClause,
              params) -> List[Dict[str, Any]]:
        rows = self._project(bindings, clause.items, params, alias_map=True)
        if clause.where:
            rows = [r for r in rows if self._truthy(self._eval(clause.where, r, params))]
        return rows

    def _return(self, bindings: List[Dict[str, Any]], clause: ReturnClause,
                params) -> List[Dict[str, Any]]:
        rows = self._project(bindings, clause.items, params)
        if clause.distinct:
            seen = set()
            dedup = []
            for r in rows:
                key = tuple(sorted((k, str(v)) for k, v in r.items()))
                if key not in seen:
                    seen.add(key)
                    dedup.append(r)
            rows = dedup
        return rows

    def _project(self, bindings, items, params, alias_map=False) -> List[Dict[str, Any]]:
        has_agg = any(self._is_aggregate(i.expression) for i in items)
        if not has_agg:
            rows = []
            for b in bindings:
                row = {}
                for item in items:
                    val = self._eval(item.expression, b, params)
                    key = item.alias or self._expr_key(item.expression)
                    row[key] = val
                rows.append(row)
            return rows
        # grouped aggregation
        group_keys = []
        for item in items:
            if not self._is_aggregate(item.expression):
                for v in self._collect_vars(item.expression):
                    if v not in group_keys:
                        group_keys.append(v)
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for b in bindings:
            gk = tuple(b.get(k) for k in group_keys)
            groups.setdefault(gk, []).append(b)
        rows = []
        for gk, group in groups.items():
            row = {}
            for item in items:
                key = item.alias or self._expr_key(item.expression)
                row[key] = self._aggregate(item.expression, group, params)
            rows.append(row)
        return rows

    def _aggregate(self, expr, group: List[Dict[str, Any]], params) -> Any:
        if isinstance(expr, FunctionCall):
            name = expr.name.lower()
            if name == "count":
                if isinstance(expr.args[0], Literal) and expr.args[0].value == "*":
                    return len(group)
                vals = [self._eval(expr.args[0], b, params) for b in group]
                return sum(1 for v in vals if v is not None)
            vals = [self._eval(expr.args[0], b, params) for b in group]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if name == "collect":
                return [self._eval(expr.args[0], b, params) for b in group]
            if name == "sum":
                return sum(vals)
            if name == "avg":
                return sum(vals) / len(vals) if vals else 0
            if name == "min":
                return min(vals) if vals else None
            if name == "max":
                return max(vals) if vals else None
        return self._eval(expr, group[0], params)

    # -- CALL procedures --------------------------------------------------- #
    def _call(self, bindings, clause: CallClause, params, analytics):
        proc = clause.procedure.lower()
        args = [self._eval(a, bindings[0] if bindings else {}, params) for a in clause.args] \
            if clause.args else []
        if proc.startswith("graph.analytics") or proc.startswith("graph.path"):
            if analytics is None:
                return bindings, [{"error": "analytics engine not configured",
                                   "procedure": clause.procedure}]
            request = {
                "type": proc.split(".")[-1] if "." in proc else "centrality",
                "procedure": clause.procedure,
                "parameters": self._call_args_to_dict(clause.args, params),
                "source_node": params.get("source_id"),
                "target_node": params.get("target_id"),
            }
            try:
                result = analytics.run_sync(request)
                return bindings, [{"result": result}]
            except Exception as e:
                return bindings, [{"error": str(e)}]
        if proc in ("shortestpath", "allshortestpaths", "graph.path.shortestpath"):
            src = params.get("source_id")
            tgt = params.get("target_id")
            md = int(params.get("max_depth", 10))
            if src and tgt:
                paths = (self.backend.shortest_paths(src, tgt, md)
                         if proc == "allshortestpaths"
                         else (self.backend.shortest_paths(src, tgt, md)[:1] or [[]]))
                return bindings, [{"paths": paths}]
        return bindings, [{"procedure": clause.procedure, "args": args}]

    @staticmethod
    def _call_args_to_dict(args, params):
        out = {}
        for a in args:
            if isinstance(a, dict):
                out.update(a)
        return out

    # -- expression evaluation --------------------------------------------- #
    def _eval(self, expr: ExprNode, binding: Dict[str, Any], params: Dict[str, Any]) -> Any:
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, Parameter):
            return params.get(expr.name)
        if isinstance(expr, Variable):
            return self._resolve_var(expr, binding, params)
        if isinstance(expr, UnaryExpr):
            if expr.op == "NOT":
                return not self._truthy(self._eval(expr.operand, binding, params))
            return -self._num(self._eval(expr.operand, binding, params))
        if isinstance(expr, BinaryExpr):
            return self._eval_binary(expr, binding, params)
        if isinstance(expr, FunctionCall):
            return self._eval_function(expr, binding, params)
        raise QueryParseError(f"Cannot evaluate expression: {expr}")

    def _eval_binary(self, expr: BinaryExpr, binding, params) -> Any:
        op = expr.op
        if op in ("AND", "OR"):
            l = self._truthy(self._eval(expr.left, binding, params))
            if op == "AND":
                return l and self._truthy(self._eval(expr.right, binding, params))
            return l or self._truthy(self._eval(expr.right, binding, params))
        l = self._eval(expr.left, binding, params)
        r = self._eval(expr.right, binding, params)
        if op == "=":
            return l == r
        if op in ("<>", "!="):
            return l != r
        if op == "<":
            return self._num(l) < self._num(r)
        if op == "<=":
            return self._num(l) <= self._num(r)
        if op == ">":
            return self._num(l) > self._num(r)
        if op == ">=":
            return self._num(l) >= self._num(r)
        if op == "CONTAINS":
            return str(r) in str(l)
        if op == "STARTS WITH":
            return str(l).startswith(str(r))
        if op == "ENDS WITH":
            return str(l).endswith(str(r))
        if op == "IN":
            return l in (r if isinstance(r, (list, tuple, set)) else [r])
        if op == "=~":
            import re as _re
            return _re.search(str(r), str(l)) is not None
        return False

    def _eval_function(self, expr: FunctionCall, binding, params) -> Any:
        name = expr.name.lower()
        args = [self._eval(a, binding, params) for a in expr.args]
        if name == "id":
            v = args[0]
            if isinstance(v, dict) and "id" in v:
                return v["id"]
            return v
        if name == "labels":
            v = args[0]
            if isinstance(v, dict):
                return v.get("labels", [])
            return []
        if name == "type":
            v = args[0]
            if isinstance(v, list) and v:
                return v[0].get("type")
            if isinstance(v, dict):
                return v.get("type")
            return None
        if name == "properties":
            v = args[0]
            if isinstance(v, dict):
                return v.get("properties", {})
            return {}
        if name == "keys":
            v = args[0]
            if isinstance(v, dict):
                return list(v.get("properties", {}).keys())
            return []
        if name in ("size", "length"):
            v = args[0]
            return len(v) if hasattr(v, "__len__") else 0
        if name == "coalesce":
            for a in args:
                if a is not None:
                    return a
            return None
        if name == "toint":
            return self._int(args[0])
        if name == "tofloat":
            return self._num(args[0])
        if name == "tostring":
            return str(args[0])
        # aggregation handled separately
        if name in _AGGREGATES:
            return self._aggregate(expr, [binding], params)
        return None

    def _resolve_var(self, var: Variable, binding, params):
        name = var.name
        if name == "*":
            return binding
        val = binding.get(name)
        if val is None:
            # maybe a literal bound elsewhere
            return None
        if isinstance(val, list) and val and isinstance(val[0], dict) and "type" in val[0]:
            if var.field:
                return val[-1].get("properties", {}).get(var.field)
            return val
        if isinstance(val, str):
            g = self.backend.as_networkx()
            if val in g:
                data = g.nodes[val]
                if var.field:
                    return data.get("properties", {}).get(var.field)
                return {"id": val, "labels": data.get("labels", []),
                        "properties": data.get("properties", {})}
            return val
        return val

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _resolve_labels(labels):
        return [l for l in labels if l]

    @staticmethod
    def _resolve_props(props, params):
        out = {}
        for k, v in props.items():
            if isinstance(v, Parameter):
                out[k] = params.get(v.name)
            elif isinstance(v, ExprNode):
                out[k] = _literal_of(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _edge_repr(edata):
        return {
            "type": edata.get("type"),
            "source": edata.get("source"),
            "target": edata.get("target"),
            "properties": edata.get("properties", {}),
        }

    @staticmethod
    def _is_aggregate(expr) -> bool:
        return isinstance(expr, FunctionCall) and expr.name.lower() in _AGGREGATES

    @staticmethod
    def _collect_vars(expr, out=None):
        if out is None:
            out = []
        if isinstance(expr, Variable) and expr.name not in ("*",):
            if expr.name not in out:
                out.append(expr.name)
        elif isinstance(expr, FunctionCall):
            for a in expr.args:
                GraphTraverser._collect_vars(a, out)
        elif isinstance(expr, BinaryExpr):
            GraphTraverser._collect_vars(expr.left, out)
            GraphTraverser._collect_vars(expr.right, out)
        elif isinstance(expr, UnaryExpr):
            GraphTraverser._collect_vars(expr.operand, out)
        return out

    @staticmethod
    def _expr_key(expr) -> str:
        if isinstance(expr, Variable):
            return expr.name + (f".{expr.field}" if expr.field else "")
        if isinstance(expr, FunctionCall):
            return expr.name
        return "expr"

    @staticmethod
    def _materialize(binding: Dict[str, Any], params) -> Dict[str, Any]:
        out = {}
        for k, v in binding.items():
            if isinstance(v, str):
                out[k] = v
            elif isinstance(v, list) and v and isinstance(v[0], dict) and "type" in v[0]:
                out[k] = v
        return out

    @staticmethod
    def _order_by(results, clause: OrderByClause, params):
        def sort_key(row):
            keys = []
            for item in clause.items:
                v = GraphTraverser._eval(item.expression, row, params)
                keys.append((GraphTraverser._num(v) if isinstance(v, (int, float)) else str(v),
                            item.descending))
            return tuple((k if not d else -k) if isinstance(k, (int, float)) else k
                        for k, d in keys)
        return sorted(results, key=sort_key)

    @staticmethod
    def _truthy(v) -> bool:
        if isinstance(v, (bool,)):
            return v
        if v is None:
            return False
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, (list, dict, str, tuple, set)):
            return len(v) > 0
        return bool(v)

    @staticmethod
    def _num(v):
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0


def _literal_of(node):
    if isinstance(node, Literal):
        return node.value
    return node


# --------------------------------------------------------------------------- #
# Execution engine
# --------------------------------------------------------------------------- #
@dataclass
class EngineConfig:
    parser_config: Dict[str, Any] = field(default_factory=dict)
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    planner_config: Dict[str, Any] = field(default_factory=dict)
    cache_config: Dict[str, Any] = field(default_factory=dict)
    stats_config: Dict[str, Any] = field(default_factory=dict)
    backend: str = "memory"
    backend_config: Dict[str, Any] = field(default_factory=dict)


class QueryExecutionEngine:
    """Parse, optimize, plan, dispatch and execute graph queries."""

    def __init__(self, config: Optional[EngineConfig] = None,
                 backend: Optional[GraphBackend] = None,
                 analytics=None):
        self.config = config or EngineConfig()
        self.query_parser = QueryParser(self.config.parser_config)
        self.query_optimizer = QueryOptimizer(self.config.optimizer_config)
        self.execution_planner = ExecutionPlanner(self.config.planner_config)
        self.result_cache = ResultCache(self.config.cache_config)
        self.statistics_collector = StatisticsCollector(self.config.stats_config)
        self.compiler = CypherCompiler()
        self._analytics = analytics
        self.translator = MultiLanguageTranslator()
        if backend is None:
            self.backend = create_backend(self.config.backend, self.config.backend_config)
        else:
            self.backend = backend
        self._monitor_hooks: List[callable] = []

    def add_monitor_hook(self, hook) -> None:
        self._monitor_hooks.append(hook)
        self.result_cache.add_monitor_hook(lambda e, d: hook(e, d))

    # -- sync API ---------------------------------------------------------- #
    def execute_query(self, query: str,
                      parameters: Optional[Dict[str, Any]] = None,
                      options: Optional[Dict[str, Any]] = None,
                      language: str = "cypher") -> Dict[str, Any]:
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("Use execute_query_async from within an async context")
        return asyncio.run(self.execute_query_async(query, parameters, options, language))

    async def execute_query_async(self, query: str,
                                  parameters: Optional[Dict[str, Any]] = None,
                                  options: Optional[Dict[str, Any]] = None,
                                  language: str = "cypher") -> Dict[str, Any]:
        options = options or {}
        parameters = parameters or {}
        start = time.time()

        # Translate non-Cypher languages into Cypher (or SPARQL).
        if language and language.lower() not in ("cypher", "opencypher"):
            try:
                query, target_backend = self.translator.translate(language, query)
            except TranslationError as e:
                return {
                    "success": False,
                    "error": f"Translation failed: {e}",
                    "backend": self.backend.name,
                    "execution_time_ms": 0.0,
                }

        # Parse + optimize + plan
        parsed = self.query_parser.parse(query)
        parsed.parameters.update(parameters)
        optimized = self.query_optimizer.optimize(parsed)
        plan: ExecutionPlan = self.execution_planner.create_plan(optimized)

        cache_key = ResultCache.generate_cache_key(
            ast_to_cypher(plan.optimized_ast), parameters)
        if not options.get("bypass_cache", False) and not options.get("no_cache", False):
            cached = self.result_cache.get(cache_key)
            if cached is not None:
                elapsed = (time.time() - start) * 1000
                self.statistics_collector.record_cache_hit(query, elapsed,
                                                          plan.backend, len(cached))
                return {
                    "success": True,
                    "results": cached,
                    "from_cache": True,
                    "backend": plan.backend,
                    "execution_time_ms": elapsed,
                    "plan": self._plan_summary(plan),
                }

        try:
            result = await self._dispatch(plan, parameters)
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            self.statistics_collector.record_query_execution(
                query, elapsed, 0, plan.backend, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "from_cache": False,
                "backend": plan.backend,
                "execution_time_ms": elapsed,
            }

        elapsed = (time.time() - start) * 1000
        self.statistics_collector.record_query_execution(
            query, elapsed, len(result), plan.backend)

        if self._should_cache(result, options):
            ttl = options.get("cache_ttl", self.result_cache.default_ttl)
            self.result_cache.set(cache_key, result, ttl)

        return {
            "success": True,
            "results": result,
            "from_cache": False,
            "backend": plan.backend,
            "execution_time_ms": elapsed,
            "plan": self._plan_summary(plan),
        }

    async def _dispatch(self, plan: ExecutionPlan, parameters: Dict[str, Any]):
        backend = plan.backend
        if backend in ("neo4j", "memgraph", "sparql") and self.backend.is_available():
            try:
                query_str = self.compiler.compile(plan.optimized_ast, backend)
                return await self._execute_remote(backend, query_str, parameters)
            except Exception as e:
                logger.warning(
                    f"Remote backend {backend} failed ({e}); degrading to in-memory")
        # In-memory execution
        traverser = GraphTraverser(self.backend)
        return traverser.execute(plan.optimized_ast, parameters, self._analytics)

    async def _execute_remote(self, backend: str, query_str: str,
                              parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Generic remote execution path (driver-based). Raises if unavailable.
        if backend == "sparql":
            raise NotImplementedError("SPARQL remote execution requires an endpoint client")
        driver = self.backend.driver
        if driver is None:
            raise RuntimeError("driver unavailable")
        # neo4j/memgraph share the neo4j driver API
        from neo4j import AsyncSession  # type: ignore
        async with driver.session() as session:
            res = await session.run(query_str, parameters)
            records = await res.data()
        return records

    @staticmethod
    def _should_cache(result, options) -> bool:
        if options.get("no_cache", False):
            return False
        if not isinstance(result, (list, dict)):
            return False
        if len(result) > options.get("max_cache_result_size", 100000):
            return False
        return True

    @staticmethod
    def _plan_summary(plan: ExecutionPlan) -> Dict[str, Any]:
        return {
            "backend": plan.backend,
            "estimated_cost": plan.estimated_cost,
            "index_hints": plan.index_hints,
            "rules_applied": plan.rules_applied,
        }

    # -- graph mutation helpers (for populating the backend) --------------- #
    def load_triples(self, triples) -> None:
        self.backend.load_triples(triples)

    def add_node(self, node_id, labels=None, properties=None):
        self.backend.add_node(node_id, labels, properties)

    def add_edge(self, source, target, edge_type, properties=None):
        self.backend.add_edge(source, target, edge_type, properties)


import asyncio  # noqa: E402  (kept at bottom for clarity)


__all__ = [
    "CypherCompiler", "GraphTraverser", "QueryExecutionEngine", "EngineConfig",
]
