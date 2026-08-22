"""
Knowledge Graph Query Optimizer, Execution Planner, Plan Optimizer and
Backend Selector.

These components operate on the :class:`CypherAst` produced by the parser and
emit an :class:`ExecutionPlan` that the execution engine consumes. All
optimizations are *rule-based* and safe to apply repeatedly (idempotent).

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

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .parser import (
    CypherAst, MatchClause, WhereClause, NodePattern, RelationshipPattern,
    BinaryExpr, Parameter, Variable, CallClause, ReturnClause, WithClause,
    OrderByClause, LimitClause, SkipClause,
)


# --------------------------------------------------------------------------- #
# Execution plan
# --------------------------------------------------------------------------- #
@dataclass
class ExecutionPlan:
    optimized_ast: CypherAst
    backend: str = "memory"
    index_hints: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    rules_applied: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Cost estimator
# --------------------------------------------------------------------------- #
class CostEstimator:
    """Heuristic cost model based on AST shape and selectivity."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def estimate(self, ast: CypherAst) -> float:
        cost = 0.0
        for m in ast.get_clauses(MatchClause):
            for p in m.patterns:
                nodes = [e for e in p.elements if isinstance(e, NodePattern)]
                rels = [e for e in p.elements if isinstance(e, RelationshipPattern)]
                # nodes with label/property filters are cheaper to seed
                for n in nodes:
                    cost += 1.0
                    if n.labels:
                        cost += 0.5
                    if n.properties:
                        cost += 0.5
                for r in rels:
                    span = (r.max_hops or 1)
                    cost += span * 2.0
        if ast.get_clauses(WhereClause):
            cost += 1.0
        if ast.get_clauses(ReturnClause):
            cost += 0.5
        return cost * 10.0


# --------------------------------------------------------------------------- #
# Index selector
# --------------------------------------------------------------------------- #
class IndexSelector:
    """Suggest index hints from node labels/properties in the AST."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def select_indexes(self, ast: CypherAst) -> List[str]:
        hints: List[str] = []
        label_props: Dict[str, set] = {}
        for m in ast.get_clauses(MatchClause):
            for p in m.patterns:
                for el in p.elements:
                    if isinstance(el, NodePattern) and el.labels:
                        key = ":".join(el.labels)
                        props = set(el.properties.keys())
                        label_props.setdefault(key, set()).update(props)
        for label, props in label_props.items():
            for prop in props:
                hints.append(f"INDEX ON :{label}({prop})")
        return hints


# --------------------------------------------------------------------------- #
# Join optimizer
# --------------------------------------------------------------------------- #
class JoinOptimizer:
    """Reorder MATCH patterns to evaluate the most selective first."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def optimize_joins(self, ast: CypherAst) -> CypherAst:
        for m in ast.get_clauses(MatchClause):
            m.patterns.sort(key=self._pattern_selectivity, reverse=True)
        return ast

    @staticmethod
    def _pattern_selectivity(pattern) -> float:
        score = 0.0
        for el in pattern.elements:
            if isinstance(el, NodePattern):
                if el.labels:
                    score += 2.0
                score += len(el.properties) * 1.5
            elif isinstance(el, RelationshipPattern):
                if el.types:
                    score += 1.0
                score += len(el.properties) * 1.0
        return score


# --------------------------------------------------------------------------- #
# Query optimizer
# --------------------------------------------------------------------------- #
class QueryOptimizer:
    """Apply rule-based optimizations to a parsed query AST."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.cost_estimator = CostEstimator(self.config.get("cost", {}))
        self.index_selector = IndexSelector(self.config.get("index", {}))
        self.join_optimizer = JoinOptimizer(self.config.get("join", {}))

    def optimize(self, parsed_query: CypherAst) -> Dict[str, Any]:
        ast = parsed_query
        rules: List[str] = []

        ast = self.push_down_selections(ast)
        rules.append("selection_pushdown")

        ast = self.join_optimizer.optimize_joins(ast)
        rules.append("join_reorder")

        ast = self.push_down_predicates(ast)
        rules.append("predicate_pushdown")

        ast = self.push_down_projections(ast)
        rules.append("projection_pushdown")

        ast = self.fold_constants(ast)
        rules.append("constant_folding")

        index_hints = self.index_selector.select_indexes(ast)
        cost = self.cost_estimator.estimate(ast)

        return {
            "optimized_ast": ast,
            "index_hints": index_hints,
            "cost_estimate": cost,
            "optimization_rules_applied": rules,
        }

    # -- rules ------------------------------------------------------------- #
    def push_down_selections(self, ast: CypherAst) -> CypherAst:
        """Move top-level WHERE equality filters into node/rel patterns."""
        where_clauses = ast.get_clauses(WhereClause)
        if not where_clauses:
            return ast
        # Collect simple equality predicates var.prop = $param
        remaining = []
        for wc in where_clauses:
            expr = wc.expression
            matched = self._extract_equality(expr, ast)
            if matched is None:
                remaining.append(wc)
        # rebuild where clauses list
        new_clauses = [c for c in ast.clauses if not isinstance(c, WhereClause)]
        new_clauses.extend(remaining)
        ast.clauses = new_clauses
        return ast

    def _extract_equality(self, expr, ast: CypherAst):
        if not isinstance(expr, BinaryExpr) or expr.op not in ("=",):
            return None
        left, right = expr.left, expr.right
        # var.prop = $param
        if (isinstance(left, Variable)
                and left.field and isinstance(right, Parameter)):
            self._apply_prop_to_pattern(left.name, left.field, right.name, ast)
            return expr
        return None

    @staticmethod
    def _apply_prop_to_pattern(var: str, prop: str, param: str, ast: CypherAst):
        for m in ast.get_clauses(MatchClause):
            for p in m.patterns:
                for el in p.elements:
                    if isinstance(el, (NodePattern, RelationshipPattern)) and el.variable == var:
                        el.properties[prop] = Parameter(param)

    def push_down_predicates(self, ast: CypherAst) -> CypherAst:
        """Merge node in-pattern property literals into seed constraints."""
        # Already largely handled by parser; here we ensure default-hop rels
        # get bounded to keep traversal cost finite when no bound exists.
        for m in ast.get_clauses(MatchClause):
            for p in m.patterns:
                for el in p.elements:
                    if isinstance(el, RelationshipPattern):
                        if el.min_hops is None and el.max_hops is None:
                            el.min_hops = 1
                            el.max_hops = 1
        return ast

    def push_down_projections(self, ast: CypherAst) -> CypherAst:
        """Mark which variables are needed by RETURN / WITH for pruning."""
        needed: set = set()
        for rc in ast.get_clauses(ReturnClause):
            for item in rc.items:
                needed |= _collect_variables(item.expression)
        for wc in ast.get_clauses(WithClause):
            for item in wc.items:
                needed |= _collect_variables(item.expression)
        ast._projection_vars = needed  # type: ignore[attr-defined]
        return ast

    def fold_constants(self, ast: CypherAst) -> CypherAst:
        """Constant fold WHERE/RETURN expressions (reuses normalizer)."""
        from .parser import QueryNormalizer
        QueryNormalizer().normalize(ast)
        return ast


def _collect_variables(node) -> set:
    out: set = set()
    if hasattr(node, "name") and hasattr(node, "field"):
        out.add(node.name)
    for child in getattr(node, "__dict__", {}).values():
        if isinstance(child, list):
            for c in child:
                out |= _collect_variables(c)
        elif hasattr(c := child, "name") if not isinstance(child, (str, int, float, bool)) else False:
            out |= _collect_variables(child)
    return out


# --------------------------------------------------------------------------- #
# Backend selector
# --------------------------------------------------------------------------- #
class BackendSelector:
    """Choose the most appropriate backend for a plan."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.preferred = self.config.get("preferred", ["memory", "neo4j", "memgraph", "sparql"])
        self.available = self.config.get("available", ["memory"])

    def select_backend(self, plan: ExecutionPlan) -> str:
        ast = plan.optimized_ast
        # SPARQL queries (CALL graph.sparql.*) route to sparql backend
        for c in ast.clauses:
            if isinstance(c, CallClause):
                if "sparql" in c.procedure.lower():
                    return "sparql" if "sparql" in self.available else "memory"
        for backend in self.preferred:
            if backend in self.available:
                return backend
        return "memory"


# --------------------------------------------------------------------------- #
# Plan optimizer
# --------------------------------------------------------------------------- #
class PlanOptimizer:
    """Optimize a physical :class:`ExecutionPlan` (operator ordering)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def optimize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        # Combine consecutive filter steps where possible.
        merged: List[Dict[str, Any]] = []
        for step in plan.steps:
            if merged and merged[-1].get("op") == "filter" and step.get("op") == "filter":
                merged[-1]["predicates"].extend(step.get("predicates", []))
            else:
                merged.append(dict(step))
        plan.steps = merged
        # Finalize estimated cost if still zero
        if plan.estimated_cost == 0.0:
            plan.estimated_cost = CostEstimator().estimate(plan.optimized_ast)
        return plan


# --------------------------------------------------------------------------- #
# Execution planner
# --------------------------------------------------------------------------- #
class ExecutionPlanner:
    """Turn an optimized AST into a physical :class:`ExecutionPlan`."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.plan_optimizer = PlanOptimizer(self.config.get("plan_optimizer", {}))
        self.backend_selector = BackendSelector(self.config.get("backend", {}))

    def create_plan(self, optimized_query: Dict[str, Any]) -> ExecutionPlan:
        ast = optimized_query["optimized_ast"]
        plan = ExecutionPlan(
            optimized_ast=ast,
            index_hints=optimized_query.get("index_hints", []),
            estimated_cost=optimized_query.get("cost_estimate", 0.0),
            rules_applied=optimized_query.get("optimization_rules_applied", []),
        )
        plan.steps = self._generate_steps(ast)
        plan = self.plan_optimizer.optimize_plan(plan)
        backend = self.backend_selector.select_backend(plan)
        plan.backend = backend
        return plan

    @staticmethod
    def _generate_steps(ast: CypherAst) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        for c in ast.clauses:
            if isinstance(c, MatchClause):
                steps.append({
                    "op": "match",
                    "patterns": c.patterns,
                    "optional": c.optional,
                })
            elif isinstance(c, WhereClause):
                steps.append({
                    "op": "filter",
                    "predicates": [c.expression] if c.expression else [],
                })
            elif isinstance(c, CallClause):
                steps.append({
                    "op": "procedure",
                    "procedure": c.procedure,
                    "args": c.args,
                    "yield_fields": c.yield_fields,
                })
            elif isinstance(c, ReturnClause):
                steps.append({
                    "op": "return",
                    "items": c.items,
                    "distinct": c.distinct,
                })
            elif isinstance(c, WithClause):
                steps.append({
                    "op": "with",
                    "items": c.items,
                    "where": c.where,
                })
            elif isinstance(c, OrderByClause):
                steps.append({"op": "order_by", "items": c.items})
            elif isinstance(c, LimitClause):
                steps.append({"op": "limit", "count": c.count})
            elif isinstance(c, SkipClause):
                steps.append({"op": "skip", "count": c.count})
            elif isinstance(c, tuple) and c[0] == "UNION":
                steps.append({"op": "union"})
        return steps


__all__ = [
    "ExecutionPlan", "CostEstimator", "IndexSelector", "JoinOptimizer",
    "QueryOptimizer", "BackendSelector", "PlanOptimizer", "ExecutionPlanner",
]
