"""Offline unit tests for the Genetic Programming symbolic-regression engine.

No LLM / network required. Fits small synthetic datasets and asserts that the
best MSE decreases over generations / the recovered expression is close, plus
that safe evaluation never crashes on division-by-zero.
"""

import math
import random

from openevolve.selection import run_symbolic_regression
from openevolve.symbolic_regression import (
    Node,
    _safe_eval,
    point_mutation,
    subtree_crossover,
    subtree_mutation,
    tree_to_string,
    SymbolicRegressionGP,
)


def test_additive_fit_decreases_mse():
    rng = random.Random(0)
    X = [[i, i * 2] for i in range(20)]
    y = [a + b for a, b in X]  # y = x0 + x1

    result = run_symbolic_regression(
        data=(X, y),
        generations=30,
        pop_size=120,
        random_state=1,
    )
    history = result.history
    assert history[0] >= history[-1], "best MSE should not increase over generations"
    # The engine should approach the true linear relationship closely.
    assert result.mse < 1e-2, f"recovered MSE too high: {result.mse}"


def test_sin_fit_recovers_close_expression():
    X = [[x] for x in [i * 0.3 for i in range(20)]]
    y = [math.sin(a[0]) for a in X]

    result = run_symbolic_regression(
        data=(X, y),
        generations=60,
        pop_size=200,
        function_set=["+", "-", "*", "/", "sin", "cos"],
        random_state=7,
    )
    # Callable should predict the sine curve within tolerance on unseen points.
    for xv in [0.11, 0.42, 0.77, 1.23]:
        pred = result.callable([xv])
        assert abs(pred - math.sin(xv)) < 0.5, (
            f"pred={pred} vs sin({xv})={math.sin(xv)}"
        )


def test_safe_eval_does_not_crash_on_division_by_zero():
    # A tree that divides by a constant terminal that could be zero.
    tree = Node("/", children=[Node("var", value=0), Node("const", value=0.0)])
    val = _safe_eval(tree, [5.0])
    assert math.isfinite(val), "division by zero must yield a finite penalty, not crash"
    assert val == 1e6

    # exp overflow guard
    big = Node("exp", children=[Node("const", value=1000.0)])
    assert math.isfinite(_safe_eval(big, []))


def test_genetic_operators_preserve_validity():
    rng = random.Random(3)
    fset = ["+", "-", "*", "/", "sin", "cos", "exp", "log"]
    p1 = Node("*", children=[Node("var", value=0), Node("const", value=2.0)])
    p2 = Node("+", children=[Node("var", value=1), Node("const", value=1.0)])

    c1, c2 = subtree_crossover(p1, p2, rng)
    assert isinstance(c1, Node) and isinstance(c2, Node)
    assert tree_to_string(c1) and tree_to_string(c2)

    mutated = subtree_mutation(c1, rng, fset, n_vars=2, const_range=(-1, 1), max_depth=6)
    assert isinstance(mutated, Node)
    assert math.isfinite(_safe_eval(mutated, [1.0, 2.0]))

    point = point_mutation(c2, rng, fset, n_vars=2, const_range=(-1, 1))
    assert isinstance(point, Node)
    assert math.isfinite(_safe_eval(point, [1.0, 2.0]))


def test_gp_class_api_returns_callable_and_history():
    gp = SymbolicRegressionGP(parsimony_coefficient=0.0)
    X = [[i] for i in range(10)]
    y = [3.0 * a[0] + 1.0 for a in X]
    res = gp.evolve(data=(X, y), generations=20, pop_size=80, random_state=5)
    assert callable(res.callable)
    assert len(res.history) == 20
    assert isinstance(res.expression, str) and res.expression
