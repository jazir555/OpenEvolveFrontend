"""Genetic Programming for Symbolic Regression.

A compact, dependency-free Genetic Programming (GP) engine that evolves
expression trees to fit (X, y) data. Documented originally in
``docs/architecture/EVOLUTION_ALGORITHM_ENHANCEMENT_SPEC.md`` (the
"Symbolic Regression" section) as ``SymbolicRegressionGP``; this is a real,
runnable implementation of that design.

Expression trees use operator nodes (``+ - * / sin cos exp log``) and
terminal nodes (variables ``x0..xn`` and random constants). The engine
provides ramped half-and-half initialization, subtree crossover, point and
subtree mutation, reproduction, tournament selection, and fitness based on
negative MSE with a parsimony (tree-size) penalty. Evaluation is guarded
against division-by-zero and overflow (large finite penalty instead of
crashing).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

# --------------------------------------------------------------------------- #
# Operator definitions
# --------------------------------------------------------------------------- #

_BINARY_OPS = ("+", "-", "*", "/")
_UNARY_OPS = ("sin", "cos", "exp", "log")
DEFAULT_FUNCTION_SET = list(_BINARY_OPS) + list(_UNARY_OPS)

# Arity of every supported operator. Terminals (var/const) have arity 0.
_ARITY = {
    "+": 2, "-": 2, "*": 2, "/": 2,
    "sin": 1, "cos": 1, "exp": 1, "log": 1,
}

_PENALTY = 1e6  # large finite value used for unsafe evaluation


class Node:
    """A node in an expression tree.

    A node is either a function (``op`` in the function set, with ``children``)
    or a terminal (``op == "var"`` with ``value`` = variable index, or
    ``op == "const"`` with ``value`` = float constant).
    """

    __slots__ = ("op", "children", "value")

    def __init__(
        self,
        op: str,
        children: Optional[List["Node"]] = None,
        value: Any = None,
    ) -> None:
        self.op = op
        self.children = children if children is not None else []
        self.value = value

    def arity(self) -> int:
        return _ARITY.get(self.op, 0)

    def is_terminal(self) -> bool:
        return self.arity() == 0

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def copy(self) -> "Node":
        return Node(
            self.op,
            [c.copy() for c in self.children],
            self.value,
        )


# --------------------------------------------------------------------------- #
# Safe evaluation
# --------------------------------------------------------------------------- #


def _safe_eval(node: Node, x: Sequence[float]) -> float:
    """Evaluate ``node`` at input vector ``x`` with safety guards.

    Division by zero returns a large finite penalty; ``exp`` input is clamped
    to avoid overflow; ``log`` of a non-positive argument returns 0. The
    result is always a finite float so the caller can never crash.
    """
    op = node.op
    if op == "var":
        try:
            return float(x[node.value])
        except (IndexError, TypeError):
            return _PENALTY
    if op == "const":
        return float(node.value)

    args = [_safe_eval(c, x) for c in node.children]
    for a in args:
        if not math.isfinite(a):
            return _PENALTY

    if op == "+":
        return args[0] + args[1]
    if op == "-":
        return args[0] - args[1]
    if op == "*":
        return args[0] * args[1]
    if op == "/":
        b = args[1]
        if abs(b) < 1e-9:
            return _PENALTY
        return args[0] / b
    if op == "sin":
        return math.sin(args[0])
    if op == "cos":
        return math.cos(args[0])
    if op == "exp":
        return math.exp(min(max(args[0], -50.0), 50.0))
    if op == "log":
        a = args[0]
        if abs(a) < 1e-9:
            return 0.0
        return math.log(abs(a))
    return _PENALTY


def tree_to_callable(node: Node, n_vars: int) -> Callable[[Sequence[float]], float]:
    """Return a callable ``f(x)`` that evaluates the tree on a vector ``x``."""

    def _func(x: Sequence[float]) -> float:
        return _safe_eval(node, x)

    return _func


def tree_to_string(node: Node) -> str:
    """Return a human-readable infix/function string for the tree."""
    op = node.op
    if op == "var":
        return f"x{node.value}"
    if op == "const":
        return repr(float(node.value))
    if op in ("sin", "cos", "exp", "log"):
        return f"{op}({tree_to_string(node.children[0])})"
    # Binary operators rendered in infix form.
    left = tree_to_string(node.children[0])
    right = tree_to_string(node.children[1])
    return f"({left} {op} {right})"


# --------------------------------------------------------------------------- #
# Tree generation and genetic operators
# --------------------------------------------------------------------------- #


def _random_terminal(rng: random.Random, n_vars: int, const_range) -> Node:
    if n_vars > 0 and rng.random() < (1.0 / (1.0 + 1.0)):  # bias toward variables
        return Node("var", value=rng.randrange(n_vars))
    lo, hi = const_range
    return Node("const", value=rng.uniform(lo, hi))


def _random_tree(
    rng: random.Random,
    max_depth: int,
    grow: bool,
    function_set: Sequence[str],
    n_vars: int,
    const_range,
) -> Node:
    """Generate a random tree using the ``grow`` or ``full`` method."""
    if max_depth <= 0:
        return _random_terminal(rng, n_vars, const_range)
    if grow and rng.random() < 0.3:
        return _random_terminal(rng, n_vars, const_range)

    op = rng.choice(list(function_set))
    arity = _ARITY[op]
    children = [
        _random_tree(rng, max_depth - 1, grow, function_set, n_vars, const_range)
        for _ in range(arity)
    ]
    return Node(op, children)


def _all_subtrees(node: Node, path: Tuple[int, ...] = ()) -> List[Tuple[Node, Tuple[int, ...]]]:
    """Return every (subtree, path) pair in the tree (root included)."""
    out = [(node, path)]
    for i, child in enumerate(node.children):
        out.extend(_all_subtrees(child, path + (i,)))
    return out


def _replace_subtree(root: Node, path: Tuple[int, ...], new_subtree: Node) -> Node:
    """Return a copy of ``root`` with the node at ``path`` replaced."""
    if not path:
        return new_subtree.copy()
    root = root.copy()
    current = root
    for step in path[:-1]:
        current = current.children[step]
    current.children[path[-1]] = new_subtree.copy()
    return root


def subtree_crossover(
    parent1: Node, parent2: Node, rng: random.Random
) -> Tuple[Node, Node]:
    """Subtree crossover: swap randomly chosen subtrees between parents."""
    subs1 = _all_subtrees(parent1)
    subs2 = _all_subtrees(parent2)
    t1, p1 = rng.choice(subs1)
    t2, p2 = rng.choice(subs2)
    child1 = _replace_subtree(parent1, p1, t2)
    child2 = _replace_subtree(parent2, p2, t1)
    return child1, child2


def subtree_mutation(
    node: Node,
    rng: random.Random,
    function_set: Sequence[str],
    n_vars: int,
    const_range,
    max_depth: int,
) -> Node:
    """Replace a randomly chosen subtree with a freshly generated tree."""
    subs = _all_subtrees(node)
    _, path = rng.choice(subs)
    remaining = max_depth - len(path)
    depth = max(1, min(remaining, 4)) if remaining > 0 else 1
    new_sub = _random_tree(rng, depth, grow=True, function_set=function_set,
                           n_vars=n_vars, const_range=const_range)
    return _replace_subtree(node, path, new_sub)


def point_mutation(
    node: Node,
    rng: random.Random,
    function_set: Sequence[str],
    n_vars: int,
    const_range,
) -> Node:
    """Point mutation: replace one node in place.

    Function nodes are swapped for another function of the same arity; terminals
    are swapped for a new random terminal. The tree is otherwise unchanged.
    """
    node = node.copy()
    subs = _all_subtrees(node)
    target, path = rng.choice(subs)

    if not path:  # root selected: replace the whole node
        if target.is_terminal():
            return _random_terminal(rng, n_vars, const_range)
        same_arity = [o for o in function_set if _ARITY[o] == target.arity()]
        replacement = rng.choice(same_arity) if same_arity else target.op
        return Node(replacement, [c.copy() for c in target.children])

    current = node
    for step in path[:-1]:
        current = current.children[step]

    if target.is_terminal():
        current.children[path[-1]] = _random_terminal(rng, n_vars, const_range)
    else:
        same_arity = [o for o in function_set if _ARITY[o] == target.arity()]
        replacement = rng.choice(same_arity) if same_arity else target.op
        current.children[path[-1]] = Node(replacement, [c.copy() for c in target.children])
    return node


def tournament_selection(
    population: List[Tuple[Node, float]],
    rng: random.Random,
    tournament_size: int,
) -> Node:
    """Return a copy of the fittest individual among ``tournament_size`` picks."""
    contenders = rng.choices(population, k=tournament_size)
    best = max(contenders, key=lambda ind: ind[1])
    return best[0].copy()


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #


def _mse(node: Node, X, y) -> float:
    """Mean squared error of ``node`` against (X, y); inf on any non-finite."""
    total = 0.0
    n = len(y)
    for i in range(n):
        pred = _safe_eval(node, X[i])
        if not math.isfinite(pred):
            return float("inf")
        diff = pred - float(y[i])
        total += diff * diff
    return total / n if n > 0 else float("inf")


def _fitness(node: Node, X, y, parsimony: float) -> float:
    """Fitness = negative MSE minus a parsimony penalty on tree size."""
    mse = _mse(node, X, y)
    if not math.isfinite(mse):
        return float("-inf")
    return -mse - parsimony * node.size()


@dataclass
class SymbolicRegressionResult:
    """Outcome of a GP symbolic-regression run."""

    expression: str
    callable: Callable[[Sequence[float]], float]
    mse: float
    fitness: float
    tree: Node
    history: List[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Main engine
# --------------------------------------------------------------------------- #


class SymbolicRegressionGP:
    """Genetic Programming engine for symbolic regression.

    Mirrors the design-only ``SymbolicRegressionGP`` from the enhancement spec:
    function/terminal sets, ramped half-and-half init, tournament selection,
    subtree crossover, point/subtree mutation, reproduction, and a parsimony
    penalty on fitness.
    """

    def __init__(
        self,
        function_set: Optional[Sequence[str]] = None,
        const_range: Tuple[float, float] = (-1.0, 1.0),
        init_depth: Tuple[int, int] = (2, 6),
        max_depth: int = 10,
        tournament_size: int = 3,
        p_crossover: float = 0.8,
        p_mutation: float = 0.15,
        p_reproduction: float = 0.05,
        parsimony_coefficient: float = 0.001,
    ) -> None:
        self.function_set = list(function_set or DEFAULT_FUNCTION_SET)
        self.const_range = const_range
        self.init_depth = init_depth
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.p_reproduction = p_reproduction
        self.parsimony_coefficient = parsimony_coefficient

    def _init_population(
        self, pop_size: int, n_vars: int, rng: random.Random
    ) -> List[Tuple[Node, float]]:
        lo, hi = self.init_depth
        pop: List[Tuple[Node, float]] = []
        for i in range(pop_size):
            depth = rng.randint(lo, max(lo, hi))
            grow = (i % 2 == 0)  # ramped half-and-half
            tree = _random_tree(
                rng, depth, grow, self.function_set, n_vars, self.const_range
            )
            pop.append((tree, 0.0))
        return pop

    def evolve(
        self,
        data: Tuple[Sequence[Sequence[float]], Sequence[float]],
        generations: int = 100,
        pop_size: int = 200,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> SymbolicRegressionResult:
        """Evolve symbolic expressions to fit ``data = (X, y)``.

        Args:
            data: tuple ``(X, y)`` where ``X`` is a 2D array-like of shape
                ``(n_samples, n_vars)`` and ``y`` is a 1D array-like of targets.
            generations: number of evolutionary generations.
            pop_size: population size.
            random_state: optional seed for reproducibility.
            verbose: if True, print per-generation best MSE.

        Returns:
            A :class:`SymbolicRegressionResult` with the best expression string,
            a callable, its MSE, and the per-generation best-MSE history.
        """
        X, y = data
        X = [[float(v) for v in row] for row in X]
        y = [float(v) for v in y]
        n_vars = len(X[0]) if X else 0

        rng = random.Random(random_state)
        population = self._init_population(pop_size, n_vars, rng)

        history: List[float] = []
        best_node: Optional[Node] = None
        best_mse = float("inf")

        for gen in range(generations):
            # Evaluate population.
            evaluated: List[Tuple[Node, float]] = []
            for tree, _ in population:
                fit = _fitness(tree, X, y, self.parsimony_coefficient)
                evaluated.append((tree, fit))

            # Track best.
            gen_best = max(evaluated, key=lambda ind: ind[1])
            gen_mse = _mse(gen_best[0], X, y)
            if math.isfinite(gen_mse) and gen_mse < best_mse:
                best_mse = gen_mse
                best_node = gen_best[0].copy()
            history.append(min(best_mse, gen_mse))

            if verbose:
                print(f"gen {gen:4d}  best_mse={min(best_mse, gen_mse):.6g}")

            # Build next generation.
            next_pop: List[Tuple[Node, float]] = []
            while len(next_pop) < pop_size:
                r = rng.random()
                if r < self.p_reproduction:
                    parent = tournament_selection(
                        evaluated, rng, self.tournament_size
                    )
                    next_pop.append((parent, 0.0))
                elif r < self.p_reproduction + self.p_crossover:
                    p1 = tournament_selection(evaluated, rng, self.tournament_size)
                    p2 = tournament_selection(evaluated, rng, self.tournament_size)
                    c1, c2 = subtree_crossover(p1, p2, rng)
                    next_pop.append((c1, 0.0))
                    if len(next_pop) < pop_size:
                        next_pop.append((c2, 0.0))
                else:
                    parent = tournament_selection(
                        evaluated, rng, self.tournament_size
                    )
                    child = parent
                    if rng.random() < 0.5:
                        child = subtree_mutation(
                            child, rng, self.function_set, n_vars,
                            self.const_range, self.max_depth,
                        )
                    else:
                        child = point_mutation(
                            child, rng, self.function_set, n_vars,
                            self.const_range,
                        )
                    next_pop.append((child, 0.0))

            # Elitism: keep the single best individual.
            if best_node is not None:
                next_pop[0] = (best_node.copy(), 0.0)
            population = next_pop

        if best_node is None:
            best_node = max(population, key=lambda ind: ind[1])[0]

        best_mse = _mse(best_node, X, y)
        best_fit = _fitness(best_node, X, y, self.parsimony_coefficient)
        return SymbolicRegressionResult(
            expression=tree_to_string(best_node),
            callable=tree_to_callable(best_node, n_vars),
            mse=best_mse,
            fitness=best_fit,
            tree=best_node,
            history=history,
        )


def evolve(
    data: Tuple[Sequence[Sequence[float]], Sequence[float]],
    generations: int = 100,
    pop_size: int = 200,
    function_set: Optional[Sequence[str]] = None,
    const_range: Tuple[float, float] = (-1.0, 1.0),
    init_depth: Tuple[int, int] = (2, 6),
    max_depth: int = 10,
    tournament_size: int = 3,
    p_crossover: float = 0.8,
    p_mutation: float = 0.15,
    p_reproduction: float = 0.05,
    parsimony_coefficient: float = 0.001,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> SymbolicRegressionResult:
    """Convenience wrapper to run :class:`SymbolicRegressionGP` with kwargs."""
    engine = SymbolicRegressionGP(
        function_set=function_set,
        const_range=const_range,
        init_depth=init_depth,
        max_depth=max_depth,
        tournament_size=tournament_size,
        p_crossover=p_crossover,
        p_mutation=p_mutation,
        p_reproduction=p_reproduction,
        parsimony_coefficient=parsimony_coefficient,
    )
    return engine.evolve(
        data,
        generations=generations,
        pop_size=pop_size,
        random_state=random_state,
        verbose=verbose,
    )
