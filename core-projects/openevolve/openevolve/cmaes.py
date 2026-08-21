"""CMA-ES: Covariance Matrix Adaptation Evolution Strategy.

Documented in ``docs/architecture/EVOLUTION_ALGORITHM_ENHANCEMENT_SPEC.md``
(the "Algorithm Variants" list of the Enhanced Evolution Engine) as one of the
algorithm variants alongside NSGA-III, MAP-Elites, Novelty Search and
Differential Evolution; this module is a real, runnable implementation of the
standard (mu/mu_w, lambda)-CMA-ES of Hansen & Ostermeier.

The implementation is pure numpy and follows Hansen's reference formulation:

* a distribution mean vector ``m`` and global step size ``sigma``,
* a full covariance matrix ``C`` factored by eigen decomposition
  (``C = B diag(D^2) B^T``) so samples are drawn as
  ``x = m + sigma * B @ (D * z)`` with ``z ~ N(0, I)``,
* weighted intermediate recombination of the ``mu`` best of ``lambda``
  offspring using logarithmic (super-linear) recombination weights,
* two evolution paths: the conjugate path ``p_sigma`` for cumulative
  step-size adaptation (CSA) and ``p_c`` for the rank-one covariance update,
* rank-one (``c_1``) plus rank-mu (``c_mu``) covariance updates including the
  Heaviside stalling correction,
* optional box bounds handled by repair (clipping) of sampled points, so the
  mean always stays inside the feasible box,
* standard termination checks (``tolfun``, ``tolx``, condition number,
  ``sigma`` divergence, target fitness).

Conventions mirror the other algorithm modules in this package
(``openevolve.nsga3``, ``openevolve.novelty_search``, ``openevolve.neat``,
``openevolve.symbolic_regression``): the objective is **minimized**, seeds are
passed as ``random_state``, and there is both an object API (:class:`CMAES`),
a functional entry point (:func:`evolve` / :func:`run_cmaes`) and a
``select_mo``-style wrapper (:func:`cmaes_selection`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "CMAES",
    "CMAESResult",
    "cmaes_selection",
    "evolve",
    "recombination_weights",
    "run_cmaes",
]

# Value substituted for non-finite objective values so a single bad evaluation
# cannot poison the covariance update (mirrors the penalty used by
# ``openevolve.symbolic_regression._safe_eval``).
_PENALTY = 1e300

BoundsLike = Union[
    None,
    Tuple[float, float],
    Tuple[Sequence[float], Sequence[float]],
    Sequence[Sequence[float]],
]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class CMAESResult:
    """Outcome of a CMA-ES run.

    Attributes:
        best_solution: best (lowest-objective) point found, shape ``(dim,)``.
        best_fitness: objective value of ``best_solution``.
        history: best-so-far objective value after each generation
            (monotonically non-increasing).
        generation_best: best objective value *within* each generation.
        mean: final distribution mean.
        sigma: final global step size.
        covariance: final covariance matrix ``C``.
        generations: number of generations actually executed.
        evaluations: number of objective evaluations performed.
        stop_reason: which termination criterion fired (``"max_generations"``
            when the loop ran to completion).
    """

    best_solution: np.ndarray
    best_fitness: float
    history: List[float] = field(default_factory=list)
    generation_best: List[float] = field(default_factory=list)
    mean: Optional[np.ndarray] = None
    sigma: float = 0.0
    covariance: Optional[np.ndarray] = None
    generations: int = 0
    evaluations: int = 0
    stop_reason: str = "max_generations"

    # Convenience aliases used by other engines in this package.
    @property
    def x(self) -> np.ndarray:
        return self.best_solution

    @property
    def fbest(self) -> float:
        return self.best_fitness


# --------------------------------------------------------------------------- #
# Strategy parameters
# --------------------------------------------------------------------------- #
def default_population_size(dim: int) -> int:
    """Default offspring count ``lambda = 4 + floor(3 ln n)``."""
    return int(4 + math.floor(3.0 * math.log(max(int(dim), 1))))


def recombination_weights(pop_size: int, mu: Optional[int] = None) -> np.ndarray:
    """Positive logarithmic recombination weights (sum to 1).

    ``w_i ~ ln(mu + 0.5) - ln(i)`` for ``i = 1..mu`` which is the standard
    super-linear weighting giving the best-ranked offspring most influence.
    """
    lam = max(int(pop_size), 2)
    mu_eff_count = int(mu) if mu is not None else lam // 2
    mu_eff_count = max(1, min(mu_eff_count, lam))
    ranks = np.arange(1, mu_eff_count + 1, dtype=float)
    weights = np.log(mu_eff_count + 0.5) - np.log(ranks)
    weights = np.maximum(weights, 1e-12)
    return weights / float(np.sum(weights))


def _parse_bounds(
    bounds: BoundsLike, dim: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Normalize the many accepted bounds spellings into (lower, upper) arrays.

    Accepts ``None``, ``(lo, hi)`` scalars, ``(lo_vec, hi_vec)`` sequences, or a
    per-dimension list of ``(lo, hi)`` pairs. ``None``/``inf`` entries mean
    unbounded in that direction.
    """
    if bounds is None:
        return None, None

    lower: np.ndarray
    upper: np.ndarray
    seq = list(bounds)  # type: ignore[arg-type]

    def _expand(spec, fill: float) -> Optional[np.ndarray]:
        if spec is None:
            return np.full(dim, fill, dtype=float)
        vec = np.asarray(spec, dtype=float)
        if vec.ndim == 0:
            return np.full(dim, float(vec), dtype=float)
        if vec.shape == (dim,):
            return vec.astype(float)
        return None

    if len(seq) == 2:
        # Preferred spelling: (lower, upper) with scalar or per-dimension limits.
        lo = _expand(seq[0], -np.inf)
        hi = _expand(seq[1], np.inf)
        if lo is not None and hi is not None:
            lower, upper = lo, hi
        else:
            pairs = np.asarray(bounds, dtype=float)
            if pairs.shape != (dim, 2):
                raise ValueError(
                    f"bounds shape {pairs.shape} incompatible with dim={dim}"
                )
            lower, upper = pairs[:, 0], pairs[:, 1]
    else:
        # Per-dimension [(lo, hi), ...] pairs.
        pairs = np.asarray(bounds, dtype=float)
        if pairs.shape != (dim, 2):
            raise ValueError(
                f"bounds shape {pairs.shape} incompatible with dim={dim}"
            )
        lower, upper = pairs[:, 0], pairs[:, 1]

    lower = np.asarray(lower, dtype=float).reshape(dim)
    upper = np.asarray(upper, dtype=float).reshape(dim)
    if np.any(upper < lower):
        raise ValueError("bounds upper limit must be >= lower limit")
    return lower, upper


# --------------------------------------------------------------------------- #
# CMA-ES engine
# --------------------------------------------------------------------------- #
class CMAES:
    """Standard (mu/mu_w, lambda)-CMA-ES for continuous minimization.

    Args:
        dim: problem dimension ``n``.
        x0: initial mean (default: zeros, or the center of ``bounds``).
        sigma0: initial global step size (coordinate-wise standard deviation).
        pop_size: offspring per generation ``lambda`` (default ``4+3 ln n``).
        mu: parents used for recombination (default ``lambda // 2``).
        bounds: optional box constraints (see :func:`_parse_bounds`).
        random_state: seed for reproducible sampling.
        tol_fitness: stop when the objective drops below this value.
        tol_fun: stop when the best-value spread over recent generations is
            smaller than this.
        tol_x: stop when all step lengths become smaller than this.
        max_condition: stop when ``cond(C)`` exceeds this value.

    Usage is either the ask/tell loop::

        es = CMAES(dim=5, random_state=0)
        while not es.stop():
            xs = es.ask()
            es.tell(xs, [f(x) for x in xs])

    or the batteries-included :meth:`evolve`.
    """

    def __init__(
        self,
        dim: int,
        x0: Optional[Sequence[float]] = None,
        sigma0: float = 0.5,
        pop_size: Optional[int] = None,
        mu: Optional[int] = None,
        bounds: BoundsLike = None,
        random_state: Optional[int] = None,
        tol_fitness: Optional[float] = None,
        tol_fun: float = 1e-12,
        tol_x: float = 1e-12,
        max_condition: float = 1e14,
    ) -> None:
        dim = int(dim)
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if sigma0 <= 0:
            raise ValueError(f"sigma0 must be > 0, got {sigma0}")

        self.dim = dim
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state

        self.lower, self.upper = _parse_bounds(bounds, dim)

        # --- initial mean --------------------------------------------------- #
        if x0 is None:
            if self.lower is not None and self.upper is not None:
                center = np.where(
                    np.isfinite(self.lower) & np.isfinite(self.upper),
                    (self.lower + self.upper) / 2.0,
                    0.0,
                )
                self.mean = np.asarray(center, dtype=float)
            else:
                self.mean = np.zeros(dim, dtype=float)
        else:
            self.mean = np.asarray(x0, dtype=float).reshape(dim).copy()
        self.mean = self._repair(self.mean)

        # --- selection / recombination parameters --------------------------- #
        self.pop_size = int(pop_size) if pop_size else default_population_size(dim)
        self.pop_size = max(4, self.pop_size)
        self.mu = int(mu) if mu else self.pop_size // 2
        self.mu = max(1, min(self.mu, self.pop_size))
        self.weights = recombination_weights(self.pop_size, self.mu)
        # Variance-effective selection mass.
        self.mu_eff = float(1.0 / np.sum(self.weights ** 2))

        n = float(dim)
        # --- adaptation constants (Hansen's reference values) --------------- #
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.damps = (
            1.0
            + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0)
            + self.c_sigma
        )
        self.c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c_1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c_1,
            2.0
            * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
            / ((n + 2.0) ** 2 + self.mu_eff),
        )
        # E||N(0,I)|| approximation.
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        # --- dynamic state -------------------------------------------------- #
        self.sigma = float(sigma0)
        self.sigma0 = float(sigma0)
        self.p_sigma = np.zeros(dim, dtype=float)
        self.p_c = np.zeros(dim, dtype=float)
        self.C = np.eye(dim, dtype=float)
        self.B = np.eye(dim, dtype=float)
        self.D = np.ones(dim, dtype=float)
        self.invsqrtC = np.eye(dim, dtype=float)
        self.eigen_eval = 0  # evaluation count at last eigen decomposition

        self.generation = 0
        self.evaluations = 0
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness = float("inf")
        self.history: List[float] = []
        self.generation_best: List[float] = []
        self.stop_reason: Optional[str] = None

        # --- termination thresholds ----------------------------------------- #
        self.tol_fitness = tol_fitness
        self.tol_fun = float(tol_fun)
        self.tol_x = float(tol_x)
        self.max_condition = float(max_condition)

        self._pending: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _repair(self, x: np.ndarray) -> np.ndarray:
        """Clip ``x`` into the feasible box (no-op when unbounded)."""
        if self.lower is None and self.upper is None:
            return x
        return np.clip(x, self.lower, self.upper)

    def _update_eigen(self) -> None:
        """Refresh ``B``, ``D`` and ``C^{-1/2}`` from the covariance matrix."""
        # Enforce exact symmetry before the decomposition.
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        self.D = np.sqrt(eigenvalues)
        self.B = eigenvectors
        self.invsqrtC = (self.B * (1.0 / self.D)) @ self.B.T
        self.eigen_eval = self.evaluations

    def _maybe_update_eigen(self) -> None:
        """Lazy eigen decomposition (Hansen's O(n^2)-amortized schedule)."""
        interval = self.pop_size / (10.0 * self.dim * (self.c_1 + self.c_mu))
        if self.evaluations - self.eigen_eval > interval:
            self._update_eigen()

    @property
    def condition_number(self) -> float:
        """Axis-ratio squared of the sampling distribution, ``cond(C)``."""
        return float((np.max(self.D) / np.min(self.D)) ** 2)

    # ------------------------------------------------------------------ #
    # ask / tell interface
    # ------------------------------------------------------------------ #
    def ask(self, n_samples: Optional[int] = None) -> np.ndarray:
        """Sample a new generation of candidate solutions.

        Returns an array of shape ``(lambda, dim)``; points are repaired into
        the bounds when bounds were supplied.
        """
        count = int(n_samples) if n_samples else self.pop_size
        z = self.rng.standard_normal((count, self.dim))
        y = (self.B * self.D) @ z.T  # shape (dim, count)
        x = self.mean[:, None] + self.sigma * y
        samples = self._repair(x.T)
        self._pending = samples
        return samples

    def tell(
        self,
        solutions: Sequence[Sequence[float]],
        fitnesses: Sequence[float],
    ) -> None:
        """Update mean, evolution paths, covariance and step size.

        Args:
            solutions: the evaluated candidate points (``(lambda, dim)``).
            fitnesses: their objective values (minimization).
        """
        x = np.asarray(solutions, dtype=float).reshape(-1, self.dim)
        f = np.asarray(fitnesses, dtype=float).reshape(-1)
        if x.shape[0] != f.shape[0]:
            raise ValueError(
                f"solutions/fitnesses length mismatch: {x.shape[0]} vs {f.shape[0]}"
            )
        if x.shape[0] < 1:
            raise ValueError("tell() requires at least one solution")

        # Guard against NaN/inf objective values.
        f = np.where(np.isfinite(f), f, _PENALTY)
        self.evaluations += x.shape[0]
        self.generation += 1

        order = np.argsort(f, kind="stable")
        x_sorted = x[order]
        f_sorted = f[order]

        # Track best-so-far.
        if f_sorted[0] < self.best_fitness:
            self.best_fitness = float(f_sorted[0])
            self.best_solution = x_sorted[0].copy()
        self.generation_best.append(float(f_sorted[0]))
        self.history.append(float(self.best_fitness))

        # --- weighted recombination of the mu best -------------------------- #
        mu_eff_count = min(self.mu, x_sorted.shape[0])
        weights = self.weights[:mu_eff_count]
        weights = weights / float(np.sum(weights))
        old_mean = self.mean.copy()
        self.mean = self._repair(weights @ x_sorted[:mu_eff_count])

        # Steps of the selected offspring, in sigma units.
        y_selected = (x_sorted[:mu_eff_count] - old_mean) / self.sigma
        y_w = weights @ y_selected

        # --- cumulative step-size adaptation path (conjugate path) ---------- #
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + math.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff
        ) * (self.invsqrtC @ y_w)

        ps_norm = float(np.linalg.norm(self.p_sigma))
        denom = math.sqrt(
            max(1.0 - (1.0 - self.c_sigma) ** (2 * self.generation), 1e-20)
        )
        h_sigma = (ps_norm / denom) / self.chi_n < (1.4 + 2.0 / (self.dim + 1.0))

        # --- anisotropic evolution path for the rank-one update -------------- #
        self.p_c = (1.0 - self.c_c) * self.p_c
        if h_sigma:
            self.p_c = self.p_c + math.sqrt(
                self.c_c * (2.0 - self.c_c) * self.mu_eff
            ) * y_w

        # --- covariance matrix update (rank-one + rank-mu) ------------------ #
        c1a = self.c_1 * (1.0 - (0.0 if h_sigma else 1.0) * self.c_c * (2.0 - self.c_c))
        rank_one = self.c_1 * np.outer(self.p_c, self.p_c)
        rank_mu = self.c_mu * (y_selected.T * weights) @ y_selected
        self.C = (1.0 - c1a - self.c_mu) * self.C + rank_one + rank_mu

        # --- step-size update ----------------------------------------------- #
        exponent = (self.c_sigma / self.damps) * (ps_norm / self.chi_n - 1.0)
        self.sigma *= math.exp(float(np.clip(exponent, -10.0, 10.0)))
        self.sigma = float(np.clip(self.sigma, 1e-300, 1e300))

        self._maybe_update_eigen()
        self._pending = None

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #
    def stop(self) -> Optional[str]:
        """Return the name of a satisfied termination criterion, else ``None``."""
        if self.tol_fitness is not None and self.best_fitness <= self.tol_fitness:
            return "tol_fitness"
        if self.generation >= 2:
            window = self.generation_best[-max(10, self.dim):]
            if len(window) >= 5 and (max(window) - min(window)) <= self.tol_fun:
                return "tol_fun"
        step = self.sigma * np.sqrt(np.maximum(np.diag(self.C), 0.0))
        if self.generation >= 1 and np.all(step < self.tol_x) and np.all(
            self.sigma * np.abs(self.p_c) < self.tol_x
        ):
            return "tol_x"
        if self.condition_number > self.max_condition:
            return "condition_number"
        if not np.isfinite(self.sigma) or self.sigma > 1e20 * self.sigma0:
            return "sigma_divergence"
        return None

    # ------------------------------------------------------------------ #
    # High-level loop
    # ------------------------------------------------------------------ #
    def evolve(
        self,
        objective_fn: Callable[[np.ndarray], float],
        generations: int = 100,
        verbose: bool = False,
    ) -> CMAESResult:
        """Minimize ``objective_fn`` for at most ``generations`` generations."""
        if generations < 0:
            raise ValueError("generations must be >= 0")

        reason = "max_generations"
        for gen in range(int(generations)):
            samples = self.ask()
            fitnesses = [self._safe_eval(objective_fn, x) for x in samples]
            self.tell(samples, fitnesses)

            if verbose:
                print(
                    f"gen {gen}: best={self.best_fitness:.6g} "
                    f"sigma={self.sigma:.3g} cond={self.condition_number:.3g}"
                )

            stop = self.stop()
            if stop is not None:
                reason = stop
                break

        self.stop_reason = reason
        best = (
            self.best_solution.copy()
            if self.best_solution is not None
            else self.mean.copy()
        )
        return CMAESResult(
            best_solution=best,
            best_fitness=float(self.best_fitness),
            history=list(self.history),
            generation_best=list(self.generation_best),
            mean=self.mean.copy(),
            sigma=float(self.sigma),
            covariance=self.C.copy(),
            generations=self.generation,
            evaluations=self.evaluations,
            stop_reason=reason,
        )

    @staticmethod
    def _safe_eval(
        objective_fn: Callable[[np.ndarray], float], x: np.ndarray
    ) -> float:
        """Evaluate the objective, converting failures into a large penalty."""
        try:
            value = float(objective_fn(x))
        except Exception:
            return _PENALTY
        if not math.isfinite(value):
            return _PENALTY
        return value


# --------------------------------------------------------------------------- #
# Functional entry points
# --------------------------------------------------------------------------- #
def evolve(
    objective_fn: Callable[[np.ndarray], float],
    dim: int,
    generations: int = 100,
    pop_size: Optional[int] = None,
    x0: Optional[Sequence[float]] = None,
    sigma0: float = 0.5,
    mu: Optional[int] = None,
    bounds: BoundsLike = None,
    random_state: Optional[int] = None,
    tol_fitness: Optional[float] = None,
    verbose: bool = False,
) -> CMAESResult:
    """Minimize a continuous objective with CMA-ES.

    Args:
        objective_fn: callable mapping a 1D numpy array to a scalar to minimize.
        dim: dimension of the search space.
        generations: maximum number of generations.
        pop_size: offspring per generation (default ``4 + 3 ln dim``).
        x0: initial mean (default zeros or the center of ``bounds``).
        sigma0: initial step size.
        mu: number of parents (default ``pop_size // 2``).
        bounds: optional box constraints.
        random_state: seed for reproducibility.
        tol_fitness: early-stop threshold on the objective value.
        verbose: print per-generation progress.

    Returns:
        :class:`CMAESResult` with the best solution and the best-so-far history.
    """
    es = CMAES(
        dim=dim,
        x0=x0,
        sigma0=sigma0,
        pop_size=pop_size,
        mu=mu,
        bounds=bounds,
        random_state=random_state,
        tol_fitness=tol_fitness,
    )
    return es.evolve(objective_fn, generations=generations, verbose=verbose)


def run_cmaes(
    objective_fn: Callable[[np.ndarray], float],
    dim: int,
    generations: int = 100,
    pop_size: Optional[int] = None,
    x0: Optional[Sequence[float]] = None,
    sigma0: float = 0.5,
    mu: Optional[int] = None,
    bounds: BoundsLike = None,
    random_state: Optional[int] = None,
    tol_fitness: Optional[float] = None,
    verbose: bool = False,
) -> CMAESResult:
    """Convenience alias of :func:`evolve` (mirrors ``run_neat``/``run_cmaes``
    style entry points used elsewhere in the package)."""
    return evolve(
        objective_fn,
        dim=dim,
        generations=generations,
        pop_size=pop_size,
        x0=x0,
        sigma0=sigma0,
        mu=mu,
        bounds=bounds,
        random_state=random_state,
        tol_fitness=tol_fitness,
        verbose=verbose,
    )


# --------------------------------------------------------------------------- #
# Selection-contract wrapper (mirrors nsga3_selection / novelty_selection)
# --------------------------------------------------------------------------- #
def cmaes_selection(
    objectives: np.ndarray,
    population_size: int,
    weights: Optional[Sequence[float]] = None,
    random_state: Optional[int] = None,
) -> List[int]:
    """CMA-ES style (mu, lambda) truncation selection over an objective matrix.

    Mirrors the ``select_mo`` contract used by NSGA-II/III and Novelty Search:
    a 2D matrix (rows = individuals, columns = minimization objectives) in, the
    indices kept for the next generation out.

    CMA-ES selection is purely rank-based: objectives are scalarized (weighted
    sum, equal weights by default) and the ``population_size`` best individuals
    are retained in rank order — exactly the ``mu`` parents that CMA-ES would
    use for weighted recombination. Ties are broken deterministically when
    ``random_state`` is given.
    """
    matrix = np.asarray(objectives, dtype=float)
    if matrix.size == 0:
        return []
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    n, m = matrix.shape
    if weights is None:
        w = np.ones(m, dtype=float) / float(m)
    else:
        w = np.asarray(weights, dtype=float).reshape(m)
        total = float(np.sum(np.abs(w)))
        if total > 0:
            w = w / total

    scalarized = matrix @ w
    scalarized = np.where(np.isfinite(scalarized), scalarized, _PENALTY)

    if random_state is not None:
        rng = np.random.default_rng(random_state)
        jitter = rng.random(n) * 1e-12
        order = np.argsort(scalarized + jitter, kind="stable")
    else:
        order = np.argsort(scalarized, kind="stable")

    keep = max(0, min(int(population_size), n))
    return [int(i) for i in order[:keep]]
