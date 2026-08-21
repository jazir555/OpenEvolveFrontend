"""math_domain - canonical mathematical-domain and evaluation-metric registries.

Flat-script module providing the shared ``MathematicalDomain`` and
``EvaluationMetric`` definitions that the ``engines/`` decomposition, workflow and
evaluation scripts expect via ``from math_domain import MathematicalDomain,
EvaluationMetric``.

``MathematicalDomain`` is a string Enum (member names match the taxonomy already
used across the engines) with lookup/detection helpers. ``EvaluationMetric`` is a
registry of metric descriptors, each with a ``compute`` helper; the standard
metrics are also exposed as class attributes so ``EvaluationMetric.CORRECTNESS``
keeps working.

Pure-Python, no external dependencies.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MathematicalDomain
# ---------------------------------------------------------------------------
class MathematicalDomain(str, Enum):
    """Taxonomy of mathematical domains used for routing and tagging."""

    GENERAL = "general"
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number_theory"
    TOPOLOGY = "topology"
    LOGIC = "logic"
    SET_THEORY = "set_theory"
    COMBINATORICS = "combinatorics"
    PROBABILITY = "probability"
    STATISTICS = "statistics"
    LINEAR_ALGEBRA = "linear_algebra"
    CATEGORY_THEORY = "category_theory"
    GRAPH_THEORY = "graph_theory"
    OPTIMIZATION = "optimization"
    COMPUTER_SCIENCE = "computer_science"

    # -- lookup ----------------------------------------------------------
    @classmethod
    def from_string(cls, value: Any) -> "MathematicalDomain":
        """Lenient lookup: accepts values, member names, aliases and near-misses.

        Always returns a member (``GENERAL`` when nothing matches) so callers can
        route without exception handling.
        """
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.GENERAL
        text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if not text:
            return cls.GENERAL
        for member in cls:
            if text == member.value or text == member.name.lower():
                return member
        if text in _DOMAIN_ALIASES:
            return _DOMAIN_ALIASES[text]
        close = difflib.get_close_matches(text, [m.value for m in cls], n=1, cutoff=0.82)
        if close:
            return cls(close[0])
        return cls.GENERAL

    @classmethod
    def all_domains(cls) -> List["MathematicalDomain"]:
        """Every domain member."""
        return list(cls)

    @classmethod
    def names(cls) -> List[str]:
        """Every domain value as a string."""
        return [m.value for m in cls]

    @classmethod
    def detect(cls, text: str) -> "MathematicalDomain":
        """Classify free text into a domain via keyword scoring."""
        return cls.rank(text)[0][0] if text else cls.GENERAL

    @classmethod
    def rank(cls, text: str) -> List[Tuple["MathematicalDomain", float]]:
        """Score every domain against ``text``, best first.

        Scores are keyword-hit counts normalized by the domain's keyword count.
        ``GENERAL`` is the fallback when nothing matches.
        """
        tokens = set(re.findall(r"[a-z_]+", str(text).lower()))
        scored: List[Tuple[MathematicalDomain, float]] = []
        for member in cls:
            keywords = DOMAIN_KEYWORDS.get(member, ())
            if not keywords:
                continue
            hits = sum(1 for kw in keywords if kw in tokens)
            if hits:
                scored.append((member, hits / len(keywords)))
        scored.sort(key=lambda pair: (-pair[1], pair[0].value))
        if not scored:
            return [(cls.GENERAL, 0.0)]
        return scored

    # -- descriptive helpers ---------------------------------------------
    @property
    def label(self) -> str:
        """Human-readable title, e.g. ``"Number Theory"``."""
        return self.value.replace("_", " ").title()

    @property
    def keywords(self) -> Tuple[str, ...]:
        """Keywords associated with this domain."""
        return DOMAIN_KEYWORDS.get(self, ())

    def describe(self) -> str:
        """One-line description of the domain."""
        return DOMAIN_DESCRIPTIONS.get(self, "General mathematics.")

    def related(self) -> List["MathematicalDomain"]:
        """Neighbouring domains commonly used together with this one."""
        return list(DOMAIN_RELATIONS.get(self, ()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "label": self.label,
            "description": self.describe(),
            "keywords": list(self.keywords),
            "related": [d.value for d in self.related()],
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


_DOMAIN_ALIASES: Dict[str, MathematicalDomain] = {
    "arithmetic": MathematicalDomain.NUMBER_THEORY,
    "numbertheory": MathematicalDomain.NUMBER_THEORY,
    "calculus": MathematicalDomain.ANALYSIS,
    "real_analysis": MathematicalDomain.ANALYSIS,
    "complex_analysis": MathematicalDomain.ANALYSIS,
    "measure_theory": MathematicalDomain.ANALYSIS,
    "abstract_algebra": MathematicalDomain.ALGEBRA,
    "group_theory": MathematicalDomain.ALGEBRA,
    "ring_theory": MathematicalDomain.ALGEBRA,
    "linalg": MathematicalDomain.LINEAR_ALGEBRA,
    "matrices": MathematicalDomain.LINEAR_ALGEBRA,
    "matrix_theory": MathematicalDomain.LINEAR_ALGEBRA,
    "differential_geometry": MathematicalDomain.GEOMETRY,
    "algebraic_geometry": MathematicalDomain.GEOMETRY,
    "point_set_topology": MathematicalDomain.TOPOLOGY,
    "proof_theory": MathematicalDomain.LOGIC,
    "model_theory": MathematicalDomain.LOGIC,
    "type_theory": MathematicalDomain.LOGIC,
    "sets": MathematicalDomain.SET_THEORY,
    "graphs": MathematicalDomain.GRAPH_THEORY,
    "discrete_math": MathematicalDomain.COMBINATORICS,
    "counting": MathematicalDomain.COMBINATORICS,
    "stats": MathematicalDomain.STATISTICS,
    "cs": MathematicalDomain.COMPUTER_SCIENCE,
    "algorithms": MathematicalDomain.COMPUTER_SCIENCE,
    "complexity": MathematicalDomain.COMPUTER_SCIENCE,
    "operations_research": MathematicalDomain.OPTIMIZATION,
    "linear_programming": MathematicalDomain.OPTIMIZATION,
    "categories": MathematicalDomain.CATEGORY_THEORY,
}


DOMAIN_KEYWORDS: Dict[MathematicalDomain, Tuple[str, ...]] = {
    MathematicalDomain.ALGEBRA: ("group", "ring", "field", "ideal", "polynomial", "module", "homomorphism"),
    MathematicalDomain.ANALYSIS: ("limit", "continuous", "derivative", "integral", "converge", "series", "measure"),
    MathematicalDomain.GEOMETRY: ("triangle", "angle", "circle", "manifold", "curvature", "distance", "polygon"),
    MathematicalDomain.NUMBER_THEORY: ("prime", "divisor", "modulo", "congruence", "integer", "gcd", "diophantine"),
    MathematicalDomain.TOPOLOGY: ("open", "closed", "compact", "homeomorphism", "neighborhood", "connected", "homotopy"),
    MathematicalDomain.LOGIC: ("proposition", "predicate", "quantifier", "satisfiable", "tautology", "inference", "axiom"),
    MathematicalDomain.SET_THEORY: ("set", "subset", "union", "intersection", "cardinality", "ordinal", "bijection"),
    MathematicalDomain.COMBINATORICS: ("permutation", "combination", "binomial", "counting", "pigeonhole", "partition", "recurrence"),
    MathematicalDomain.PROBABILITY: ("probability", "random", "expectation", "variance", "distribution", "markov", "stochastic"),
    MathematicalDomain.STATISTICS: ("sample", "estimator", "regression", "hypothesis", "confidence", "variance", "correlation"),
    MathematicalDomain.LINEAR_ALGEBRA: ("matrix", "vector", "eigenvalue", "determinant", "basis", "rank", "linear"),
    MathematicalDomain.CATEGORY_THEORY: ("functor", "morphism", "category", "monad", "colimit", "adjoint", "natural"),
    MathematicalDomain.GRAPH_THEORY: ("graph", "vertex", "edge", "tree", "cycle", "clique", "coloring"),
    MathematicalDomain.OPTIMIZATION: ("minimize", "maximize", "objective", "constraint", "feasible", "gradient", "convex"),
    MathematicalDomain.COMPUTER_SCIENCE: ("algorithm", "complexity", "automaton", "program", "recursion", "datastructure", "turing"),
}


DOMAIN_DESCRIPTIONS: Dict[MathematicalDomain, str] = {
    MathematicalDomain.GENERAL: "General or cross-cutting mathematics.",
    MathematicalDomain.ALGEBRA: "Algebraic structures: groups, rings, fields and modules.",
    MathematicalDomain.ANALYSIS: "Limits, continuity, differentiation, integration and measure.",
    MathematicalDomain.GEOMETRY: "Shape, size, relative position and geometric structure.",
    MathematicalDomain.NUMBER_THEORY: "Properties of the integers, primes and congruences.",
    MathematicalDomain.TOPOLOGY: "Spaces, continuity and invariants under deformation.",
    MathematicalDomain.LOGIC: "Formal systems, proof theory and satisfiability.",
    MathematicalDomain.SET_THEORY: "Sets, cardinality, ordinals and foundations.",
    MathematicalDomain.COMBINATORICS: "Counting, arrangements and discrete structures.",
    MathematicalDomain.PROBABILITY: "Random phenomena, distributions and stochastic processes.",
    MathematicalDomain.STATISTICS: "Inference, estimation and hypothesis testing from data.",
    MathematicalDomain.LINEAR_ALGEBRA: "Vector spaces, matrices and linear transformations.",
    MathematicalDomain.CATEGORY_THEORY: "Categories, functors and universal constructions.",
    MathematicalDomain.GRAPH_THEORY: "Graphs, networks and their structural properties.",
    MathematicalDomain.OPTIMIZATION: "Objective functions subject to constraints.",
    MathematicalDomain.COMPUTER_SCIENCE: "Algorithms, complexity and computation.",
}


DOMAIN_RELATIONS: Dict[MathematicalDomain, Tuple[MathematicalDomain, ...]] = {
    MathematicalDomain.ALGEBRA: (MathematicalDomain.LINEAR_ALGEBRA, MathematicalDomain.NUMBER_THEORY, MathematicalDomain.CATEGORY_THEORY),
    MathematicalDomain.ANALYSIS: (MathematicalDomain.TOPOLOGY, MathematicalDomain.PROBABILITY),
    MathematicalDomain.GEOMETRY: (MathematicalDomain.TOPOLOGY, MathematicalDomain.LINEAR_ALGEBRA),
    MathematicalDomain.NUMBER_THEORY: (MathematicalDomain.ALGEBRA, MathematicalDomain.COMBINATORICS),
    MathematicalDomain.TOPOLOGY: (MathematicalDomain.ANALYSIS, MathematicalDomain.GEOMETRY),
    MathematicalDomain.LOGIC: (MathematicalDomain.SET_THEORY, MathematicalDomain.COMPUTER_SCIENCE),
    MathematicalDomain.SET_THEORY: (MathematicalDomain.LOGIC, MathematicalDomain.COMBINATORICS),
    MathematicalDomain.COMBINATORICS: (MathematicalDomain.GRAPH_THEORY, MathematicalDomain.PROBABILITY),
    MathematicalDomain.PROBABILITY: (MathematicalDomain.STATISTICS, MathematicalDomain.ANALYSIS),
    MathematicalDomain.STATISTICS: (MathematicalDomain.PROBABILITY, MathematicalDomain.OPTIMIZATION),
    MathematicalDomain.LINEAR_ALGEBRA: (MathematicalDomain.ALGEBRA, MathematicalDomain.OPTIMIZATION),
    MathematicalDomain.CATEGORY_THEORY: (MathematicalDomain.ALGEBRA, MathematicalDomain.LOGIC),
    MathematicalDomain.GRAPH_THEORY: (MathematicalDomain.COMBINATORICS, MathematicalDomain.COMPUTER_SCIENCE),
    MathematicalDomain.OPTIMIZATION: (MathematicalDomain.LINEAR_ALGEBRA, MathematicalDomain.COMPUTER_SCIENCE),
    MathematicalDomain.COMPUTER_SCIENCE: (MathematicalDomain.LOGIC, MathematicalDomain.GRAPH_THEORY),
}


# ---------------------------------------------------------------------------
# EvaluationMetric
# ---------------------------------------------------------------------------
def _ratio_scorer(value: Any = None, **kwargs: Any) -> float:
    """Score from ``passed``/``total``, a raw fraction, or text similarity."""
    passed = kwargs.get("passed")
    total = kwargs.get("total")
    if passed is not None and total:
        return _clamp(float(passed) / float(total))
    prediction = kwargs.get("prediction")
    reference = kwargs.get("reference")
    if prediction is not None and reference is not None:
        return _text_similarity(prediction, reference)
    return _coerce_score(value)


def _penalty_scorer(value: Any = None, **kwargs: Any) -> float:
    """Score that decreases with a reported ``violations``/``errors`` count."""
    for key in ("violations", "errors", "failures", "issues"):
        count = kwargs.get(key)
        if count is not None:
            return _clamp(1.0 / (1.0 + float(count)))
    return _coerce_score(value)


def _delta_scorer(value: Any = None, **kwargs: Any) -> float:
    """Score an improvement from ``before``/``after`` values."""
    before, after = kwargs.get("before"), kwargs.get("after")
    if before is not None and after is not None:
        before_f, after_f = float(before), float(after)
        if before_f == 0:
            return 1.0 if after_f > 0 else 0.0
        return _clamp((after_f - before_f) / abs(before_f))
    return _coerce_score(value)


def _clamp(score: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(score)))


def _coerce_score(value: Any) -> float:
    """Best-effort conversion of an arbitrary value into a 0..1 score."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        score = float(value)
        # Treat 0..100 inputs as percentages.
        if score > 1.0:
            score = score / 100.0 if score <= 100.0 else 1.0
        return _clamp(score)
    if isinstance(value, (list, tuple)):
        values = [_coerce_score(v) for v in value]
        return sum(values) / len(values) if values else 0.0
    if isinstance(value, dict):
        for key in ("score", "value", "ratio"):
            if key in value:
                return _coerce_score(value[key])
        return 0.0
    text = str(value).strip().lower()
    named = {
        "pass": 1.0, "passed": 1.0, "true": 1.0, "yes": 1.0, "excellent": 1.0,
        "good": 0.75, "fair": 0.5, "poor": 0.25,
        "fail": 0.0, "failed": 0.0, "false": 0.0, "no": 0.0,
    }
    if text in named:
        return named[text]
    try:
        return _clamp(float(text))
    except ValueError:
        return 0.0


def _text_similarity(prediction: Any, reference: Any) -> float:
    return difflib.SequenceMatcher(
        None, str(prediction).strip(), str(reference).strip()
    ).ratio()


@dataclass
class _Metric:
    """Descriptor for a single evaluation metric."""

    name: str
    description: str = ""
    higher_is_better: bool = True
    weight: float = 1.0
    scorer: Optional[Callable[..., float]] = field(default=None, repr=False, compare=False)
    domains: Tuple[MathematicalDomain, ...] = ()

    def compute(self, value: Any = None, **kwargs: Any) -> float:
        """Compute this metric's normalized 0..1 score.

        Accepts either a raw ``value`` or keyword evidence such as
        ``passed``/``total``, ``prediction``/``reference``, ``before``/``after``
        or ``violations``. Never raises: unusable input scores 0.0.
        """
        try:
            scorer = self.scorer or _ratio_scorer
            score = _clamp(scorer(value, **kwargs))
        except Exception as exc:  # noqa: BLE001 - metrics must not break callers
            logger.debug("Metric %s failed to compute: %s", self.name, exc)
            return 0.0
        return score if self.higher_is_better else _clamp(1.0 - score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "higher_is_better": self.higher_is_better,
            "weight": self.weight,
            "domains": [d.value for d in self.domains],
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


class EvaluationMetric:
    """Registry of evaluation metrics.

    The standard metrics are available as class attributes
    (``EvaluationMetric.CORRECTNESS``), by lookup
    (``EvaluationMetric.get("correctness")``) and in bulk
    (``EvaluationMetric.all()``). Each is a :class:`_Metric` with a ``compute``
    helper::

        EvaluationMetric.CORRECTNESS.compute(passed=8, total=10)   # -> 0.8
        EvaluationMetric.compute_all({"correctness": 0.9})          # -> {...}
    """

    _registry: ClassVar[Dict[str, _Metric]] = {}

    # Populated below via _register(); declared for readability/introspection.
    OVERALL_QUALITY: ClassVar[_Metric]
    CORRECTNESS: ClassVar[_Metric]
    CLARITY: ClassVar[_Metric]
    COMPLETENESS: ClassVar[_Metric]
    EFFECTIVENESS: ClassVar[_Metric]
    EFFICIENCY: ClassVar[_Metric]
    MAINTAINABILITY: ClassVar[_Metric]
    SCALABILITY: ClassVar[_Metric]
    ROBUSTNESS: ClassVar[_Metric]
    SECURITY: ClassVar[_Metric]
    COMPLIANCE: ClassVar[_Metric]
    AESTHETICS: ClassVar[_Metric]
    IMPROVEMENT_GAIN: ClassVar[_Metric]
    RIGOR: ClassVar[_Metric]
    NOVELTY: ClassVar[_Metric]

    def __new__(cls, *args: Any, **kwargs: Any) -> _Metric:  # type: ignore[misc]
        """``EvaluationMetric("foo")`` builds and registers a metric."""
        return cls.register(*args, **kwargs)

    # -- registry --------------------------------------------------------
    @classmethod
    def register(
        cls,
        name: str,
        description: str = "",
        higher_is_better: bool = True,
        weight: float = 1.0,
        scorer: Optional[Callable[..., float]] = None,
        domains: Iterable[MathematicalDomain] = (),
    ) -> _Metric:
        """Create/overwrite a metric and return the descriptor."""
        metric = _Metric(
            name=str(name).strip().lower(),
            description=description,
            higher_is_better=higher_is_better,
            weight=weight,
            scorer=scorer,
            domains=tuple(domains),
        )
        cls._registry[metric.name] = metric
        setattr(cls, metric.name.upper(), metric)
        return metric

    @classmethod
    def get(cls, name: Any, default: Optional[_Metric] = None) -> Optional[_Metric]:
        """Look a metric up by name (case/format insensitive)."""
        if isinstance(name, _Metric):
            return name
        key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        metric = cls._registry.get(key)
        if metric is not None:
            return metric
        close = difflib.get_close_matches(key, list(cls._registry), n=1, cutoff=0.85)
        return cls._registry[close[0]] if close else default

    @classmethod
    def all(cls) -> List[_Metric]:
        """Every registered metric, name-sorted."""
        return [cls._registry[k] for k in sorted(cls._registry)]

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._registry)

    @classmethod
    def for_domain(cls, domain: Any) -> List[_Metric]:
        """Metrics tagged for ``domain`` (plus untagged, universal metrics)."""
        target = MathematicalDomain.from_string(domain)
        return [m for m in cls.all() if not m.domains or target in m.domains]

    # -- computation -----------------------------------------------------
    @classmethod
    def compute(cls, name: Any, value: Any = None, **kwargs: Any) -> float:
        """Compute a single metric by name. Unknown metrics score 0.0."""
        metric = cls.get(name)
        if metric is None:
            logger.debug("Unknown evaluation metric: %s", name)
            return 0.0
        return metric.compute(value, **kwargs)

    @classmethod
    def compute_all(cls, values: Dict[Any, Any]) -> Dict[str, float]:
        """Compute every metric present in ``values``, keyed by metric name."""
        results: Dict[str, float] = {}
        for key, raw in (values or {}).items():
            metric = cls.get(key)
            if metric is None:
                continue
            if isinstance(raw, dict):
                results[metric.name] = metric.compute(**raw)
            else:
                results[metric.name] = metric.compute(raw)
        return results

    @classmethod
    def aggregate(cls, scores: Dict[Any, Any]) -> float:
        """Weighted mean of the computed metric scores in ``scores``."""
        computed = cls.compute_all(scores)
        if not computed:
            return 0.0
        total_weight = 0.0
        total = 0.0
        for name, score in computed.items():
            metric = cls.get(name)
            weight = metric.weight if metric else 1.0
            total += score * weight
            total_weight += weight
        return _clamp(total / total_weight) if total_weight else 0.0


def _bootstrap_metrics() -> None:
    """Register the standard metric set used across the engines."""
    standard: List[Tuple[str, str, float, Optional[Callable[..., float]]]] = [
        ("overall_quality", "Aggregate quality of the artifact.", 1.5, None),
        ("correctness", "Functional/logical correctness.", 2.0, _ratio_scorer),
        ("clarity", "Readability and comprehensibility.", 1.0, None),
        ("completeness", "Coverage of all required elements.", 1.25, _ratio_scorer),
        ("effectiveness", "Degree to which the goal is achieved.", 1.25, None),
        ("efficiency", "Resource and time economy.", 1.0, None),
        ("maintainability", "Ease of future modification.", 1.0, None),
        ("scalability", "Behaviour as load or size grows.", 1.0, None),
        ("robustness", "Tolerance of adverse or edge conditions.", 1.25, _penalty_scorer),
        ("security", "Resistance to misuse and attack.", 1.5, _penalty_scorer),
        ("compliance", "Adherence to specs, standards and policy.", 1.25, _penalty_scorer),
        ("aesthetics", "Presentation and structural elegance.", 0.5, None),
        ("improvement_gain", "Measured gain over the baseline.", 1.0, _delta_scorer),
        ("rigor", "Formal soundness of the reasoning.", 1.5, _ratio_scorer),
        ("novelty", "Originality relative to prior art.", 0.75, None),
    ]
    for name, description, weight, scorer in standard:
        EvaluationMetric.register(
            name=name, description=description, weight=weight, scorer=scorer
        )


_bootstrap_metrics()


__all__ = [
    "MathematicalDomain",
    "EvaluationMetric",
    "DOMAIN_KEYWORDS",
    "DOMAIN_DESCRIPTIONS",
    "DOMAIN_RELATIONS",
]
