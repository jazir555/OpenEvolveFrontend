"""
Public facade for the AI-powered ensemble strategy selector.

Exposes :class:`EnsembleStrategySelector` at
``openevolve.unified.EnsembleStrategySelector`` with the documented
``recommend_with_confidence`` / ``learn_from_run`` API.

When the upstream ``knowledge_engine.core.strategy_recommender`` implementation is
importable it is reused (so the full ensemble / online-learning logic is honored);
otherwise a self-contained rule-based ensemble is used so the symbol is always
available. Both paths return the documented :class:`StrategyRecommendation`
shape::

    from openevolve.unified import EnsembleStrategySelector

    selector = EnsembleStrategySelector()
    recommendation = await selector.recommend_with_confidence(
        problem="Optimize portfolio allocation",
        domain="finance",
        constraints={"max_evaluations": 50},
    )
    print(recommendation["mode"], recommendation["confidence"])

    await selector.learn_from_run(
        problem="Optimize portfolio allocation",
        domain="finance",
        strategy_used=recommendation["mode"],
        result=result,
    )
"""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

# Mode-specific expected-improvement messages (shown in StrategyRecommendation).
_EXPECTED_IMPROVEMENT = {
    "pes": "60% fewer evaluations via plan-execute-summarize",
    "qd": "Broad, diverse solution archive via MAP-Elites",
    "mo": "Pareto-optimal trade-offs across objectives",
    "adversarial": "Robustness via adversarial co-evolution",
    "standard": "Stable single-objective optimization",
    "hybrid": "Blended OpenEvolve + LoongFlow execution",
}

# Domains that benefit from diverse solution archives.
_DIVERSITY_DOMAINS = {"finance", "trading", "science", "engineering", "pharma"}
# Domains that need robustness / adversarial hardening.
_ROBUSTNESS_DOMAINS = {"engineering", "pharma", "finance"}
# Domains (or keywords) whose evaluations are expensive -> favour PES.
_EXPENSIVE_DOMAINS = {"science", "engineering", "pharma", "finance", "trading"}
_EXPENSIVE_KEYWORDS = (
    "backtest",
    "simulation",
    "experiment",
    "training",
    "monte carlo",
    "docking",
    "finite element",
)

# Best-effort reuse of the upstream ensemble implementation.
_REAL_SELECTOR = None
try:  # pragma: no cover - depends on optional install
    from knowledge_engine.core.strategy_recommender import (  # type: ignore
        EnsembleStrategySelector as _UpstreamSelector,
    )

    _REAL_SELECTOR = _UpstreamSelector
except Exception:  # pragma: no cover - best-effort reuse
    _REAL_SELECTOR = None


class EnsembleStrategySelector:
    """
    AI-powered strategy selector that recommends optimal evolutionary modes.

    Wraps the upstream ``knowledge_engine`` ensemble selector when available and
    otherwise falls back to a self-contained rule-based ensemble. The public
    surface (``recommend_with_confidence`` / ``learn_from_run``) is identical.
    """

    def __init__(
        self,
        knowledge_engine: Any = None,
        llm_client: Any = None,
        learning_enabled: bool = True,
        **kwargs: Any,
    ):
        self.knowledge_engine = knowledge_engine
        self.llm_client = llm_client
        self.learning_enabled = learning_enabled
        self._history: List[Dict[str, Any]] = []
        self._wrapped: Any = None

        if _REAL_SELECTOR is not None:
            try:
                self._wrapped = _REAL_SELECTOR(
                    knowledge_engine=knowledge_engine,
                    llm_client=llm_client,
                    learning_enabled=learning_enabled,
                    **kwargs,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Upstream EnsembleStrategySelector unavailable: {exc}")
                self._wrapped = None

    async def recommend_with_confidence(
        self,
        problem: str,
        domain: str,
        constraints: Optional[Dict[str, Any]] = None,
        objectives: Optional[List[str]] = None,
        evaluation_cost: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recommend an evolutionary mode with a confidence estimate.

        Returns a :class:`StrategyRecommendation`-shaped dict::

            {
                "mode": "pes",
                "confidence": 0.9,
                "reason": "Expensive evaluations, PES reduces cost by 60%",
                "expected_improvement": "60% fewer evaluations",
                "config": {...},
            }
        """
        constraints = constraints or {}
        objectives = objectives or list(constraints.get("objectives", []) or [])

        if self._wrapped is not None:
            try:
                return await self._recommend_via_wrapped(
                    problem, domain, constraints, objectives, evaluation_cost
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Upstream recommend failed, using built-in: {exc}")

        return self._recommend_builtin(
            problem, domain, constraints, objectives, evaluation_cost
        )

    async def _recommend_via_wrapped(
        self,
        problem: str,
        domain: str,
        constraints: Dict[str, Any],
        objectives: List[str],
        evaluation_cost: Optional[str],
    ) -> Dict[str, Any]:
        """Delegate to the upstream ensemble and normalize its result."""
        prediction = await self._wrapped.recommend_with_ensemble(
            problem_description=problem,
            domain=domain,
            constraints=constraints,
        )

        strategy = getattr(prediction, "strategy", ("openevolve", "standard"))
        if isinstance(strategy, (tuple, list)):
            mode = strategy[1] if len(strategy) > 1 else "standard"
        else:
            mode = str(strategy)

        disagreement = getattr(prediction, "disagreement_ratio", 0.0) or 0.0
        point_estimate = getattr(prediction, "point_estimate", 0.75) or 0.75
        reasoning = getattr(prediction, "reasoning", "") or ""

        confidence = max(0.0, min(1.0, 1.0 - float(disagreement)))
        if confidence <= 0.0:
            confidence = max(0.0, min(1.0, float(point_estimate)))

        config = {
            "evolution_mode": mode,
            "domain": domain,
            **constraints,
        }

        return {
            "mode": mode,
            "confidence": round(confidence, 3),
            "reason": reasoning,
            "expected_improvement": _EXPECTED_IMPROVEMENT.get(
                mode, "Improved evolutionary efficiency"
            ),
            "config": config,
        }

    def _recommend_builtin(
        self,
        problem: str,
        domain: str,
        constraints: Dict[str, Any],
        objectives: List[str],
        evaluation_cost: Optional[str],
    ) -> Dict[str, Any]:
        """Self-contained rule-based ensemble recommendation."""
        problem_l = (problem or "").lower()

        if len(objectives) > 1:
            mode, confidence, reason = (
                "mo",
                0.90,
                "Multiple objectives require Pareto optimization",
            )
        elif evaluation_cost in ("expensive", "very_expensive") or domain in _EXPENSIVE_DOMAINS or any(
            kw in problem_l for kw in _EXPENSIVE_KEYWORDS
        ):
            mode, confidence, reason = (
                "pes",
                0.85,
                "Expensive evaluations: PES reduces evaluation cost",
            )
        elif domain in _DIVERSITY_DOMAINS:
            mode, confidence, reason = (
                "qd",
                0.80,
                "Diverse solutions required: use MAP-Elites (QD)",
            )
        elif domain in _ROBUSTNESS_DOMAINS or constraints.get("safety_critical"):
            mode, confidence, reason = (
                "adversarial",
                0.85,
                "Safety-critical: adversarial co-evolution",
            )
        else:
            mode, confidence, reason = (
                "standard",
                0.75,
                f"Default to standard mode for {domain}",
            )

        return {
            "mode": mode,
            "confidence": confidence,
            "reason": reason,
            "expected_improvement": _EXPECTED_IMPROVEMENT.get(
                mode, "Improved evolutionary efficiency"
            ),
            "config": {
                "evolution_mode": mode,
                "domain": domain,
                **constraints,
            },
        }

    async def learn_from_run(
        self,
        problem: str,
        domain: str,
        strategy_used: str,
        result: Any,
    ) -> None:
        """
        Record the outcome of a run so future recommendations can adapt.

        Args:
            problem: Problem statement that was solved.
            domain: Problem domain.
            strategy_used: Mode that was actually used.
            result: The :class:`EvolutionResult`-shaped dict for the run.
        """
        record = {
            "problem": problem,
            "domain": domain,
            "strategy_used": strategy_used,
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._history.append(record)

        if self._wrapped is not None:
            try:
                await self._wrapped.learn_from_run(
                    {
                        "run_id": (result or {}).get(
                            "run_id", f"run_{len(self._history)}"
                        ),
                        "domain": domain,
                        "strategy_used": strategy_used,
                        "final_score": (result or {}).get("fitness", 0.0),
                        "iterations": (result or {}).get("iterations", 0),
                        "evaluations": (result or {}).get("evaluations", 0),
                        "complexity": "medium",
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Upstream learn_from_run failed: {exc}")
