"""
Knowledge Extraction & Learning (Sovereign-Grade Decomposition Workflow: Stage 7 / §3.7)
and an analytics/metrics aggregator for the Analytics Dashboard (§6.2).

This module owns:
  1. ``extract_workflow_knowledge`` - turn a completed/partial workflow run into
     ``KnowledgeArtifact`` objects (successful decomposition patterns, effective
     gauntlet configs, failure modes from red/gold critiques, reusable sub-problem
     solutions).
  2. ``WorkflowLearningStore`` - a lightweight, file-backed learning store that records
     which (team, gauntlet, evolution_mode) combos succeeded and recommends a
     ``best_strategy_for(problem_type)`` to feed future runs.
  3. Analytics aggregation: ``aggregate_workflow_metrics`` + ``collect_step_metrics``
     producing serializable ``PerformanceMetrics`` for the dashboard.

All integration with ``knowledge_engine`` is import-guarded and NEVER raises at import
time. If the knowledge engine is unavailable (or fails at runtime), artifacts are
persisted to a local JSONL file as a fallback.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Guarded data-model imports. We reuse the canonical dataclasses that live in
# ``openevolve.kernel.schema`` and are re-exported by ``workflow_structures``.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised at import time
    from workflow_structures import (
        CritiqueReport,
        DecompositionPlan,
        KnowledgeArtifact,
        PerformanceMetrics,
        SolutionAttempt,
        SubProblem,
        VerificationReport,
        WorkflowState,
    )
except Exception:  # pragma: no cover - extreme fallback only
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional

    @dataclass
    class KnowledgeArtifact:  # type: ignore
        artifact_id: str = ""
        artifact_type: str = ""
        source_workflow_id: str = ""
        source_stage: int = 6
        timestamp: Any = None
        confidence: float = 0.0
        title: str = ""
        description: str = ""
        content: Dict[str, Any] = field(default_factory=dict)
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class PerformanceMetrics:  # type: ignore
        workflow_id: str = ""
        execution_time: float = 0.0
        resource_usage: Dict[str, Any] = field(default_factory=dict)
        success_rate: float = 0.0
        error_count: int = 0
        quality_score: float = 0.0
        accuracy: float = 0.0
        efficiency: float = 0.0
        reliability: float = 0.0
        throughput: float = 0.0
        latency: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class SubProblem:  # type: ignore
        id: str = ""

    @dataclass
    class SolutionAttempt:  # type: ignore
        id: str = ""

    @dataclass
    class CritiqueReport:  # type: ignore
        pass

    @dataclass
    class VerificationReport:  # type: ignore
        pass

    @dataclass
    class DecompositionPlan:  # type: ignore
        pass

    @dataclass
    class WorkflowState:  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Guarded knowledge_engine integration. Must NEVER raise at import time.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import-time guard
    from knowledge_engine import get_knowledge_engine  # type: ignore  # noqa: F401

    _KE_MODULE_PRESENT = True
except Exception:  # pragma: no cover
    _KE_MODULE_PRESENT = False


# Default JSONL fallback location (next to this module).
_DEFAULT_JSONL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".workflow_knowledge_store.jsonl"
)


# ===========================================================================
# Storage backends
# ===========================================================================
def _artifact_to_record(artifact: Any) -> Dict[str, Any]:
    """Serialize an artifact to a JSON-friendly record."""
    if hasattr(artifact, "to_dict"):
        try:
            return artifact.to_dict()
        except Exception:
            pass
    if isinstance(artifact, dict):
        return artifact
    return {
        "artifact_id": getattr(artifact, "artifact_id", None),
        "artifact_type": getattr(artifact, "artifact_type", None),
        "source_workflow_id": getattr(artifact, "source_workflow_id", None),
        "title": getattr(artifact, "title", None),
        "content": getattr(artifact, "content", None),
    }


def _write_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str, ensure_ascii=False))
        fh.write("\n")


class JsonlKnowledgeStore:
    """File-backed (JSONL) knowledge store used as the offline fallback."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or _DEFAULT_JSONL_PATH

    def store(self, artifact: Any) -> None:
        _write_jsonl(self.path, _artifact_to_record(artifact))

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out


class KnowledgeEngineBackedStore:
    """
    Attempts to persist artifacts via ``knowledge_engine``. Any failure (engine
    unavailable at runtime, async errors, timeouts) silently falls back to the
    JSONL file so the pipeline is never blocked.
    """

    def __init__(self, fallback_path: Optional[str] = None):
        self._fallback_path = fallback_path or _DEFAULT_JSONL_PATH
        self._engine: Any = None
        self._ready: bool = False

    def _get_engine(self) -> Any:
        if self._ready:
            return self._engine
        self._ready = True
        try:
            from knowledge_engine import get_knowledge_engine  # type: ignore

            import concurrent.futures as _cf

            def _build() -> Any:
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(get_knowledge_engine())
                finally:
                    loop.close()

            # Bound the engine construction in a thread so a slow/heavy init
            # (or unavailable runtime) can never hang the offline pipeline.
            with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_build)
                self._engine = fut.result(timeout=5)
        except Exception:
            self._engine = None
        return self._engine

    def store(self, artifact: Any) -> bool:
        engine = self._get_engine()
        if engine is None:
            _write_jsonl(self._fallback_path, _artifact_to_record(artifact))
            return False
        try:
            data = (
                artifact.to_dict()
                if hasattr(artifact, "to_dict")
                else artifact
            )
            result = engine.store_artifact(data)
            if hasattr(result, "__await__"):
                asyncio.wait_for(result, timeout=5)
            return True
        except Exception:
            _write_jsonl(self._fallback_path, _artifact_to_record(artifact))
            return False


def _build_knowledge_engine_store(fallback_path: Optional[str]) -> Optional[Any]:
    """Return a knowledge-engine-backed store, or None if it cannot be used."""
    if not _KE_MODULE_PRESENT:
        return None
    try:
        from knowledge_engine import get_knowledge_engine  # type: ignore  # noqa: F401

        return KnowledgeEngineBackedStore(fallback_path)
    except Exception:
        return None


def _resolve_store(
    store: Optional[Any], fallback_path: Optional[str]
) -> Any:
    if store is not None:
        return store
    ke = _build_knowledge_engine_store(fallback_path)
    if ke is not None:
        return ke
    return JsonlKnowledgeStore(fallback_path or _DEFAULT_JSONL_PATH)


# ===========================================================================
# Artifact construction helpers
# ===========================================================================
def _make_artifact(
    artifact_type: str,
    source_workflow_id: str,
    title: str,
    description: str,
    content: Dict[str, Any],
    *,
    confidence: float = 0.5,
    problem_type: Optional[str] = None,
    domain: Optional[str] = None,
    tags: Optional[List[str]] = None,
    effectiveness_score: Optional[float] = None,
) -> KnowledgeArtifact:
    return KnowledgeArtifact(
        artifact_id="ka_" + uuid_hex(),
        artifact_type=artifact_type,
        source_workflow_id=source_workflow_id or "unknown",
        source_stage=6,
        timestamp=datetime.now(),
        confidence=float(confidence),
        title=title,
        description=description,
        content=content,
        metadata={
            "problem_type": problem_type,
            "domain": domain,
        },
        tags=tags or [],
        effectiveness_score=effectiveness_score,
    )


def uuid_hex() -> str:
    import uuid

    return uuid.uuid4().hex[:12]


# ===========================================================================
# 1. Knowledge extraction (Stage 7 / §3.7)
# ===========================================================================
def extract_workflow_knowledge(
    workflow_state: Any,
    decomposition_plan: Any,
    solution_attempts: List[Any],
    reports: List[Any],
    *,
    fallback_path: Optional[str] = None,
    store: Optional[Any] = None,
) -> List[KnowledgeArtifact]:
    """
    Turn a completed/partial workflow run into a list of ``KnowledgeArtifact``
    objects (doc §3.7 Knowledge Extraction & Learning).

    Args:
        workflow_state: A ``WorkflowState`` (may be ``None``).
        decomposition_plan: A ``DecompositionPlan`` (may be ``None``).
        solution_attempts: A list of ``SolutionAttempt`` objects.
        reports: A list of reports (``CritiqueReport`` / ``VerificationReport``,
            or anything with a recognizable shape). May be ``None``.
        fallback_path: Optional JSONL path used when the knowledge engine is
            unavailable.
        store: Optional explicit store backend. When omitted, the knowledge
            engine is used when available and JSONL otherwise.

    Returns:
        A (possibly empty) list of ``KnowledgeArtifact``.
    """
    artifacts: List[KnowledgeArtifact] = []

    solution_attempts = list(solution_attempts or [])
    reports = list(reports or [])

    source_workflow_id = (
        getattr(workflow_state, "workflow_id", None) if workflow_state else None
    ) or (getattr(decomposition_plan, "problem_id", None) if decomposition_plan else None) or "unknown"

    problem_type = _infer_problem_type(workflow_state, decomposition_plan)
    domain = _infer_domain(workflow_state, decomposition_plan)

    # Merge reports from the explicit arg and from the workflow state.
    all_reports: List[Any] = list(reports)
    if workflow_state is not None:
        all_reports.extend(getattr(workflow_state, "all_critique_reports", []) or [])
        all_reports.extend(
            getattr(workflow_state, "all_verification_reports", []) or []
        )

    # --- Decomposition strategy / plan pattern -----------------------------
    if decomposition_plan is not None:
        sub_problems = getattr(decomposition_plan, "sub_problems", []) or []
        strategy = getattr(decomposition_plan, "strategy", None)
        strategy_name = getattr(strategy, "value", None) or str(strategy) if strategy else "unknown"
        artifacts.append(
            _make_artifact(
                "decomposition_strategy",
                source_workflow_id,
                "Decomposition strategy pattern",
                "Effective decomposition strategy and plan structure for this run.",
                {
                    "strategy": strategy_name,
                    "sub_problem_count": len(sub_problems),
                    "problem_statement": getattr(
                        decomposition_plan, "problem_statement", ""
                    ),
                    "teams": {
                        "content_analyzer": getattr(
                            decomposition_plan, "content_analyzer_team_name", ""
                        ),
                        "planner": getattr(decomposition_plan, "planner_team_name", ""),
                        "assembler": getattr(
                            decomposition_plan, "assembler_team_name", ""
                        ),
                    },
                    "gauntlets": {
                        "final_red": getattr(
                            decomposition_plan, "final_red_team_gauntlet_name", None
                        ),
                        "final_gold": getattr(
                            decomposition_plan, "final_gold_team_gauntlet_name", ""
                        ),
                    },
                },
                confidence=0.6,
                problem_type=problem_type,
                domain=domain,
                tags=["decomposition", strategy_name],
            )
        )

    # --- Solution patterns from successful solution attempts ---------------
    solved: List[Any] = []
    for attempt in solution_attempts:
        status = getattr(attempt, "status", None)
        status_val = getattr(status, "value", None) if status else None
        is_solved = status_val in ("verified", "solved") or status in (
            "verified",
            "solved",
        )
        if is_solved:
            solved.append(attempt)

    if solved:
        for attempt in solved:
            approach = (
                getattr(attempt, "solution_approach", None)
                or getattr(attempt, "approach", None)
                or "unknown"
            )
            quality = _avg_quality(getattr(attempt, "quality_metrics", None))
            artifacts.append(
                _make_artifact(
                    "solution_pattern",
                    source_workflow_id,
                    "Reusable solution pattern",
                    "A successful solution approach worth reusing for similar sub-problems.",
                    {
                        "sub_problem_id": getattr(attempt, "sub_problem_id", None),
                        "approach": approach,
                        "quality_score": quality,
                        "team_id": getattr(attempt, "team_id", None),
                        "content_excerpt": (getattr(attempt, "content", "") or "")[
                            :500
                        ],
                    },
                    confidence=max(0.4, min(1.0, quality or 0.5)),
                    problem_type=problem_type,
                    domain=domain,
                    tags=["solution", str(approach)],
                    effectiveness_score=quality,
                )
            )
    elif solution_attempts:
        # Even without explicit "solved" status, record the best attempt.
        best = max(
            solution_attempts,
            key=lambda a: _avg_quality(getattr(a, "quality_metrics", None)),
        )
        artifacts.append(
            _make_artifact(
                "solution_pattern",
                source_workflow_id,
                "Candidate solution pattern",
                "Best-effort solution pattern extracted from available attempts.",
                {
                    "sub_problem_id": getattr(best, "sub_problem_id", None),
                    "approach": getattr(best, "solution_approach", None)
                    or getattr(best, "approach", None)
                    or "unknown",
                    "quality_score": _avg_quality(
                        getattr(best, "quality_metrics", None)
                    ),
                },
                confidence=0.4,
                problem_type=problem_type,
                domain=domain,
                tags=["solution", "candidate"],
            )
        )

    # --- Critique insights / failure modes from red/gold critiques ----------
    critique_insights = 0
    for report in all_reports:
        if isinstance(report, CritiqueReport) or _is_critique(report):
            approved = getattr(report, "is_approved", None)
            if approved is None:
                approved = getattr(report, "passed", None)
            if not approved:
                flaws = getattr(report, "identified_flaws", None) or getattr(
                    report, "flaws", []
                )
                artifacts.append(
                    _make_artifact(
                        "critique_insight",
                        source_workflow_id,
                        "Failure mode / critique insight",
                        "Recurring weakness surfaced by red-team critique.",
                        {
                            "gauntlet_name": getattr(report, "gauntlet_name", ""),
                            "solution_attempt_id": getattr(
                                report, "solution_attempt_id", None
                            )
                            or getattr(report, "solution_id", None),
                            "flaws": [
                                f
                                if isinstance(f, (str, int, float, bool))
                                else (getattr(f, "get", lambda k: None)("type") or str(f))
                                for f in (flaws or [])
                            ],
                            "severity_scores": getattr(
                                report, "flaw_severity_scores", {}
                            )
                            or {},
                            "overall_score": getattr(report, "overall_score", 0.0),
                        },
                        confidence=0.55,
                        problem_type=problem_type,
                        domain=domain,
                        tags=["critique", "failure_mode"],
                    )
                )
                critique_insights += 1

    # --- Gauntlet effectiveness from verification (gold) reports -----------
    gauntlet_effect: Dict[str, Dict[str, Any]] = {}
    for report in all_reports:
        if isinstance(report, VerificationReport) or _is_verification(report):
            name = getattr(report, "gauntlet_name", "") or "unknown"
            approved = getattr(report, "is_approved", None)
            if approved is None:
                approved = getattr(report, "verified", None)
            score = getattr(report, "average_score", None)
            if score is None:
                score = getattr(report, "confidence", 0.0)
            entry = gauntlet_effect.setdefault(
                name,
                {"gauntlet_name": name, "passed": 0, "failed": 0, "scores": []},
            )
            if approved:
                entry["passed"] += 1
            else:
                entry["failed"] += 1
            entry["scores"].append(float(score or 0.0))

    for name, entry in gauntlet_effect.items():
        scores = entry["scores"] or [0.0]
        avg = sum(scores) / len(scores)
        artifacts.append(
            _make_artifact(
                "gauntlet_effectiveness",
                source_workflow_id,
                "Gauntlet effectiveness",
                "Effectiveness of a verification/gauntlet configuration on this run.",
                {
                    **entry,
                    "avg_score": avg,
                    "total": entry["passed"] + entry["failed"],
                },
                confidence=0.6,
                problem_type=problem_type,
                domain=domain,
                tags=["gauntlet", name],
                effectiveness_score=avg,
            )
        )

    # --- Team performance (if gauntlet/team metadata is present) -----------
    team_perf = _collect_team_performance(workflow_state, decomposition_plan)
    if team_perf:
        artifacts.append(
            _make_artifact(
                "team_performance",
                source_workflow_id,
                "Team performance summary",
                "Aggregated team assignment performance for this run.",
                team_perf,
                confidence=0.5,
                problem_type=problem_type,
                domain=domain,
                tags=["team"],
            )
        )

    # --- Persist artifacts -------------------------------------------------
    backend = _resolve_store(store, fallback_path)
    for artifact in artifacts:
        try:
            backend.store(artifact)
        except Exception:
            # Last-resort direct JSONL write so we never lose knowledge.
            try:
                _write_jsonl(
                    fallback_path or _DEFAULT_JSONL_PATH,
                    _artifact_to_record(artifact),
                )
            except Exception:
                pass

    # Attach to the workflow state if possible.
    if workflow_state is not None and hasattr(workflow_state, "knowledge_artifacts"):
        try:
            workflow_state.knowledge_artifacts.extend(artifacts)
        except Exception:
            pass

    return artifacts


def _is_critique(report: Any) -> bool:
    if report is None:
        return False
    return "critique" in type(report).__name__.lower()


def _is_verification(report: Any) -> bool:
    if report is None:
        return False
    name = type(report).__name__.lower()
    return "verification" in name or "gold" in name


def _avg_quality(metrics: Any) -> float:
    if not metrics:
        return 0.0
    if isinstance(metrics, dict):
        vals = [v for v in metrics.values() if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else 0.0
    return float(metrics) if isinstance(metrics, (int, float)) else 0.0


def _infer_problem_type(workflow_state: Any, decomposition_plan: Any) -> Optional[str]:
    ps = getattr(workflow_state, "problem_statement", None) if workflow_state else None
    if not ps and decomposition_plan:
        ps = getattr(decomposition_plan, "problem_statement", None)
    return "general" if ps else None


def _infer_domain(workflow_state: Any, decomposition_plan: Any) -> Optional[str]:
    if decomposition_plan:
        ctx = getattr(decomposition_plan, "analyzed_context", None)
        if isinstance(ctx, dict):
            return ctx.get("domain")
    return None


def _collect_team_performance(
    workflow_state: Any, decomposition_plan: Any
) -> Dict[str, Any]:
    perf: Dict[str, Any] = {}
    if decomposition_plan:
        for key in (
            "content_analyzer_team_name",
            "planner_team_name",
            "assembler_team_name",
            "final_gold_team_gauntlet_name",
        ):
            val = getattr(decomposition_plan, key, None)
            if val:
                perf[key] = val
    return perf


# ===========================================================================
# 2. Lightweight learning store
# ===========================================================================
@dataclass
class _OutcomeRecord:
    team: str
    gauntlet: str
    evolution_mode: str
    problem_type: str
    success: bool
    quality_score: float
    timestamp: float = field(default_factory=time.time)


class WorkflowLearningStore:
    """
    Records which (team, gauntlet, evolution_mode) combos succeeded for a given
    problem type and recommends the best strategy for future runs.

    Pure, file-backed, no external dependencies.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._outcomes: List[_OutcomeRecord] = []
        if self.path:
            self._load()

    # ---- persistence ------------------------------------------------------
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for item in raw.get("outcomes", []):
                self._outcomes.append(_OutcomeRecord(**item))
        except Exception:
            self._outcomes = []

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "outcomes": [o.__dict__ for o in self._outcomes],
                    },
                    fh,
                    default=str,
                    indent=2,
                )
        except Exception:
            pass

    # ---- API --------------------------------------------------------------
    def record_outcome(
        self,
        team: str,
        gauntlet: str,
        evolution_mode: str,
        problem_type: str,
        success: bool,
        quality_score: float = 0.0,
    ) -> None:
        """Record a single outcome for a (team, gauntlet, evolution_mode) combo."""
        self._outcomes.append(
            _OutcomeRecord(
                team=team,
                gauntlet=gauntlet,
                evolution_mode=evolution_mode,
                problem_type=problem_type,
                success=bool(success),
                quality_score=float(quality_score),
            )
        )
        self._save()

    def best_strategy_for(
        self, problem_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Return the (team, gauntlet, evolution_mode) combo with the best observed
        score for ``problem_type``, or ``None`` if no data exists.

        Score blends success rate (0-1) with average quality, weighted 70/30.
        """
        relevant = [o for o in self._outcomes if o.problem_type == problem_type]
        if not relevant:
            return None

        groups: Dict[Tuple[str, str, str], List[_OutcomeRecord]] = {}
        for o in relevant:
            groups.setdefault((o.team, o.gauntlet, o.evolution_mode), []).append(o)

        best: Optional[Dict[str, Any]] = None
        best_score = -1.0
        for (team, gauntlet, mode), recs in groups.items():
            n = len(recs)
            success_rate = sum(1 for r in recs if r.success) / n
            avg_quality = sum(r.quality_score for r in recs) / n
            blended = 0.7 * success_rate + 0.3 * avg_quality
            if blended > best_score:
                best_score = blended
                best = {
                    "team": team,
                    "gauntlet": gauntlet,
                    "evolution_mode": mode,
                    "success_rate": success_rate,
                    "avg_quality": avg_quality,
                    "score": blended,
                    "samples": n,
                }
        return best

    def stats(self) -> Dict[str, Any]:
        return {
            "total_outcomes": len(self._outcomes),
            "problem_types": sorted(
                {o.problem_type for o in self._outcomes}
            ),
        }


# ===========================================================================
# 3. Analytics aggregator (for the Analytics Dashboard, §6.2)
# ===========================================================================
def collect_step_metrics(run_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregates step-level metrics across a workflow run. ``run_results`` is a
    plain, serializable dict (no objects required). Returns a serializable dict.
    """
    total = run_results.get("total_sub_problems")
    if total is None:
        attempts = run_results.get("solution_attempts") or []
        total = len(attempts)
    solved = run_results.get("solved_count", 0)
    failed = run_results.get("failed_count", 0)
    quality_scores = run_results.get("quality_scores") or []
    avg_quality = (
        sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    )

    return {
        "sub_problems_total": total,
        "sub_problems_solved": solved,
        "sub_problems_failed": failed,
        "critiques_total": run_results.get("critiques_total", 0),
        "verifications_total": run_results.get("verifications_total", 0),
        "refinement_loops": run_results.get("refinement_loops", 0),
        "avg_quality": avg_quality,
        "error_count": run_results.get("error_count", 0),
        "stage_durations": run_results.get("step_durations", {}) or {},
    }


def aggregate_workflow_metrics(run_results: Dict[str, Any]) -> PerformanceMetrics:
    """
    Produce a serializable ``PerformanceMetrics`` for the Analytics Dashboard.

    ``run_results`` is a plain dict with optional keys:
        workflow_id, execution_time, total_sub_problems, solved_count,
        failed_count, error_count, quality_scores (list[float]),
        resource_usage (dict), step_durations (dict), solution_attempts (list),
        critiques_total, verifications_total, refinement_loops.
    """
    steps = collect_step_metrics(run_results)

    total = steps["sub_problems_total"] or 0
    solved = steps["sub_problems_solved"]
    exec_time = float(run_results.get("execution_time", 0.0) or 0.0)
    error_count = int(run_results.get("error_count", 0) or 0)

    success_rate = (solved / total) if total > 0 else 0.0
    throughput = (solved / exec_time) if exec_time > 0 else 0.0
    latency = (exec_time / total) if total > 0 else 0.0
    efficiency = throughput

    metrics = PerformanceMetrics(
        workflow_id=str(run_results.get("workflow_id", "") or ""),
        execution_time=exec_time,
        resource_usage=dict(run_results.get("resource_usage", {}) or {}),
        success_rate=success_rate,
        error_count=error_count,
        throughput=throughput,
        latency=latency,
        accuracy=success_rate,
        efficiency=efficiency,
        quality_score=steps["avg_quality"],
        reliability=success_rate,
        scalability=0.0,
        metadata={
            "step_metrics": steps,
            "solved": solved,
            "failed": steps["sub_problems_failed"],
            "refinement_loops": steps["refinement_loops"],
        },
    )
    return metrics


__all__ = [
    "extract_workflow_knowledge",
    "WorkflowLearningStore",
    "aggregate_workflow_metrics",
    "collect_step_metrics",
    "JsonlKnowledgeStore",
    "KnowledgeEngineBackedStore",
    "_DEFAULT_JSONL_PATH",
]
