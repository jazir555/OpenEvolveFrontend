"""
Gold Team Evaluator (Gauntlet Round 3)
======================================

Consensus verification of a candidate solution. The evaluator mirrors the logic
used by
:class:`openevolve.gauntlets.three_round_orchestrator.ThreeRoundGauntletOrchestrator`
for Round 3 so it can be slotted into the orchestrator:

1. Static verification (:func:`openevolve.gauntlets.llm_judge.verify_solution`)
   stands in for a formal (Lean 4) proof, which is unavailable offline.
2. An LLM judge panel (:class:`openevolve.gauntlets.llm_judge.GauntletJudge`)
   votes on certification. With no API key configured it runs on the offline
   deterministic mock backend, which returns a deterministic verdict.

Public API
----------
    from openevolve.gauntlets.gold_team import GoldTeamEvaluator

    evaluator = GoldTeamEvaluator()
    result = await evaluator.verify(solution, problem, domain, judges=["gpt-4", "claude"])
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import asyncio
import logging
import time

from .llm_judge import (
    GauntletJudge,
    consensus_score,
    verify_solution,
)

logger = logging.getLogger(__name__)

# Weight given to static verification in the final consensus score.
VERIFICATION_WEIGHT = 0.2


@dataclass
class GoldTeamResult:
    """Outcome of the Gold Team consensus round."""

    passed: bool
    score: float
    consensus_score: float
    formal_verification_passed: bool
    judge_score: Optional[float]
    evaluator_count: int
    feedback: str
    evaluator_votes: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Any] = field(default_factory=list)
    evaluator_type: str = "gold_team"
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a dictionary."""
        return {
            "round": "round3_gold_team",
            "passed": self.passed,
            "score": self.score,
            "consensus_score": self.consensus_score,
            "formal_verification_passed": self.formal_verification_passed,
            "judge_score": self.judge_score,
            "evaluator_count": len(self.evaluator_votes),
            "feedback": self.feedback,
            "evaluator_type": self.evaluator_type,
            "timestamp": self.timestamp,
        }


class GoldTeamEvaluator:
    """
    Consensus (Gold Team) evaluator for Gauntlet Round 3.

    Args:
        llm_config: Optional judge model configuration. When omitted (or when no
            API key is available) the offline deterministic mock judge is used.
        threshold: Minimum blended score required to pass Round 3.
        language: Language of the candidate solution.
    """

    def __init__(
        self,
        llm_config: Any = None,
        threshold: float = 0.7,
        language: str = "python",
    ):
        self.llm_config = llm_config
        self.threshold = threshold
        self.language = language
        self.judge: Optional[GauntletJudge] = None
        try:
            self.judge = GauntletJudge(llm_config=llm_config)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Gold Team judge initialization failed: {exc}")
            self.judge = None

    async def verify(
        self,
        solution: str,
        problem: str,
        domain: str,
        judges: Optional[Sequence[Any]] = None,
    ) -> GoldTeamResult:
        """
        Verify ``solution`` via static checks plus a judge consensus vote.

        Args:
            solution: Candidate solution source.
            problem: Problem statement.
            domain: Problem domain.
            judges: Optional explicit judge panel. May be a list of model names
                (strings) or model configuration dicts. When provided a fresh
                judge ensemble is built from them; the offline mock is used when no
                credentials are available.

        Returns:
            GoldTeamResult with the consensus score and vote breakdown.
        """
        start_time = time.time()
        logger.info("Gold Team consensus verification starting")

        # 1. Static verification stands in for formal (Lean 4) verification
        verification = verify_solution(solution, language=self.language)
        formal_verification_passed = bool(verification.get("passed"))

        # 2. Collect one certification vote per judge model
        judge = self._resolve_judge(judges)
        votes = []
        if judge is not None:
            try:
                votes = await judge.gold_team(
                    solution=solution,
                    problem=problem,
                    domain=domain,
                    language=self.language,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(f"Gold Team judge failed: {exc}")
                votes = []

        usable_votes = [vote for vote in votes if vote.parsed]

        if usable_votes:
            judge_score = sum(v.score for v in usable_votes) / len(usable_votes)
            agreement = consensus_score(usable_votes)
        else:
            # No judge available: fall back to the verification checks alone
            checks = verification.get("checks", {})
            judge_score = (
                sum(1.0 for passed in checks.values() if passed) / len(checks)
                if checks
                else 0.0
            )
            agreement = judge_score

        consensus = agreement
        score = (1.0 - VERIFICATION_WEIGHT) * judge_score + VERIFICATION_WEIGHT * (
            1.0 if formal_verification_passed else 0.0
        )

        passed = bool(score >= self.threshold)

        feedback = (
            f"Gold Team consensus: {consensus:.2f} across {len(usable_votes)} judge(s). "
            f"Judge score: {judge_score:.2f}. "
            f"Static verification: {'PASSED' if formal_verification_passed else 'FAILED'} "
            f"({verification.get('detail')}). "
            f"Final score: {score:.2f}"
        )

        return GoldTeamResult(
            passed=passed,
            score=score,
            consensus_score=consensus,
            formal_verification_passed=formal_verification_passed,
            judge_score=judge_score if usable_votes else None,
            evaluator_count=len(usable_votes),
            feedback=feedback,
            evaluator_votes=[vote.to_dict() for vote in votes],
            artifacts=[verification],
        )

    def _resolve_judge(
        self, judges: Optional[Sequence[Any]]
    ) -> Optional[GauntletJudge]:
        """Pick the judge to use, optionally building one from ``judges``."""
        if not judges:
            return self.judge
        try:
            if isinstance(judges, (list, tuple)):
                models = [
                    {"name": j} if isinstance(j, str) else j for j in judges
                ]
                return GauntletJudge(llm_config={"models": models})
            return GauntletJudge(llm_config=judges)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to build explicit judge panel: {exc}")
            return self.judge

    # Aliases so the evaluator can be dropped into the orchestrator in place of a
    # GauntletJudge (which exposes ``gold_team`` / ``evaluate``).
    async def gold_team(self, solution: str, problem: str, domain: str, language: str = "python", prior_findings=None) -> GoldTeamResult:
        return await self.verify(solution, problem, domain)

    async def evaluate(self, solution: str, problem: str, domain: str, **kwargs: Any) -> GoldTeamResult:
        return await self.verify(solution, problem, domain, **kwargs)
