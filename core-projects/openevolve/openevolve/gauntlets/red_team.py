"""
Red Team Evaluator (Gauntlet Round 2)
=====================================

Adversarial evaluation of a candidate solution. The evaluator combines two real
signals, mirroring the logic used by
:class:`openevolve.gauntlets.three_round_orchestrator.ThreeRoundGauntletOrchestrator`
for Round 2 so it can be slotted into the orchestrator:

1. Deterministic adversarial probes (:func:`openevolve.gauntlets.llm_judge.probe_solution`)
   that attack the candidate locally and report which attacks succeeded.
2. An LLM judge (:class:`openevolve.gauntlets.llm_judge.GauntletJudge`) prompted to
   adversarially critique the candidate. With no API key configured it runs on the
   offline deterministic mock backend.

Public API
----------
    from openevolve.gauntlets.red_team import RedTeamEvaluator

    evaluator = RedTeamEvaluator()
    result = await evaluator.attack(solution, problem, domain, num_rounds=5)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import asyncio
import logging
import time

from .llm_judge import (
    GauntletJudge,
    describe_attacks,
    probe_solution,
    robustness_from_probes,
    successful_attacks,
)

logger = logging.getLogger(__name__)

# Blend of LLM judge verdict vs. deterministic static analysis.
JUDGE_WEIGHT = 0.5
STATIC_WEIGHT = 0.5


@dataclass
class RedTeamResult:
    """Outcome of the Red Team adversarial round."""

    passed: bool
    score: float
    attacks_attempted: int
    attacks_successful: int
    robustness_score: float
    judge_score: Optional[float]
    rounds_completed: int
    feedback: str
    attack_details: List[Dict[str, Any]] = field(default_factory=list)
    evaluator_type: str = "red_team"
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a dictionary."""
        return {
            "round": "round" if False else "round2_red_team",
            "passed": self.passed,
            "score": self.score,
            "attacks_attempted": self.attacks_attempted,
            "attacks_successful": self.attacks_successful,
            "robustness_score": self.robustness_score,
            "judge_score": self.judge_score,
            "rounds_completed": self.rounds_completed,
            "feedback": self.feedback,
            "attack_count": len(self.attack_details),
            "evaluator_type": self.evaluator_type,
            "timestamp": self.timestamp,
        }


class RedTeamEvaluator:
    """
    Adversarial (Red Team) evaluator for Gauntlet Round 2.

    Args:
        llm_config: Optional judge model configuration. When omitted (or when no
            API key is available) the offline deterministic mock judge is used.
        threshold: Minimum blended score required to pass Round 2.
        language: Language of the candidate solution.
    """

    def __init__(
        self,
        llm_config: Any = None,
        threshold: float = 0.6,
        language: str = "python",
    ):
        self.llm_config = llm_config
        self.threshold = threshold
        self.language = language
        self.judge: Optional[GauntletJudge] = None
        try:
            self.judge = GauntletJudge(llm_config=llm_config)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Red Team judge initialization failed: {exc}")
            self.judge = None

    async def attack(
        self,
        solution: str,
        problem: str,
        domain: str,
        num_rounds: int = 5,
    ) -> RedTeamResult:
        """
        Run the adversarial campaign against ``solution``.

        Args:
            solution: Candidate solution source.
            problem: Problem statement.
            domain: Problem domain.
            num_rounds: Number of adversarial judge passes to run. The deterministic
                probes are run once; the LLM judge is consulted ``num_rounds`` times
                so the verdict is hardened against a single flaky response.

        Returns:
            RedTeamResult with the blended score and attack details.
        """
        start_time = time.time()
        logger.info(f"Red Team attack starting (num_rounds={num_rounds})")

        # 1. Deterministic attack vectors
        probes = probe_solution(solution, language=self.language)
        static_robustness = robustness_from_probes(probes)
        probe_findings = describe_attacks(probes)

        attack_details: List[Dict[str, Any]] = list(probes)
        attacks_attempted = len(probes)
        attacks_successful = len(successful_attacks(probes))

        # 2. LLM judge adversarial critique (offline mock when no model configured)
        judge_score: Optional[float] = None
        judge_feedback = "Red Team judge unavailable"
        rounds_completed = 0

        if self.judge is not None:
            for _ in range(max(1, num_rounds)):
                try:
                    verdict = await self.judge.red_team(
                        solution=solution,
                        problem=problem,
                        domain=domain,
                        language=self.language,
                        prior_findings=probe_findings,
                    )
                    rounds_completed += 1
                    if verdict.parsed:
                        judge_score = (
                            verdict.score
                            if judge_score is None
                            else (judge_score + verdict.score) / 2.0
                        )
                        judge_feedback = verdict.feedback
                        for finding in verdict.findings:
                            attack_details.append(
                                {
                                    "name": "llm_red_team_finding",
                                    "description": finding,
                                    "severity": "medium",
                                    "successful": True,
                                    "evidence": verdict.model,
                                }
                            )
                            attacks_attempted += 1
                            attacks_successful += 1
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(f"Red Team judge pass failed: {exc}")
                    break

        # 3. Blend the two signals
        if judge_score is None:
            score = static_robustness
        else:
            score = STATIC_WEIGHT * static_robustness + JUDGE_WEIGHT * judge_score

        passed = bool(score >= self.threshold)

        feedback = (
            f"Red Team evaluation complete. "
            f"{attacks_attempted} attacks attempted, "
            f"{attacks_successful} successful. "
            f"Static robustness: {static_robustness:.2f}, "
            f"judge score: {'n/a' if judge_score is None else f'{judge_score:.2f}'}. "
            f"Judge: {judge_feedback}"
        )

        return RedTeamResult(
            passed=passed,
            score=score,
            attacks_attempted=attacks_attempted,
            attacks_successful=attacks_successful,
            robustness_score=static_robustness,
            judge_score=judge_score,
            rounds_completed=max(1, rounds_completed),
            feedback=feedback,
            attack_details=attack_details,
        )

    # Aliases so the evaluator can be dropped into the orchestrator in place of a
    # GauntletJudge (which exposes ``red_team`` / ``evaluate``).
    async def red_team(self, solution: str, problem: str, domain: str, language: str = "python", prior_findings=None) -> RedTeamResult:
        return await self.attack(solution, problem, domain, num_rounds=5)

    async def evaluate(self, solution: str, problem: str, domain: str, **kwargs: Any) -> RedTeamResult:
        return await self.attack(solution, problem, domain, **kwargs)
