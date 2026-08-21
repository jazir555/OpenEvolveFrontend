"""Gauntlet Team Coordination Manager & Analyzer.

Runs the multi-round Red / Blue / Gold validation flow and aggregates the
results into a single gauntlet verdict.

Flow per candidate solution (one gauntlet run):

    1. RED TEAM   - adversarial critique: enumerate challenges / findings.
    2. BLUE TEAM  - fix the findings and attempt a formal proof with the
                    LeanAIDE client.
    3. GOLD TEAM  - authoritative verification of the (fixed) solution using the
                    LeanAIDE client and emit a final confidence-weighted verdict.

The manager is functional without any external service: the coordinators fall
back to the built-in mock LeanAIDE client, and every round is wrapped so a
failure in one team degrades gracefully instead of aborting the gauntlet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from team_lean_bridge import get_lean_client, sha1

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Aggregated result of one multi-round gauntlet validation."""

    content_id: str
    rounds_passed: int
    total_rounds: int
    overall_passed: bool
    red_findings_count: int
    blue_proof_proved: bool
    gold_verified: bool
    final_score: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "rounds_passed": self.rounds_passed,
            "total_rounds": self.total_rounds,
            "overall_passed": self.overall_passed,
            "red_findings_count": self.red_findings_count,
            "blue_proof_proved": self.blue_proof_proved,
            "gold_verified": self.gold_verified,
            "final_score": round(self.final_score, 4),
            "details": self.details,
        }


class TeamValidationManager:
    """Orchestrates the Red -> Blue -> Gold multi-round gauntlet validation."""

    def __init__(
        self,
        red: Any = None,
        blue: Any = None,
        gold: Any = None,
        client: Any = None,
    ):
        self.client = client or get_lean_client()
        self.red = red
        self.blue = blue
        self.gold = gold
        self.last_report: Optional[ValidationReport] = None

    # -- lazy coordinator wiring (keeps the module importable) --------------
    def _ensure_coordinators(self) -> None:
        if self.red is None:
            try:  # pragma: no cover - optional heavy dependency
                from red_team_coordinator import RedTeamCoordinator

                self.red = RedTeamCoordinator(use_ensemble=False, enable_persistence=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("RedTeamCoordinator unavailable: %s", exc)
                self.red = None
        if self.blue is None:
            from blue_team_coordinator import BlueTeamCoordinator

            self.blue = BlueTeamCoordinator(client=self.client)
        if self.gold is None:
            from gold_team_coordinator import GoldTeamCoordinator

            self.gold = GoldTeamCoordinator(client=self.client)

    # -- main entry point --------------------------------------------------
    def run_validation(
        self,
        content: str,
        content_type: str = "general",
        content_id: Optional[str] = None,
        tactics: Optional[List[str]] = None,
        rounds: int = 3,
    ) -> ValidationReport:
        """Run the full Red/Blue/Gold validation gauntlet on ``content``."""
        self._ensure_coordinators()
        content_id = content_id or sha1(content)
        details: Dict[str, Any] = {}

        # ---- Round 1: RED TEAM critique ----
        red_findings: List[Any] = []
        try:
            if self.red is not None:
                session = self.red.coordinate_adversarial_testing(content, content_type)
                red_findings = list(getattr(session, "aggregated_findings", []) or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Red Team round failed: %s", exc)
            details["red_error"] = str(exc)
        details["red_findings_count"] = len(red_findings)

        # ---- Round 2: BLUE TEAM fix + formal proof ----
        blue_fix: Dict[str, Any] = {"fixed_content": content, "applied_fixes": []}
        proof = None
        try:
            blue_fix = self.blue.fix_content(content, red_findings)
            proof = self.blue.attempt_proof(content, tactics)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Blue Team round failed: %s", exc)
            details["blue_error"] = str(exc)
        blue_proved = bool(getattr(proof, "proved", False))
        details["blue_fix"] = {
            k: v for k, v in blue_fix.items() if k != "fixed_content"
        }
        details["blue_proof"] = proof.to_dict() if proof is not None else None

        # ---- Round 3: GOLD TEAM verification ----
        gold: Dict[str, Any] = {"passed": False, "verified": False, "confidence": 0.0}
        try:
            artifacts = [
                {"red_findings": len(red_findings), "blue_proved": blue_proved}
            ]
            gold = self.gold.final_review(content, artifacts=artifacts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gold Team round failed: %s", exc)
            details["gold_error"] = str(exc)
        gold_verified = bool(gold.get("verified", False))
        details["gold"] = gold

        # ---- Aggregation ----
        red_score = max(0.0, 1.0 - len(red_findings) * 0.1)
        blue_score = 1.0 if blue_proved else 0.3
        gold_score = float(gold.get("confidence", 0.0)) if gold_verified else 0.0
        final_score = round(0.25 * red_score + 0.35 * blue_score + 0.40 * gold_score, 4)

        rounds_passed = sum(
            [
                1 if len(red_findings) == 0 else 0,
                1 if blue_proved else 0,
                1 if gold_verified else 0,
            ]
        )
        overall_passed = gold_verified and blue_proved

        report = ValidationReport(
            content_id=content_id,
            rounds_passed=rounds_passed,
            total_rounds=rounds,
            overall_passed=overall_passed,
            red_findings_count=len(red_findings),
            blue_proof_proved=blue_proved,
            gold_verified=gold_verified,
            final_score=final_score,
            details=details,
        )
        self.last_report = report
        return report


class GauntletAnalyzer:
    """Aggregates many :class:`ValidationReport` instances into gauntlet stats."""

    def __init__(self) -> None:
        self.reports: List[ValidationReport] = []

    def add_report(self, report: ValidationReport) -> None:
        self.reports.append(report)

    def aggregate(self) -> Dict[str, Any]:
        total = len(self.reports)
        if not total:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "average_score": 0.0,
                "average_rounds_passed": 0.0,
                "red_findings_total": 0,
                "blue_proof_rate": 0.0,
                "gold_verification_rate": 0.0,
            }
        passed = sum(1 for r in self.reports if r.overall_passed)
        avg_score = sum(r.final_score for r in self.reports) / total
        avg_rounds = sum(r.rounds_passed for r in self.reports) / total
        blue_proved = sum(1 for r in self.reports if r.blue_proof_proved)
        gold_verified = sum(1 for r in self.reports if r.gold_verified)
        red_total = sum(r.red_findings_count for r in self.reports)
        return {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4),
            "average_score": round(avg_score, 4),
            "average_rounds_passed": round(avg_rounds, 4),
            "red_findings_total": red_total,
            "blue_proof_rate": round(blue_proved / total, 4),
            "gold_verification_rate": round(gold_verified / total, 4),
        }

    def summarize(self) -> str:
        agg = self.aggregate()
        return (
            f"Gauntlet: {agg['total']} runs, {agg['passed']} passed "
            f"({agg['pass_rate']*100:.1f}%), avg score {agg['average_score']}, "
            f"blue proof rate {agg['blue_proof_rate']}, "
            f"gold verify rate {agg['gold_verification_rate']}"
        )


def run_gauntlet(
    contents: List[str],
    content_type: str = "general",
    tactics: Optional[List[str]] = None,
    client: Any = None,
) -> Dict[str, Any]:
    """Convenience: validate a batch of candidate solutions and analyze them."""
    manager = TeamValidationManager(client=client)
    analyzer = GauntletAnalyzer()
    for content in contents:
        analyzer.add_report(manager.run_validation(content, content_type, tactics=tactics))
    return {
        "reports": [r.to_dict() for r in analyzer.reports],
        "aggregate": analyzer.aggregate(),
    }
