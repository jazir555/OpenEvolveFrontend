"""
OpenEvolve Gauntlets Package

Multi-round evaluation systems with sophisticated state management,
decision logic, and artifact fusion.

This package provides:
- Multi-round orchestration across different gauntlet types
- State management and tracking across rounds
- Decision logic for continue/terminate decisions
- Score normalization across different scoring scales
- Artifact fusion with consensus detection
- Progress reporting and performance metrics
"""

from .loongflow_gauntlet import (
    LoongFlowGauntletEvaluator,
    LoongFlowGauntletConfig,
    GauntletEvaluationResult,
)

from .three_round_orchestrator import (
    ThreeRoundGauntletOrchestrator,
    ThreeRoundConfig,
    FullGauntletResult,
    Round1Result,
    Round2Result,
    Round3Result,
)

from .red_team import RedTeamEvaluator, RedTeamResult
from .gold_team import GoldTeamEvaluator, GoldTeamResult

from .multi_round_orchestrator import (
    MultiRoundGauntletOrchestrator,
    GauntletState,
    FusedArtifacts,
)

from .llm_judge import (
    GauntletJudge,
    JudgeVerdict,
    build_judge_ensemble,
    probe_solution,
    robustness_from_probes,
    verify_solution,
)

__all__ = [
    # LoongFlow Gauntlet
    'LoongFlowGauntletEvaluator',
    'LoongFlowGauntletConfig',
    'GauntletEvaluationResult',

    # Three Round Orchestrator
    'ThreeRoundGauntletOrchestrator',
    'ThreeRoundConfig',
    'FullGauntletResult',
    'Round1Result',
    'Round2Result',
    'Round3Result',

    # Multi Round Orchestrator
    'MultiRoundGauntletOrchestrator',
    'GauntletState',
    'FusedArtifacts',

    # Red Team / Gold Team judging
    'GauntletJudge',
    'JudgeVerdict',
    'build_judge_ensemble',
    'probe_solution',
    'robustness_from_probes',
    'verify_solution',

    # Red Team (Round 2) and Gold Team (Round 3) evaluators
    'RedTeamEvaluator',
    'RedTeamResult',
    'GoldTeamEvaluator',
    'GoldTeamResult',
]
