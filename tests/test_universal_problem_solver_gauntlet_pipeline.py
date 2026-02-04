import json
from unittest.mock import patch

from openevolve_structures import Team, ModelConfig, GauntletDefinition, GauntletRoundRule
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from universal_problem_solver import UniversalProblemSolver


def _make_team(name: str, role: str) -> Team:
    return Team(
        name=name,
        role=role,
        members=[ModelConfig(model_id=f"{name}-model", api_key="test-key")],
    )


def _make_gauntlet(name: str, team_name: str) -> GauntletDefinition:
    rounds = [
        GauntletRoundRule(
            round_number=1,
            quorum_required_approvals=1,
            quorum_from_panel_size=1,
            min_overall_confidence=0.5,
        )
    ]
    return GauntletDefinition(name=name, team_name=team_name, rounds=rounds)


@patch("workflow_engine._request_openai_compatible_chat")
def test_universal_problem_solver_gauntlet_pipeline(mock_chat):
    mock_chat.return_value = json.dumps({
        "score": 0.9,
        "justification": "Looks good",
        "targeted_feedback": [],
    })

    team_manager = TeamManager()
    gauntlet_manager = GauntletManager()

    blue_team = _make_team("BlueTeam", "Blue")
    red_team = _make_team("RedTeam", "Red")
    gold_team = _make_team("GoldTeam", "Gold")

    team_manager.teams = {
        blue_team.name: blue_team,
        red_team.name: red_team,
        gold_team.name: gold_team,
    }

    gauntlet_manager.gauntlets = {
        "SolverGen": _make_gauntlet("SolverGen", blue_team.name),
        "SubRed": _make_gauntlet("SubRed", red_team.name),
        "SubGold": _make_gauntlet("SubGold", gold_team.name),
        "FinalRed": _make_gauntlet("FinalRed", red_team.name),
        "FinalGold": _make_gauntlet("FinalGold", gold_team.name),
    }

    solver = UniversalProblemSolver(
        team_manager=team_manager,
        gauntlet_manager=gauntlet_manager,
        gauntlet_config={
            "solver_generation_gauntlet": "SolverGen",
            "sub_problem_red_gauntlet": "SubRed",
            "sub_problem_gold_gauntlet": "SubGold",
            "final_red_gauntlet": "FinalRed",
            "final_gold_gauntlet": "FinalGold",
        },
    )

    result = solver.solve(
        problem_statement="Design an API for order processing",
        domain="software",
        constraints=["auth", "latency"],
        success_criteria=["correctness", "reliability"],
        run_gauntlets=True,
    )

    assert result.gauntlet_summary
    assert result.gauntlet_results["sub_problems"]
    assert "red" in result.gauntlet_results["final"]
    assert "gold" in result.gauntlet_results["final"]
    assert result.gauntlet_summary["total_runs"] > 0
