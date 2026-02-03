from roma_openevolve_integration import create_roma_adapter


def test_roma_reassembly_fallback_uses_enhanced_engine():
    adapter = create_roma_adapter(enable_roma=False)
    solutions = [
        {
            "id": "sp_alpha",
            "solution": "Define SharedAPI and expose authentication flow.",
            "dependencies": [],
        },
        {
            "id": "sp_beta",
            "solution": "Consume SharedAPI for session management.",
            "dependencies": ["sp_alpha"],
        },
    ]

    result = adapter.reassemble_solutions(solutions, problem_statement="Test ROMA fallback")

    assert result["status"] == "completed"
    assert result.get("reassembly_method") == "enhanced_recomposition"
    assert "## Section" in result.get("final_solution", "")
