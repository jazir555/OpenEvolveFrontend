import pytest

from sop_generator_research_quest import (
    get_research_quest_stage,
    list_research_quest_stages,
    load_stage_template,
    ResearchQuestSOPGenerator,
)


def test_stage_registry_contains_eight_stages():
    stages = list_research_quest_stages()
    assert len(stages) == 8
    assert stages[0].stage_id == 1
    assert stages[-1].stage_id == 8


def test_get_research_quest_stage_invalid():
    with pytest.raises(ValueError):
        get_research_quest_stage(9)


def test_stage_template_exists():
    template = load_stage_template(1)
    assert "Stage 1" in template


def test_requirement_builder_contains_stage_name():
    generator = ResearchQuestSOPGenerator()
    requirement = generator.build_stage_requirement(
        research_question="What is the best validation strategy?",
        stage_id=3,
        context="Use evidence from prior experiments.",
    )
    assert "Stage 3" in requirement
    assert "Hypothesis Planning" in requirement
    assert "Use evidence from prior experiments." in requirement


def test_quality_weight_uses_metadata():
    generator = ResearchQuestSOPGenerator()
    dummy_sop = type("Dummy", (), {"metadata": {"quality_score": 0.5}})()
    weighted = generator.compute_quality_weight(0.6, dummy_sop)
    assert 0.4 <= weighted <= 0.6
