# SOP Generator + Research-Quest Integration

This guide describes the SOP Generator integration for the 8-stage Research-Quest methodology.

## What It Does

- Maps each Research-Quest stage to SOP requirements.
- Generates turnkey SOPs per stage using MAKER.
- Produces hypothesis test protocols (Stage 3) and evidence collection SOPs (Stage 4).
- Refines SOPs based on reflection feedback (Stage 8).
- Applies SOP quality as a confidence weight.

## Core Modules

- `sop_generator_research_quest.py`:
  - `ResearchQuestSOPGenerator`
  - `ResearchQuestWorkflowManager`
  - Stage definitions and templates loader
- `sop_templates/research_quest_stages/`:
  - Stage 1-8 templates
- `sop_templates/domains/` and `sop_templates/methods/`:
  - Domain and method templates

## Basic Usage

```python
from sop_generator_research_quest import ResearchQuestWorkflowManager

manager = ResearchQuestWorkflowManager()
runs = await manager.run_full_workflow(
    research_question="How can we improve dataset reproducibility?",
    base_confidence=0.6,
    context="Focus on open-source benchmark datasets."
)

stage_3 = runs[3]
print(stage_3.sop.to_markdown()[:1000])
```

## Integration Notes

- SOP quality scores are stored in `sop.metadata["quality_score"]`.
- Confidence weighting is applied in `compute_quality_weight`.
- Reflection feedback uses `refine_stage_sop`.

## Expected Outputs

- Stage SOPs for all 8 Research-Quest stages
- SOP quality scores for tracking improvements
- Optional hypothesis and evidence protocol SOPs
