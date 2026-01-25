# SOP + Research-Quest User Guide

This guide shows how to generate SOPs for Research-Quest stages and refine them over time.

## Quick Start

```python
from sop_generator_research_quest import ResearchQuestSOPGenerator

generator = ResearchQuestSOPGenerator(domain="research")
sop = await generator.generate_stage_sop(
    research_question="How to improve evaluation reproducibility?",
    stage_id=1
)
print(sop.to_markdown())
```

## Stage SOPs

Each stage has a template under `sop_templates/research_quest_stages/`.
You can add additional constraints or context per stage.

## Hypothesis Protocols

```python
protocols = await generator.generate_hypothesis_protocols(
    research_question="How to reduce benchmark leakage?",
    hypotheses=["Leakage stems from dataset overlap", "Leakage stems from prompt leakage"]
)
```

## Evidence Collection SOPs

```python
evidence_sop = await generator.generate_evidence_collection_sop(
    research_question="What data sources matter?",
    evidence_sources=["arXiv", "OpenML", "PapersWithCode"]
)
```

## Reflection and Refinement

```python
refined = await generator.refine_stage_sop(
    stage_id=8,
    research_question="Improve evaluation reproducibility",
    feedback="Missing verification steps for data provenance",
    existing_sop=sop
)
```

## Best Practices

- Provide clear constraints for equipment and resources.
- Keep stage context short and targeted.
- Track SOP quality via `sop.metadata["quality_score"]`.
