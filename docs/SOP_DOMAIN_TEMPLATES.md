# SOP Domain Templates

Domain templates provide guidance for specialized SOP generation.

## Available Templates

- `sop_templates/domains/biology/wet_lab_experiment.md`
- `sop_templates/domains/physics/experimental_setup.md`
- `sop_templates/domains/cs/benchmarking.md`
- `sop_templates/domains/social_science/survey_design.md`

## Method Templates

- `sop_templates/methods/experimental.md`
- `sop_templates/methods/observational.md`
- `sop_templates/methods/computational.md`

## How to Use

Include the template content as context when building requirements:

```python
from sop_generator_research_quest import load_stage_template

template = load_stage_template(3)
requirement = f"{template}\nInclude domain template: benchmarking SOP"
```
