# SOP + Research-Quest API Reference

## Stage Registry

```python
get_research_quest_stage(stage_id: int) -> ResearchQuestStageDefinition
list_research_quest_stages() -> List[ResearchQuestStageDefinition]
load_stage_template(stage_id: int) -> str
```

## ResearchQuestSOPGenerator

```python
class ResearchQuestSOPGenerator:
    def build_stage_requirement(
        research_question: str,
        stage_id: int,
        context: Optional[str] = None
    ) -> str

    async def generate_stage_sop(
        research_question: str,
        stage_id: int,
        constraints: Optional[List[str]] = None,
        equipment_available: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> StandardOperatingProcedure

    async def generate_full_workflow_sops(
        research_question: str,
        constraints: Optional[List[str]] = None,
        equipment_available: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> Dict[int, StandardOperatingProcedure]

    async def generate_hypothesis_protocols(
        research_question: str,
        hypotheses: List[str],
        constraints: Optional[List[str]] = None,
        equipment_available: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> Dict[str, StandardOperatingProcedure]

    async def generate_evidence_collection_sop(
        research_question: str,
        evidence_sources: List[str],
        constraints: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> StandardOperatingProcedure

    async def refine_stage_sop(
        stage_id: int,
        research_question: str,
        feedback: str,
        existing_sop: StandardOperatingProcedure,
        constraints: Optional[List[str]] = None
    ) -> StandardOperatingProcedure
```

## ResearchQuestWorkflowManager

```python
class ResearchQuestWorkflowManager:
    async def generate_stage(...)
    async def refine_stage_from_reflection(...)
    async def run_full_workflow(...)
```
