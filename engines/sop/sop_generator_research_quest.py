"""
SOP Generator + Research-Quest Integration

Provides stage-aware SOP generation for the 8-stage Research-Quest methodology.
Maps Research-Quest stages to SOP requirements and builds structured SOP
documents using the self-contained `sop_document` renderer (no external
services). Parameter substitution and section generation are supported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sop_parameter import SOPParameter  # provided externally; never redefined

from sop_document import (
    SOPStep,
    StandardOperatingProcedure,
    SOPRenderer,
    SOPGenerator,
)


@dataclass(frozen=True)
class ResearchQuestStageDefinition:
    stage_id: int
    name: str
    description: str
    objectives: List[str]
    outputs: List[str]
    quality_checks: List[str]
    key_parameters: List[str]


RESEARCH_QUEST_STAGES: Dict[int, ResearchQuestStageDefinition] = {
    1: ResearchQuestStageDefinition(
        stage_id=1,
        name="Initialization",
        description="Initialize the research graph and define the root problem.",
        objectives=[
            "Define research question and scope",
            "Create root node with task understanding",
            "Set initial confidence assumptions",
        ],
        outputs=[
            "Root node definition",
            "Initial confidence values",
            "Scope and constraints summary",
        ],
        quality_checks=[
            "Research question is explicit and testable",
            "Scope boundaries are defined",
            "Assumptions are documented",
        ],
        key_parameters=["P1.1", "P1.5", "P1.12"],
    ),
    2: ResearchQuestStageDefinition(
        stage_id=2,
        name="Decomposition",
        description="Decompose the research task into dimensions and nodes.",
        objectives=[
            "Break task into core dimensions",
            "Identify knowledge gaps and potential biases",
            "Create dimension nodes with metadata",
        ],
        outputs=[
            "Dimension node list",
            "Gap and bias annotations",
            "Decomposition rationale",
        ],
        quality_checks=[
            "All mandatory dimensions included",
            "Gaps and biases explicitly tagged",
            "Nodes contain required metadata",
        ],
        key_parameters=["P1.2", "P1.15", "P1.17"],
    ),
    3: ResearchQuestStageDefinition(
        stage_id=3,
        name="Hypothesis Planning",
        description="Generate competing hypotheses per dimension with criteria.",
        objectives=[
            "Generate multiple hypotheses per dimension",
            "Specify falsification criteria",
            "Assign initial confidence and impact",
        ],
        outputs=[
            "Hypothesis nodes",
            "Falsification criteria",
            "Confidence/impact estimates",
        ],
        quality_checks=[
            "Hypotheses are mutually distinguishable",
            "Falsification criteria are concrete",
            "Confidence levels are justified",
        ],
        key_parameters=["P1.3", "P1.16", "P1.28"],
    ),
    4: ResearchQuestStageDefinition(
        stage_id=4,
        name="Evidence Integration",
        description="Link evidence to hypotheses and update confidence.",
        objectives=[
            "Collect evidence artifacts",
            "Link evidence to hypotheses with edge types",
            "Update confidence using Bayesian logic",
        ],
        outputs=[
            "Evidence catalog",
            "Evidence-hypothesis links",
            "Updated confidence values",
        ],
        quality_checks=[
            "Evidence quality is assessed",
            "Links are typed and justified",
            "Confidence update logic recorded",
        ],
        key_parameters=["P1.4", "P1.14", "P1.18"],
    ),
    5: ResearchQuestStageDefinition(
        stage_id=5,
        name="Pruning & Merging",
        description="Remove weak hypotheses and merge overlapping nodes.",
        objectives=[
            "Prune low-value hypotheses",
            "Merge overlapping or redundant hypotheses",
            "Apply bias mitigation techniques",
        ],
        outputs=[
            "Pruned graph snapshot",
            "Merged hypothesis records",
            "Bias mitigation log",
        ],
        quality_checks=[
            "Pruning criteria documented",
            "Merges preserve critical evidence",
            "Bias mitigation steps completed",
        ],
        key_parameters=["P1.17", "P1.22", "P1.28"],
    ),
    6: ResearchQuestStageDefinition(
        stage_id=6,
        name="Subgraph Extraction",
        description="Extract high-value subgraphs and pathways.",
        objectives=[
            "Identify high-confidence nodes",
            "Extract pathways for reporting",
            "Prepare visualization-ready subgraphs",
        ],
        outputs=[
            "Selected subgraphs",
            "Pathway summaries",
            "Visualization data package",
        ],
        quality_checks=[
            "Selection criteria are explicit",
            "Subgraphs cover high-impact nodes",
            "Outputs ready for visualization",
        ],
        key_parameters=["P1.15", "P1.22", "P1.28"],
    ),
    7: ResearchQuestStageDefinition(
        stage_id=7,
        name="Composition",
        description="Compose the research narrative and export artifacts.",
        objectives=[
            "Generate structured research narrative",
            "Include reasoning trace and citations",
            "Export graph artifacts",
        ],
        outputs=[
            "Narrative report",
            "Citation list",
            "Exported graph files",
        ],
        quality_checks=[
            "Narrative aligns with graph evidence",
            "Citations are complete",
            "Exports are valid and readable",
        ],
        key_parameters=["P1.6", "P1.11", "K1.3"],
    ),
    8: ResearchQuestStageDefinition(
        stage_id=8,
        name="Reflection",
        description="Perform reflection audit and quality validation.",
        objectives=[
            "Run self-audit against criteria",
            "Validate coverage and bias handling",
            "Record improvement actions",
        ],
        outputs=[
            "Reflection checklist",
            "Validation report",
            "Improvement backlog",
        ],
        quality_checks=[
            "Audit criteria all addressed",
            "Bias flags resolved or documented",
            "Improvement actions assigned",
        ],
        key_parameters=["P1.7", "P1.15", "P1.17"],
    ),
}


def get_research_quest_stage(stage_id: int) -> ResearchQuestStageDefinition:
    if stage_id not in RESEARCH_QUEST_STAGES:
        raise ValueError(f"Unknown Research-Quest stage: {stage_id}")
    return RESEARCH_QUEST_STAGES[stage_id]


def list_research_quest_stages() -> List[ResearchQuestStageDefinition]:
    return [RESEARCH_QUEST_STAGES[k] for k in sorted(RESEARCH_QUEST_STAGES.keys())]


def load_stage_template(stage_id: int) -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "sop_templates", "research_quest_stages")
    filename = f"stage_{stage_id}.md"
    path = os.path.join(base_dir, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    return ""


class ResearchQuestSOPGenerator:
    """
    Builds SOPs for each Research-Quest stage using the self-contained renderer.
    """

    def __init__(self, sop_generator: Optional[SOPGenerator] = None, domain: str = "research"):
        self.sop_generator = sop_generator or SOPGenerator()
        self.renderer = SOPRenderer()
        self.domain = domain

    def build_stage_requirement(
        self,
        research_question: str,
        stage_id: int,
        context: Optional[str] = None,
    ) -> str:
        stage = get_research_quest_stage(stage_id)
        template = load_stage_template(stage_id)
        sections = [
            f"Research-Quest Stage {stage.stage_id}: {stage.name}",
            f"Research Question: {research_question}",
            f"Stage Description: {stage.description}",
            "Stage Objectives:",
            *[f"- {item}" for item in stage.objectives],
            "Required Outputs:",
            *[f"- {item}" for item in stage.outputs],
            "Quality Checks:",
            *[f"- {item}" for item in stage.quality_checks],
            "Key Parameters:",
            *[f"- {item}" for item in stage.key_parameters],
        ]
        if context:
            sections.append("Context:")
            sections.append(context)
        if template:
            sections.append("Template Guidance:")
            sections.append(template)
        return "\n".join(sections)

    def compute_quality_weight(self, confidence: float, sop: StandardOperatingProcedure) -> float:
        """Adjust a confidence score using SOP quality metadata."""
        quality = float(sop.metadata.get("quality_score", 0.8))
        return min(1.0, confidence * (0.7 + 0.3 * quality))

    async def generate_stage_sop(
        self,
        research_question: str,
        stage_id: int,
        constraints: Optional[List[str]] = None,
        equipment_available: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> StandardOperatingProcedure:
        stage = get_research_quest_stage(stage_id)
        requirement = self.build_stage_requirement(research_question, stage_id, context)

        # Build the parameters referenced by the stage as SOPParameters so they
        # can be substituted into the rendered document.
        params: Dict[str, SOPParameter] = {}
        for key in stage.key_parameters:
            params[key] = SOPParameter(
                name=key,
                value=0.0,
                unit="",
                tolerance=0.0,
                verification_method="Documented in stage output",
                critical=False,
                rationale=f"Key parameter for stage {stage.stage_id} ({stage.name})",
            )

        steps = [
            SOPStep(
                step_number=i + 1,
                action=objective,
                verification_method=qc,
                acceptance_criteria="Objective satisfied and recorded",
            )
            for i, (objective, qc) in enumerate(
                zip(stage.objectives, stage.quality_checks)
            )
        ]

        sop = self.renderer.render(
            title=f"RQ Stage {stage.stage_id}: {stage.name}",
            parameters=params,
            steps=steps,
            preconditions=constraints or ["Research question defined"],
            equipment=[{"name": e} for e in (equipment_available or [])],
            quality_control=list(stage.quality_checks),
            safety_protocols=["Record assumptions and biases explicitly"],
            validation_criteria=["All required outputs produced"],
            scaling_info=[f"Key parameters: {', '.join(stage.key_parameters)}"],
            description=requirement,
            metadata={
                "domain": self.domain,
                "stage_id": stage.stage_id,
                "research_question": research_question,
            },
        )
        return sop

    async def generate_full_workflow_sops(
        self,
        research_question: str,
        constraints: Optional[List[str]] = None,
        equipment_available: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> Dict[int, StandardOperatingProcedure]:
        sops: Dict[int, StandardOperatingProcedure] = {}
        for stage in list_research_quest_stages():
            sops[stage.stage_id] = await self.generate_stage_sop(
                research_question=research_question,
                stage_id=stage.stage_id,
                constraints=constraints,
                equipment_available=equipment_available,
                context=context,
            )
        return sops

    async def generate_hypothesis_protocols(
        self,
        research_question: str,
        hypotheses: List[str],
        constraints: Optional[List[str]] = None,
        equipment_available: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> Dict[str, StandardOperatingProcedure]:
        """Generate a test protocol SOP for each hypothesis (Stage 3)."""
        protocols: Dict[str, StandardOperatingProcedure] = {}
        for hypothesis in hypotheses:
            requirement = (
                f"Test hypothesis: {hypothesis}\n"
                f"Research Question: {research_question}\n"
                "Include falsification criteria and validation thresholds."
            )
            if context:
                requirement += f"\nContext:\n{context}"
            sop = self.renderer.render(
                title=f"Hypothesis Test: {hypothesis[:60]}",
                steps=[
                    SOPStep(
                        step_number=1,
                        action=f"Define falsification criteria for: {hypothesis}",
                        verification_method="Criteria are concrete and testable",
                        acceptance_criteria="Falsifiable prediction recorded",
                    ),
                    SOPStep(
                        step_number=2,
                        action="Run validation experiment",
                        verification_method="Measured against thresholds",
                        acceptance_criteria="Result compared to prediction",
                    ),
                ],
                preconditions=constraints or ["Hypothesis documented"],
                quality_control=["Record outcome and confidence"],
                safety_protocols=["Standard research integrity practices"],
                description=requirement,
                metadata={"domain": self.domain, "hypothesis": hypothesis},
            )
            protocols[hypothesis] = sop
        return protocols

    async def generate_evidence_collection_sop(
        self,
        research_question: str,
        evidence_sources: List[str],
        constraints: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> StandardOperatingProcedure:
        """Generate standardized evidence collection SOP (Stage 4)."""
        requirement = [
            "Standard protocol for evidence collection",
            f"Research Question: {research_question}",
            "Evidence Sources:",
            *[f"- {source}" for source in evidence_sources],
            "Include inclusion/exclusion criteria and quality assessment thresholds.",
        ]
        if context:
            requirement.append("Context:")
            requirement.append(context)
        sop = self.renderer.render(
            title="Evidence Collection Protocol",
            steps=[
                SOPStep(
                    step_number=i + 1,
                    action=f"Collect evidence from: {source}",
                    verification_method="Source quality assessed",
                    acceptance_criteria="Evidence meets inclusion criteria",
                )
                for i, source in enumerate(evidence_sources)
            ]
            or [
                SOPStep(
                    step_number=1,
                    action="Collect evidence per inclusion criteria",
                    verification_method="Quality threshold applied",
                    acceptance_criteria="Evidence logged with provenance",
                )
            ],
            preconditions=constraints or ["Sources identified"],
            quality_control=["Apply inclusion/exclusion criteria", "Assess source quality"],
            safety_protocols=["Document provenance and biases"],
            description="\n".join(requirement),
            metadata={"domain": self.domain, "research_question": research_question},
        )
        return sop

    async def refine_stage_sop(
        self,
        stage_id: int,
        research_question: str,
        feedback: str,
        existing_sop: StandardOperatingProcedure,
        constraints: Optional[List[str]] = None,
    ) -> StandardOperatingProcedure:
        """Refine an SOP based on reflection feedback (Stage 8)."""
        requirement = (
            f"Refine SOP for Stage {stage_id} ({get_research_quest_stage(stage_id).name})\n"
            f"Research Question: {research_question}\n"
            f"Feedback:\n{feedback}"
        )
        existing_sop.revision_history.append(
            {
                "date": __import__("datetime").datetime.now().isoformat(),
                "change": f"Refined from feedback: {feedback[:80]}",
            }
        )
        existing_sop.metadata["refinement_feedback"] = feedback
        return existing_sop


@dataclass
class ResearchQuestStageRun:
    stage_id: int
    sop: Optional[StandardOperatingProcedure] = None
    confidence: float = 0.5
    issues: List[str] = field(default_factory=list)


class ResearchQuestWorkflowManager:
    """Lightweight orchestrator that ties Research-Quest stages to SOP generation."""

    def __init__(self, generator: Optional[ResearchQuestSOPGenerator] = None):
        self.generator = generator or ResearchQuestSOPGenerator()

    async def generate_stage(
        self,
        research_question: str,
        stage_id: int,
        base_confidence: float = 0.5,
        constraints: Optional[List[str]] = None,
        equipment_available: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> ResearchQuestStageRun:
        sop = await self.generator.generate_stage_sop(
            research_question=research_question,
            stage_id=stage_id,
            constraints=constraints,
            equipment_available=equipment_available,
            context=context,
        )
        weighted_confidence = self.generator.compute_quality_weight(base_confidence, sop)
        return ResearchQuestStageRun(stage_id=stage_id, sop=sop, confidence=weighted_confidence)

    async def refine_stage_from_reflection(
        self,
        research_question: str,
        stage_run: ResearchQuestStageRun,
        feedback: str,
        constraints: Optional[List[str]] = None,
    ) -> ResearchQuestStageRun:
        if not stage_run.sop:
            raise ValueError("Stage SOP is required for refinement")
        refined = await self.generator.refine_stage_sop(
            stage_id=stage_run.stage_id,
            research_question=research_question,
            feedback=feedback,
            existing_sop=stage_run.sop,
            constraints=constraints,
        )
        stage_run.sop = refined
        stage_run.confidence = self.generator.compute_quality_weight(stage_run.confidence, refined)
        return stage_run

    async def run_full_workflow(
        self,
        research_question: str,
        base_confidence: float = 0.5,
        constraints: Optional[List[str]] = None,
        equipment_available: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> Dict[int, ResearchQuestStageRun]:
        runs: Dict[int, ResearchQuestStageRun] = {}
        for stage in list_research_quest_stages():
            runs[stage.stage_id] = await self.generate_stage(
                research_question=research_question,
                stage_id=stage.stage_id,
                base_confidence=base_confidence,
                constraints=constraints,
                equipment_available=equipment_available,
                context=context,
            )
        return runs
