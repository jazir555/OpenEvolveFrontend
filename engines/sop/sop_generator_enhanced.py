"""
Enhanced SOP Generator

Provides SOP generation (manufacturing, assembly, testing, maintenance, safety)
without external services. It renders Standard Operating Procedure documents
from `SOPParameter` values plus templates, performs parameter substitution and
section generation, and validates parameters.

`SOPParameter` is provided externally (another agent) as
`engines/other/sop_parameter.py` and is imported here; it is never redefined.
The SOP document model and renderer live in `sop_document` (also service-free).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sop_parameter import SOPParameter

from sop_document import (
    SOPStep,
    StandardOperatingProcedure,
    SOPRenderer,
    SOPEvaluator,
    SOPGenerator,
    format_parameter,
)

logger = logging.getLogger(__name__)


class SOPType(Enum):
    MANUFACTURING = "manufacturing"
    QUALITY_CONTROL = "quality_control"
    SAFETY = "safety"
    ASSEMBLY = "assembly"
    TESTING = "testing"
    MAINTENANCE = "maintenance"
    CALIBRATION = "calibration"
    CLEANING = "cleaning"
    TROUBLESHOOTING = "troubleshooting"


class IndustryStandard(Enum):
    ISO_9001 = "ISO 9001"
    ISO_13485 = "ISO 13485"
    AS9100 = "AS9100"
    GMP = "GMP"
    FDA_21_CFR_11 = "FDA 21 CFR Part 11"
    OSHA = "OSHA"
    IEC_62304 = "IEC 62304"


@dataclass
class AssemblyInstruction:
    step_number: int
    description: str
    components: List[str]
    tools_required: List[str]
    torque_specifications: Optional[Dict[str, float]] = None
    visual_check: str = ""
    functional_test: str = ""


@dataclass
class TestProcedure:
    test_name: str
    test_type: str
    equipment_required: List[str]
    test_parameters: Dict[str, SOPParameter]
    acceptance_criteria: str
    preconditions: List[str]
    procedure_steps: List[str]
    data_recording: str
    pass_fail_criteria: str


class LLM4IASIntegration:
    """
    Local, service-free stand-in for LLM4IAS. Generates structured SOP
    fragments (manufacturing process, QC plan, safety protocols, maintenance
    schedule) deterministically from the supplied specifications.
    """

    def __init__(self):
        self.available = False  # No external service required.

    def is_available(self) -> bool:
        return self.available

    def generate_manufacturing_sop(
        self,
        product_spec: Dict[str, Any],
        equipment_list: List[str],
        industry_standard: IndustryStandard = IndustryStandard.ISO_9001,
    ) -> Dict[str, Any]:
        steps = []
        process_steps = product_spec.get("process_steps") or [
            "Material preparation",
            "Primary processing",
            "Final assembly",
        ]
        for i, op in enumerate(process_steps, 1):
            steps.append(
                {
                    "step_number": i,
                    "operation": op,
                    "equipment": equipment_list,
                    "cycle_time": product_spec.get("cycle_time", 30),
                    "quality_checks": ["Dimensional inspection", "Visual inspection"],
                }
            )
        return {
            "process_name": product_spec.get("name", "Manufacturing Process"),
            "industry_standard": industry_standard.value,
            "steps": steps,
            "total_cycle_time": product_spec.get("cycle_time", 30) * len(steps),
        }

    def generate_quality_control_plan(
        self,
        product_spec: Dict[str, Any],
        critical_characteristics: List[str],
        aql: float = 0.01,
    ) -> Dict[str, Any]:
        procedures = []
        for characteristic in critical_characteristics:
            procedures.append(
                {
                    "inspection_point": characteristic,
                    "measurement_method": f"Measure {characteristic} with calibrated instrument",
                    "acceptance_criteria": "Within specification +/- tolerance",
                    "sampling_plan": f"AQL {aql * 100:.2f}%",
                    "frequency": "Every unit" if aql < 0.01 else "Statistical sampling",
                }
            )
        return {
            "qc_procedures": procedures,
            "inspection_levels": {"incoming": 100, "in_process": 100, "final": 100},
            "documentation": "All measurements recorded in the quality log",
        }

    def generate_safety_protocols(
        self, hazards: List[Dict[str, Any]], industry: str = "manufacturing"
    ) -> List[Dict[str, Any]]:
        protocols = []
        for hazard in hazards:
            protocols.append(
                {
                    "hazard_type": hazard.get("type", "general"),
                    "hazard_description": hazard.get("description", "Unknown hazard"),
                    "risk_level": hazard.get("risk", "Medium"),
                    "required_ppe": hazard.get("ppe", ["Safety glasses", "Gloves"]),
                    "engineering_controls": hazard.get("controls", ["Guards", "Ventilation"]),
                    "administrative_controls": ["Training", "Procedures"],
                    "emergency_procedures": hazard.get(
                        "emergency", ["Evacuate", "Call supervisor"]
                    ),
                }
            )
        return protocols

    def generate_maintenance_schedule(
        self, equipment_specs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        schedules = []
        for equip in equipment_specs:
            schedules.append(
                {
                    "equipment_id": equip.get("id", "EQ001"),
                    "equipment_name": equip.get("name", "Equipment"),
                    "maintenance_type": equip.get("maintenance_type", "Preventive"),
                    "frequency": equip.get("frequency", "Monthly"),
                    "procedures": equip.get("procedures", ["Inspect", "Clean", "Lubricate"]),
                    "estimated_duration": equip.get("duration", 1.0),
                    "technician_skill_level": equip.get("skill", "Trained"),
                }
            )
        return schedules


class EnhancedSOPGenerator:
    """
    Enhanced SOP generator (service-free).

    Produces manufacturing, assembly, testing and maintenance SOPs by rendering
    `SOPParameter` values through `SOPRenderer`, substituting parameter tokens
    and validating the result.
    """

    def __init__(self, config: Any = None):
        self.config = config
        self.llm4ias = LLM4IASIntegration()
        self.renderer = SOPRenderer()
        self.evaluator = SOPEvaluator()
        logger.info("EnhancedSOPGenerator initialized (service-free)")

    async def generate_manufacturing_sop(
        self,
        product_name: str,
        product_spec: Dict[str, Any],
        equipment_list: List[str],
        industry_standard: IndustryStandard = IndustryStandard.ISO_9001,
        include_qc: bool = True,
        include_safety: bool = True,
    ) -> StandardOperatingProcedure:
        mfg = self.llm4ias.generate_manufacturing_sop(
            product_spec, equipment_list, industry_standard
        )
        params: Dict[str, SOPParameter] = {}
        for k, v in (product_spec.get("parameters") or {}).items():
            if isinstance(v, SOPParameter):
                params[k] = v
            elif isinstance(v, dict):
                params[k] = SOPParameter(
                    name=v.get("name", k),
                    value=float(v.get("value", 0)),
                    unit=v.get("unit", ""),
                    tolerance=float(v.get("tolerance", 0)),
                    verification_method=v.get("verification_method", ""),
                    critical=v.get("critical", True),
                    rationale=v.get("rationale", ""),
                )

        steps = [
            SOPStep(
                step_number=s["step_number"],
                action=s["operation"],
                duration=float(s.get("cycle_time", 0)) * 60,
                verification_method="Measurement against tolerance",
                acceptance_criteria="Operation completed within specified limits",
                contingency_action="Quarantine batch and notify supervisor",
            )
            for s in mfg["steps"]
        ]

        qc = self.llm4ias.generate_quality_control_plan(
            product_spec, product_spec.get("critical_characteristics", [])
        )
        safety = self.llm4ias.generate_safety_protocols(product_spec.get("hazards", []))

        sop = self.renderer.render(
            title=f"Manufacturing Procedure - {product_name}",
            parameters=params,
            steps=steps,
            preconditions=["Work area prepared", "Personnel trained"],
            equipment=[{"name": e} for e in equipment_list],
            quality_control=[
                f"{p['inspection_point']}: {p['acceptance_criteria']} ({p['sampling_plan']})"
                for p in qc.get("qc_procedures", [])
            ]
            if include_qc
            else [],
            safety_protocols=[
                f"{p['hazard_type']} ({p['risk_level']}): PPE {', '.join(p['required_ppe'])}"
                for p in safety
            ]
            if include_safety
            else [],
            description=f"Manufacturing SOP for {product_name} per {industry_standard.value}",
            metadata={
                "sop_type": SOPType.MANUFACTURING.value,
                "industry_standard": industry_standard.value,
                "total_cycle_time": mfg.get("total_cycle_time"),
            },
        )
        return sop

    async def generate_assembly_sop(
        self,
        assembly_name: str,
        bill_of_materials: List[Dict[str, Any]],
        assembly_sequence: List[Dict[str, Any]],
        tools_required: List[str],
    ) -> StandardOperatingProcedure:
        steps = [
            SOPStep(
                step_number=i + 1,
                action=step.get("description", f"Assembly step {i + 1}"),
                verification_method=step.get("visual_check", "Visual inspection"),
                acceptance_criteria=step.get("functional_test", "Fits and functions"),
                contingency_action="Disassemble and rework if defective",
            )
            for i, step in enumerate(assembly_sequence)
        ]
        sop = self.renderer.render(
            title=f"Assembly Procedure - {assembly_name}",
            steps=steps,
            preconditions=["All components available", "Tools calibrated"],
            equipment=[{"name": t} for t in tools_required],
            materials=bill_of_materials,
            quality_control=["Torque verification", "Functional test"],
            safety_protocols=["Wear appropriate PPE", "Follow tool safety rules"],
            description=f"Assembly SOP for {assembly_name}",
            metadata={"sop_type": SOPType.ASSEMBLY.value},
        )
        return sop

    async def generate_testing_sop(
        self,
        test_name: str,
        test_type: str,
        test_parameters: Dict[str, Any],
        acceptance_criteria: str,
        equipment_required: List[str],
    ) -> StandardOperatingProcedure:
        params: Dict[str, SOPParameter] = {}
        for name, spec in (test_parameters or {}).items():
            if isinstance(spec, SOPParameter):
                params[name] = spec
            elif isinstance(spec, dict):
                params[name] = SOPParameter(
                    name=spec.get("name", name),
                    value=float(spec.get("value", 0)),
                    unit=spec.get("unit", ""),
                    tolerance=float(spec.get("tolerance", 0)),
                    verification_method=spec.get("verification_method", ""),
                )
        procedure = TestProcedure(
            test_name=test_name,
            test_type=test_type,
            equipment_required=equipment_required,
            test_parameters=params,
            acceptance_criteria=acceptance_criteria,
            preconditions=["Equipment calibrated", "Sample prepared"],
            procedure_steps=[
                "Set up test equipment per manufacturer specifications",
                "Configure test parameters",
                "Run test sequence",
                "Record all measurements",
                "Evaluate against acceptance criteria",
            ],
            data_recording="Record all measurements in the test log",
            pass_fail_criteria=acceptance_criteria,
        )
        steps = [
            SOPStep(
                step_number=i + 1,
                action=s,
                verification_method="Data captured",
                acceptance_criteria=acceptance_criteria,
            )
            for i, s in enumerate(procedure.procedure_steps)
        ]
        param_lines = [f"{n}: {format_parameter(p)}" for n, p in params.items()]
        sop = self.renderer.render(
            title=f"Testing Procedure - {test_name}",
            parameters=params,
            steps=steps,
            equipment=[{"name": e} for e in equipment_required],
            quality_control=param_lines + [f"Acceptance: {acceptance_criteria}"],
            safety_protocols=["Follow equipment safety procedures"],
            description=f"{test_type} test for {test_name}",
            metadata={"sop_type": SOPType.TESTING.value},
        )
        return sop

    async def generate_maintenance_sop(
        self, equipment_specs: List[Dict[str, Any]]
    ) -> StandardOperatingProcedure:
        schedules = self.llm4ias.generate_maintenance_schedule(equipment_specs)
        qc = [f"{s['equipment_name']}: {s['maintenance_type']} ({s['frequency']})" for s in schedules]
        sop = self.renderer.render(
            title="Maintenance Procedure",
            steps=[
                SOPStep(
                    step_number=i + 1,
                    action=f"{s['equipment_name']} - " + "; ".join(s["procedures"]),
                    verification_method="Function check after service",
                    acceptance_criteria="Equipment operates within spec",
                )
                for i, s in enumerate(schedules)
            ],
            quality_control=qc,
            safety_protocols=["Lockout/Tagout before service", "Use PPE"],
            description="Preventive maintenance SOP",
            metadata={"sop_type": SOPType.MAINTENANCE.value},
        )
        return sop

    async def generate_complete_invention_sop(
        self,
        invention_spec: Dict[str, Any],
        include_all_sections: bool = True,
    ) -> StandardOperatingProcedure:
        product_name = invention_spec.get("name", "Invention")
        sections_text: List[str] = [f"Standard Operating Procedure - {product_name}"]
        equipment = invention_spec.get("equipment", [])

        if include_all_sections and "manufacturing" in invention_spec:
            mfg = await self.generate_manufacturing_sop(
                product_name,
                invention_spec["manufacturing"],
                equipment,
                IndustryStandard.ISO_9001,
            )
            sections_text.append(mfg.to_markdown())

        if include_all_sections and "assembly" in invention_spec:
            asm = await self.generate_assembly_sop(
                product_name,
                invention_spec["assembly"].get("bom", []),
                invention_spec["assembly"].get("sequence", []),
                invention_spec["assembly"].get("tools", []),
            )
            sections_text.append(asm.to_markdown())

        if include_all_sections and "testing" in invention_spec:
            t = invention_spec["testing"]
            test = await self.generate_testing_sop(
                product_name + " Test",
                t.get("type", "Functional"),
                t.get("parameters", {}),
                t.get("acceptance", "Pass all tests"),
                t.get("equipment", []),
            )
            sections_text.append(test.to_markdown())

        combined = "\n\n".join(sections_text)
        # Re-render as a single SOP via token substitution on the combined text.
        sop = self.renderer.render(
            title=f"Standard Operating Procedure - {product_name}",
            steps=[
                SOPStep(
                    step_number=1,
                    action="Follow the generated section documents in order",
                    verification_method="Section completion sign-off",
                    acceptance_criteria="All sections executed successfully",
                )
            ],
            preconditions=["All sub-procedure documents reviewed"],
            description=combined[:2000],
            metadata={
                "document_number": f"SOP-{abs(hash(product_name)) % 10000:04d}",
                "revision": "1.0",
                "approval_status": "Draft",
                "sections": list(invention_spec.keys()),
            },
        )
        return sop


async def generate_industrial_sop(
    invention_goal: str,
    domain: str = "manufacturing",
    specifications: Optional[Dict[str, Any]] = None,
    include_qc: bool = True,
    include_safety: bool = True,
) -> StandardOperatingProcedure:
    """Convenience function for industrial SOP generation (service-free)."""
    generator = EnhancedSOPGenerator()
    spec = specifications or {}
    spec["name"] = invention_goal
    spec["domain"] = domain
    equipment = spec.get("equipment", ["Workstation", "Measurement tools", "Assembly tools"])
    if "manufacturing" in spec or "process_steps" in spec:
        return await generator.generate_manufacturing_sop(
            invention_goal, spec, equipment,
            include_qc=include_qc, include_safety=include_safety,
        )
    return await generator.generate_complete_invention_sop(spec)


__all__ = [
    "EnhancedSOPGenerator",
    "LLM4IASIntegration",
    "AssemblyInstruction",
    "TestProcedure",
    "SOPType",
    "IndustryStandard",
    "generate_industrial_sop",
]
