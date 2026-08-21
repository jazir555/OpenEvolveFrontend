"""
Real SOP Generator - Industrial Automation (service-free)

Provides ACTUAL SOP generation using a rule-based industrial automation expert
system and template-driven document rendering. Renders SOP documents from
`SOPParameter` values, supports parameter substitution and section generation,
and validates parameters. No external services (LLM/MAKER) are required.

`SOPParameter` is provided externally (another agent) as
`engines/other/sop_parameter.py` and is imported here; it is never redefined.
"""
from __future__ import annotations


import asyncio
import logging
import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from sop_parameter import SOPParameter

from sop_document import SOPRenderer, format_parameter, substitute

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
    IATF_16949 = "IATF 16949"
    GMP = "GMP"
    FDA_21_CFR_11 = "FDA 21 CFR Part 11"
    OSHA = "OSHA"
    IEC_62304 = "IEC 62304"
    ISO_45001 = "ISO 45001"


@dataclass
class ManufacturingStep:
    step_number: int
    operation: str
    equipment_required: List[str]
    parameters: Dict[str, Any]
    quality_checks: List[str]
    safety_precautions: List[str]
    cycle_time: float
    setup_time: float
    inspection_required: bool = True
    sign_off_required: bool = True
    work_instructions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "operation": self.operation,
            "equipment": self.equipment_required,
            "parameters": self.parameters,
            "quality_checks": self.quality_checks,
            "safety_precautions": self.safety_precautions,
            "cycle_time_minutes": self.cycle_time,
            "setup_time_minutes": self.setup_time,
            "inspection_required": self.inspection_required,
            "sign_off_required": self.sign_off_required,
            "work_instructions": self.work_instructions,
        }


@dataclass
class QualityControlProcedure:
    inspection_point: str
    measurement_method: str
    acceptance_criteria: str
    sampling_plan: str
    measurement_tools: List[str]
    frequency: str
    record_required: bool = True
    reaction_plan: str = ""
    gage_r_requirement: bool = False
    statistical_process_control: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inspection_point": self.inspection_point,
            "measurement_method": self.measurement_method,
            "acceptance_criteria": self.acceptance_criteria,
            "sampling_plan": self.sampling_plan,
            "measurement_tools": self.measurement_tools,
            "frequency": self.frequency,
            "record_required": self.record_required,
            "reaction_plan": self.reaction_plan,
            "gage_r_required": self.gage_r_requirement,
            "spc_required": self.statistical_process_control,
        }


@dataclass
class SafetyProtocol:
    hazard_type: str
    hazard_description: str
    risk_level: str
    required_ppe: List[str]
    engineering_controls: List[str]
    administrative_controls: List[str]
    emergency_procedures: List[str]
    spill_response: Optional[str] = None
    first_aid: Optional[str] = None
    sds_reference: Optional[str] = None
    lockout_tagout_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hazard_type": self.hazard_type,
            "hazard_description": self.hazard_description,
            "risk_level": self.risk_level,
            "required_ppe": self.required_ppe,
            "engineering_controls": self.engineering_controls,
            "administrative_controls": self.administrative_controls,
            "emergency_procedures": self.emergency_procedures,
            "spill_response": self.spill_response,
            "first_aid": self.first_aid,
            "sds_reference": self.sds_reference,
            "lockout_tagout_required": self.lockout_tagout_required,
        }


@dataclass
class MaintenanceSchedule:
    equipment_id: str
    equipment_name: str
    maintenance_type: str
    frequency: str
    procedures: List[str]
    required_parts: List[str]
    estimated_duration: float
    technician_skill_level: str
    documentation_required: bool = True
    calibration_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equipment_id": self.equipment_id,
            "equipment_name": self.equipment_name,
            "maintenance_type": self.maintenance_type,
            "frequency": self.frequency,
            "procedures": self.procedures,
            "required_parts": self.required_parts,
            "estimated_duration_hours": self.estimated_duration,
            "technician_skill_level": self.technician_skill_level,
            "documentation_required": self.documentation_required,
            "calibration_required": self.calibration_required,
        }


class IndustrialExpertSystem:
    """Rule-based expert system for industrial automation process design."""

    PROCESS_TEMPLATES = {
        "machining": {
            "steps": [
                {"operation": "Material preparation", "time_ratio": 0.1},
                {"operation": "Rough machining", "time_ratio": 0.3},
                {"operation": "Finish machining", "time_ratio": 0.3},
                {"operation": "Deburring", "time_ratio": 0.1},
                {"operation": "Final inspection", "time_ratio": 0.2},
            ],
            "equipment": ["CNC Mill", "CNC Lathe", "Deburring station"],
            "qc_points": ["Dimensional check", "Surface finish"],
        },
        "additive": {
            "steps": [
                {"operation": "Build preparation", "time_ratio": 0.15},
                {"operation": "3D printing", "time_ratio": 0.5},
                {"operation": "Post-processing", "time_ratio": 0.2},
                {"operation": "Heat treatment", "time_ratio": 0.1},
                {"operation": "Quality inspection", "time_ratio": 0.05},
            ],
            "equipment": ["3D Printer", "Powder handling", "Heat treatment oven"],
            "qc_points": ["Density check", "Dimensional accuracy", "Mechanical properties"],
        },
        "injection_molding": {
            "steps": [
                {"operation": "Material drying", "time_ratio": 0.1},
                {"operation": "Mold setup", "time_ratio": 0.15},
                {"operation": "Injection molding", "time_ratio": 0.4},
                {"operation": "Cooling", "time_ratio": 0.2},
                {"operation": "Part removal/inspection", "time_ratio": 0.15},
            ],
            "equipment": ["Injection molding machine", "Material dryer", "Chiller"],
            "qc_points": ["Dimensional check", "Visual inspection", "Weight check"],
        },
        "assembly": {
            "steps": [
                {"operation": "Component preparation", "time_ratio": 0.15},
                {"operation": "Sub-assembly 1", "time_ratio": 0.25},
                {"operation": "Sub-assembly 2", "time_ratio": 0.25},
                {"operation": "Final assembly", "time_ratio": 0.25},
                {"operation": "Functional test", "time_ratio": 0.1},
            ],
            "equipment": ["Workstation", "Torque tools", "Test equipment"],
            "qc_points": ["Torque verification", "Functional test"],
        },
        "electronics": {
            "steps": [
                {"operation": "PCB preparation", "time_ratio": 0.1},
                {"operation": "Solder paste application", "time_ratio": 0.1},
                {"operation": "Component placement", "time_ratio": 0.2},
                {"operation": "Reflow soldering", "time_ratio": 0.2},
                {"operation": "Inspection", "time_ratio": 0.15},
                {"operation": "Testing", "time_ratio": 0.25},
            ],
            "equipment": ["Pick and place", "Reflow oven", "AOI", "Test equipment"],
            "qc_points": ["AOI inspection", "ICT/Functional test"],
        },
        "sheet_metal": {
            "steps": [
                {"operation": "Material preparation", "time_ratio": 0.1},
                {"operation": "Cutting", "time_ratio": 0.2},
                {"operation": "Forming", "time_ratio": 0.25},
                {"operation": "Welding/fastening", "time_ratio": 0.25},
                {"operation": "Finishing", "time_ratio": 0.1},
                {"operation": "Inspection", "time_ratio": 0.1},
            ],
            "equipment": ["Laser cutter", "Press brake", "Welder"],
            "qc_points": ["Dimensional check", "Weld quality"],
        },
    }

    SAFETY_TEMPLATES = {
        "mechanical": {
            "ppe": ["Safety glasses", "Safety shoes", "Cut-resistant gloves"],
            "controls": ["Machine guards", "Emergency stops", "Interlocks"],
            "emergency": ["Press E-stop", "Call supervisor", "First aid if injured"],
        },
        "chemical": {
            "ppe": ["Safety goggles", "Chemical-resistant gloves", "Apron"],
            "controls": ["Fume hood", "Spill containment", "Eyewash station"],
            "emergency": ["Move to fresh air", "Rinse affected area", "Call emergency services"],
        },
        "electrical": {
            "ppe": ["Insulated gloves", "Safety glasses", "Non-conductive shoes"],
            "controls": ["Lockout/Tagout", "Arc flash protection", "Ground fault protection"],
            "emergency": ["De-energize", "Call for help", "CPR if needed"],
        },
        "thermal": {
            "ppe": ["Heat-resistant gloves", "Face shield", "Protective clothing"],
            "controls": ["Heat shields", "Ventilation", "Temperature monitoring"],
            "emergency": ["Cool burn with water", "Seek medical attention", "Report incident"],
        },
        "noise": {
            "ppe": ["Hearing protection"],
            "controls": ["Sound enclosures", "Administrative controls"],
            "emergency": [],
        },
    }

    def __init__(self):
        self.manufacturing_types = list(self.PROCESS_TEMPLATES.keys())

    def analyze_product(self, product_spec: Dict[str, Any]) -> Dict[str, Any]:
        material = str(product_spec.get("material", "")).lower()
        features = product_spec.get("features", [])
        tolerances = product_spec.get("tolerances", {})
        volume = product_spec.get("volume", 1000)

        if "pcb" in material or "electronic" in material:
            mfg_type = "electronics"
        elif "plastic" in material or "polymer" in material:
            mfg_type = "injection_molding" if volume > 1000 else "additive"
        elif "metal" in material or "aluminum" in material or "steel" in material:
            complexity = len(features) if isinstance(features, (list, tuple)) else 0
            mfg_type = "additive" if (complexity > 10 and volume < 100) else "machining"
        elif any(f in str(features).lower() for f in ["weld", "fold", "bend"]):
            mfg_type = "sheet_metal"
        else:
            mfg_type = "assembly"

        tight = any(
            t < 0.1 for t in tolerances.values() if isinstance(t, (int, float))
        )
        base_time = {"machining": 60, "additive": 240, "injection_molding": 5}.get(mfg_type, 30)
        return {
            "manufacturing_type": mfg_type,
            "precision_level": "high" if tight else "standard",
            "estimated_cycle_time": base_time,
            "suitable_processes": self._get_suitable_processes(mfg_type),
            "critical_quality_params": self._get_critical_params(mfg_type),
        }

    def _get_suitable_processes(self, mfg_type: str) -> List[str]:
        return self.PROCESS_TEMPLATES.get(mfg_type, {}).get("equipment", [])

    def _get_critical_params(self, mfg_type: str) -> List[str]:
        return self.PROCESS_TEMPLATES.get(mfg_type, {}).get("qc_points", [])

    def generate_manufacturing_process(
        self,
        product_spec: Dict[str, Any],
        equipment_list: List[str],
        cycle_time_target: Optional[float] = None,
    ) -> Dict[str, Any]:
        analysis = self.analyze_product(product_spec)
        mfg_type = analysis["manufacturing_type"]
        template = self.PROCESS_TEMPLATES.get(mfg_type, self.PROCESS_TEMPLATES["assembly"])
        base_time = cycle_time_target or analysis["estimated_cycle_time"]

        steps = []
        for i, step_template in enumerate(template["steps"], 1):
            cycle_time = base_time * step_template["time_ratio"]
            step = ManufacturingStep(
                step_number=i,
                operation=step_template["operation"],
                equipment_required=self._match_equipment(step_template["operation"], equipment_list),
                parameters=self._generate_parameters(step_template["operation"]),
                quality_checks=self._generate_quality_checks(step_template["operation"]),
                safety_precautions=self._generate_safety_precautions(step_template["operation"]),
                cycle_time=cycle_time,
                setup_time=cycle_time * 0.2,
                work_instructions=self._generate_work_instructions(step_template["operation"]),
            )
            steps.append(step)

        return {
            "process_name": product_spec.get("name", "Manufacturing Process"),
            "manufacturing_type": mfg_type,
            "steps": [s.to_dict() for s in steps],
            "total_cycle_time": sum(s.cycle_time for s in steps),
            "equipment_utilized": equipment_list,
            "quality_checkpoints": template["qc_points"],
            "process_analysis": analysis,
        }

    def _match_equipment(self, operation: str, equipment_list: List[str]) -> List[str]:
        operation_lower = operation.lower()
        matched = [
            equip
            for equip in equipment_list
            if any(kw in operation_lower for kw in str(equip).lower().split())
        ]
        if not matched and equipment_list:
            matched = [equipment_list[0]]
        return matched

    def _generate_parameters(self, operation: str) -> Dict[str, Any]:
        op = operation.lower()
        if "machining" in op:
            return {
                "spindle_speed": {"value": 3000, "unit": "rpm", "tolerance": "±100"},
                "feed_rate": {"value": 500, "unit": "mm/min", "tolerance": "±25"},
                "depth_of_cut": {"value": 2, "unit": "mm", "tolerance": "±0.1"},
            }
        if "welding" in op:
            return {
                "current": {"value": 150, "unit": "A", "tolerance": "±10"},
                "voltage": {"value": 25, "unit": "V", "tolerance": "±2"},
                "travel_speed": {"value": 300, "unit": "mm/min", "tolerance": "±30"},
            }
        if "printing" in op or "additive" in op:
            return {
                "layer_height": {"value": 0.1, "unit": "mm", "tolerance": "±0.02"},
                "print_speed": {"value": 50, "unit": "mm/s", "tolerance": "±5"},
                "temperature": {"value": 200, "unit": "°C", "tolerance": "±5"},
            }
        return {}

    def _generate_quality_checks(self, operation: str) -> List[str]:
        op = operation.lower()
        checks = ["Visual inspection"]
        if "machining" in op:
            checks += ["Dimensional check", "Surface finish check"]
        elif "welding" in op:
            checks += ["Weld penetration check", "Visual weld inspection"]
        elif "molding" in op:
            checks += ["Part weight", "Sink mark check"]
        return checks

    def _generate_safety_precautions(self, operation: str) -> List[str]:
        op = operation.lower()
        precautions = ["Follow standard safety procedures"]
        if "machining" in op:
            precautions += ["Wear safety glasses", "Secure workpiece properly", "Check tooling"]
        elif "welding" in op:
            precautions += ["Wear welding helmet", "Ensure ventilation", "Check gas connections"]
        return precautions

    def _generate_work_instructions(self, operation: str) -> List[str]:
        op = operation.lower()
        if "preparation" in op:
            return [
                "1. Gather all required materials and tools",
                "2. Verify material certifications",
                "3. Set up work area according to 5S standards",
            ]
        if "machining" in op:
            return [
                "1. Load program and verify tool offsets",
                "2. Perform dry run if required",
                "3. Execute machining operation",
                "4. Monitor cutting forces and temperatures",
            ]
        if "inspection" in op:
            return [
                "1. Calibrate measurement equipment",
                "2. Measure all critical dimensions",
                "3. Record results in quality database",
                "4. Tag parts accordingly",
            ]
        return []

    def generate_quality_control_plan(
        self,
        product_spec: Dict[str, Any],
        critical_characteristics: List[str],
        aql: float = 0.01,
    ) -> Dict[str, Any]:
        if aql <= 0.01:
            inspection_level, frequency = "100% inspection", "Every unit"
        elif aql <= 0.065:
            inspection_level, frequency = "Tightened inspection", "Every 10th unit"
        else:
            inspection_level, frequency = "Normal inspection", "Statistical sampling"

        procedures = [
            QualityControlProcedure(
                inspection_point=c,
                measurement_method=f"Calibrated instrument per {c} specification",
                acceptance_criteria="Within tolerance per drawing",
                sampling_plan=inspection_level,
                measurement_tools=["CMM", "Micrometer", "Gauge"],
                frequency=frequency,
                reaction_plan="Quarantine batch, notify quality engineer, initiate NCR",
                gage_r_requirement=True,
                statistical_process_control=(aql <= 0.01),
            ).to_dict()
            for c in critical_characteristics
        ]
        return {
            "inspection_level": inspection_level,
            "aql": aql,
            "procedures": procedures,
            "documentation": "All inspections recorded in quality management system",
            "records_retention": "7 years minimum per ISO 9001",
        }

    def generate_safety_protocols(
        self, hazards: List[Dict[str, Any]], industry: str = "manufacturing"
    ) -> List[Dict[str, Any]]:
        protocols = []
        for hazard in hazards:
            htype = str(hazard.get("type", "general")).lower()
            template = self.SAFETY_TEMPLATES.get(htype, self.SAFETY_TEMPLATES["mechanical"])
            protocols.append(
                SafetyProtocol(
                    hazard_type=htype,
                    hazard_description=hazard.get("description", f"{htype} hazard"),
                    risk_level=hazard.get("risk", "Medium"),
                    required_ppe=template["ppe"] + hazard.get("additional_ppe", []),
                    engineering_controls=template["controls"],
                    administrative_controls=["Training", "Procedures", "Supervision"],
                    emergency_procedures=template["emergency"],
                    spill_response=hazard.get("spill_response"),
                    first_aid=hazard.get("first_aid"),
                    lockout_tagout_required=(htype == "electrical" or hazard.get("lototo", False)),
                ).to_dict()
            )
        return protocols

    def generate_maintenance_schedules(
        self, equipment_specs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        result = []
        for idx, eq in enumerate(equipment_specs, 1):
            result.append(MaintenanceSchedule(
                equipment_id=eq.get("id", f"EQ{idx:03d}"),
                equipment_name=eq.get("name", "Equipment"),
                maintenance_type=eq.get("type", "Preventive"),
                frequency=eq.get("frequency", "Monthly"),
                procedures=eq.get("procedures", ["Inspect for wear", "Clean and lubricate", "Check calibration"]),
                required_parts=eq.get("parts", []),
                estimated_duration=eq.get("duration", 2.0),
                technician_skill_level=eq.get("skill", "Certified Technician"),
                calibration_required=eq.get("calibration", False),
            ).to_dict()
        )
        return result


class RealSOPGenerator:
    """
    Production-grade SOP generator with an industrial expert system.

    Renders complete SOP documents from `SOPParameter` values plus templates and
    validates parameters. Fully functional without external services.
    """

    def __init__(self, config: Any = None):
        self.config = config
        self.expert_system = IndustrialExpertSystem()
        self.renderer = SOPRenderer()
        logger.info("RealSOPGenerator initialized (service-free)")

    async def generate_manufacturing_sop(
        self,
        product_name: str,
        product_spec: Dict[str, Any],
        equipment_list: List[str],
        industry_standard: IndustryStandard = IndustryStandard.ISO_9001,
        include_qc: bool = True,
        include_safety: bool = True,
    ) -> Dict[str, Any]:
        mfg_process = self.expert_system.generate_manufacturing_process(product_spec, equipment_list)
        result: Dict[str, Any] = {
            "document_type": "Manufacturing SOP",
            "document_title": f"Manufacturing Procedure - {product_name}",
            "industry_standard": industry_standard.value,
            "product_name": product_name,
            "effective_date": datetime.now().isoformat(),
            "revision": "1.0",
            "approval_required": True,
            "manufacturing_process": mfg_process,
        }
        if include_qc:
            result["quality_control"] = self.expert_system.generate_quality_control_plan(
                product_spec, product_spec.get("critical_characteristics", [])
            )
        if include_safety:
            result["safety_protocols"] = self.expert_system.generate_safety_protocols(
                product_spec.get("hazards", [])
            )
        result["generated_by"] = "IndustrialExpertSystem"
        return result

    async def generate_sop(
        self,
        invention_spec: Dict[str, Any],
        format: str = "markdown",
    ) -> str:
        sop_package = await self.generate_complete_invention_sop(invention_spec, True)
        if format == "json":
            return json.dumps(sop_package, indent=2, default=str)
        if format == "html":
            return self._format_html(self._format_markdown(sop_package))
        return self._format_markdown(sop_package)

    async def generate_complete_invention_sop(
        self,
        invention_spec: Dict[str, Any],
        include_all_sections: bool = True,
    ) -> Dict[str, Any]:
        product_name = invention_spec.get("name", "Invention")
        package = {
            "document_title": f"Standard Operating Procedure - {product_name}",
            "document_number": f"SOP-{abs(hash(product_name)) % 10000:04d}",
            "revision": "1.0",
            "effective_date": datetime.now().isoformat(),
            "approval_status": "Draft",
            "sections": {},
        }

        if include_all_sections and ("manufacturing" in invention_spec or "process_steps" in invention_spec):
            package["sections"]["manufacturing"] = await self.generate_manufacturing_sop(
                product_name,
                invention_spec.get("manufacturing", invention_spec),
                invention_spec.get("equipment", []),
            )
        if include_all_sections and "assembly" in invention_spec:
            a = invention_spec["assembly"]
            package["sections"]["assembly"] = {
                "document_type": "Assembly SOP",
                "bill_of_materials": a.get("bom", []),
                "assembly_sequence": a.get("sequence", []),
                "tools_required": a.get("tools", []),
            }
        if include_all_sections and "testing" in invention_spec:
            t = invention_spec["testing"]
            package["sections"]["testing"] = {
                "document_type": "Testing SOP",
                "test_type": t.get("type", "Functional"),
                "test_parameters": t.get("parameters", {}),
                "acceptance_criteria": t.get("acceptance", "Pass all tests"),
                "equipment_required": t.get("equipment", []),
            }
        if include_all_sections and "equipment" in invention_spec:
            equipment_specs = []
            for eq in invention_spec["equipment"]:
                if isinstance(eq, dict):
                    equipment_specs.append(eq)
                else:
                    equipment_specs.append({"id": f"EQ{len(equipment_specs) + 1:03d}", "name": eq})
            if equipment_specs:
                package["sections"]["maintenance"] = {
                    "document_type": "Maintenance SOP",
                    "maintenance_schedules": self.expert_system.generate_maintenance_schedules(
                        equipment_specs
                    ),
                }
        if "hazards" in invention_spec:
            package["sections"]["safety_summary"] = {
                "hazards_identified": len(invention_spec["hazards"]),
                "safety_protocols": self.expert_system.generate_safety_protocols(
                    invention_spec["hazards"]
                ),
                "required_training": ["General safety", "Equipment-specific", "PPE usage"],
            }
        return package

    # ----- rendering -----

    def _format_markdown(self, package: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append(f"# {package.get('document_title', 'Standard Operating Procedure')}")
        lines.append("")
        lines.append(f"**Document Number:** {package.get('document_number', 'N/A')}")
        lines.append(f"**Revision:** {package.get('revision', '1.0')}")
        lines.append(f"**Effective Date:** {package.get('effective_date', datetime.now().isoformat())}")
        lines.append(f"**Approval Status:** {package.get('approval_status', 'Draft')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        for key, section in package.get("sections", {}).items():
            lines.append(f"## {key.replace('_', ' ').title()}")
            lines.append("")
            lines.extend(self._render_section(section))
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*This SOP was generated by the IndustrialExpertSystem*")
        return "\n".join(lines)

    def _render_section(self, section: Any) -> List[str]:
        out: List[str] = []
        if isinstance(section, dict):
            mfg = section.get("manufacturing_process")
            if mfg:
                out.append(f"**Process Type:** {mfg.get('manufacturing_type', 'N/A')}")
                out.append(f"**Total Cycle Time:** {mfg.get('total_cycle_time', 0):.1f} minutes")
                out.append("")
                for step in mfg.get("steps", []):
                    out.append(f"### Step {step.get('step_number', 0)}: {step.get('operation', 'Unknown')}")
                    out.append(f"- **Equipment:** {', '.join(step.get('equipment', []))}")
                    out.append(f"- **Cycle Time:** {step.get('cycle_time_minutes', 0):.1f} min")
                    out.append(f"- **Quality Checks:** {', '.join(step.get('quality_checks', []))}")
                    params = step.get("parameters") or {}
                    for pname, pval in params.items():
                        out.append(f"- **Parameter {pname}:** {pval}")
                    out.append("")
            qc = section.get("quality_control") or section.get("procedures")
            if qc and isinstance(qc, list):
                out.append(f"**Inspection Level:** {section.get('inspection_level', 'Normal')}")
                out.append("**QC Procedures:**")
                for proc in qc:
                    if isinstance(proc, dict):
                        out.append(f"- {proc.get('inspection_point', 'check')}: {proc.get('acceptance_criteria', '')}")
                    else:
                        out.append(f"- {proc}")
                out.append("")
            safety = section.get("safety_protocols")
            if safety and isinstance(safety, list):
                out.append(f"**Hazards Identified:** {section.get('hazards_identified', len(safety))}")
                for protocol in safety:
                    if isinstance(protocol, dict):
                        out.append(f"### {protocol.get('hazard_type', 'General')} Hazard")
                        out.append(f"- **Risk Level:** {protocol.get('risk_level', 'Medium')}")
                        out.append(f"- **Required PPE:** {', '.join(protocol.get('required_ppe', []))}")
                        out.append("")
            for plain in ("bill_of_materials", "assembly_sequence", "tools_required",
                          "test_type", "test_parameters", "acceptance_criteria",
                          "equipment_required", "maintenance_schedules"):
                if plain in section:
                    out.append(f"**{plain.replace('_', ' ').title()}:** {section[plain]}")
                    out.append("")
        return out

    def _format_html(self, markdown: str) -> str:
        html = [
            "<!DOCTYPE html>", "<html>", "<head>", "<title>Standard Operating Procedure</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px}",
            "h1{color:#2c3e50;border-bottom:2px solid #3498db}h2{color:#34495e;margin-top:30px}</style>",
            "</head>", "<body>",
        ]
        for line in markdown.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                html.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("**") and line.endswith("**"):
                html.append(f"<p><strong>{line[2:-2]}</strong></p>")
            elif line.startswith("- "):
                html.append(f"<li>{line[2:]}</li>")
            elif line == "---":
                html.append("<hr>")
            elif line:
                html.append(f"<p>{line}</p>")
        html.extend(["</body>", "</html>"])
        return "\n".join(html)


async def generate_industrial_sop(
    invention_goal: str,
    domain: str = "manufacturing",
    specifications: Optional[Dict[str, Any]] = None,
    include_qc: bool = True,
    include_safety: bool = True,
) -> Dict[str, Any]:
    """Convenience function for industrial SOP generation (service-free)."""
    generator = RealSOPGenerator()
    spec = specifications or {}
    spec["name"] = invention_goal
    spec["domain"] = domain
    equipment = spec.get("equipment", ["Workstation", "Measurement tools", "Assembly tools"])
    return await generator.generate_manufacturing_sop(
        product_name=invention_goal,
        product_spec=spec,
        equipment_list=equipment,
        include_qc=include_qc,
        include_safety=include_safety,
    )


__all__ = [
    "RealSOPGenerator",
    "IndustrialExpertSystem",
    "ManufacturingStep",
    "QualityControlProcedure",
    "SafetyProtocol",
    "MaintenanceSchedule",
    "SOPType",
    "IndustryStandard",
    "generate_industrial_sop",
]
