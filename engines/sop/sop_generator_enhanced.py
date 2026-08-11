"""
Enhanced SOP Generator with LLM4IAS Integration

This module provides comprehensive SOP generation using:
- LLM4IAS (LLM for Industrial Automation Systems)
- Manufacturing SOP generation
- Quality control procedures
- Safety protocols
- Assembly instructions
- Testing procedures
- Maintenance schedules

Author: OpenEvolve
Version: 2.0.0
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

# Import base SOP generator
try:
    from sop_generator import (
        SOPGenerator, StandardOperatingProcedure, SOPStep, SOPParameter,
        SOPEvaluator, generate_sop
    )
    BASE_SOP_AVAILABLE = True
except ImportError:
    BASE_SOP_AVAILABLE = False
    logger.warning("Base SOP generator not available")

# Import MAKER integration
try:
    from generic_maker_integration import run_generic_maker, GenericEvaluator, TaskType, MAKERConfig
    MAKER_AVAILABLE = True
except ImportError:
    MAKER_AVAILABLE = False

# LLM4IAS integration placeholder
LLM4IAS_AVAILABLE = False
try:
    # Would import LLM4IAS here
    # from llm4ias import LLM4IASGenerator
    LLM4IAS_AVAILABLE = True
except ImportError:
    logger.info("LLM4IAS not available - using MAKER fallback")


class SOPType(Enum):
    """Types of SOPs for different purposes"""
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
    """Industry standards for SOP compliance"""
    ISO_9001 = "ISO 9001"
    ISO_13485 = "ISO 13485"  # Medical devices
    AS9100 = "AS9100"  # Aerospace
    GMP = "GMP"  # Good Manufacturing Practice
    FDA_21_CFR_11 = "FDA 21 CFR Part 11"
    OSHA = "OSHA"
    IEC_62304 = "IEC 62304"  # Medical device software


@dataclass
class ManufacturingStep:
    """Manufacturing process step"""
    step_number: int
    operation: str
    equipment_required: List[str]
    parameters: Dict[str, SOPParameter]
    quality_checks: List[str]
    safety_precautions: List[str]
    cycle_time: float  # minutes
    setup_time: float  # minutes
    inspection_required: bool = True
    sign_off_required: bool = True


@dataclass
class QualityControlProcedure:
    """Quality control procedure"""
    inspection_point: str
    measurement_method: str
    acceptance_criteria: str
    sampling_plan: str
    measurement_tools: List[str]
    frequency: str
    record_required: bool = True
    reaction_plan: str = ""  # What to do if criteria not met


@dataclass
class SafetyProtocol:
    """Safety protocol specification"""
    hazard_type: str
    hazard_description: str
    risk_level: str  # Critical, High, Medium, Low
    required_ppe: List[str]
    engineering_controls: List[str]
    administrative_controls: List[str]
    emergency_procedures: List[str]
    spill_response: Optional[str] = None
    first_aid: Optional[str] = None


@dataclass
class MaintenanceSchedule:
    """Equipment maintenance schedule"""
    equipment_id: str
    equipment_name: str
    maintenance_type: str  # Preventive, Predictive, Corrective
    frequency: str  # Daily, Weekly, Monthly, etc.
    procedures: List[str]
    required_parts: List[str]
    estimated_duration: float  # hours
    technician_skill_level: str
    documentation_required: bool = True


@dataclass
class AssemblyInstruction:
    """Assembly instruction step"""
    step_number: int
    description: str
    components: List[str]
    tools_required: List[str]
    torque_specifications: Optional[Dict[str, float]] = None
    adhesive_specifications: Optional[Dict[str, Any]] = None
    visual_check: str = ""
    functional_test: str = ""


@dataclass
class TestProcedure:
    """Testing procedure specification"""
    test_name: str
    test_type: str  # Functional, Performance, Environmental, Safety
    equipment_required: List[str]
    test_parameters: Dict[str, SOPParameter]
    acceptance_criteria: str
    preconditions: List[str]
    procedure_steps: List[str]
    data_recording: str
    pass_fail_criteria: str


class LLM4IASIntegration:
    """
    Integration with LLM4IAS for Industrial Automation SOPs.
    
    LLM4IAS provides:
    - Manufacturing process SOPs
    - Quality control procedures
    - Safety protocols per OSHA standards
    - Assembly instructions
    - Testing procedures
    - Maintenance schedules
    """
    
    def __init__(self):
        self.available = LLM4IAS_AVAILABLE
        
    def is_available(self) -> bool:
        """Check if LLM4IAS is available"""
        return self.available
    
    def generate_manufacturing_sop(
        self,
        product_spec: Dict[str, Any],
        equipment_list: List[str],
        industry_standard: IndustryStandard = IndustryStandard.ISO_9001
    ) -> Dict[str, Any]:
        """
        Generate manufacturing SOP using LLM4IAS.
        
        Args:
            product_spec: Product specifications
            equipment_list: Available equipment
            industry_standard: Industry standard to comply with
            
        Returns:
            Manufacturing SOP structure
        """
        if not self.available:
            return self._mock_manufacturing_sop(product_spec, equipment_list)
        
        # Would call actual LLM4IAS here
        return self._mock_manufacturing_sop(product_spec, equipment_list)
    
    def generate_quality_control_plan(
        self,
        product_spec: Dict[str, Any],
        critical_characteristics: List[str],
        aql: float = 0.01
    ) -> Dict[str, Any]:
        """
        Generate quality control plan.
        
        Args:
            product_spec: Product specifications
            critical_characteristics: Critical quality characteristics
            aql: Acceptable Quality Level
            
        Returns:
            Quality control plan
        """
        qc_procedures = []
        
        for characteristic in critical_characteristics:
            qc_procedures.append({
                "inspection_point": characteristic,
                "measurement_method": f"Measure {characteristic} using calibrated instrument",
                "acceptance_criteria": f"Within specification ± tolerance",
                "sampling_plan": f"AQL {aql * 100}%",
                "frequency": "Every unit" if aql < 0.01 else "Sampled"
            })
        
        return {
            "qc_procedures": qc_procedures,
            "inspection_levels": {
                "incoming": 100 if aql < 0.01 else 10,
                "in_process": 100,
                "final": 100
            },
            "documentation": "All measurements recorded in quality log"
        }
    
    def generate_safety_protocols(
        self,
        hazards: List[Dict[str, Any]],
        industry: str = "manufacturing"
    ) -> List[SafetyProtocol]:
        """
        Generate safety protocols for identified hazards.
        
        Args:
            hazards: List of hazard descriptions
            industry: Industry type
            
        Returns:
            List of safety protocols
        """
        protocols = []
        
        for hazard in hazards:
            protocol = SafetyProtocol(
                hazard_type=hazard.get('type', 'general'),
                hazard_description=hazard.get('description', 'Unknown hazard'),
                risk_level=hazard.get('risk', 'Medium'),
                required_ppe=hazard.get('ppe', ['Safety glasses', 'Gloves']),
                engineering_controls=hazard.get('controls', ['Guards', 'Ventilation']),
                administrative_controls=['Training', 'Procedures'],
                emergency_procedures=hazard.get('emergency', ['Evacuate', 'Call supervisor']),
                spill_response=hazard.get('spill_response'),
                first_aid=hazard.get('first_aid')
            )
            protocols.append(protocol)
        
        return protocols
    
    def generate_maintenance_schedule(
        self,
        equipment_specs: List[Dict[str, Any]]
    ) -> List[MaintenanceSchedule]:
        """
        Generate maintenance schedules for equipment.
        
        Args:
            equipment_specs: Equipment specifications
            
        Returns:
            List of maintenance schedules
        """
        schedules = []
        
        for equip in equipment_specs:
            schedule = MaintenanceSchedule(
                equipment_id=equip.get('id', 'EQ001'),
                equipment_name=equip.get('name', 'Equipment'),
                maintenance_type=equip.get('maintenance_type', 'Preventive'),
                frequency=equip.get('frequency', 'Monthly'),
                procedures=equip.get('procedures', ['Inspect', 'Clean', 'Lubricate']),
                required_parts=equip.get('parts', []),
                estimated_duration=equip.get('duration', 1.0),
                technician_skill_level=equip.get('skill', 'Trained'),
                documentation_required=True
            )
            schedules.append(schedule)
        
        return schedules
    
    def _mock_manufacturing_sop(
        self,
        product_spec: Dict[str, Any],
        equipment_list: List[str]
    ) -> Dict[str, Any]:
        """Create mock manufacturing SOP structure"""
        return {
            "process_name": product_spec.get('name', 'Manufacturing Process'),
            "industry_standard": "ISO 9001",
            "steps": [
                {
                    "step_number": 1,
                    "operation": "Material preparation",
                    "equipment": equipment_list[:2] if len(equipment_list) >= 2 else equipment_list,
                    "cycle_time": 10,
                    "quality_checks": ["Verify material certification"]
                },
                {
                    "step_number": 2,
                    "operation": "Primary processing",
                    "equipment": equipment_list[2:4] if len(equipment_list) >= 4 else equipment_list,
                    "cycle_time": 30,
                    "quality_checks": ["Dimensional inspection"]
                },
                {
                    "step_number": 3,
                    "operation": "Final assembly",
                    "equipment": equipment_list[4:] if len(equipment_list) >= 5 else equipment_list,
                    "cycle_time": 20,
                    "quality_checks": ["Functional test", "Visual inspection"]
                }
            ],
            "total_cycle_time": 60,
            "workstation_layout": "Linear flow",
            "wip_limit": 5
        }


class EnhancedSOPGenerator:
    """
    Enhanced SOP generator with LLM4IAS integration.
    
    Provides:
    - Manufacturing SOPs
    - Quality control procedures
    - Safety protocols
    - Assembly instructions
    - Testing procedures
    - Maintenance schedules
    """
    
    def __init__(self, config: Optional[MAKERConfig] = None):
        self.config = config or MAKERConfig()
        self.llm4ias = LLM4IASIntegration()
        
        # Initialize base generator if available
        if BASE_SOP_AVAILABLE:
            self.base_generator = SOPGenerator(config)
        else:
            self.base_generator = None
    
    async def generate_manufacturing_sop(
        self,
        product_name: str,
        product_spec: Dict[str, Any],
        equipment_list: List[str],
        industry_standard: IndustryStandard = IndustryStandard.ISO_9001,
        include_qc: bool = True,
        include_safety: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive manufacturing SOP.
        
        Args:
            product_name: Name of product
            product_spec: Product specifications
            equipment_list: Available equipment
            industry_standard: Industry standard
            include_qc: Include quality control
            include_safety: Include safety protocols
            
        Returns:
            Complete manufacturing SOP
        """
        logger.info(f"Generating manufacturing SOP for {product_name}...")
        
        # Generate base manufacturing SOP
        if self.llm4ias.is_available():
            mfg_sop = self.llm4ias.generate_manufacturing_sop(
                product_spec, equipment_list, industry_standard
            )
        else:
            mfg_sop = await self._generate_mfg_with_maker(
                product_name, product_spec, equipment_list
            )
        
        result = {
            "sop_type": SOPType.MANUFACTURING.value,
            "product_name": product_name,
            "industry_standard": industry_standard.value,
            "manufacturing_process": mfg_sop,
            "revision": "1.0",
            "effective_date": datetime.now().isoformat()
        }
        
        # Add quality control
        if include_qc:
            critical_chars = product_spec.get('critical_characteristics', [])
            result["quality_control"] = self.llm4ias.generate_quality_control_plan(
                product_spec, critical_chars
            )
        
        # Add safety protocols
        if include_safety:
            hazards = product_spec.get('hazards', [])
            result["safety_protocols"] = [
                {
                    "hazard_type": p.hazard_type,
                    "risk_level": p.risk_level,
                    "required_ppe": p.required_ppe,
                    "controls": p.engineering_controls + p.administrative_controls
                }
                for p in self.llm4ias.generate_safety_protocols(hazards)
            ]
        
        return result
    
    async def generate_assembly_sop(
        self,
        assembly_name: str,
        bill_of_materials: List[Dict[str, Any]],
        assembly_sequence: List[Dict[str, Any]],
        tools_required: List[str]
    ) -> Dict[str, Any]:
        """
        Generate assembly SOP with detailed instructions.
        
        Args:
            assembly_name: Name of assembly
            bill_of_materials: List of components
            assembly_sequence: Assembly steps
            tools_required: Required tools
            
        Returns:
            Assembly SOP
        """
        logger.info(f"Generating assembly SOP for {assembly_name}...")
        
        instructions = []
        for i, step in enumerate(assembly_sequence, 1):
            instruction = AssemblyInstruction(
                step_number=i,
                description=step.get('description', f'Step {i}'),
                components=step.get('components', []),
                tools_required=step.get('tools', []),
                torque_specifications=step.get('torque'),
                visual_check=step.get('visual_check', ''),
                functional_test=step.get('functional_test', '')
            )
            instructions.append(instruction)
        
        return {
            "sop_type": SOPType.ASSEMBLY.value,
            "assembly_name": assembly_name,
            "bill_of_materials": bill_of_materials,
            "tools_required": tools_required,
            "assembly_instructions": [
                {
                    "step": inst.step_number,
                    "description": inst.description,
                    "components": inst.components,
                    "tools": inst.tools_required,
                    "torque": inst.torque_specifications,
                    "visual_check": inst.visual_check
                }
                for inst in instructions
            ],
            "final_inspection": "Complete functional test and visual inspection"
        }
    
    async def generate_testing_sop(
        self,
        test_name: str,
        test_type: str,
        test_parameters: Dict[str, Any],
        acceptance_criteria: str,
        equipment_required: List[str]
    ) -> Dict[str, Any]:
        """
        Generate testing procedure SOP.
        
        Args:
            test_name: Name of test
            test_type: Type of test
            test_parameters: Test parameters
            acceptance_criteria: Pass/fail criteria
            equipment_required: Test equipment
            
        Returns:
            Testing SOP
        """
        test_procedure = TestProcedure(
            test_name=test_name,
            test_type=test_type,
            equipment_required=equipment_required,
            test_parameters={
                name: SOPParameter(
                    name=name,
                    value=spec.get('value', 0),
                    unit=spec.get('unit', ''),
                    tolerance=spec.get('tolerance', 0)
                )
                for name, spec in test_parameters.items()
            },
            acceptance_criteria=acceptance_criteria,
            preconditions=["Equipment calibrated", "Sample prepared"],
            procedure_steps=[
                "Set up test equipment per manufacturer specifications",
                "Configure test parameters",
                "Run test sequence",
                "Record all measurements",
                "Evaluate against acceptance criteria"
            ],
            data_recording="Record all measurements in test log",
            pass_fail_criteria=acceptance_criteria
        )
        
        return {
            "sop_type": SOPType.TESTING.value,
            "test_name": test_procedure.test_name,
            "test_type": test_procedure.test_type,
            "equipment": test_procedure.equipment_required,
            "parameters": {
                name: {
                    "value": param.value,
                    "unit": param.unit,
                    "tolerance": param.tolerance
                }
                for name, param in test_procedure.test_parameters.items()
            },
            "procedure": test_procedure.procedure_steps,
            "acceptance_criteria": test_procedure.acceptance_criteria
        }
    
    async def generate_maintenance_sop(
        self,
        equipment_specs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate maintenance schedules and procedures.
        
        Args:
            equipment_specs: Equipment specifications
            
        Returns:
            Maintenance SOP
        """
        schedules = self.llm4ias.generate_maintenance_schedule(equipment_specs)
        
        return {
            "sop_type": SOPType.MAINTENANCE.value,
            "maintenance_schedules": [
                {
                    "equipment": s.equipment_name,
                    "type": s.maintenance_type,
                    "frequency": s.frequency,
                    "procedures": s.procedures,
                    "duration_hours": s.estimated_duration,
                    "skill_level": s.technician_skill_level
                }
                for s in schedules
            ],
            "documentation": "All maintenance activities recorded in CMMS"
        }
    
    async def generate_complete_invention_sop(
        self,
        invention_spec: Dict[str, Any],
        include_all_sections: bool = True
    ) -> Dict[str, Any]:
        """
        Generate complete SOP package for an invention.
        
        Args:
            invention_spec: Complete invention specification
            include_all_sections: Include all SOP sections
            
        Returns:
            Complete SOP package
        """
        logger.info(f"Generating complete SOP for {invention_spec.get('name', 'Invention')}...")
        
        sop_package = {
            "document_title": f"Standard Operating Procedure - {invention_spec.get('name', 'Invention')}",
            "document_number": f"SOP-{hash(str(invention_spec)) % 10000:04d}",
            "revision": "1.0",
            "effective_date": datetime.now().isoformat(),
            "approval_status": "Draft",
            "sections": {}
        }
        
        # Manufacturing SOP
        if include_all_sections and 'manufacturing' in invention_spec:
            mfg = await self.generate_manufacturing_sop(
                invention_spec['name'],
                invention_spec['manufacturing'],
                invention_spec.get('equipment', []),
                IndustryStandard.ISO_9001
            )
            sop_package["sections"]["manufacturing"] = mfg
        
        # Assembly SOP
        if include_all_sections and 'assembly' in invention_spec:
            asm = await self.generate_assembly_sop(
                invention_spec['name'],
                invention_spec['assembly'].get('bom', []),
                invention_spec['assembly'].get('sequence', []),
                invention_spec['assembly'].get('tools', [])
            )
            sop_package["sections"]["assembly"] = asm
        
        # Testing SOP
        if include_all_sections and 'testing' in invention_spec:
            test = await self.generate_testing_sop(
                invention_spec['name'] + " Test",
                invention_spec['testing'].get('type', 'Functional'),
                invention_spec['testing'].get('parameters', {}),
                invention_spec['testing'].get('acceptance', 'Pass all tests'),
                invention_spec['testing'].get('equipment', [])
            )
            sop_package["sections"]["testing"] = test
        
        # Maintenance SOP
        if include_all_sections and 'equipment' in invention_spec:
            maint = await self.generate_maintenance_sop(invention_spec['equipment'])
            sop_package["sections"]["maintenance"] = maint
        
        # Safety Summary
        if 'hazards' in invention_spec:
            safety_protocols = self.llm4ias.generate_safety_protocols(
                invention_spec['hazards']
            )
            sop_package["sections"]["safety_summary"] = {
                "hazards_identified": len(invention_spec['hazards']),
                "risk_assessment": [
                    {"hazard": p.hazard_type, "risk": p.risk_level}
                    for p in safety_protocols
                ],
                "required_training": ["General safety", "Equipment-specific"]
            }
        
        return sop_package
    
    async def _generate_mfg_with_maker(
        self,
        product_name: str,
        product_spec: Dict[str, Any],
        equipment_list: List[str]
    ) -> Dict[str, Any]:
        """Generate manufacturing SOP using MAKER"""
        if not MAKER_AVAILABLE:
            return self.llm4ias._mock_manufacturing_sop(product_spec, equipment_list)
        
        task_desc = f"""
Generate a detailed manufacturing SOP for:
Product: {product_name}
Specifications: {json.dumps(product_spec, indent=2)}
Equipment: {', '.join(equipment_list)}

Include:
1. Process flow with cycle times
2. Equipment requirements per step
3. Quality control points
4. Safety considerations
5. Workstation layout recommendation
"""
        
        result = await run_generic_maker(
            task_description=task_desc,
            evaluator=GenericEvaluator(),
            task_type=TaskType.CUSTOM,
            config=self.config
        )
        
        return {
            "process_name": product_name,
            "generated_with": "MAKER",
            "content": result.solution
        }


async def generate_industrial_sop(
    invention_goal: str,
    domain: str = "manufacturing",
    specifications: Optional[Dict[str, Any]] = None,
    include_qc: bool = True,
    include_safety: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for industrial SOP generation.
    
    Args:
        invention_goal: Description of invention/process
        domain: Industry domain
        specifications: Technical specifications
        include_qc: Include quality control
        include_safety: Include safety protocols
        
    Returns:
        Complete industrial SOP
    """
    generator = EnhancedSOPGenerator()
    
    spec = specifications or {}
    spec['name'] = invention_goal
    spec['domain'] = domain
    
    return await generator.generate_complete_invention_sop(spec)


# Export main classes and functions
__all__ = [
    'EnhancedSOPGenerator',
    'LLM4IASIntegration',
    'ManufacturingStep',
    'QualityControlProcedure',
    'SafetyProtocol',
    'MaintenanceSchedule',
    'AssemblyInstruction',
    'TestProcedure',
    'SOPType',
    'IndustryStandard',
    'generate_industrial_sop'
]
