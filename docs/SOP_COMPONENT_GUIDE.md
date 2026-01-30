# SOP Component System - Complete Guide

## Table of Contents

1. [Introduction](#introduction)
2. [What Can Be Improved](#what-can-be-improved)
3. [Component Types](#component-types)
4. [Generation Examples](#generation-examples)
5. [Refinement Examples](#refinement-examples)
6. [Optimization Examples](#optimization-examples)
7. [Integration with Full SOP](#integration-with-full-sop)
8. [API Reference](#api-reference)

---

## Introduction

The SOP Component System provides **granular control** over every aspect of Standard Operating Procedure generation and refinement. Instead of generating an entire SOP at once, you can:

1. **Generate individual components** independently
2. **Refine specific components** based on feedback
3. **Optimize components** through evolutionary methods
4. **Test components** for safety issues
5. **Apply all integrations** at the component level

### Why Component-Level?

| Traditional SOP Generation | Component-Level Generation |
|---------------------------|----------------------------|
| Regenerate entire SOP for one change | Modify only what needs changing |
| Cannot optimize individual parameters | Optimize each parameter independently |
| Fixed structure | Flexible, iterative refinement |
| All-or-nothing approach | Granular improvements |

---

## What Can Be Improved

### Every SOP Component:

1. **Environmental Conditions** - Each parameter (temperature, humidity, pressure, etc.)
2. **Equipment Specifications** - Each piece of equipment
3. **Materials** - Each reagent/material
4. **Protocol Steps** - Each individual step
5. **Quality Control** - Each QC procedure
6. **Safety Protocols** - Each safety measure
7. **Validation Criteria** - Each validation criterion
8. **Scaling Information** - Each scaling guideline
9. **Preconditions** - Each precondition

### How Each Can Be Improved:

1. **Generation** - Create from scratch
2. **Refinement** - Improve based on feedback
3. **Optimization** - Evolve toward optimal values
4. **Testing** - Find safety issues
5. **Verification** - Formal verification (if mathematical)

---

## Component Types

### 1. Environmental Conditions

**What**: Parameters like temperature, humidity, pressure, atmosphere

**Can Improve**:
- Target value (what temperature is optimal?)
- Tolerance (how precise can we be?)
- Verification method (how do we measure it?)
- Criticality (is this parameter critical?)
- Rationale (why does this matter?)

### 2. Equipment Specifications

**What**: Required equipment with models and specs

**Can Improve**:
- Model selection (which specific model?)
- Specifications (what capabilities needed?)
- Features (what features are required?)
- Calibration requirements
- Maintenance considerations

### 3. Materials

**What**: Reagents, consumables, materials

**Can Improve**:
- Purity/grade specifications
- Amount and tolerance
- Storage requirements
- Safety considerations
- Alternative options

### 4. Protocol Steps

**What**: Individual execution steps

**Can Improve**:
- Action clarity and specificity
- Duration estimates
- Verification methods
- Acceptance criteria
- Contingency actions
- Substep decomposition

### 5. Quality Control Procedures

**What**: QC checks and procedures

**Can Improve**:
- What to check
- How to check it
- Acceptance criteria
- Frequency/timing
- Documentation requirements

### 6. Safety Protocols

**What**: Safety measures and procedures

**Can Improve**:
- PPE requirements
- Engineering controls
- Administrative controls
- Emergency procedures
- First aid measures

### 7. Validation Criteria

**What**: Validation and verification criteria

**Can Improve**:
- What is being validated
- How to measure it
- Acceptance criteria
- Measurement methods

### 8. Scaling Information

**What**: Scaling guidance for different scales

**Can Improve**:
- Linear scaling relationships
- Non-linear considerations
- Equipment limitations
- Practical ranges

---

## Generation Examples

### Generate Environmental Condition

```python
from sop_component_system import generate_sop_component, SOPComponentType

# Generate a temperature parameter
temp_param = await generate_sop_component(
    component_type=SOPComponentType.ENVIRONMENTAL_CONDITION,
    component_name="Temperature",
    context={
        "purpose": "Nanoparticle synthesis",
        "equipment": ["Temperature controller", "Hotplate"]
    },
    domain="chemistry"
)

print(f"Temperature: {temp_param.format_spec()}")
print(f"Verification: {temp_param.verification_method}")
print(f"Critical: {temp_param.critical}")
print(f"Rationale: {temp_param.rationale}")
```

**Output**:
```
Temperature: 75.0 °C ± 2.0 °C
Verification: Calibrated digital thermometer (±0.1°C)
Critical: True
Rationale: Temperature controls particle size and crystallinity
```

### Generate Equipment Specification

```python
# Generate equipment specification
stirrer = await generate_sop_component(
    component_type=SOPComponentType.EQUIPMENT_SPECIFICATION,
    component_name="Magnetic Stirrer with Hotplate",
    context={
        "purpose": "Mixing and heating chemical solutions",
        "requirements": [
            "Temperature range: RT-250°C",
            "Speed control: 100-2000 RPM",
            "Temperature accuracy: ±1°C"
        ]
    },
    domain="chemistry"
)

print(f"Model: {stirrer['model']}")
print(f"Specifications: {stirrer['specifications']}")
print(f"Features: {stirrer['features']}")
```

### Generate Material Specification

```python
# Generate material specification
material = await generate_sop_component(
    component_type=SOPComponentType.MATERIAL,
    component_name="Iron(III) chloride hexahydrate",
    context={
        "purpose": "Precursor for iron oxide nanoparticles",
        "requirements": [
            "High purity",
            "Anhydrous preferred",
            "Storable at room temperature"
        ]
    },
    domain="chemistry"
)

print(f"Purity: {material['purity']}")
print(f"Grade: {material['grade']}")
print(f"Amount: {material['amount']} ± {material['tolerance']} {material['unit']}")
print(f"Safety: {material['safety']}")
```

### Generate Protocol Step

```python
# Generate a protocol step
step = await generate_sop_component(
    component_type=SOPComponentType.PROTOCOL_STEP,
    component_name="Prepare iron chloride solution",
    context={
        "step_number": 1,
        "equipment": ["Beaker", "Magnetic stirrer", "Balance"],
        "materials": ["Iron chloride", "Deionized water"]
    },
    domain="chemistry"
)

print(f"Step {step.step_number}: {step.action}")
print(f"Duration: {step.duration/60:.1f} ± {step.duration_tolerance/60:.1f} min")
print(f"Verification: {step.verification_method}")
print(f"Acceptance: {step.acceptance_criteria}")
print(f"Contingency: {step.contingency_action}")
```

---

## Refinement Examples

### Refine Environmental Condition

```python
from sop_component_system import SOPComponentGenerator

generator = SOPComponentGenerator()

# Current parameter (wide tolerance)
current = SOPParameter(
    name="Temperature",
    value=75.0,
    unit="°C",
    tolerance=5.0,  # Wide tolerance
    verification_method="Thermometer",
    critical=True,
    rationale="Controls particle size"
)

# Refine for tighter tolerance
refined = await generator.refine_environmental_condition(
    param=current,
    refinement_goal="Tighten tolerance to ±2°C for better particle size control",
    context={
        "domain": "chemistry",
        "equipment": ["Precision temperature controller"]
    }
)

print(f"Before: {current.format_spec()}")
print(f"After:  {refined.format_spec()}")
```

**Output**:
```
Before: 75.0 °C ± 5.0 °C
After:  75.0 °C ± 2.0 °C
```

### Refine Protocol Step

```python
# Current step (basic)
current_step = SOPStep(
    step_number=1,
    action="Mix the chemicals",
    duration=None,
    verification_method="",
    acceptance_criteria="",
    contingency_action=""
)

# Refine for completeness
refined_step = await generator.refine_protocol_step(
    step=current_step,
    refinement_goal="Add specific duration, verification method, and acceptance criteria",
    context={"domain": "chemistry"}
)

print(f"Before: {current_step.action}")
print(f"After:  {refined_step.action}")
print(f"Duration: {refined_step.duration/60:.1f} min")
print(f"Verification: {refined_step.verification_method}")
print(f"Acceptance: {refined_step.acceptance_criteria}")
```

---

## Optimization Examples

### Optimize Parameter Tolerance

```python
from sop_component_system import SOPComponentGenerator, SOPIntegratedConfig

# Enable evolutionary optimization
config = SOPIntegratedConfig(
    enable_evolution=True,
    evolution_generations=20,
    evolution_population_size=15
)

generator = SOPComponentGenerator(config)

# Parameter with loose tolerance
param = SOPParameter(
    name="Reaction Temperature",
    value=75.0,
    unit="°C",
    tolerance=10.0,  # Very loose
    verification_method="Standard thermometer"
)

# Optimize for tighter tolerance
optimized = await generator.optimize_component(
    component=param,
    component_type=SOPComponentType.ENVIRONMENTAL_CONDITION,
    optimization_goal="Minimize tolerance while maintaining achievability",
    context={
        "domain": "chemistry",
        "equipment": ["Standard temperature controller"]
    }
)

print(f"Original: {param.format_spec()}")
print(f"Optimized: {optimized.format_spec()}")
print(f"Improvement: {(1 - optimized.tolerance/param.tolerance)*100:.1f}% tighter")
```

**Output**:
```
Original: 75.0 °C ± 10.0 °C
Optimized: 75.0 °C ± 3.5 °C
Improvement: 65.0% tighter
```

### Optimize Step Duration

```python
# Step with long duration
step = SOPStep(
    step_number=2,
    action="Heat to reaction temperature",
    duration=1800.0,  # 30 minutes
    duration_tolerance=300.0  # ±5 min
)

# Optimize for shorter duration
optimized_step = await generator.optimize_component(
    component=step,
    component_type=SOPComponentType.PROTOCOL_STEP,
    optimization_goal="Minimize duration while ensuring complete heating",
    context={
        "domain": "chemistry",
        "equipment": ["Rapid heating system"]
    }
)

print(f"Original duration: {step.duration/60:.1f} ± {step.duration_tolerance/60:.1f} min")
print(f"Optimized duration: {optimized_step.duration/60:.1f} ± {optimized_step.duration_tolerance/60:.1f} min")
```

---

## Integration with Full SOP

### Build Complete SOP from Components

```python
from sop_component_system import SOPComponentGenerator
from sop_generator import StandardOperatingProcedure

generator = SOPComponentGenerator()

# Create SOP
sop = StandardOperatingProcedure(
    title="Nanoparticle Synthesis",
    version="1.0",
    status="DRAFT",
    effective_date="2025-01-15",
    description="Built from individual components"
)

# Add environmental conditions (generate and optimize each)
temp = await generator.generate_environmental_condition(
    "Temperature",
    {"purpose": "Nanoparticle synthesis"},
    "chemistry"
)
sop.environmental_conditions["Temperature"] = temp

humidity = await generator.generate_environmental_condition(
    "Humidity",
    {"purpose": "Nanoparticle synthesis"},
    "chemistry"
)
sop.environmental_conditions["Humidity"] = humidity

# Add equipment
stirrer = await generator.generate_equipment_specification(
    "Magnetic Stirrer",
    "Mixing",
    {"requirements": ["Temperature control"]},
    "chemistry"
)
sop.equipment.append(stirrer)

# Add materials
iron = await generator.generate_material(
    "Iron(III) chloride",
    "Precursor",
    {"requirements": ["High purity"]},
    "chemistry"
)
sop.materials.append(iron)

# Add protocol steps
step1 = await generator.generate_protocol_step(
    1, "Prepare precursor solution",
    {"equipment": ["Beaker"]},
    [],
    "chemistry"
)
sop.protocols.append(step1)

step2 = await generator.generate_protocol_step(
    2, "Heat to reaction temperature",
    {"equipment": ["Hotplate"]},
    [step1],
    "chemistry"
)
sop.protocols.append(step2)

# Add quality control
qc = await generator.generate_quality_control_procedure(
    "Particle size verification",
    {},
    "chemistry"
)
sop.quality_control.append(qc)

# Add safety
safety = await generator.generate_safety_protocol(
    "Handling corrosive chemicals",
    {},
    "chemistry"
)
sop.safety_protocols.append(safety)

# Export
print(sop.to_markdown())
```

---

## API Reference

### SOPComponentType

```python
class SOPComponentType(Enum):
    ENVIRONMENTAL_CONDITION = "environmental_condition"
    EQUIPMENT_SPECIFICATION = "equipment_specification"
    MATERIAL = "material"
    PROTOCOL_STEP = "protocol_step"
    QUALITY_CONTROL = "quality_control"
    SAFETY_PROTOCOL = "safety_protocol"
    VALIDATION_CRITERION = "validation_criterion"
    SCALING_INFO = "scaling_info"
    PRECONDITION = "precondition"
```

### Main Functions

#### generate_sop_component()

```python
async def generate_sop_component(
    component_type: SOPComponentType,
    component_name: str,
    context: Dict[str, Any],
    domain: str = "general",
    config: SOPIntegratedConfig = None
) -> Any
```

Generate any SOP component independently.

**Parameters**:
- `component_type`: Type of component to generate
- `component_name`: Name/description
- `context`: Additional context (equipment, materials, etc.)
- `domain`: Domain (chemistry, manufacturing, etc.)
- `config`: Optional configuration

**Returns**: Generated component (type depends on component_type)

---

### SOPComponentGenerator

```python
class SOPComponentGenerator:
    def __init__(self, config: SOPIntegratedConfig = None)

    # Environmental conditions
    async def generate_environmental_condition(
        self,
        parameter_name: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> SOPParameter

    async def refine_environmental_condition(
        self,
        param: SOPParameter,
        refinement_goal: str,
        context: Dict[str, Any] = None
    ) -> SOPParameter

    # Equipment
    async def generate_equipment_specification(
        self,
        equipment_name: str,
        purpose: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> Dict[str, str]

    # Materials
    async def generate_material(
        self,
        material_name: str,
        purpose: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> Dict[str, Any]

    # Protocol steps
    async def generate_protocol_step(
        self,
        step_number: int,
        action_description: str,
        context: Dict[str, Any],
        previous_steps: List[SOPStep] = None,
        domain: str = "general"
    ) -> SOPStep

    async def refine_protocol_step(
        self,
        step: SOPStep,
        refinement_goal: str,
        context: Dict[str, Any] = None
    ) -> SOPStep

    # Quality control, safety, validation, scaling
    async def generate_quality_control_procedure(
        self,
        qc_focus: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> str

    async def generate_safety_protocol(
        self,
        hazard_type: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> str

    async def generate_validation_criterion(
        self,
        criterion_focus: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> str

    async def generate_scaling_info(
        self,
        base_process: str,
        context: Dict[str, Any],
        domain: str = "general"
    ) -> str

    # Optimization and testing
    async def optimize_component(
        self,
        component: Any,
        component_type: SOPComponentType,
        optimization_goal: str,
        context: Dict[str, Any] = None
    ) -> Any

    async def test_component_safety(
        self,
        component: Any,
        component_type: SOPComponentType,
        context: Dict[str, Any] = None
    ) -> Tuple[bool, List[str]]

    def get_statistics(self) -> Dict[str, Any]
```

---

## Summary

The SOP Component System provides **complete granular control** over SOP generation and improvement:

✅ **9 Component Types** - Every SOP component can be generated/refined
✅ **5 Operations** - Generate, refine, optimize, test, verify
✅ **Integration Ready** - Works with all MAKER/MDAP integrations
✅ **Flexible** - Use independently or build complete SOPs
✅ **Trackable** - Component-level statistics

**This addresses the requirement:**
> "the SOP procedures, materials and every other potential component of the SOP must be able to be improved by the system as well"

The component system ensures:
- ✓ Every procedure can be generated and refined
- ✓ Every material can be specified and optimized
- ✓ Every equipment can be specified with alternatives
- ✓ Every environmental parameter can be tuned
- ✓ Every step can be detailed and verified
- ✓ Every QC procedure can be validated
- ✓ Every safety protocol can be tested
- ✓ Every validation criterion can be specified
- ✓ Every scaling guideline can be provided

---

**Version**: 1.0.0
**Last Updated**: 2025-12-30
**Paper**: arXiv:2511.09030
