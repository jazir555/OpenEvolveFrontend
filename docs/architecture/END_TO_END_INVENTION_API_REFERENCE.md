# End-to-End Invention Planner - API Reference

## Table of Contents

1. [Overview](#overview)
2. [Main Functions](#main-functions)
3. [Classes](#classes)
4. [Data Models](#data-models)
5. [Enums](#enums)
6. [Configuration](#configuration)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

---

## Overview

The End-to-End Invention Planner API provides a comprehensive interface for generating bulletproof invention plans from natural language prompts. The API is organized around several key components:

- **Main Entry Point**: `plan_invention()` function
- **Core Class**: `EndToEndInventionPlanner`
- **Data Models**: `BulletproofSOP`, `InventionGoal`, `ValidatedMath`, `ErrorSource`, `SuccessCriterion`
- **Configuration**: `MAKERConfig` for behavior customization
- **Supporting Systems**: SOP generators, LeanAide integration, adversarial testing

---

## Main Functions

### `plan_invention()`

**Async function** - Main entry point for invention planning.

```python
async def plan_invention(
    prompt: str,
    domain: str = "general",
    constraints: List[str] = None,
    available_equipment: List[str] = None
) -> BulletproofSOP
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | `str` | Yes | - | Natural language description of the invention to plan. Should be specific and detailed. |
| `domain` | `str` | No | `"general"` | Technical domain for the invention. Affects knowledge retrieval and validation. |
| `constraints` | `List[str]` | No | `None` | Specific constraints or requirements for the invention. |
| `available_equipment` | `List[str]` | No | `None` | Equipment available for executing the invention plan. |

**Returns:**
- `BulletproofSOP`: Complete bulletproof invention plan with all validations.

**Raises:**
- `ValueError`: If prompt is empty or invalid
- `TimeoutError`: If planning exceeds configured timeout
- `ImportError`: If required dependencies are missing

**Supported Domains:**
- `"physics"` - Physics inventions, materials, devices
- `"chemistry"` - Chemical synthesis, reactions, materials
- `"biology"` - Biotechnology, genetics, medicine
- `"materials_science"` - Alloys, polymers, composites
- `"engineering"` - Mechanical, electrical, software
- `"general"` - Multi-domain or unspecified

**Example:**

```python
from end_to_end_invention_planner import plan_invention

# Basic usage
plan = await plan_invention(
    prompt="Create a plan to invent high-temperature superconducting wire",
    domain="physics"
)

# Advanced usage with constraints
plan = await plan_invention(
    prompt="Create a plan to invent iron oxide magnetic nanoparticles",
    domain="chemistry",
    constraints=[
        "Must be biocompatible",
        "Particle size 10-15 nm",
        "Water dispersible"
    ],
    available_equipment=[
        "Standard chemistry lab",
        "Furnace",
        "Centrifuge"
    ]
)

# Access results
print(f"Confidence: {plan.validation_summary['confidence']:.1%}")
print(f"Ready for execution: {plan.validation_summary['ready_for_execution']}")
document = plan.to_executable_document()
```

---

### `get_invention_planner_capabilities()`

**Function** - Query system capabilities and availability of integrations.

```python
def get_invention_planner_capabilities() -> Dict[str, Any]
```

**Returns:**

```python
{
    "end_to_end_planning": bool,           # Always True
    "prompt_understanding": bool,           # Always True
    "knowledge_retrieval": bool,            # Always True
    "decomposition": bool,                  # If decomposition engine available
    "math_formalization": bool,             # If LeanAide available
    "physics_validation": bool,             # Always True
    "error_analysis": bool,                 # Always True
    "adversarial_testing": bool,            # Always True
    "binary_success_criteria": bool,        # Always True
    "turnkey_ready": bool,                  # Always True
    "supported_domains": List[str],         # List of supported domains
    "pipeline_stages": List[str],           # Pipeline stage names
    "output": "Bulletproof SOP with all validations"
}
```

**Example:**

```python
from end_to_end_invention_planner import get_invention_planner_capabilities

capabilities = get_invention_planner_capabilities()

print("System Capabilities:")
for capability, available in capabilities.items():
    if isinstance(available, bool):
        status = "✓" if available else "✗"
        print(f"  {status} {capability}")
    elif isinstance(available, list):
        print(f"  {capability}: {', '.join(available)}")
```

---

## Classes

### `EndToEndInventionPlanner`

Main class for end-to-end invention planning. Provides full control over the planning process.

#### Constructor

```python
EndToEndInventionPlanner(config: MAKERConfig = None)
```

**Parameters:**
- `config`: Optional configuration object. If not provided, uses default configuration.

**Example:**

```python
from end_to_end_invention_planner import EndToEndInventionPlanner
from generic_maker_integration import MAKERConfig

# Default configuration
planner = EndToEndInventionPlanner()

# Custom configuration
config = MAKERConfig(
    enable_voting=True,
    voting_threshold=5,
    enable_decomposition=True,
    max_generations=50,
    population_size=30
)
planner = EndToEndInventionPlanner(config=config)
```

#### Methods

##### `plan_invention()`

Main method for generating invention plans.

```python
async def plan_invention(
    self,
    prompt: str,
    domain: str = "general",
    constraints: List[str] = None,
    available_equipment: List[str] = None
) -> BulletproofSOP
```

See [`plan_invention()`](#plan_invention) function documentation for details.

##### `get_statistics()`

Get planning statistics from the planner instance.

```python
def get_statistics(self) -> Dict[str, Any]
```

**Returns:**

```python
{
    "prompts_processed": int,          # Number of prompts processed
    "inventions_planned": int,         # Number of inventions planned
    "math_formalized": int,            # Total math theorems formalized
    "errors_identified": int,          # Total error sources identified
    "red_team_findings": int,          # Total red team findings
    "blue_team_fixes": int,            # Total blue team fixes
    "total_planning_time": float       # Total time spent planning (seconds)
}
```

**Example:**

```python
planner = EndToEndInventionPlanner()
await planner.plan_invention("Create a plan for superconductors")

stats = planner.get_statistics()
print(f"Inventions planned: {stats['inventions_planned']}")
print(f"Math formalized: {stats['math_formalized']}")
print(f"Errors identified: {stats['errors_identified']}")
print(f"Average planning time: {stats['total_planning_time'] / stats['inventions_planned']:.1f}s")
```

---

### `InventionEvaluator`

Evaluator class for assessing invention planning quality. Extends `GenericEvaluator`.

#### Methods

##### `evaluate()`

Evaluate a solution for a given task.

```python
def evaluate(self, solution: str, task: GenericTask) -> float
```

**Parameters:**
- `solution`: The solution text to evaluate
- `task`: The task object

**Returns:**
- `float`: Evaluation score from 0.0 to 1.0

**Evaluation Criteria:**
- Solution length (prefer detailed solutions)
- Presence of "step" keyword
- Presence of "error" keyword
- Presence of validation keywords
- Presence of criteria keywords

##### `get_evaluation_details()`

Get details about the evaluation process.

```python
def get_evaluation_details(self) -> Dict[str, Any]
```

**Returns:**
```python
{"type": "invention_planner"}
```

---

## Data Models

### `BulletproofSOP`

Complete bulletproof invention plan.

```python
@dataclass
class BulletproofSOP:
    invention_goal: InventionGoal
    knowledge_base: List[str]
    decomposition: Dict[str, Any]
    formalized_math: List[ValidatedMath]
    physics_validation: Dict[str, bool]
    error_sources: List[ErrorSource]
    red_team_findings: List[str]
    blue_team_fixes: List[str]
    success_criteria: List[SuccessCriterion]
    sop: StandardOperatingProcedure
    validation_summary: Dict[str, Any]
    created_at: float
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `invention_goal` | `InventionGoal` | Parsed invention goal from prompt |
| `knowledge_base` | `List[str]` | Relevant scientific knowledge |
| `decomposition` | `Dict[str, Any]` | Atomic step decomposition |
| `formalized_math` | `List[ValidatedMath]` | Formalized mathematical relationships |
| `physics_validation` | `Dict[str, bool]` | Physics/logic validation results |
| `error_sources` | `List[ErrorSource]` | All identified error sources |
| `red_team_findings` | `List[str]` | Red team vulnerability findings |
| `blue_team_fixes` | `List[str]` | Blue team fix implementations |
| `success_criteria` | `List[SuccessCriterion]` | Binary success/fail criteria |
| `sop` | `StandardOperatingProcedure` | Complete SOP for execution |
| `validation_summary` | `Dict[str, Any]` | Overall validation results |
| `created_at` | `float` | Unix timestamp of creation |

#### Methods

##### `to_executable_document()`

Generate complete turnkey-ready document.

```python
def to_executable_document(self) -> str
```

**Returns:**
- `str`: Complete Markdown document ready for execution

**Example:**

```python
plan = await plan_invention("Create a plan for superconductors", domain="physics")

# Generate document
document = plan.to_executable_document()

# Save to file
with open("superconductor_plan.md", "w") as f:
    f.write(document)

# Print key sections
print(f"Goal: {plan.invention_goal.target}")
print(f"Knowledge sources: {len(plan.knowledge_base)}")
print(f"Decomposition steps: {len(plan.decomposition.get('steps', []))}")
print(f"Math formalized: {len(plan.formalized_math)}")
print(f"Error sources: {len(plan.error_sources)}")
print(f"Success criteria: {len(plan.success_criteria)}")
```

---

### `InventionGoal`

Parsed invention goal from natural language prompt.

```python
@dataclass
class InventionGoal:
    goal_type: str                    # "technology", "material", "device", "process"
    target: str                       # Specific invention target
    domain: str                       # Technical domain
    key_requirements: List[str]       # Key requirements
    constraints: List[str]            # Constraints on the invention
    success_definition: str           # Definition of success
    complexity_score: float           # 0.0 to 1.0
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `goal_type` | `str` | Type of invention (technology/material/device/process) |
| `target` | `str` | Specific invention target description |
| `domain` | `str` | Technical domain (physics/chemistry/biology/etc) |
| `key_requirements` | `List[str]` | Key requirements for the invention |
| `constraints` | `List[str]` | Constraints on the invention |
| `success_definition` | `str` | Definition of successful invention |
| `complexity_score` | `float` | Complexity score from 0.0 (simple) to 1.0 (complex) |

---

### `ValidatedMath`

Mathematical relationship formalized in Lean 4.

```python
@dataclass
class ValidatedMath:
    description: str                  # Description of the math
    lean_theorem: str                 # Lean theorem statement
    lean_proof: str                   # Lean proof code
    variables: Dict[str, str]         # Variable definitions
    assumptions: List[str]            # Mathematical assumptions
    verification_method: str          # How to verify
    confidence: float                 # Confidence in formalization (0-1)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `description` | `str` | Human-readable description |
| `lean_theorem` | `str` | Lean 4 theorem statement |
| `lean_proof` | `str` | Lean 4 proof code |
| `variables` | `Dict[str, str]` | Variable name -> definition |
| `assumptions` | `List[str]` | List of assumptions |
| `verification_method` | `str` | Method for verification |
| `confidence` | `float` | Confidence score (0.0 to 1.0) |

---

### `ErrorSource`

Potential source of error in invention execution.

```python
@dataclass
class ErrorSource:
    error_type: str                   # Type of error
    description: str                  # Error description
    probability: float                # Estimated probability (0-1)
    impact: str                       # "critical", "high", "medium", "low"
    mitigation_strategy: str          # How to mitigate
    verification_method: str          # How to verify
    acceptance_criteria: str          # Acceptance criteria
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `error_type` | `str` | Type of error (equipment/human/material/etc) |
| `description` | `str` | Error description |
| `probability` | `float` | Estimated probability (0.0 to 1.0) |
| `impact` | `str` | Impact level (critical/high/medium/low) |
| `mitigation_strategy` | `str` | Mitigation strategy |
| `verification_method` | `str` | Verification method |
| `acceptance_criteria` | `str` | Acceptance criteria |

---

### `SuccessCriterion`

Binary success criterion for invention validation.

```python
@dataclass
class SuccessCriterion:
    criterion: str                    # Criterion description
    measurement_method: str           # How to measure
    pass_threshold: float             # Pass threshold value
    units: str                        # Measurement units
    verification: str                 # Verification method
    fallback_criteria: List[str]      # Alternative criteria
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `criterion` | `str` | Criterion description |
| `measurement_method` | `str` | Measurement method |
| `pass_threshold` | `float` | Pass threshold value |
| `units` | `str` | Measurement units |
| `verification` | `str` | Verification method |
| `fallback_criteria` | `List[str]` | Alternative criteria if primary fails |

---

## Enums

### `PipelineStage`

End-to-end pipeline stages.

```python
class PipelineStage(Enum):
    PROMPT_ANALYSIS = "prompt_analysis"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    DECOMPOSITION = "decomposition"
    MATH_FORMALIZATION = "math_formalization"
    PHYSICS_VALIDATION = "physics_validation"
    ERROR_ANALYSIS = "error_analysis"
    RED_BLUE_TEAM = "red_blue_team"
    SOP_GENERATION = "sop_generation"
    SUCCESS_CRITERIA = "success_criteria"
```

**Values:**
- `PROMPT_ANALYSIS`: Stage 1 - Analyze user prompt
- `KNOWLEDGE_RETRIEVAL`: Stage 2 - Retrieve scientific knowledge
- `DECOMPOSITION`: Stage 3 - Decompose into atomic steps
- `MATH_FORMALIZATION`: Stage 4 - Formalize mathematics in Lean
- `PHYSICS_VALIDATION`: Stage 5 - Validate physics/logic
- `ERROR_ANALYSIS`: Stage 6 - Analyze all error sources
- `RED_BLUE_TEAM`: Stage 7 - Red/blue team adversarial testing
- `SOP_GENERATION`: Stage 8 - Generate bulletproof SOP
- `SUCCESS_CRITERIA`: Stage 9 - Define binary success criteria

---

### `TaskType`

Types of tasks for MAKER system.

```python
class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REFACTORING = "code_refactoring"
    DOCUMENT_PROCESSING = "document_processing"
    TEXT_SUMMARIZATION = "text_summarization"
    DATA_ANALYSIS = "data_analysis"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    OPTIMIZATION = "optimization"
    CUSTOM = "custom"
```

---

## Configuration

### `MAKERConfig`

Configuration for MAKER/MDAP system behavior.

```python
@dataclass
class MAKERConfig:
    enable_voting: bool = True
    voting_threshold: int = 5
    enable_decomposition: bool = True
    max_generations: int = 50
    population_size: int = 30
    timeout_seconds: int = 300
    # ... additional fields
```

**Key Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_voting` | `bool` | `True` | Enable multi-agent voting |
| `voting_threshold` | `int` | `5` | First-K-ahead voting threshold (1-10) |
| `enable_decomposition` | `bool` | `True` | Enable MDAP task decomposition |
| `max_generations` | `int` | `50` | Max evolutionary generations |
| `population_size` | `int` | `30` | Population size for evolution |
| `timeout_seconds` | `int` | `300` | Timeout per operation (seconds) |

**Example:**

```python
from generic_maker_integration import MAKERConfig

# High-precision configuration (physics, critical applications)
high_precision = MAKERConfig(
    enable_voting=True,
    voting_threshold=7,
    enable_decomposition=True,
    max_generations=100,
    population_size=50
)

# Fast configuration (prototyping, exploration)
fast_config = MAKERConfig(
    enable_voting=True,
    voting_threshold=3,
    enable_decomposition=False,
    max_generations=20,
    population_size=15
)

# Balanced configuration (default)
balanced = MAKERConfig()
```

---

## Error Handling

### Common Exceptions

#### `ValueError`

Raised when input parameters are invalid.

```python
# Empty prompt
try:
    plan = await plan_invention("", domain="physics")
except ValueError as e:
    print(f"Invalid input: {e}")
```

#### `TimeoutError`

Raised when planning exceeds configured timeout.

```python
# Operation timeout
try:
    plan = await plan_invention(complex_prompt, domain="physics")
except TimeoutError:
    print("Planning timed out - try reducing complexity or increasing timeout")
```

#### `ImportError`

Raised when required dependencies are missing.

```python
# Missing dependency
try:
    from end_to_end_invention_planner import plan_invention
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install -r requirements.txt")
```

### Error Handling Best Practices

```python
import asyncio
from end_to_end_invention_planner import plan_invention

async def safe_plan_invention(prompt, domain="general"):
    """Safely plan an invention with error handling"""
    try:
        # Validate inputs
        if not prompt or len(prompt.strip()) < 10:
            raise ValueError("Prompt too short - be more specific")

        # Attempt planning
        plan = await plan_invention(prompt, domain)

        # Check validation
        if not plan.validation_summary['ready_for_execution']:
            print(f"Warning: Plan not ready (confidence: {plan.validation_summary['confidence']:.1%})")
            print("Review validation issues before proceeding")

        return plan

    except ValueError as e:
        print(f"Input validation error: {e}")
        return None

    except TimeoutError:
        print("Planning timed out. Try:")
        print("  1. Reducing prompt complexity")
        print("  2. Increasing timeout in configuration")
        print("  3. Reducing voting threshold")
        return None

    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return None

# Usage
plan = await safe_plan_invention(
    "Create a plan for superconducting wire",
    domain="physics"
)

if plan:
    print("Planning successful!")
    document = plan.to_executable_document()
```

---

## Examples

### Example 1: Basic Usage

```python
import asyncio
from end_to_end_invention_planner import plan_invention

async def main():
    # Plan a simple invention
    plan = await plan_invention(
        prompt="Create a plan to invent magnetic nanoparticles",
        domain="chemistry"
    )

    # Check results
    print(f"Goal: {plan.invention_goal.target}")
    print(f"Complexity: {plan.invention_goal.complexity_score:.2f}")
    print(f"Confidence: {plan.validation_summary['confidence']:.1%}")

    # Save document
    with open("plan.md", "w") as f:
        f.write(plan.to_executable_document())

asyncio.run(main())
```

### Example 2: Advanced Usage with Custom Config

```python
import asyncio
from end_to_end_invention_planner import EndToEndInventionPlanner
from generic_maker_integration import MAKERConfig

async def main():
    # Custom configuration for high precision
    config = MAKERConfig(
        enable_voting=True,
        voting_threshold=7,
        max_generations=100,
        population_size=50
    )

    planner = EndToEndInventionPlanner(config=config)

    # Plan complex invention
    plan = await planner.plan_invention(
        prompt="Create a plan for room-temperature superconductor",
        domain="physics",
        constraints=["Critical temperature > 77 K", "Standard lab equipment"]
    )

    # Review critical error sources
    critical_errors = [e for e in plan.error_sources if e.impact == "critical"]
    print(f"Critical errors: {len(critical_errors)}")

    # Save document
    with open("superconductor_plan.md", "w") as f:
        f.write(plan.to_executable_document())

    # Check statistics
    stats = planner.get_statistics()
    print(f"Total planning time: {stats['total_planning_time']:.1f}s")

asyncio.run(main())
```

### Example 3: Batch Processing

```python
import asyncio
from end_to_end_invention_planner import EndToEndInventionPlanner

async def plan_multiple_inventions(inventions):
    """Plan multiple inventions in sequence"""
    planner = EndToEndInventionPlanner()

    results = []
    for invention in inventions:
        print(f"Planning: {invention['name']}")
        plan = await planner.plan_invention(
            prompt=invention['prompt'],
            domain=invention.get('domain', 'general')
        )

        results.append({
            'name': invention['name'],
            'confidence': plan.validation_summary['confidence'],
            'ready': plan.validation_summary['ready_for_execution'],
            'document': plan.to_executable_document()
        })

    return results

async def main():
    inventions = [
        {
            'name': 'Magnetic Nanoparticles',
            'prompt': 'Create a plan to invent iron oxide magnetic nanoparticles',
            'domain': 'chemistry'
        },
        {
            'name': 'Superconductor',
            'prompt': 'Create a plan to invent high-temperature superconducting wire',
            'domain': 'physics'
        },
        {
            'name': 'Aluminum Alloy',
            'prompt': 'Create a plan to invent lightweight high-strength aluminum alloy',
            'domain': 'materials_science'
        }
    ]

    results = await plan_multiple_inventions(inventions)

    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Ready: {result['ready']}")

        # Save each plan
        filename = f"{result['name'].lower().replace(' ', '_')}_plan.md"
        with open(filename, 'w') as f:
            f.write(result['document'])

asyncio.run(main())
```

### Example 4: Domain-Specific Planning

```python
import asyncio
from end_to_end_invention_planner import plan_invention

async def plan_by_domain():
    """Demonstrate domain-specific planning"""

    # Physics invention
    physics_plan = await plan_invention(
        prompt="Create a plan to invent a quantum dot solar cell with >30% efficiency",
        domain="physics"
    )
    print(f"Physics plan confidence: {physics_plan.validation_summary['confidence']:.1%}")

    # Chemistry invention
    chemistry_plan = await plan_invention(
        prompt="Create a plan to invent a catalytic converter for reducing NOx emissions",
        domain="chemistry"
    )
    print(f"Chemistry plan confidence: {chemistry_plan.validation_summary['confidence']:.1%}")

    # Biology invention
    biology_plan = await plan_invention(
        prompt="Create a plan to invent a CRISPR-based diagnostic test for malaria",
        domain="biology"
    )
    print(f"Biology plan confidence: {biology_plan.validation_summary['confidence']:.1%}")

    # Materials science invention
    materials_plan = await plan_invention(
        prompt="Create a plan to invent a shape memory alloy for biomedical implants",
        domain="materials_science"
    )
    print(f"Materials plan confidence: {materials_plan.validation_summary['confidence']:.1%}")

asyncio.run(plan_by_domain())
```

### Example 5: Error Analysis and Review

```python
import asyncio
from end_to_end_invention_planner import plan_invention

async def analyze_plan_thoroughly():
    """Plan and thoroughly analyze the result"""
    plan = await plan_invention(
        prompt="Create a plan to invent a high-efficiency perovskite solar cell",
        domain="physics"
    )

    # Analyze by impact level
    print("Error Analysis by Impact:")
    for impact in ['critical', 'high', 'medium', 'low']:
        errors = [e for e in plan.error_sources if e.impact == impact]
        print(f"\n{impact.upper()} ({len(errors)} errors):")
        for error in errors[:5]:  # Show top 5
            print(f"  - {error.description}")
            print(f"    Probability: {error.probability:.1%}")
            print(f"    Mitigation: {error.mitigation_strategy[:80]}...")

    # Check red team findings
    print(f"\nRed Team Findings ({len(plan.red_team_findings)}):")
    for finding in plan.red_team_findings[:10]:
        print(f"  - {finding}")

    # Check blue team fixes
    print(f"\nBlue Team Fixes ({len(plan.blue_team_fixes)}):")
    for fix in plan.blue_team_fixes[:10]:
        print(f"  - {fix}")

    # Check binary success criteria
    print(f"\nBinary Success Criteria ({len(plan.success_criteria)}):")
    for criterion in plan.success_criteria:
        print(f"  - {criterion.criterion}")
        print(f"    Pass if: {criterion.pass_threshold} {criterion.units}")
        print(f"    Binary: PASS or FAIL")

asyncio.run(analyze_plan_thoroughly())
```

---

## Type Hints

The API uses Python type hints for better IDE support and type checking.

```python
from typing import List, Dict, Any
from end_to_end_invention_planner import (
    plan_invention,
    BulletproofSOP,
    InventionGoal,
    ValidatedMath,
    ErrorSource,
    SuccessCriterion
)

async def create_plan(
    prompt: str,
    domain: str
) -> BulletproofSOP:
    """Type-hinted function"""
    plan: BulletproofSOP = await plan_invention(prompt, domain)
    return plan

# Access typed fields
goal: InventionGoal = plan.invention_goal
math: List[ValidatedMath] = plan.formalized_math
errors: List[ErrorSource] = plan.error_sources
criteria: List[SuccessCriterion] = plan.success_criteria
```

---

**Version**: 1.0.0
**Last Updated**: 2025-12-30
**Paper**: arXiv:2511.09030
