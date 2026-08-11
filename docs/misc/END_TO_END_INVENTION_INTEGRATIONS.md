# End-to-End Invention Planner - Integration Guide

## Table of Contents

1. [Overview](#overview)
2. [Core Integrations](#core-integrations)
3. [Optional Integrations](#optional-integrations)
4. [Integration Architecture](#integration-architecture)
5. [Data Flow](#data-flow)
6. [Custom Integrations](#custom-integrations)
7. [Integration Examples](#integration-examples)

---

## Overview

The End-to-End Invention Planner integrates with multiple OpenEvolve systems to provide comprehensive invention planning capabilities. This guide describes all available integrations, how they work, and how to configure them.

### Integration Categories

1. **Core Integrations** (Required)
   - Generic MAKER Integration - Multi-agent voting and decomposition
   - SOP Generator - Base SOP generation
   - SOP Component System - Component-level generation
   - SOP Integrated System - Full integration mode

2. **Optional Integrations** (Enhanced capabilities)
   - LeanAide - Mathematical formalization in Lean 4
   - Decomposition Engine - MDAP task decomposition
   - Knowledge Engine - Scientific knowledge retrieval
   - Red/Blue Team - Adversarial testing
   - BubbleLabs - Analytics and persistence
   - CrewAI - Distributed task delegation

---

## Core Integrations

### 1. Generic MAKER Integration

**Purpose**: Provides zero-error guarantees through multi-agent voting and task decomposition.

**Module**: `generic_maker_integration.py`

**Key Features**:
- First-to-ahead-by-k voting for solution selection
- MDAP task decomposition for complex problems
- Red-flagging of unreliable outputs
- Statistical convergence guarantees

**Configuration**:

```python
from generic_maker_integration import MAKERConfig, TaskType

config = MAKERConfig(
    enable_voting=True,           # Enable multi-agent voting
    voting_threshold=5,           # First-K-ahead (1-10)
    enable_decomposition=True,    # Use MDAP decomposition
    max_generations=50,          # Evolutionary generations
    population_size=30,          # Population size
    timeout_seconds=300          # Operation timeout
)
```

**How It Works**:

1. **Voting**: Multiple agents generate solutions independently
2. **Selection**: First solution to get K votes ahead of others wins
3. **Decomposition**: Complex tasks broken into sub-tasks via MDAP
4. **Aggregation**: Results combined with validation

**Invention Planner Usage**:

```python
from end_to_end_invention_planner import EndToEndInventionPlanner
from generic_maker_integration import MAKERConfig

# Configure MAKER behavior
config = MAKERConfig(
    voting_threshold=7,  # Higher threshold for more confidence
    max_generations=100  # More iterations for complex problems
)

planner = EndToEndInventionPlanner(config=config)
plan = await planner.plan_invention("your prompt", domain="physics")
```

**Voting Threshold Guide**:

| Threshold | Use Case | Confidence | Speed |
|-----------|----------|------------|-------|
| 3 | Rapid prototyping, exploration | Medium | Fast |
| 5 | Standard use (default) | High | Medium |
| 7 | Critical applications, physics | Very High | Slow |
| 10 | Maximum reliability, production | Highest | Very Slow |

---

### 2. SOP Generator

**Purpose**: Base Standard Operating Procedure generation.

**Module**: `sop_generator.py`

**Key Features**:
- Generate complete SOPs from requirements
- Support for multiple domains
- Quality evaluation and optimization
- Protocol step generation

**Classes**:

```python
from sop_generator import (
    SOPGenerator,
    StandardOperatingProcedure,
    SOPParameter,
    SOPStep,
    SOPEvaluator
)
```

**How It Works**:

1. Parse requirements and constraints
2. Generate environmental conditions
3. Specify equipment and materials
4. Create detailed protocol steps
5. Add quality control and safety measures
6. Optimize for clarity and completeness

**Configuration**:

```python
from sop_generator import SOPGenerator, SOPConfig

config = SOPConfig(
    include_quality_control=True,
    include_safety_measures=True,
    include_contingencies=True,
    detail_level="comprehensive"
)

generator = SOPGenerator(config)
sop = await generator.generate_sop(requirements, domain)
```

---

### 3. SOP Component System

**Purpose**: Generate individual SOP components with full integration.

**Module**: `sop_component_system.py`

**Key Features**:
- Component-level generation (environment, equipment, materials, protocols)
- Domain-specific optimization
- Multi-LLM ensemble for each component
- Quality validation

**Component Types**:

```python
from sop_component_system import SOPComponentGenerator, SOPComponentType

generator = SOPComponentGenerator(config)

# Generate specific components
environment = await generator.generate_environmental_condition(
    "Temperature", context, domain
)

equipment = await generator.generate_equipment_specification(
    "Centrifuge", context, domain
)

materials = await generator.generate_materials(
    ["Iron oxide", "Water"], context, domain
)

protocol = await generator.generate_protocol_step(
    step_number=1,
    description="Mix reactants",
    context=context,
    domain=domain
)
```

**Optimization**:

Each component can be optimized for:
- Clarity and readability
- Completeness (all required info)
- Verifiability (can be tested)
- Robustness (error handling)

---

### 4. SOP Integrated System

**Purpose**: Full integration of all SOP components with all OpenEvolve systems.

**Module**: `sop_integrated_system.py`

**Key Features**:
- Full integration mode (all systems)
- Domain-specific templates
- Multi-LLM consensus
- Complete validation

**Integration Modes**:

```python
from sop_integrated_system import IntegratedSOPGenerator, SOPIntegratedConfig, SOPIntegrationMode

# Full integration mode (all systems)
config_full = SOPIntegratedConfig(mode=SOPIntegrationMode.FULL)

# Basic mode (minimal systems)
config_basic = SOPIntegratedConfig(mode=SOPIntegrationMode.BASIC)

# Domain-specific mode
config_domain = SOPIntegratedConfig(
    mode=SOPIntegrationMode.DOMAIN_SPECIFIC,
    domain="chemistry"
)

generator = IntegratedSOPGenerator(config_full)
sop = await generator.generate_sop(requirement, domain, constraints)
```

**Full Integration Includes**:
- MAKER/MDAP voting and decomposition
- LeanAide math formalization
- Knowledge engine integration
- Adversarial testing (red/blue team)
- Evolutionary optimization
- Quality assessment
- Analytics tracking

---

## Optional Integrations

### 5. LeanAide Integration

**Purpose**: Formal mathematical verification using Lean 4 theorem prover.

**Status**: Optional (falls back to simulation if unavailable)

**Module**: `leanaide_client.py`

**Key Features**:
- Natural language to Lean 4 translation
- Automated proof generation and verification
- Mathematical problem detection
- Formal verification of solutions

**Availability Check**:

```python
try:
    from leanaide_client import LeanAideClient, LeanAideConfig
    LEANAIDE_AVAILABLE = True
    print("LeanAide available - full math formalization enabled")
except ImportError:
    LEANAIDE_AVAILABLE = False
    print("LeanAide not available - math will be simulated")
```

**Configuration**:

```python
from leanaide_client import LeanAideConfig

config = LeanAideConfig(
    server_url="http://localhost:8080",
    timeout=30,
    max_retries=3
)

client = LeanAideClient(config)
```

**Usage in Invention Planner**:

The planner auto-detects LeanAide availability. If available:
- Equations are formalized in Lean 4
- Theorems are stated and proved
- Formal verification is performed
- Confidence scores are based on proof verification

If not available:
- Math is simulated (placeholders used)
- Confidence scores are based on LLM consensus
- System still functional but less rigorous

**Starting LeanAide Server** (optional):

```bash
cd LeanAide
python leanaide_server.py
```

---

### 6. Decomposition Engine

**Purpose**: MDAP (Maximal Agentic Decomposition) for complex task breakdown.

**Status**: Optional (enhanced decomposition if available)

**Module**: `decomposition_engine.py`

**Key Features**:
- Atomic step decomposition
- Dependency graph construction
- Critical path analysis
- Parallelization opportunities

**Availability Check**:

```python
try:
    from decomposition_engine import DecompositionEngine
    DECOMPOSITION_AVAILABLE = True
    print("Decomposition engine available")
except ImportError:
    DECOMPOSITION_AVAILABLE = False
    print("Decomposition engine not available")
```

**Usage**:

```python
from decomposition_engine import DecompositionEngine

engine = DecompositionEngine()

# Decompose invention into atomic steps
atomic_steps = await engine.decompose_into_atomic_steps(goal)
dependency_graph = await engine.build_dependency_graph(atomic_steps)
validated_steps = await engine.validate_atomicity(atomic_steps)

# Get critical path
critical_path = await engine.find_critical_path(atomic_steps, dependency_graph)
```

**Benefits**:
- More granular decomposition
- Better dependency tracking
- Optimization opportunities
- Parallel execution planning

---

### 7. Knowledge Engine

**Purpose**: Scientific knowledge retrieval for domain expertise.

**Status**: Optional (enhanced knowledge retrieval if available)

**Modules**:
- `knowledge_engine/bedrock_kb.py` - AWS Bedrock knowledge base
- `knowledge_engine/elasticsearch_search.py` - Elasticsearch search
- `knowledge_engine/indexer.py` - Knowledge indexing

**Availability Check**:

```python
try:
    from knowledge_engine.bedrock_kb import BedrockKB
    from knowledge_engine.elasticsearch_search import ElasticsearchSearch
    KB_AVAILABLE = True
except ImportError:
    KB_AVAILABLE = False
```

**Configuration**:

```python
# Bedrock Knowledge Base
from knowledge_engine.bedrock_kb import BedrockKB

kb = BedrockKB(region="us-east-1")
knowledge = await kb.search_similar_inventions(goal)
domain_knowledge = await kb.get_domain_knowledge(goal.domain)

# Elasticsearch
from knowledge_engine.elasticsearch_search import ElasticsearchSearch

es = ElasticsearchSearch(url="http://localhost:9200")
results = await es.search(query, index="scientific_papers")
```

**Usage in Invention Planner**:

If knowledge engine is available:
- Real scientific literature retrieved
- Prior art searched and referenced
- Domain-specific knowledge incorporated
- More accurate and grounded plans

---

### 8. Red/Blue Team Integration

**Purpose**: Adversarial testing to find and fix vulnerabilities.

**Status**: Core (built-in, always available)

**Modules**:
- `red_team.py` - Red team (attackers)
- `blue_team.py` - Blue team (defenders)
- `adversarial.py` - Adversarial system coordinator

**How It Works**:

```python
from red_team import RedTeam
from blue_team import BlueTeam

# Red team: Find vulnerabilities
red_team = RedTeam(config=aggressive_config)
vulnerabilities = await red_team.attack_plan(invention_plan)

# Blue team: Generate fixes
blue_team = BlueTeam(config=defensive_config)
for vuln in vulnerabilities:
    root_cause = await blue_team.analyze_root_cause(vuln)
    fix = await blue_team.generate_fix(root_cause, sop)
    # Apply fix and re-test
```

**Red Team Strategies**:
- Parameter perturbation testing
- Edge case exploration
- Failure mode injection
- Boundary condition testing
- Stress testing
- Chaos testing

**Blue Team Strategies**:
- Root cause analysis
- Fix generation
- Verification testing
- Contingency planning
- Hardening procedures

---

### 9. BubbleLabs Integration

**Purpose**: Analytics, persistence, and validation tracking.

**Status**: Optional (enhanced analytics if available)

**Modules**:
- `bubblelabs_analytics.py` - Analytics tracking
- `bubblelabs_persistence.py` - SOP version storage
- `bubblelabs_validation.py` - Validation tracking

**Configuration**:

```python
from bubblelabs_analytics import BubbleLabsAnalytics
from bubblelabs_persistence import BubbleLabsPersistence
from bubblelabs_validation import BubbleLabsValidation

analytics = BubbleLabsAnalytics()
persistence = BubbleLabsPersistence()
validation = BubbleLabsValidation()

# Track SOP generation
await analytics.track_sop_generation(sop, metadata)

# Save SOP version
await persistence.save_sop_version(sop, version=1)

# Validate SOP
validation_results = await validation.validate_sop(sop)
```

**Features**:
- Success rate tracking
- Error frequency analysis
- Optimization history
- Version control for SOPs
- Performance metrics

---

### 10. CrewAI Integration

**Purpose**: Distributed task delegation for heavy computations.

**Status**: Optional (performance enhancement if available)

**Modules**:
- `crewai_client.py` - CrewAI client
- `crewai_integration.py` - Integration layer

**Configuration**:

```python
from crewai_client import CrewAIClient

crewai = CrewAIClient(url="http://localhost:9000")

# Delegate heavy tasks
math_formalization = await crewai.delegate(
    task="formalize_math",
    data={"equations": equations, "domain": domain}
)

error_analysis = await crewai.delegate(
    task="analyze_errors",
    data={"steps": atomic_steps}
)
```

**Use Cases**:
- CPU-intensive math formalization
- Large-scale error analysis
- Extensive red team testing
- Evolutionary optimization iterations

---

## Integration Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  End-to-End Invention Planner                    │
└─────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌──────────────┐      ┌──────────────────┐      ┌──────────────┐
│   MAKER/     │      │   SOP Systems    │      │   Optional   │
│   MDAP       │      │                  │      │   Systems    │
│              │      │                  │      │              │
│ • Voting     │      │ • SOP Generator  │      │ • LeanAide   │
│ • Decomp     │      │ • Components     │      │ • Knowledge  │
│ • Evolution  │      │ • Integrated     │      │ • CrewAI │
└──────────────┘      └──────────────────┘      └──────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  Red/Blue Team   │
                        │  (Adversarial)   │
                        └──────────────────┘
```

### Data Flow

```
1. User Prompt
       ↓
2. Prompt Analysis (MAKER)
       ↓
3. Knowledge Retrieval (Knowledge Engine)
       ↓
4. Decomposition (MDAP + Decomposition Engine)
       ↓
5. Math Formalization (LeanAide)
       ↓
6. Physics Validation (Domain validators)
       ↓
7. Error Analysis (MAKER + Error Analyzer)
       ↓
8. Red Team Testing (Red Team)
       ↓
9. Blue Team Fixes (Blue Team)
       ↓
10. SOP Generation (SOP Integrated System)
        ↓
11. Success Criteria (MAKER)
        ↓
12. Bulletproof SOP (Final Output)
```

---

## Custom Integrations

### Creating a Custom Evaluator

```python
from generic_maker_integration import GenericEvaluator, GenericTask

class MyCustomEvaluator(GenericEvaluator):
    """Custom evaluator for specific domain"""

    def evaluate(self, solution: str, task: GenericTask) -> float:
        score = 0.0

        # Your custom evaluation logic
        if "safety" in solution.lower():
            score += 0.3
        if "validation" in solution.lower():
            score += 0.3
        if len(solution) > 1000:
            score += 0.2
        if "error" in solution.lower():
            score += 0.2

        return min(1.0, score)

    def get_evaluation_details(self) -> Dict[str, Any]:
        return {
            "type": "custom_evaluator",
            "version": "1.0"
        }

# Usage
from end_to_end_invention_planner import EndToEndInventionPlanner

planner = EndToEndInventionPlanner()
# In the internal methods, replace InventionEvaluator with MyCustomEvaluator
```

### Creating a Custom Validator

```python
class PhysicsValidator:
    """Custom physics validation"""

    def validate_energy_conservation(self, steps, equations):
        """Check energy conservation"""
        # Your validation logic
        energy_in = sum(extract_energy_inputs(steps))
        energy_out = sum(extract_energy_outputs(steps))
        return abs(energy_in - energy_out) < tolerance

    def validate_thermodynamics(self, process):
        """Check 2nd law compliance"""
        # Your validation logic
        entropy_change = calculate_entropy(process)
        return entropy_change >= 0

# Usage in planner
validator = PhysicsValidator()
energy_ok = validator.validate_energy_conservation(steps, equations)
thermo_ok = validator.validate_thermodynamics(process)
```

### Integrating with External APIs

```python
import aiohttp

class ExternalKnowledgeSource:
    """Integration with external knowledge API"""

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    async def search_papers(self, query: str) -> List[Dict]:
        """Search scientific papers"""
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            params = {"query": query, "limit": 10}

            async with session.get(
                self.api_url + "/search",
                headers=headers,
                params=params
            ) as response:
                result = await response.json()
                return result["papers"]

# Usage in planner
external_kb = ExternalKnowledgeSource(
    api_url="https://api.example.com",
    api_key="your_api_key"
)

papers = await external_kb.search_papers("superconductivity")
# Incorporate papers into knowledge base
```

---

## Integration Examples

### Example 1: Full Integration Setup

```python
import asyncio
from end_to_end_invention_planner import EndToEndInventionPlanner
from generic_maker_integration import MAKERConfig

async def full_integration_example():
    """Example with all integrations enabled"""

    # Configure for maximum reliability
    config = MAKERConfig(
        enable_voting=True,
        voting_threshold=7,  # High confidence
        enable_decomposition=True,
        max_generations=100,
        population_size=50
    )

    # Initialize planner
    planner = EndToEndInventionPlanner(config=config)

    # Plan invention
    plan = await planner.plan_invention(
        prompt="Create a plan to invent room-temperature superconductor",
        domain="physics",
        constraints=["Critical temperature > 77 K"],
        available_equipment=["Standard physics lab"]
    )

    # Check integration results
    print(f"Math formalized: {len(plan.formalized_math)} theorems")
    print(f"Error sources: {len(plan.error_sources)} analyzed")
    print(f"Red team findings: {len(plan.red_team_findings)}")
    print(f"Blue team fixes: {len(plan.blue_team_fixes)}")

    # Check if LeanAide was used
    if any(m.lean_proof != "by sorry" for m in plan.formalized_math):
        print("✓ LeanAide formalization used")
    else:
        print("✗ Simulated formalization (LeanAide not available)")

    return plan

asyncio.run(full_integration_example())
```

### Example 2: Minimal Integration (Fast)

```python
async def minimal_integration_example():
    """Example with minimal integrations for speed"""

    # Configure for speed
    config = MAKERConfig(
        enable_voting=True,
        voting_threshold=3,  # Low threshold, fast
        enable_decomposition=False,  # Skip decomposition
        max_generations=20,
        population_size=15
    )

    planner = EndToEndInventionPlanner(config=config)

    plan = await planner.plan_invention(
        prompt="Create a plan for simple synthesis",
        domain="chemistry"
    )

    print(f"Planning complete in minimal mode")
    print(f"Confidence: {plan.validation_summary['confidence']:.1%}")

    return plan

asyncio.run(minimal_integration_example())
```

### Example 3: Domain-Specific Integration

```python
async def physics_integration_example():
    """Example optimized for physics inventions"""

    # Physics requires high precision
    config = MAKERConfig(
        enable_voting=True,
        voting_threshold=7,
        enable_decomposition=True,
        max_generations=100,
        population_size=50
    )

    planner = EndToEndInventionPlanner(config=config)

    plan = await planner.plan_invention(
        prompt="Create a plan to invent quantum dot solar cell",
        domain="physics",
        constraints=["Efficiency > 30%"]
    )

    # Check physics-specific validations
    print("Physics Validation:")
    for aspect, validated in plan.physics_validation.items():
        status = "✓" if validated else "✗"
        print(f"  {status} {aspect}")

    # Check math formalization
    print(f"\nFormalized Math ({len(plan.formalized_math)} theorems):")
    for math in plan.formalized_math:
        if math.confidence > 0.9:
            print(f"  ✓ {math.description} ({math.confidence:.1%})")

asyncio.run(physics_integration_example())
```

### Example 4: Biology Integration

```python
async def biology_integration_example():
    """Example optimized for biology inventions"""

    # Biology focuses on safety and validation
    config = MAKERConfig(
        enable_voting=True,
        voting_threshold=5,
        enable_decomposition=True,
        max_generations=50,
        population_size=30
    )

    planner = EndToEndInventionPlanner(config=config)

    plan = await planner.plan_invention(
        prompt="Create a plan to invent CRISPR gene therapy for DMD",
        domain="biology",
        constraints=["Safe for human trials"]
    )

    # Check safety validation
    print("Safety Validation:")
    if plan.physics_validation.get("safety_constraints"):
        print("  ✓ Safety constraints validated")
    else:
        print("  ✗ Safety review needed")

    # Check error sources (critical for biology)
    critical_errors = [e for e in plan.error_sources if e.impact == "critical"]
    print(f"\nCritical Safety Issues: {len(critical_errors)}")
    for error in critical_errors:
        print(f"  - {error.description}")
        print(f"    Mitigation: {error.mitigation_strategy}")

asyncio.run(biology_integration_example())
```

### Example 5: Checking Integration Availability

```python
def check_all_integrations():
    """Check which integrations are available"""

    print("Integration Availability Check")
    print("=" * 50)

    # Core integrations (always available)
    print("\nCore Integrations:")
    print("  ✓ Generic MAKER Integration")
    print("  ✓ SOP Generator")
    print("  ✓ SOP Component System")
    print("  ✓ SOP Integrated System")
    print("  ✓ Red/Blue Team")

    # Optional integrations
    print("\nOptional Integrations:")

    # LeanAide
    try:
        from leanaide_client import LeanAideClient
        print("  ✓ LeanAide (math formalization)")
    except ImportError:
        print("  ✗ LeanAide (math will be simulated)")

    # Decomposition Engine
    try:
        from decomposition_engine import DecompositionEngine
        print("  ✓ Decomposition Engine")
    except ImportError:
        print("  ✗ Decomposition Engine")

    # Knowledge Engine
    try:
        from knowledge_engine.bedrock_kb import BedrockKB
        print("  ✓ Bedrock Knowledge Base")
    except ImportError:
        print("  ✗ Bedrock Knowledge Base")

    try:
        from knowledge_engine.elasticsearch_search import ElasticsearchSearch
        print("  ✓ Elasticsearch Search")
    except ImportError:
        print("  ✗ Elasticsearch Search")

    # BubbleLabs
    try:
        from bubblelabs_analytics import BubbleLabsAnalytics
        print("  ✓ BubbleLabs Analytics")
    except ImportError:
        print("  ✗ BubbleLabs Analytics")

    # CrewAI
    try:
        from crewai_client import CrewAIClient
        print("  ✓ CrewAI (distributed)")
    except ImportError:
        print("  ✗ CrewAI")

    print("\n" + "=" * 50)
    print("Core integrations sufficient for basic operation.")
    print("Optional integrations enhance capabilities.")

check_all_integrations()
```

---

## Best Practices

### 1. Choose Right Configuration

```python
# High precision (critical inventions)
config = MAKERConfig(voting_threshold=7, max_generations=100)

# Balanced (standard use)
config = MAKERConfig()  # Default settings

# Fast (prototyping)
config = MAKERConfig(voting_threshold=3, max_generations=20)
```

### 2. Enable Optional Integrations When Available

```python
# Check and use LeanAide if available
try:
    from leanaide_client import LeanAideClient
    # Will auto-detect and use
except ImportError:
    print("LeanAide not available - will use simulation")
```

### 3. Use Appropriate Domain

```python
# Correct domain = better knowledge retrieval
plan = await plan_invention(
    prompt="...",
    domain="physics"  # Choose appropriate domain
)
```

### 4. Check Validation Results

```python
# Always check before using plan
if plan.validation_summary['ready_for_execution']:
    print("Ready to execute!")
else:
    print(f"Confidence: {plan.validation_summary['confidence']:.1%}")
    print("Review issues before proceeding")
```

### 5. Review Critical Error Sources

```python
# Check critical errors before execution
critical = [e for e in plan.error_sources if e.impact == "critical"]
if critical:
    print(f"WARNING: {len(critical)} critical error sources")
    for error in critical:
        print(f"  {error.description}")
```

---

**Version**: 1.0.0
**Last Updated**: 2025-12-30
**Paper**: arXiv:2511.09030
