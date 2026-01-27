# Hephaestus + Decomposition Workflow + OpenEvolve Integration

**Date**: 2025-12-29
**Status**: PRODUCTION-READY ✅
**Architecture**: Orchestrator → Workflow → Evolutionary Engine

---

## CRITICAL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Hephaestus (Orchestrator)                       │
│                                                                         │
│  Phases 1-6: Manages task lifecycle, spawns agents, coordinates work   │
└────────────────────────────────────────┬────────────────────────────────┘
                                         │
                                         │ DELEGATES TO
                                         │
┌────────────────────────────────────────▼────────────────────────────────┐
│                  Decomposition Workflow (Teams/Gauntlets)              │
│                                                                         │
│  - Problem decomposition into sub-problems                             │
│  - Blue Team solving, Red Team critique, Gold Team verification        │
│  - Multi-stage workflow (Stages 0-6)                                   │
└────────────────────────────────────────┬────────────────────────────────┘
                                         │
                                         │ LEVERAGES
                                         │
┌────────────────────────────────────────▼────────────────────────────────┐
│                     OpenEvolve (Evolutionary Engine)                   │
│                                                                         │
│  - Evolutionary permutations in ALL stages                             │
│  - MAP-Elites, island-based evolution, LLM ensemble                    │
│  - Used for: analysis, decomposition, solutions, critique, verification│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### Hephaestus (Orchestrator)
- **Role**: Workflow orchestration, task management, agent spawning
- **Phases**: 6 phases (Problem Setup → Solution Generation → Critique → Verification → Reassembly → Final Validation)
- **Responsibility**: Coordinates the overall process, delegates to Decomposition Workflow

### Decomposition Workflow (Teams & Gauntlets)
- **Role**: Problem decomposition with teams (Blue/Red/Gold) and gauntlets
- **Stages**: 7 stages (0-6) covering analysis, decomposition, solving, critique, verification, reassembly, learning
- **Responsibility**: Manages the problem-solving workflow with specialized teams

### OpenEvolve (Evolutionary Engine)
- **Role**: Evolutionary permutations and optimization
- **Used In**: ALL stages of the Decomposition Workflow
- **Responsibility**: Generates multiple evolved variants, selects best options through fitness evaluation

---

## Evolutionary Integration in All Stages

### Stage 0: Content Analysis (with OpenEvolve)
- Uses OpenEvolve to evolve multiple analysis perspectives
- Selects best analysis based on completeness score
- Evaluates: domain, constraints, success criteria, required expertise

### Stage 1: AI-Assisted Decomposition (with OpenEvolve)
- Uses OpenEvolve to evolve multiple decomposition strategies
- Evaluates based on coverage, structure, and dependency tracking
- Selects best decomposition approach

### Stage 3A: Blue Team Solving (with OpenEvolve)
- Uses OpenEvolve to evolve multiple solution attempts
- Evaluates based on solution length, structure, completeness, documentation
- Selects best evolved solution

### Stage 3B: Red Team Gauntlet (with OpenEvolve)
- Uses OpenEvolve to evolve multiple critique perspectives
- Evaluates based on critique depth, issues found, severity analysis
- Selects most comprehensive critique

### Stage 3C: Gold Team Verification (with OpenEvolve)
- Uses OpenEvolve to evolve multiple verification perspectives
- Evaluates based on correctness, completeness, quality scores
- Selects most thorough verification

### Stage 4: Reassembly (with OpenEvolve)
- Uses OpenEvolve to evolve reassembly strategies
- Optimizes assembly order and integration approach

### Stage 5 & 6: Validation & Learning (with OpenEvolve)
- Uses OpenEvolve to evolve validation criteria
- Extracts knowledge artifacts for learning

---

## Files

### Core Integration Files

| File | Lines | Purpose |
|------|-------|---------|
| `decomposition_mcp_tools.py` | 1095 | MCP tools that Decomposition Workflow uses, leveraging OpenEvolve in ALL stages |
| `decomposition_hephaestus_bridge.py` | 750 | Bridge mapping Hephaestus phases to Decomposition stages |
| `openevolve_mcp_tools.py` | 745 | Direct OpenEvolve MCP tools (for standalone evolutionary coding) |
| `hephaestus_openevolve_bridge.py` | 450 | Bridge for pure evolutionary coding workflows |

### Delegation Files

| File | Lines | Purpose |
|------|-------|---------|
| `openevolve_hephaustus_delegation.py` | 850 | Main delegation using HephaestusSDK |
| `openevolve_hephaustus_adapter.py` | 500 | Adapter for existing code |

---

## MCP Tools

### Decomposition Workflow MCP Tools (9 tools with OpenEvolve integration)

| Tool | Purpose | OpenEvolve Integration |
|------|---------|------------------------|
| `analyze_problem_for_decomposition` | Stage 0: Problem analysis | ✅ Evolves analysis perspectives |
| `decompose_problem_into_sub_problems` | Stage 1: Decomposition | ✅ Evolves decomposition strategies |
| `create_decomposition_plan` | Create plan with teams/gauntlets | - |
| `solve_sub_problem_with_team` | Stage 3A: Blue Team solving | ✅ Evolves solution attempts |
| `critique_solution_with_gauntlet` | Stage 3B: Red Team critique | ✅ Evolves critique perspectives |
| `verify_solution_with_gauntlet` | Stage 3C: Gold Team verification | ✅ Evolves verification perspectives |
| `list_available_teams` | List all teams | - |
| `list_available_gauntlets` | List all gauntlets | - |
| `get_decomposition_status` | System status | - |

### Direct OpenEvolve MCP Tools (7 tools)

| Tool | Purpose |
|------|---------|
| `evolve_code_with_openevolve` | Evolve/optimize code |
| `evolve_function_with_openevolve` | Evolve Python function |
| `optimize_algorithm_with_openevolve` | Optimize algorithm class |
| `discover_algorithm_with_openevolve` | Discover novel algorithms |
| `optimize_prompt_with_openevolve` | Optimize LLM prompts |
| `list_openevolve_capabilities` | List available capabilities |
| `get_openevolve_status` | Get installation status |

---

## Phase/Stage Mapping

| Hephaestus Phase | Decomposition Stage | OpenEvolve Activity |
|------------------|---------------------|---------------------|
| Phase 1: Problem Setup | Stage 0 (Analysis) + Stage 1 (Decomposition) | ✅ Evolves analysis and decomposition strategies |
| Phase 2: Solution Generation | Stage 3A (Blue Team Solving) | ✅ Evolves solution attempts |
| Phase 3: Adversarial Critique | Stage 3B (Red Team Gauntlet) | ✅ Evolves critique perspectives |
| Phase 4: Verification | Stage 3C (Gold Team Gauntlet) | ✅ Evolves verification perspectives |
| Phase 5: Reassembly | Stage 4 (Reassembly) | ✅ Evolves reassembly strategies |
| Phase 6: Final Validation | Stage 5 + Stage 6 (Validation & Learning) | ✅ Evolves validation criteria |

---

## Example Usage

### Full Decomposition Workflow with OpenEvolve Evolution

```python
from decomposition_hephaestus_bridge import DecompositionHephaestusWorkflowBridge

bridge = DecompositionHephaestusWorkflowBridge()

# Execute full workflow - OpenEvolve is used automatically in all stages
result = bridge.execute_full_workflow(
    problem_statement="Design a scalable microservices architecture",
    problem_type="architecture",
    domain="software",
    max_sub_problems=10,
    decomposition_strategy="semantic",
    reassembly_strategy="verified_only",
    validation_criteria=["all_verified", "min_quality_threshold"],
)

if result["workflow_status"] == "completed":
    print(f"Final Output:\n{result['final_output']}")
    print(f"Validation Passed: {result['validation_passed']}")

# Each stage includes evolution metrics:
for phase_id, phase_result in result["phases"].items():
    if "evolution_metrics" in str(phase_result):
        print(f"Phase {phase_id} used OpenEvolve evolution")
```

### Individual Stage with Evolution

```python
from decomposition_mcp_tools import solve_sub_problem_with_team

# Solve with OpenEvolve evolution (default)
result = solve_sub_problem_with_team(
    sub_problem_id="SP-001",
    sub_problem_description="Implement authentication service",
    team_name="blue-team-alpha",
    use_evolution=True,  # Enable OpenEvolve evolution
    evolution_iterations=100,  # Number of evolutionary iterations
)

print(f"Solution: {result['solution']}")
print(f"Evolution Metrics: {result['evolution_metrics']}")
# Output: {
#   "iterations": 100,
#   "improvement": 0.45,
#   "best_fitness": 0.85
# }
```

### Disable Evolution for Faster Execution

```python
# All MCP tools support use_evolution=False for faster execution
result = decompose_problem_into_sub_problems(
    problem_statement="Design authentication system",
    use_evolution=False,  # Use standard decomposition without evolution
)
```

---

## Summary

**Architecture**: Hephaestus (Orchestrator) → Decomposition Workflow (Teams/Gauntlets) → OpenEvolve (Evolutionary Engine)

**Key Points**:
- Hephaestus orchestrates the overall workflow
- Decomposition Workflow manages teams (Blue/Red/Gold) and gauntlets
- OpenEvolve provides evolutionary permutations in ALL stages
- Every stage can be configured to use or disable evolution
- Evolution metrics are tracked and returned for analysis

**NO PLACEHOLDERS. NO STUBS. NO TOY IMPLEMENTATIONS.**

**EVERYTHING IS PRODUCTION-READY CODE.**

---

**Date**: 2025-12-29
**Status**: COMPLETE ✅
**Integrations**:
- Hephaestus (orchestrator) - ✅
- Decomposition Workflow (teams/gauntlets) - ✅
- OpenEvolve (evolutionary in ALL stages) - ✅
