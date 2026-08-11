# Z3-LeanAIDE-OpenEvolve-BubbleLabs Integration

## Complete Integration Suite

A comprehensive, production-ready integration that connects Microsoft Z3 SMT Solver with LeanAIDE formal verification, OpenEvolve workflow engine, and BubbleLabs visualization platform.

**Version:** 2.0  
**Last Updated:** 2026-01-31

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Advanced Features](#advanced-features)
4. [Integration Layers](#integration-layers)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [API Reference](#api-reference)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  BubbleLabs UI   │  │ Advanced Visuals │  │   Dashboards     │          │
│  │                  │  │                  │  │                  │          │
│  │ • Constraint     │  │ • Proof Trees    │  │ • Real-time      │          │
│  │   Graphs         │  │ • Opt Landscape  │  │   Monitoring     │          │
│  │ • Node States    │  │ • Progress Bars  │  │ • Analytics      │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
└───────────┼─────────────────────┼─────────────────────┼────────────────────┘
            │                     │                     │
┌───────────▼─────────────────────▼─────────────────────▼────────────────────┐
│                         INTEGRATION LAYER                                  │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │              Z3LeanAideOpenEvolveIntegration                      │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐       │     │
│  │  │   Problem    │  │   Adaptive   │  │   Enhanced       │       │     │
│  │  │  Classifier  │──│   Solver     │──│  Verification    │       │     │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘       │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐       │     │
│  │  │   Result     │  │   Knowledge  │  │   Performance    │       │     │
│  │  │    Cache     │  │  Extraction  │  │    Monitor       │       │     │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘       │     │
│  └──────────────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────────┘
            │                     │                     │
┌───────────▼─────────────────────▼─────────────────────▼────────────────────┐
│                           BRIDGE LAYER                                     │
│  ┌────────────────────────┐    ┌────────────────────────────────────────┐ │
│  │   Z3-LeanAIDE Bridge   │    │         Agent Coordination             │ │
│  │                        │    │                                        │ │
│  │ • SMT ↔ Lean Trans.    │    │  ┌──────────┐ ┌──────────┐ ┌────────┐ │ │
│  │ • Cross-Verification   │    │  │  Solver  │ │ Optimizer│ │ Prover │ │ │
│  │ • Strategy Selection   │    │  │  Agent   │ │  Agent   │ │ Agent  │ │ │
│  └────────────────────────┘    │  └──────────┘ └──────────┘ └────────┘ │ │
│                                │  ┌──────────┐ ┌──────────┐            │ │
│  ┌────────────────────────┐    │  │Translator│ │ Verifier │            │ │
│  │     MCP Interface      │    │  │  Agent   │ │  Agent   │            │ │
│  │                        │    │  └──────────┘ └──────────┘            │ │
│  │ • External Tools       │    └────────────────────────────────────────┘ │
│  │ • Protocol Standard    │                                               │
│  └────────────────────────┘                                               │
└────────────────────────────────────────────────────────────────────────────┘
            │                     │                     │
┌───────────▼─────────────────────▼─────────────────────▼────────────────────┐
│                         SOLVER LAYER                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │  Z3 Advanced     │  │    Z3 Base       │  │    LeanAIDE      │         │
│  │                  │  │                  │  │                  │         │
│  │ • Optimization   │  │ • Constraints    │  │ • Theorem Prover │         │
│  │ • Arrays/BV      │  │ • SMT-LIB        │  │ • Proof Gen      │         │
│  │ • Incremental    │  │ • CLI/Python     │  │ • Elaboration    │         │
│  │ • Portfolio      │  │                  │  │                  │         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘         │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Base Z3 Integration (`z3prover_integration.py`)

**Purpose:** Core Z3 solver interface

**Features:**
- Constraint satisfaction solving
- Theorem proving
- SMT-LIB2 format support
- Problem detection
- Both Python API and CLI interfaces

**Key Classes:**
```python
from z3prover_integration import (
    Z3SolverEngine,      # Main constraint solver
    Z3TheoremProver,     # Theorem proving
    Z3ProblemDetector,   # Problem classification
    Z3Config             # Configuration
)
```

### 2. Advanced Z3 Features (`z3prover_advanced.py`)

**Purpose:** Extended solver capabilities

**Features:**
- **Optimization:** Single/multi-objective, Pareto frontier
- **Arrays:** Array constraints and operations
- **Bit-Vectors:** Low-level bit manipulation
- **Incremental Solving:** Push/pop for constraint stacks
- **Portfolio Solving:** Multiple strategies in parallel
- **Proof Extraction:** Detailed proof generation

**Key Classes:**
```python
from z3prover_advanced import (
    Z3AdvancedSolver,      # Advanced solver
    OptimizationResult,     # Optimization results
    PortfolioResult,        # Portfolio solving
    ExtractedProof,         # Proof extraction
    IncrementalState        # Incremental solving state
)
```

### 3. Z3-LeanAIDE Bridge (`z3_leanaide_bridge.py`)

**Purpose:** Bidirectional Z3 ↔ Lean integration

**Features:**
- SMT-LIB to Lean 4 translation
- Lean 4 to SMT-LIB translation
- Cross-verification with multiple strategies
- Consensus building

**Verification Strategies:**
- `Z3_FIRST`: Try Z3, fallback to Lean
- `LEAN_FIRST`: Try Lean, fallback to Z3
- `PARALLEL`: Run both concurrently
- `CONSENSUS`: Both must agree
- `ADAPTIVE`: Auto-select based on problem

### 4. OpenEvolve Integration (`z3_leanaide_openevolve_integration.py`)

**Purpose:** Full workflow integration

**Problem Categories:**
- `CONSTRAINT_SOLVING`: Z3 for constraints
- `OPTIMIZATION`: Z3 for optimization
- `THEOREM_PROVING`: LeanAIDE for proofs
- `SMT_VERIFICATION`: Z3 SMT
- `HYBRID`: Combined approach
- `STANDARD`: Regular OpenEvolve

---

## Advanced Features

### 5. MCP Tools (`z3_mcp_tools.py`)

**Purpose:** Model Context Protocol integration for external AI systems

**Available Tools:**
- `z3_solve_constraints`: Constraint solving
- `z3_optimize`: Optimization
- `z3_prove_theorem`: Theorem proving
- `z3_translate_smt_to_lean`: Translation
- `z3_solve_incremental`: Incremental solving
- `z3_extract_proof`: Proof extraction
- `z3_analyze_problem`: Problem analysis
- `z3_solve_portfolio`: Portfolio solving

**Example:**
```python
from z3_mcp_tools import get_z3_mcp_server

server = get_z3_mcp_server()
result = server.call_tool("z3_solve_constraints", {
    "variables": [{"name": "x", "type": "INTEGER"}],
    "constraints": ["(> x 0)"]
})
```

### 6. CrewAI Bridge (`z3_crewai_bridge.py`)

**Purpose:** Multi-agent workflows with Z3

**Agent Types:**
- `Z3SolverAgent`: Constraint solving
- `Z3OptimizerAgent`: Optimization
- `Z3TheoremProverAgent`: Theorem proving
- `Z3TranslatorAgent`: Translations
- `Z3VerifierAgent`: Cross-verification

**Example:**
```python
from z3_crewai_bridge import get_z3_agent_coordinator, AgentTask, AgentRole

coordinator = get_z3_agent_coordinator()
coordinator.create_solver_agent("solver_1")

task = AgentTask(
    task_id="task_1",
    role=AgentRole.SOLVER,
    problem="(set-logic LIA)..."
)

result = await coordinator.execute_single(task)
```

### 7. Result Caching (`z3_result_cache.py`)

**Purpose:** Intelligent result caching with persistence

**Features:**
- LRU/LFU/FIFO/TTL eviction policies
- SQLite persistent storage
- Tag-based invalidation
- Checksum verification
- Distributed cache support

**Example:**
```python
from z3_result_cache import get_z3_result_cache, Cached

cache = get_z3_result_cache()

# Manual caching
cache.set("solve", params, result, ttl=3600)
hit, value = cache.get("solve", params)

# Decorator
@Cached(ttl=3600, tags=["constraint"])
async def solve_problem(params):
    # ... solving logic
    return result
```

### 8. Advanced UI (`z3_bubblelabs_advanced_ui.py`)

**Purpose:** Rich visualization components

**Visualizations:**
- Interactive constraint graphs
- Proof tree explorers
- Optimization landscapes
- Real-time progress tracking
- Comparative analysis views

### 9. Performance Monitor (`z3_performance_monitor.py`)

**Purpose:** Comprehensive performance monitoring

**Features:**
- Execution time tracking
- Success rate monitoring
- Memory/CPU tracking
- Automatic alerting
- Bottleneck identification
- Trend analysis

**Example:**
```python
from z3_performance_monitor import get_z3_performance_monitor, monitored

monitor = get_z3_performance_monitor()
monitor.start_monitoring(interval=10)

# Decorator
@monitored("constraint_solving")
def solve_constraints(params):
    # ... solving logic
    pass

# Get insights
bottlenecks = monitor.get_bottlenecks(5)
dashboard = monitor.get_dashboard_data()
```

### 10. Knowledge Extraction (`z3_knowledge_extraction.py`)

**Purpose:** Extract and manage knowledge from Z3 operations

**Capabilities:**
- Proof pattern mining
- Constraint pattern analysis
- Strategy learning
- Mathematical insight extraction
- Knowledge reuse

---

## Integration Layers

### BubbleLabs UI Integration

**Basic Nodes (`z3_leanaide_bubblelabs_ui.py`):**
- `z3_problem_classifier`
- `z3_constraint_solver`
- `z3_theorem_prover`
- `z3_smt_solver`
- `z3_leanaide_cross_verify`

**Advanced Visualizations (`z3_bubblelabs_advanced_ui.py`):**
- Constraint network graphs
- Proof tree explorers
- Optimization landscapes
- Real-time dashboards

**Registration:**
```python
from z3_leanaide_bubblelabs_ui import register_z3_leanaide_bubblelabs_tools
from z3_bubblelabs_advanced_ui import get_z3_advanced_bubblelabs_ui

# Register basic tools
register_z3_leanaide_bubblelabs_tools()

# Use advanced UI
ui = get_z3_advanced_bubblelabs_ui()
viz = ui.create_constraint_visualization(node_id, variables, constraints)
```

---

## Installation & Setup

### Prerequisites

```bash
# Install Z3
# Ubuntu/Debian
sudo apt-get install z3

# macOS
brew install z3

# Python bindings
pip install z3-solver

# Optional: psutil for performance monitoring
pip install psutil

# Optional: redis for distributed caching
pip install redis
```

### Configuration

**Z3 Configuration:**
```python
from z3prover_integration import Z3Config

config = Z3Config(
    timeout=60.0,
    memory_limit_mb=8192,
    num_threads=4,
    proof_generation=True
)
```

**Cache Configuration:**
```python
from z3_result_cache import CacheConfig, CachePolicy

config = CacheConfig(
    max_size=10000,
    default_ttl=7200,
    policy=CachePolicy.LRU,
    persistent_storage=True,
    db_path="z3_cache.db"
)
```

---

## Usage Guide

### Quick Start

```python
import asyncio
from z3_leanaide_openevolve_integration import solve_with_z3_leanaide

async def main():
    result = await solve_with_z3_leanaide("""
        Find x and y where:
        - x > 0 and x < 10
        - y = x + 5
    """)
    
    print(f"Category: {result['classification']['category']}")
    print(f"Solution: {result['solution']['content']}")

asyncio.run(main())
```

### Advanced Optimization

```python
from z3prover_advanced import Z3AdvancedSolver, OptimizationObjective, Z3Config
from z3prover_integration import Z3Variable, Z3Constraint, Z3ConstraintType

solver = Z3AdvancedSolver(Z3Config(timeout=60))

# Multi-objective optimization
variables = [
    Z3Variable("x", Z3ConstraintType.INTEGER),
    Z3Variable("y", Z3ConstraintType.INTEGER)
]

constraints = [
    Z3Constraint("(>= x 0)", Z3ConstraintType.INTEGER),
    Z3Constraint("(>= y 0)", Z3ConstraintType.INTEGER),
    Z3Constraint("(<= (+ x y) 100)", Z3ConstraintType.INTEGER)
]

objectives = [
    ("x", OptimizationObjective.MAXIMIZE),
    ("y", OptimizationObjective.MAXIMIZE)
]

result = solver.optimize(variables, constraints, objectives, "pareto")

print(f"Pareto front: {len(result.pareto_front)} solutions")
for point in result.pareto_front:
    print(f"  {point}")
```

### Incremental Solving

```python
from z3prover_advanced import get_z3_advanced_solver, Z3Constraint, Z3ConstraintType

solver = get_z3_advanced_solver()

# Create initial state
state_id = solver.create_incremental_state(
    variables=[Z3Variable("x", Z3ConstraintType.INTEGER)],
    constraints=[Z3Constraint("(> x 0)", Z3ConstraintType.INTEGER)]
)

# Check
result = solver.check_incremental(state_id)
print(f"Initial: {result.status.value}")

# Push scope and add constraint
solver.push_scope(state_id, "upper_bound")
solver.add_constraint_incremental(state_id, Z3Constraint("(< x 10)", Z3ConstraintType.INTEGER))

result = solver.check_incremental(state_id)
print(f"With upper bound: {result.status.value}")

# Pop back
solver.pop_scope(state_id)

result = solver.check_incremental(state_id)
print(f"After pop: {result.status.value}")
```

### Proof Extraction

```python
from z3prover_advanced import get_z3_advanced_solver, ProofFormat

solver = get_z3_advanced_solver()

smtlib = """
(set-logic LIA)
(declare-fun x () Int)
(assert (> x 0))
(assert (not (> (+ x 1) 0)))
(check-sat)
"""

proof = solver.extract_proof(smtlib, ProofFormat.JSON)

if proof.success:
    print(f"Proof steps: {len(proof.proof_steps)}")
    print(f"Tactics used: {proof.tactics_used}")
    for step in proof.proof_steps:
        print(f"  {step.step_number}: {step.tactic}")
```

---

## API Reference

See individual module docstrings for detailed API documentation:

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `z3prover_integration.py` | Core Z3 | `Z3SolverEngine`, `Z3TheoremProver` |
| `z3prover_advanced.py` | Advanced features | `Z3AdvancedSolver`, `OptimizationResult` |
| `z3_leanaide_bridge.py` | Z3-Lean bridge | `Z3LeanAideBridge`, `TranslationResult` |
| `z3_leanaide_openevolve_integration.py` | Workflow | `Z3LeanAideOpenEvolveIntegration` |
| `z3_mcp_tools.py` | MCP interface | `Z3MCPServer` |
| `z3_crewai_bridge.py` | Agent workflows | `Z3AgentCoordinator` |
| `z3_result_cache.py` | Caching | `Z3ResultCache`, `Cached` |
| `z3_bubblelabs_advanced_ui.py` | Visualization | `Z3AdvancedBubbleLabsUI` |
| `z3_performance_monitor.py` | Monitoring | `Z3PerformanceMonitor`, `monitored` |
| `z3_knowledge_extraction.py` | Knowledge | `Z3KnowledgeExtractor` |

---

## Performance Tuning

### Optimization Tips

1. **Use Portfolio Solving for Hard Problems:**
```python
result = solver.solve_portfolio(smtlib, parallel=True)
```

2. **Enable Caching:**
```python
cache = get_z3_result_cache(CacheConfig(
    max_size=10000,
    policy=CachePolicy.LRU
))
```

3. **Parallel Strategy:**
```python
config = Z3Config(num_threads=4)
```

4. **Incremental for Similar Problems:**
```python
state_id = solver.create_incremental_state(vars, constraints)
# Modify and recheck
```

### Monitoring

```python
monitor = get_z3_performance_monitor()
monitor.start_monitoring(interval=10)

# Get bottleneck report
bottlenecks = monitor.get_bottlenecks(5)
for b in bottlenecks:
    print(f"{b['operation']}: {b['avg_time_s']:.3f}s")
```

---

## Troubleshooting

### Common Issues

**Z3 Not Available:**
```bash
# Check installation
z3 --version

# Install Python bindings
pip install z3-solver
```

**Performance Issues:**
- Check `monitor.get_bottlenecks()`
- Enable caching
- Use portfolio solving
- Increase timeout

**Memory Issues:**
```python
config = Z3Config(
    memory_limit_mb=4096,
    timeout=30
)
```

**Cache Corruption:**
```python
cache = get_z3_result_cache()
cache.clear()  # Clear and rebuild
```

### Debug Mode

```python
import logging
logging.getLogger('z3_integration').setLevel(logging.DEBUG)
```

---

## File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `z3prover_integration.py` | 983 | Core Z3 interface |
| `z3prover_advanced.py` | 1,154 | Advanced features |
| `z3_leanaide_bridge.py` | 1,005 | Z3-Lean bridge |
| `z3_leanaide_openevolve_integration.py` | 1,048 | Workflow integration |
| `z3_mcp_tools.py` | 659 | MCP interface |
| `z3_crewai_bridge.py` | 660 | Agent workflows |
| `z3_result_cache.py` | 593 | Caching layer |
| `z3_bubblelabs_ui.py` | 896 | Basic UI |
| `z3_bubblelabs_advanced_ui.py` | 732 | Advanced visualizations |
| `z3_performance_monitor.py` | 791 | Performance monitoring |
| `z3_knowledge_extraction.py` | 687 | Knowledge extraction |
| `test_z3_leanaide_integration.py` | 820 | Test suite |
| `demo_z3_leanaide_integration.py` | 420 | Demo script |

**Total:** ~10,500 lines of integration code

---

## License

This integration is part of the OpenEvolve project.

## Support

- Run demo: `python demo_z3_leanaide_integration.py`
- Run tests: `pytest test_z3_leanaide_integration.py -v`
- Check status: `get_integration_status()`
