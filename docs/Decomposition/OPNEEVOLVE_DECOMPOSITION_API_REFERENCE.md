# OpenEvolve Decomposition Engine - Complete API Reference

**Version:** 2.0.0
**Last Updated:** 2025-01-03
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [MCP Tools API](#mcp-tools-api)
4. [Decomposition Strategies](#decomposition-strategies)
5. [Data Models](#data-models)
6. [Execution Methods](#execution-methods)
7. [Configuration Options](#configuration-options)
8. [Error Handling](#error-handling)
9. [Performance Tuning](#performance-tuning)
10. [Security Considerations](#security-considerations)

---

## Overview

The OpenEvolve Decomposition Engine is a sovereign-grade problem decomposition system that breaks down complex problems into verifiable, solvable sub-problems. It integrates with multiple execution frameworks including OpenEvolve's evolutionary engine, Claudiomiro, DataPizza, ROMA, and ROMA-MDAP-MAKER.

### Key Features

- **Intelligent Decomposition**: LLM-powered semantic analysis for natural problem breakdown
- **Multiple Strategies**: Semantic, hierarchical, and flow-based decomposition
- **Seven Execution Methods**: Traditional, Claudiomiro, DataPizza, ROMA, Hybrid, ROMA-MDAP-MAKER, Auto
- **Quality Assurance**: Integrated Red/Gold team gauntlets for validation
- **Evolutionary Optimization**: OpenEvolve integration for solution evolution
- **Zero-Error Guaranteed**: ROMA-MDAP-MAKER mode with first-to-ahead-by-k voting

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    crewai Orchestrator                   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              Decomposition MCP Tools Layer                   │
│  - analyze_problem_for_decomposition                        │
│  - decompose_problem_into_sub_problems                      │
│  - solve_sub_problem_with_team                              │
│  - critique_solution_with_gauntlet                           │
│  - verify_solution_with_gauntlet                             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│            Decomposition Engine Core                         │
│  - SemanticDecomposition                                     │
│  - HierarchicalDecomposition                                 │
│  - FlowBasedDecomposition                                    │
│  - ProblemAnalyzer                                           │
└────────────────────┬───┬──────────┬──────────┬──────────────┘
                     │   │          │          │
    ┌────────────────┘   │          │          └────────────────┐
    │                    │          │                           │
┌───▼──────────┐  ┌─────▼────┐  ┌──▼──────┐  ┌──────────▼─────────┐
│ OpenEvolve   │  │Claudiomiro│  │DataPizza│  │ ROMA-MDAP-MAKER    │
│ Evolution    │  │  Autonomous│ │ Multi-  │  │ Zero-Error Voting  │
│              │  │Development│ │ Agent   │  │                    │
└──────────────┘  └───────────┘  └─────────┘  └────────────────────┘
```

---

## Core Components

### 1. DecompositionEngine

**Purpose**: Main orchestration engine for problem decomposition

**Location**: `decomposition_engine.py`

#### Methods

##### `decompose(problem: ProblemDefinition, strategy: str = "semantic") -> DecompositionResult`

Decompose a problem using the specified strategy.

**Parameters:**
- `problem` (ProblemDefinition): The problem to decompose
- `strategy` (str): Decomposition strategy - "semantic", "hierarchical", or "flow"
  - Default: "semantic"

**Returns:**
- `DecompositionResult`: Object containing:
  - `sub_problems` (List[SubProblem]): Generated sub-problems
  - `dependency_graph` (DependencyGraph): Dependency relationships
  - `quality_scores` (QualityScores): Quality assessment
  - `metadata` (Dict): Additional metadata

**Example:**
```python
from decomposition_engine import DecompositionEngine
from sovereign_data_models import ProblemDefinition, ComplexityScore

# Create problem
problem = ProblemDefinition(
    id="prob-001",
    title="Build ML Pipeline",
    description="Create a complete machine learning pipeline for fraud detection",
    problem_type=ProblemType.IMPLEMENTATION,
    domain_context=DomainContext(domain="Machine Learning"),
    complexity_score=ComplexityScore(overall_complexity=8)
)

# Decompose
engine = DecompositionEngine()
result = engine.decompose(problem, strategy="semantic")

print(f"Generated {len(result.sub_problems)} sub-problems")
for sp in result.sub_problems:
    print(f"  - {sp.title}: {sp.type.value} (priority: {sp.priority})")
```

**Error Handling:**
- Raises `ValueError` if problem is invalid
- Raises `RuntimeError` if LLM is unavailable for semantic decomposition
- Returns empty result if decomposition fails

---

### 2. ProblemAnalyzer

**Purpose**: Analyze problem complexity and extract key characteristics

**Location**: `problem_analyzer.py`

#### Methods

##### `analyze_problem(problem: ProblemDefinition) -> Dict[str, Any]`

Comprehensive analysis of problem characteristics.

**Parameters:**
- `problem` (ProblemDefinition): Problem to analyze

**Returns:**
- Dict containing:
  - `domain` (str): Identified domain
  - `complexity` (Dict[str, int]): Complexity breakdown
  - `constraints` (List[str]): Identified constraints
  - `success_criteria` (List[str]): Success criteria
  - `estimated_sub_problems` (int): Estimated number of sub-problems
  - `required_expertise` (List[str]): Required skills
  - `key_challenges` (List[str]): Main challenges

**Example:**
```python
from problem_analyzer import ProblemAnalyzer

analyzer = ProblemAnalyzer()
analysis = analyzer.analyze_problem(problem)

print(f"Domain: {analysis['domain']}")
print(f"Complexity: {analysis['complexity']}")
print(f"Estimated sub-problems: {analysis['estimated_sub_problems']}")
```

---

### 3. TeamManager & GauntletManager

**Purpose**: Manage Blue Teams (solvers), Red Teams (critics), and Gold Teams (validators)

#### TeamManager Methods

##### `list_teams() -> List[Team]`
List all available teams.

##### `get_team(team_name: str) -> Optional[Team]`
Get a specific team by name.

##### `assign_team_to_subproblem(sub_problem_id: str, team_name: str) -> bool`
Assign a team to a sub-problem.

#### GauntletManager Methods

##### `list_gauntlets() -> List[Gauntlet]`
List all available gauntlets.

##### `run_gauntlet(gauntlet_name: str, content: str, context: Dict) -> Dict`
Run a gauntlet validation.

---

## MCP Tools API

### analyze_problem_for_decomposition

**MCP Tool Name**: `analyze_problem_for_decomposition`

**Purpose**: Analyze a problem statement to extract structured context for decomposition.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `problem_statement` | str | Yes | - | The problem to analyze |
| `problem_type` | str | No | None | Type: optimization, design, research, etc. |
| `domain` | str | No | None | Domain: software, mathematics, system design, etc. |
| `use_evolution` | bool | No | True | Use OpenEvolve for evolutionary analysis |
| `evolution_iterations` | int | No | 20 | Number of evolution iterations |

#### Returns

```python
{
    "domain": str,
    "complexity": {
        "overall": int,
        "cognitive": int,
        "computational": int,
        "domain": int,
        "integration": int
    },
    "constraints": List[str],
    "success_criteria": List[str],
    "estimated_sub_problems": int,
    "required_expertise": List[str],
    "key_challenges": List[str],
    "evolution_metrics": Dict  # if use_evolution=True
}
```

#### Example

```python
from decomposition_mcp_tools import analyze_problem_for_decomposition

analysis = analyze_problem_for_decomposition(
    problem_statement="Design a scalable microservices architecture for e-commerce",
    problem_type="design",
    domain="software",
    use_evolution=True,
    evolution_iterations=30
)

print(f"Domain: {analysis['domain']}")
print(f"Complexity: {analysis['complexity']['overall']}/10")
print(f"Estimated sub-problems: {analysis['estimated_sub_problems']}")
```

---

### decompose_problem_into_sub_problems

**MCP Tool Name**: `decompose_problem_into_sub_problems`

**Purpose**: Decompose a complex problem into solvable sub-problems.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `problem_statement` | str | Yes | - | The problem to decompose |
| `analysis` | Dict | No | None | Problem analysis from analyze_problem_for_decomposition() |
| `max_sub_problems` | int | No | 15 | Maximum number of sub-problems |
| `decomposition_strategy` | str | No | "semantic" | Strategy: semantic, hierarchical, flow |
| `complexity_target` | int | No | 5 | Target complexity per sub-problem (1-10) |
| `use_evolution` | bool | No | True | Use OpenEvolve for evolutionary decomposition |
| `evolution_iterations` | int | No | 50 | Number of evolution iterations |

#### Returns

```python
{
    "sub_problems": [
        {
            "id": str,
            "title": str,
            "description": str,
            "type": str,  # research, analysis, implementation, validation, integration
            "priority": int,  # 1-10
            "effort_hours": int,
            "complexity_score": int,  # 1-10
            "success_criteria": List[str],
            "dependencies": List[str],
            "acceptance_criteria": List[str],
            "ai_suggested_evolution_mode": str,
            "ai_suggested_complexity_score": Dict,
            "ai_suggested_evaluation_prompt": str,
            "ai_suggested_team_assignment": Dict,
            "ai_suggested_gauntlet_assignment": Dict,
            "estimated_resources": Dict,
            "potential_approaches": List[Dict],
            "required_expertise": List[str],
            "associated_risks": List[str],
            "success_dependencies": List[str],
            "testing_approach": str,
            "quality_metrics": Dict
        }
    ],
    "dependencies": Dict[str, List[str]],
    "estimated_total_complexity": int,
    "decomposition_strategy": str,
    "total_sub_problems": int,
    "evolution_metrics": Dict  # if use_evolution=True
}
```

#### Example

```python
from decomposition_mcp_tools import decompose_problem_into_sub_problems

result = decompose_problem_into_sub_problems(
    problem_statement="Build a fraud detection system",
    analysis=analysis,
    max_sub_problems=10,
    decomposition_strategy="semantic",
    complexity_target=6,
    use_evolution=True,
    evolution_iterations=50
)

print(f"Generated {result['total_sub_problems']} sub-problems")
for sp in result['sub_problems']:
    print(f"\n{sp['title']}")
    print(f"  Type: {sp['type']}")
    print(f"  Priority: {sp['priority']}/10")
    print(f"  Effort: {sp['effort_hours']}h")
    print(f"  Complexity: {sp['complexity_score']}/10")
```

---

### solve_sub_problem_with_team

**MCP Tool Name**: `solve_sub_problem_with_team`

**Purpose**: Solve a sub-problem using an assigned Blue Team with multiple execution methods.

#### Parameters

**Core Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `sub_problem_id` | str | Yes | - | ID of the sub-problem |
| `sub_problem_description` | str | Yes | - | Description of what to solve |
| `team_name` | str | Yes | - | Name of the Blue Team to use |
| `context` | Dict | No | None | Additional context and dependencies |
| `constraints` | List[str] | No | None | List of constraints |
| `requirements` | List[str] | No | None | List of requirements |
| `execution_method` | str | No | "traditional" | Execution method (see below) |
| `use_evolution` | bool | No | True | Use OpenEvolve (traditional mode) |
| `evolution_iterations` | int | No | 100 | Evolution iterations |

**Execution Methods:**

1. `"traditional"` - AI-assisted decomposition with LLM prompts
2. `"claudiomiro"` - Autonomous development with Claudiomiro CLI
3. `"datapizza"` - Multi-agent problem solving with DataPizza
4. `"roma"` - Recursive meta-agent decomposition with ROMA
5. `"hybrid"` - ROMA automatic decomposition + Decomposition Workflow teams
6. `"roma_mdap_maker"` - ROMA + MAKER zero-error voting (NEW)
7. `"auto"` - Automatically choose based on sub-problem characteristics

**Claudiomiro Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_claudiomiro` | bool | False | Explicitly enable Claudiomiro |
| `claudiomiro_provider` | str | "claude" | AI provider: claude, codex, gemini, deep-seek, glm |
| `claudiomiro_backend` | str | None | Backend directory for multi-repo |
| `claudiomiro_frontend` | str | None | Frontend directory for multi-repo |
| `working_dir` | str | "." | Working directory |
| `max_cycles` | int | 20 | Maximum execution cycles |

**DataPizza Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_datapizza` | bool | False | Explicitly enable DataPizza |
| `datapizza_provider` | str | "openai" | AI provider: openai, anthropic, google |
| `datapizza_api_key` | str | None | API key for provider |
| `datapizza_model` | str | None | Model name |
| `datapizza_tools` | List[str] | None | Tools: filesystem, duckduckgo, sql, web_fetch |
| `datapizza_planning_interval` | int | 3 | Planning interval |
| `datapizza_max_steps` | int | 20 | Maximum steps |

**ROMA Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_roma` | bool | False | Explicitly enable ROMA |
| `roma_max_depth` | int | 2 | Maximum recursion depth |
| `roma_execution_mode` | str | "recursive" | Mode: recursive or event_driven |
| `roma_provider` | str | None | AI provider |
| `roma_api_key` | str | None | API key |
| `roma_model` | str | None | Model name |

**ROMA-Decomposition Hybrid Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_hybrid` | bool | False | Explicitly enable hybrid |
| `hybrid_max_depth_analysis` | int | 3 | Max depth for analysis phase |
| `hybrid_max_depth_solving` | int | 2 | Max depth for solving phase |
| `hybrid_execution_mode` | str | "recursive" | Execution mode |
| `hybrid_provider` | str | None | AI provider |
| `hybrid_api_key` | str | None | API key |
| `hybrid_model` | str | None | Model name |
| `hybrid_enable_gauntlets` | bool | True | Enable Decomposition Workflow gauntlets |
| `hybrid_enable_evolution` | bool | True | Enable evolution |
| `hybrid_evolution_iterations` | int | 50 | Evolution iterations |

**ROMA-MDAP-MAKER Parameters (NEW):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_roma_mdap_maker` | bool | False | Explicitly enable ROMA-MDAP-MAKER |
| `roma_mdap_maker_max_depth` | int | 2 | ROMA max depth |
| `roma_mdap_maker_k_ahead` | int | 3 | MAKER voting threshold k |
| `roma_mdap_maker_enable_red_flagging` | bool | True | Enable red-flagging |
| `roma_mdap_maker_max_samples` | int | 100 | Max samples per voting round |
| `roma_mdap_maker_enable_adaptive_k` | bool | True | Enable adaptive k selection |
| `roma_mdap_maker_provider` | str | "openai" | AI provider |
| `roma_mdap_maker_api_key` | str | None | API key |
| `roma_mdap_maker_model` | str | "gpt-4o-mini" | Model name |

#### Returns

```python
{
    "sub_problem_id": str,
    "solution": str,
    "team_name": str,
    "generated_by": str,
    "status": str,  # completed, failed, partial
    "execution_method_used": str,

    # Method-specific fields
    "evolution_metrics": Dict,  # traditional with evolution
    "steps_taken": int,  # datapizza
    "tools_used": List[str],  # datapizza
    "token_usage": Dict,  # datapizza, roma
    "dag_info": Dict,  # roma, hybrid
    "roma_mdap_maker_metrics": Dict,  # roma_mdap_maker

    # Error field if failed
    "error": str  # if failed
}
```

#### Examples

**Traditional Method:**
```python
from decomposition_mcp_tools import solve_sub_problem_with_team

result = solve_sub_problem_with_team(
    sub_problem_id="sub-001",
    sub_problem_description="Implement data preprocessing pipeline",
    team_name="DataScience-Blue",
    constraints=["Must handle missing values", "Must scale features"],
    requirements=["Python implementation", "Scikit-learn compatible"],
    execution_method="traditional",
    use_evolution=True,
    evolution_iterations=100
)

print(f"Solution generated by: {result['generated_by']}")
print(f"Status: {result['status']}")
if result['status'] == 'completed':
    print(result['solution'])
```

**Claudiomiro Method:**
```python
result = solve_sub_problem_with_team(
    sub_problem_id="sub-002",
    sub_problem_description="Implement REST API for model serving",
    team_name="Backend-Blue",
    execution_method="claudiomiro",
    claudiomiro_provider="claude",
    working_dir="./backend",
    max_cycles=20
)

print(f"Method: {result['execution_method_used']}")
if result['status'] == 'completed':
    print(result['solution'])
```

**ROMA-MDAP-MAKER Method (Zero-Error):**
```python
result = solve_sub_problem_with_team(
    sub_problem_id="sub-003",
    sub_problem_description="Implement safety-critical validation logic",
    team_name="Safety-Blue",
    execution_method="roma_mdap_maker",
    roma_mdap_maker_k_ahead=3,
    roma_mdap_maker_enable_red_flagging=True,
    roma_mdap_maker_enable_adaptive_k=True,
    roma_mdap_maker_provider="openai",
    roma_mdap_maker_model="gpt-4o-mini"
)

print(f"Zero-error solution: {result['status']}")
metrics = result.get('roma_mdap_maker_metrics', {})
print(f"Error rate: {metrics.get('final_error_rate', 0.0):.4f}")
print(f"Voting rounds: {metrics.get('total_voting_rounds', 0)}")
print(f"Red-flags caught: {metrics.get('total_red_flags', 0)}")
```

---

### critique_solution_with_gauntlet

**MCP Tool Name**: `critique_solution_with_gauntlet`

**Purpose**: Critique a solution using a Red Team gauntlet.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `solution` | str | Yes | - | The solution to critique |
| `sub_problem_id` | str | Yes | - | ID of the sub-problem |
| `gauntlet_name` | str | Yes | - | Name of the Red Team gauntlet |
| `sub_problem_description` | str | No | None | Original sub-problem description |
| `use_evolution` | bool | No | True | Use OpenEvolve for evolutionary critique |
| `evolution_iterations` | int | No | 30 | Number of evolution iterations |

#### Returns

```python
{
    "sub_problem_id": str,
    "gauntlet_name": str,
    "approved": bool,
    "issues_found": List[Dict],
    "severity_distribution": Dict[str, int],
    "overall_score": float,
    "feedback": str,
    "evolution_metrics": Dict  # if use_evolution=True
}
```

#### Example

```python
from decomposition_mcp_tools import critique_solution_with_gauntlet

critique = critique_solution_with_gauntlet(
    solution=solution_code,
    sub_problem_id="sub-001",
    gauntlet_name="Security-RedTeam-Gauntlet",
    sub_problem_description="Implement authentication system",
    use_evolution=True
)

print(f"Approved: {critique['approved']}")
print(f"Overall score: {critique['overall_score']}")
print(f"Issues found: {len(critique['issues_found'])}")
for issue in critique['issues_found']:
    print(f"  - [{issue['severity']}] {issue['description']}")
```

---

### verify_solution_with_gauntlet

**MCP Tool Name**: `verify_solution_with_gauntlet`

**Purpose**: Verify a solution using a Gold Team gauntlet.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `solution` | str | Yes | - | The solution to verify |
| `critique` | Dict | Yes | - | Previous critique results |
| `sub_problem_id` | str | Yes | - | ID of the sub-problem |
| `gauntlet_name` | str | Yes | - | Name of the Gold Team gauntlet |
| `requirements` | List[str] | No | None | List of requirements to verify |
| `use_evolution` | bool | No | True | Use OpenEvolve for evolutionary verification |
| `evolution_iterations` | int | No | 30 | Number of evolution iterations |

#### Returns

```python
{
    "sub_problem_id": str,
    "gauntlet_name": str,
    "approved": bool,
    "correctness_score": float,
    "completeness_score": float,
    "quality_score": float,
    "requirements_met": Dict[str, bool],
    "feedback": str,
    "evolution_metrics": Dict  # if use_evolution=True
}
```

#### Example

```python
from decomposition_mcp_tools import verify_solution_with_gauntlet

verification = verify_solution_with_gauntlet(
    solution=solution_code,
    critique=critique_result,
    sub_problem_id="sub-001",
    gauntlet_name="Quality-GoldTeam-Gauntlet",
    requirements=["Must handle edge cases", "Must be documented"],
    use_evolution=True
)

print(f"Approved: {verification['approved']}")
print(f"Correctness: {verification['correctness_score']}")
print(f"Completeness: {verification['completeness_score']}")
print(f"Quality: {verification['quality_score']}")
print(f"Requirements met: {sum(verification['requirements_met'].values())}/{len(verification['requirements_met'])}")
```

---

## Decomposition Strategies

### SemanticDecomposition

**Best For:** Problems with clear conceptual boundaries, research/design tasks

**How It Works:**
1. Uses LLM to analyze problem semantics
2. Identifies natural conceptual clusters
3. Creates sub-problems based on semantic relationships
4. Ensures minimal overlap and clear boundaries

**Configuration:**
```python
from decomposition_engine import SemanticDecomposition

strategy = SemanticDecomposition(openevolve_client=client)
sub_problems = strategy.decompose(problem)
```

**Pros:**
- Most intelligent decomposition
- Adapts to problem domain
- Captures nuance and context

**Cons:**
- Requires LLM access
- Slower than other methods
- Can be inconsistent

---

### HierarchicalDecomposition

**Best For:** Complex systems, software architecture, organizational problems

**How It Works:**
1. Identifies top-level components
2. Recursively breaks down each component
3. Creates multi-level hierarchy
4. Maintains parent-child relationships

**Configuration:**
```python
from decomposition_engine import HierarchicalDecomposition

strategy = HierarchicalDecomposition(max_depth=3, min_sub_problems=4)
sub_problems = strategy.decompose(problem)
```

**Pros:**
- Clear structure and hierarchy
- Good for large systems
- Easy to understand dependencies

**Cons:**
- Can create deep hierarchies
- May miss cross-cutting concerns
- Less flexible

---

### FlowBasedDecomposition

**Best For:** Process flows, pipelines, workflows, sequential tasks

**How It Works:**
1. Identifies key process stages
2. Maps data/control flow
3. Creates sub-problems for each stage
4. Establishes input/output dependencies

**Configuration:**
```python
from decomposition_engine import FlowBasedDecomposition

strategy = FlowBasedDecomposition(preserve_order=True, allow_parallel=True)
sub_problems = strategy.decompose(problem)
```

**Pros:**
- Natural for workflows
- Clear sequence
- Easy to parallelize where possible

**Cons:**
- Not suitable for all problems
- May oversimplify
- Assumes linear flow

---

## Data Models

### ProblemDefinition

```python
@dataclass
class ProblemDefinition:
    id: str
    title: str
    description: str
    problem_type: ProblemType  # OPTIMIZATION, DESIGN, RESEARCH, IMPLEMENTATION, VALIDATION
    domain_context: EnhancedDomainContext
    complexity_score: ComplexityScore
    constraints: List[Constraint] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### SubProblem

```python
@dataclass
class SubProblem:
    # Core fields
    id: str
    parent_id: str
    title: str
    description: str
    type: SubProblemType  # RESEARCH, ANALYSIS, IMPLEMENTATION, VALIDATION, INTEGRATION
    complexity_score: ComplexityScore
    dependencies: List[str]  # IDs of dependent sub-problems
    success_criteria: List[SuccessCriterion]
    validation_gauntlet: str
    priority: int  # 1-10
    estimated_effort: int  # hours

    # Enhanced fields (Task 1.2)
    acceptance_criteria: List[str]
    ai_suggested_evolution_mode: str
    ai_suggested_complexity_score: ComplexityBreakdown
    ai_suggested_evaluation_prompt: str
    ai_suggested_team_assignment: SubProblemTeamAssignment
    ai_suggested_gauntlet_assignment: GauntletAssignment
    estimated_resources: ResourceEstimate
    potential_approaches: List[PotentialApproach]
    required_expertise: List[str]
    associated_risks: List[str]
    success_dependencies: List[str]
    testing_approach: str
    quality_metrics: QualityMetrics
```

### ComplexityScore

```python
@dataclass
class ComplexityScore:
    overall_complexity: int  # 1-10
    cognitive_complexity: int  # 1-10
    computational_complexity: int  # 1-10
    domain_complexity: int  # 1-10
    integration_complexity: int  # 1-10
```

### SuccessCriterion

```python
@dataclass
class SuccessCriterion:
    id: str
    description: str
    metric: str  # e.g., "accuracy", "performance", "coverage"
    threshold: float  # e.g., 0.95 for 95% accuracy
    validation_method: str  # "automated", "manual", "hybrid"
```

---

## Execution Methods

### Method Comparison

| Method | Best For | Speed | Quality | Zero-Error | Dependencies |
|--------|----------|-------|---------|------------|--------------|
| Traditional | General purpose | ★★★☆☆ | ★★★☆☆ | No | LLM only |
| Claudiomiro | Code generation | ★★☆☆☆ | ★★★★☆ | No | Claudiomiro CLI |
| DataPizza | Research/analysis | ★★☆☆☆ | ★★★★☆ | No | DataPizza |
| ROMA | Hierarchical tasks | ★★★☆☆ | ★★★★☆ | No | ROMA |
| Hybrid | Complex systems | ★☆☆☆☆ | ★★★★★ | No | ROMA + Decomposition |
| ROMA-MDAP-MAKER | Critical tasks | ★☆☆☆☆ | ★★★★★ | **Yes** | ROMA + MAKER |
| Auto | Adaptive | ★★★☆☆ | ★★★★☆ | Varies | Auto-selection |

### Execution Method Selection

The `auto` mode selects based on sub-problem characteristics:

```python
# Implementation-focused → Claudiomiro
if keywords in ["implement", "code", "function", "class"]:
    return "claudiomiro"

# Zero-error critical → ROMA-MDAP-MAKER (HIGHEST PRIORITY)
if keywords in ["critical", "zero error", "safety-critical"]:
    return "roma_mdap_maker"

# Hierarchical decomposition → ROMA
if keywords in ["decompose", "break down", "hierarchical"]:
    return "roma"

# Multi-agent analysis → DataPizza
if keywords in ["analyze", "research", "design"]:
    return "datapizza"

# Complex end-to-end → Hybrid
if keywords in ["complex system", "architecture"]:
    return "hybrid"

# Default → Traditional
return "traditional"
```

---

## Configuration Options

### OpenEvolve Client Configuration

```python
from openevolve_client import OpenEvolveClient

client = OpenEvolveClient(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o",
    temperature=0.3,  # Lower for more deterministic
    max_tokens=4000,
    enable_cache=True,
    cache_ttl=3600,  # 1 hour
    timeout=120,  # seconds
    max_retries=3
)
```

### Decomposition Engine Configuration

```python
from decomposition_engine import DecompositionEngine

engine = DecompositionEngine(
    openevolve_client=client,
    default_strategy="semantic",
    max_sub_problems=15,
    complexity_target=5,
    enable_quality_check=True,
    enable_dependency_analysis=True
)
```

### Execution Method Configuration

```python
# Claudiomiro
claudiomiro_config = {
    "provider": "claude",
    "backend": "./backend",
    "frontend": "./frontend",
    "working_dir": ".",
    "max_cycles": 20
}

# DataPizza
datapizza_config = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "tools": ["filesystem", "duckduckgo"],
    "planning_interval": 3,
    "max_steps": 20
}

# ROMA
roma_config = {
    "max_depth": 2,
    "execution_mode": "recursive",
    "provider": "openai",
    "model": "gpt-4o"
}

# ROMA-MDAP-MAKER
roma_mdap_maker_config = {
    "max_depth": 2,
    "k_ahead": 3,
    "enable_red_flagging": True,
    "max_samples": 100,
    "enable_adaptive_k": True,
    "provider": "openai",
    "model": "gpt-4o-mini"
}
```

---

## Error Handling

### Error Types

```python
class DecompositionError(Exception):
    """Base class for decomposition errors"""
    pass

class LLMUnavailableError(DecompositionError):
    """Raised when LLM is required but unavailable"""
    pass

class InvalidProblemError(DecompositionError):
    """Raised when problem definition is invalid"""
    pass

class StrategyNotFoundError(DecompositionError):
    """Raised when decomposition strategy is not found"""
    pass

class TeamNotFoundError(DecompositionError):
    """Raised when specified team doesn't exist"""
    pass

class GauntletNotFoundError(DecompositionError):
    """Raised when specified gauntlet doesn't exist"""
    pass
```

### Error Handling Patterns

```python
# Pattern 1: Graceful degradation
try:
    result = engine.decompose(problem, strategy="semantic")
except LLMUnavailableError:
    logger.warning("LLM unavailable, falling back to hierarchical")
    result = engine.decompose(problem, strategy="hierarchical")

# Pattern 2: Retry with different parameters
try:
    result = solve_sub_problem_with_team(
        sub_problem_id="sub-001",
        execution_method="roma_mdap_maker"
    )
except Exception as e:
    logger.error(f"ROMA-MDAP-MAKER failed: {e}, trying traditional")
    result = solve_sub_problem_with_team(
        sub_problem_id="sub-001",
        execution_method="traditional"
    )

# Pattern 3: Validation before execution
if not DECOMPOSITION_AVAILABLE:
    raise RuntimeError("Decomposition engine not available")

if not OPENEVOLVE_AVAILABLE:
    logger.warning("OpenEvolve not available, evolutionary features disabled")
```

---

## Performance Tuning

### Caching

```python
# Enable OpenEvolve caching
client = OpenEvolveClient(
    enable_cache=True,
    cache_ttl=3600,  # Cache for 1 hour
    cache_dir="./cache/openevolve"
)

# Clear cache
client.clear_cache()
```

### Parallel Execution

```python
from concurrent.futures import ThreadPoolExecutor

# Solve sub-problems in parallel (respecting dependencies)
def solve_sub_problem(sp):
    return solve_sub_problem_with_team(
        sub_problem_id=sp['id'],
        sub_problem_description=sp['description'],
        team_name="Default-Blue",
        execution_method="traditional"
    )

# Get sub-problems with no dependencies
independent_sps = [sp for sp in sub_problems if not sp['dependencies']]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(solve_sub_problem, independent_sps)
```

### Batch Processing

```python
# Analyze multiple problems at once
problems = [problem1, problem2, problem3]

from decomposition_mcp_tools import analyze_problem_for_decomposition

analyses = [
    analyze_problem_for_decomposition(
        problem_statement=p.description,
        use_evolution=False  # Faster without evolution
    )
    for p in problems
]
```

---

## Security Considerations

### API Key Management

```python
# NEVER hardcode API keys
# Use environment variables
import os

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenEvolveClient(api_key=api_key)
```

### Safe Code Execution

The decomposition engine uses restricted execution environments for evolved code:

```python
# Safe globals for code execution
SAFE_GLOBALS = {
    "__builtins__": {
        "dict": dict,
        "list": list,
        "str": str,
        "int": int,
        "float": float,
        "len": len,
        "range": range,
        # Only safe builtins
    }
}

# Execute with restricted environment
exec(code, SAFE_GLOBALS, local_vars)
```

### Input Validation

```python
# Validate problem statement
def validate_problem_statement(statement: str) -> bool:
    if not statement or len(statement.strip()) < 10:
        raise ValueError("Problem statement too short")

    if len(statement) > 10000:
        raise ValueError("Problem statement too long (max 10000 chars)")

    # Check for malicious patterns
    dangerous_patterns = ["__import__", "eval(", "exec("]
    for pattern in dangerous_patterns:
        if pattern in statement:
            raise ValueError(f"Dangerous pattern detected: {pattern}")

    return True
```

### Rate Limiting

```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=100, period=60)  # 100 calls per minute
def call_llm_with_rate_limit(prompt: str):
    return client.evolve(content=prompt)
```

---

## Quick Reference

### Common Workflows

**1. Simple Decomposition**
```python
from decomposition_mcp_tools import (
    analyze_problem_for_decomposition,
    decompose_problem_into_sub_problems
)

# Analyze
analysis = analyze_problem_for_decomposition(
    problem_statement="Build a REST API for user management"
)

# Decompose
decomposition = decompose_problem_into_sub_problems(
    problem_statement="Build a REST API for user management",
    analysis=analysis,
    decomposition_strategy="semantic"
)

print(f"Generated {decomposition['total_sub_problems']} sub-problems")
```

**2. Full Pipeline with Validation**
```python
from decomposition_mcp_tools import (
    analyze_problem_for_decomposition,
    decompose_problem_into_sub_problems,
    solve_sub_problem_with_team,
    critique_solution_with_gauntlet,
    verify_solution_with_gauntlet
)

# Decompose
decomp = decompose_problem_into_sub_problems(problem_statement)

# Solve each sub-problem
for sp in decomp['sub_problems']:
    # Solve
    solution = solve_sub_problem_with_team(
        sub_problem_id=sp['id'],
        sub_problem_description=sp['description'],
        team_name="Default-Blue",
        execution_method="traditional"
    )

    if solution['status'] != 'completed':
        continue

    # Critique
    critique = critique_solution_with_gauntlet(
        solution=solution['solution'],
        sub_problem_id=sp['id'],
        gauntlet_name="Default-RedTeam-Gauntlet"
    )

    # Verify
    verification = verify_solution_with_gauntlet(
        solution=solution['solution'],
        critique=critique,
        sub_problem_id=sp['id'],
        gauntlet_name="Default-GoldTeam-Gauntlet"
    )

    if verification['approved']:
        print(f"✓ {sp['title']}: APPROVED")
    else:
        print(f"✗ {sp['title']}: NEEDS REVISION")
```

**3. Zero-Error Critical Path**
```python
# Use ROMA-MDAP-MAKER for critical sub-problems
solution = solve_sub_problem_with_team(
    sub_problem_id="critical-sub-001",
    sub_problem_description="Implement safety-critical validation",
    execution_method="roma_mdap_maker",
    roma_mdap_maker_k_ahead=3,
    roma_mdap_maker_enable_red_flagging=True,
    roma_mdap_maker_enable_adaptive_k=True
)

# Check error rate
metrics = solution.get('roma_mdap_maker_metrics', {})
error_rate = metrics.get('final_error_rate', 1.0)

if error_rate < 0.001:  # 99.9% accuracy
    print("Zero-error achieved!")
```

---

## Appendix

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="..."

# OpenRouter
export OPENROUTER_API_KEY="..."
```

### Default Configurations

```python
# Default decomposition parameters
DEFAULT_MAX_SUB_PROBLEMS = 15
DEFAULT_COMPLEXITY_TARGET = 5
DEFAULT_STRATEGY = "semantic"

# Default evolution parameters
DEFAULT_EVOLUTION_ITERATIONS = 50
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4000

# Default execution parameters
DEFAULT_EXECUTION_METHOD = "traditional"
DEFAULT_TEAM = "Default-Blue"
DEFAULT_GAUNTLET_RED = "Default-RedTeam-Gauntlet"
DEFAULT_GAUNTLET_GOLD = "Default-GoldTeam-Gauntlet"
```

### Status Codes

| Code | Meaning |
|------|---------|
| `completed` | Successfully completed |
| `failed` | Failed with error |
| `partial` | Partially completed |
| `pending` | Waiting to start |
| `in_progress` | Currently executing |

### Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("decomposition_engine")
logger.setLevel(logging.DEBUG)
```

---

**Document Version:** 2.0.0
**Last Updated:** 2025-01-03
**Maintained By:** OpenEvolve Development Team
