# OpenEvolve Ensemble Functionality - Comprehensive Analysis

**Document Version:** 1.0
**Date:** 2025-01-04
**Author:** Claude Code Analysis
**Target Audience:** Blue Team, Red Team, and Evaluator Team Developers

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Ensemble Architecture](#core-ensemble-architecture)
3. [Team Coordination Framework](#team-coordination-framework)
4. [API Reference](#api-reference)
5. [Configuration Guide](#configuration-guide)
6. [Integration Guide](#integration-guide)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Advanced Patterns](#advanced-patterns)
10. [Performance Optimization](#performance-optimization)

---

## Executive Summary

OpenEvolve provides a sophisticated ensemble and team coordination system enabling:

1. **Multi-Model LLM Ensembles** - Coordinate multiple LLMs with weighted sampling
2. **Island-Based Evolution** - Parallel population evolution with migration
3. **Team Member Coordination** - Specialized agents working in parallel
4. **Result Aggregation** - Intelligent combination of multiple outputs
5. **Load Balancing** - Adaptive task distribution among team members

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `LLMEnsemble` | `openevolve/llm/ensemble.py` | Multi-model LLM coordination |
| `ProgramDatabase` | `openevolve/database.py` | Island-based MAP-Elites |
| `ProcessParallelController` | `openevolve/process_parallel.py` | Parallel execution |
| `BlueTeamCoordinator` | `blue_team_coordinator.py` | Blue team orchestration |
| `EvaluatorTeamCoordinator` | `evaluator_team_coordinator.py` | Evaluator orchestration |

---

## Core Ensemble Architecture

### 1. LLM Ensemble (`openevolve/llm/ensemble.py`)

The `LLMEnsemble` class manages multiple LLM models with configurable weights.

```python
class LLMEnsemble:
    """Ensemble of LLMs with weighted sampling"""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        self.models = [...]  # List of initialized LLM models
        self.weights = [...]  # Normalized weights for sampling
        self.random_state = random.Random()  # Deterministic selection
```

**Key Features:**
- **Weighted random model selection** - Sample models based on capability weights
- **Deterministic sampling** - Configurable random seed for reproducibility
- **Parallel generation** - Execute multiple models concurrently
- **Dual ensembles** - Separate models for evolution and evaluation

**Core Methods:**

```python
# Single generation (weighted sampling)
async def generate(self, prompt: str, **kwargs) -> str:
    """Generate using one weighted-sampled model"""
    model = self._sample_model()
    return await model.generate(prompt, **kwargs)

# Multiple generations
async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
    """Generate n variations in parallel"""
    tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
    return await asyncio.gather(*tasks)

# Batch prompts
async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
    """Generate responses for multiple prompts in parallel"""
    tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
    return await asyncio.gather(*tasks)

# All models (consensus)
async def generate_all_with_context(
    self, system_message: str, messages: List[Dict[str, str]], **kwargs
) -> List[str]:
    """Generate using all available models"""
    responses = []
    for model in self.models:
        responses.append(
            await model.generate_with_context(system_message, messages, **kwargs)
        )
    return responses
```

### 2. Island-Based Evolution (`openevolve/database.py`)

The `ProgramDatabase` implements MAP-Elites algorithm with multiple islands.

```python
class ProgramDatabase:
    """Island-based population model with MAP-Elites"""

    def __init__(self, config: DatabaseConfig):
        # Multiple isolated populations
        self.islands: List[Set[str]] = [set() for _ in range(config.num_islands)]

        # Feature grids for diversity
        self.island_feature_maps: List[Dict[str, str]] = [
            {} for _ in range(config.num_islands)
        ]

        # Best program per island
        self.island_best_programs: List[Optional[str]] = [
            None for _ in range(config.num_islands)
        ]

        # Migration settings
        self.migration_interval: int = config.migration_interval
        self.migration_rate: float = config.migration_rate
```

**Key Features:**
- **Multiple islands** - Isolated populations evolve independently
- **Feature mapping** - MAP-Elites grid maintains diversity
- **Periodic migration** - Programs migrate between islands
- **Best tracking** - Each island tracks its best program

**Core Operations:**

```python
# Add program to specific island
def add(self, program: Program, island_id: Optional[int] = None):
    if island_id is None:
        island_id = self.current_island
    self.islands[island_id].add(program.id)

# Sample from island
def sample_from_island(self, island_id: int, n_samples: int = 1):
    island_programs = [
        self.programs[pid] for pid in self.islands[island_id]
    ]
    return random.sample(island_programs, n_samples)

# Migration between islands
def migrate_population(self):
    for source_island in range(self.num_islands):
        target_island = (source_island + 1) % self.num_islands
        # Migrate best programs
        best = self.get_island_best(source_island)
        if best:
            self.islands[target_island].add(best.id)
```

### 3. Parallel Processing Controller (`openevolve/process_parallel.py`)

The `ProcessParallelController` manages true parallelism across processes.

```python
class ProcessParallelController:
    """Process-based parallel execution for true parallelism"""

    def __init__(
        self,
        config: Config,
        evaluation_file: str,
        database: ProgramDatabase,
        max_workers: int = 4
    ):
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, Future] = {}
        self.shutdown_event = asyncio.Event()
```

**Key Features:**
- **Process isolation** - Each iteration runs in separate process
- **Task queue** - Manage multiple concurrent iterations
- **Graceful shutdown** - Handle Ctrl+C and termination signals
- **Checkpoint integration** - Save state during execution

---

## Team Coordination Framework

### Blue Team Coordinator Pattern

The `BlueTeamCoordinator` orchestrates multiple specialized fixer agents.

**Location:** `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\blue_team_coordinator.py`

```python
class BlueTeamCoordinator:
    """Coordinate multiple Blue Team members for parallel fixing"""

    def __init__(
        self,
        blue_team: Optional[BlueTeam] = None,
        max_concurrent_tasks: int = 5,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED,
        task_timeout: int = 300,
        enable_persistence: bool = True,
        persistence_path: str = "./blue_team_coordinator_state.pkl"
    ):
        # Team management
        self.team_members: List[BlueTeamMember] = []
        self.member_metrics: Dict[str, TeamMemberMetrics] = {}

        # Task management
        self.task_queue: queue.Queue = queue.Queue()
        self.active_tasks: Dict[str, CoordinationTask] = {}
        self.completed_tasks: Dict[str, CoordinationTask] = {}

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
```

**Core Capabilities:**

1. **Task Distribution**
   - `ROUND_ROBIN` - Cyclic assignment
   - `LEAST_LOADED` - Assign to member with fewest active tasks
   - `SPECIALIZATION_BASED` - Match task type to member expertise
   - `RANDOM` - Random assignment
   - `ADAPTIVE` - Dynamic adjustment based on performance

2. **Member Management**
   ```python
   @dataclass
   class TeamMemberMetrics:
       member_name: str
       tasks_completed: int = 0
       tasks_failed: int = 0
       total_time_spent: float = 0.0
       average_task_time: float = 0.0
       current_load: int = 0
       specialization_scores: Dict[FixType, float] = field(default_factory=dict)
       reliability_score: float = 1.0
   ```

3. **Result Aggregation**
   - Fix combination from multiple members
   - Consensus building
   - Conflict resolution
   - Quality scoring

**Main Entry Point:**

```python
def coordinate_decomposition_fixes(
    self,
    problem_statement: str,
    sub_problems: List[Dict[str, Any]],
    content_items: Dict[str, str],
    issues_dict: Dict[str, List[Any]],
    content_types: Optional[Dict[str, str]] = None,
    strategy: BlueTeamStrategy = BlueTeamStrategy.COMPREHENSIVE,
    progress_callback: Optional[Callable] = None
) -> CoordinationSession:
    """
    Coordinate fixes for a decomposed problem.

    Main entry point for integrating with DecompositionEngine.
    Automatically fixes issues found during decomposition.
    """
```

### Evaluator Team Coordinator Pattern

Similar pattern for coordinating multiple evaluators.

**Location:** `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\evaluator_team_coordinator.py`

```python
class EvaluatorTeamCoordinator:
    """Coordinate multiple evaluators for comprehensive assessment"""

    def __init__(
        self,
        evaluators: List[EvaluatorMember],
        max_concurrent_evaluations: int = 5,
        aggregation_method: str = "weighted_average",
        consensus_threshold: float = 0.7
    ):
        self.evaluators = evaluators
        self.evaluator_metrics: Dict[str, EvaluatorMetrics] = {}
        self.aggregation_method = aggregation_method
        self.consensus_threshold = consensus_threshold
```

**Key Features:**
- Parallel evaluation execution
- Statistical aggregation (mean, median, weighted)
- Consensus detection
- Variance analysis

```python
def coordinate_evaluations(
    self,
    content: str,
    content_type: str = "general",
    evaluator_ids: Optional[List[str]] = None,
    evaluation_criteria: Optional[List[EvaluationCriterion]] = None,
    aggregation_method: str = "weighted_average"
) -> IntegratedEvaluation:
    """
    Coordinate parallel evaluations and aggregate results.

    Returns:
        IntegratedEvaluation with consensus scores and variance analysis
    """
```

---

## API Reference

### LLM Ensemble API

#### Initialize Ensemble

```python
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.config import LLMModelConfig

# Configure models
models = [
    LLMModelConfig(
        name="gpt-4",
        api_key="sk-...",
        weight=0.7,  # 70% selection probability
        temperature=0.7,
        max_tokens=2048
    ),
    LLMModelConfig(
        name="gpt-3.5-turbo",
        api_key="sk-...",
        weight=0.3,  # 30% selection probability
        temperature=0.8,
        max_tokens=4096
    )
]

# Create ensemble
ensemble = LLMEnsemble(models)
```

#### Generate with Ensemble

```python
import asyncio

# Single generation (weighted sampling)
response = await ensemble.generate(
    prompt="Optimize this function for performance..."
)

# Parallel generations (n variations)
variations = await ensemble.generate_multiple(
    prompt="Refactor this function:",
    n=5  # Generate 5 variations
)

# Batch prompts (different prompts)
responses = await ensemble.parallel_generate(
    prompts=[prompt1, prompt2, prompt3]
)

# Generate with context (system message)
response = await ensemble.generate_with_context(
    system_message="You are a code optimization expert",
    messages=[
        {"role": "user", "content": "Optimize this function"}
    ]
)

# Get all model responses (consensus)
all_responses = await ensemble.generate_all_with_context(
    system_message="You are a code optimization expert",
    messages=[...]
)
# Returns list of responses from all models
```

### Team Coordinator API

#### Initialize Coordinator

```python
from blue_team_coordinator import BlueTeamCoordinator, LoadBalancingStrategy

coordinator = BlueTeamCoordinator(
    blue_team=blue_team_instance,
    max_concurrent_tasks=5,
    load_balancing_strategy=LoadBalancingStrategy.LEAST_LOADED,
    task_timeout=300,
    enable_persistence=True
)
```

#### Coordinate Tasks

```python
# Main entry point for decomposed problems
session = coordinator.coordinate_decomposition_fixes(
    problem_statement="Fix security vulnerabilities",
    sub_problems=[
        {"id": "sub1", "description": "Fix SQL injection"},
        {"id": "sub2", "description": "Fix XSS"}
    ],
    content_items={
        "sub1": "code with SQL injection",
        "sub2": "code with XSS"
    },
    issues_dict={
        "sub1": [issue1, issue2],
        "sub2": [issue3]
    },
    strategy=BlueTeamStrategy.COMPREHENSIVE,
    progress_callback=lambda update: print(f"Progress: {update}")
)

# Session contains:
# - session_id: Unique identifier
# - tasks: List of coordinated tasks
# - status: Overall session status
# - aggregated_result: Combined fixes
```

#### Task Management

```python
# Submit individual task
task_id = coordinator.submit_task(
    sub_problem_id="sub1",
    content="code to fix",
    issues=[issue1, issue2],
    priority=TaskPriority.HIGH
)

# Wait for completion
result = coordinator.wait_for_task(task_id, timeout=300)

# Cancel task
coordinator.cancel_task(task_id)

# Query session status
status = coordinator.get_session_status(session_id)
```

### Database/Island API

#### Initialize Database with Islands

```python
from openevolve.config import DatabaseConfig
from openevolve.database import ProgramDatabase

config = DatabaseConfig(
    num_islands=4,  # 4 separate populations
    population_size=20,
    migration_interval=10,  # Migrate every 10 generations
    migration_rate=0.1  # 10% of population migrates
)

database = ProgramDatabase(config)
```

#### Island Operations

```python
# Assign program to island
database.add(program, island_id=0)

# Sample from specific island
program = database.sample_from_island(
    island_id=0,
    n_samples=3
)

# Get best program from island
best = database.get_island_best(island_id=0)

# Migrate between islands
database.migrate_population()

# Get island status
database.log_island_status()
```

---

## Configuration Guide

### LLM Ensemble Configuration

YAML configuration for multi-model ensemble:

```yaml
llm:
  api_base: "https://api.openai.com/v1"
  api_key: "sk-..."
  temperature: 0.7
  max_tokens: 2048

  # Evolution ensemble (primary models)
  models:
    - name: "gpt-4"
      weight: 0.6
      temperature: 0.7
      max_tokens: 2048

    - name: "gpt-3.5-turbo"
      weight: 0.3
      temperature: 0.8
      max_tokens: 4096

    - name: "claude-3-opus"
      api_base: "https://api.anthropic.com/v1"
      weight: 0.1
      temperature: 0.7
      max_tokens: 2048

  # Evaluator ensemble (can be different)
  evaluator_models:
    - name: "gpt-4"
      weight: 0.8
      temperature: 0.3  # Lower temperature for evaluation

    - name: "claude-3-opus"
      weight: 0.2
      temperature: 0.3
```

### Team Coordinator Configuration

```yaml
blue_team_coordinator:
  max_concurrent_tasks: 5
  load_balancing_strategy: "least_loaded"  # Options: round_robin, least_loaded, specialization_based, random, adaptive
  task_timeout: 300
  enable_persistence: true
  persistence_path: "./blue_team_state.pkl"

  # Auto-scaling configuration
  auto_scale: true
  min_members: 2
  max_members: 10

  # Team member configuration
  team_members:
    - name: "Security Expert"
      specializations: ["security_patch", "input_validation", "error_handling"]
      expertise_level: 9
      strategy: "comprehensive"

    - name: "Performance Optimizer"
      specializations: ["performance_optimization", "code_refactoring"]
      expertise_level: 8
      strategy: "targeted"

    - name: "Code Quality Specialist"
      specializations: ["clarity_improvement", "documentation_addition", "maintainability_improvement"]
      expertise_level: 7
      strategy: "minimal"
```

### Island Configuration

```yaml
database:
  # Island configuration
  num_islands: 4  # Number of separate populations
  population_size: 20  # Programs per island
  migration_interval: 10  # Generations between migrations
  migration_rate: 0.1  # Percentage that migrates

  # MAP-Elites configuration
  archive_size: 100
  feature_dimensions: ["complexity", "diversity"]
  feature_bins: 10

  # Diversity maintenance
  diversity_reference_size: 20
```

---

## Integration Guide

### For Blue Team Integration

#### Step 1: Define Team Members

```python
from blue_team import BlueTeamMember, FixType, BlueTeamStrategy

# Create specialized members
security_expert = BlueTeamMember(
    name="Security Expert",
    specializations=[
        FixType.SECURITY_PATCH,
        FixType.INPUT_VALIDATION,
        FixType.ERROR_HANDLING
    ],
    expertise_level=9,
    strategy=BlueTeamStrategy.DEFENSIVE
)

performance_specialist = BlueTeamMember(
    name="Performance Specialist",
    specializations=[
        FixType.PERFORMANCE_OPTIMIZATION,
        FixType.CODE_REFACTORING
    ],
    expertise_level=8,
    strategy=BlueTeamStrategy.TARGETED
)
```

#### Step 2: Initialize Coordinator

```python
from blue_team import BlueTeam
from blue_team_coordinator import BlueTeamCoordinator

blue_team = BlueTeam(team_members=[security_expert, performance_specialist])

coordinator = BlueTeamCoordinator(
    blue_team=blue_team,
    max_concurrent_tasks=5,
    load_balancing_strategy=LoadBalancingStrategy.SPECIALIZATION_BASED
)
```

#### Step 3: Integrate with Decomposition

```python
from decomposition_engine import DecompositionEngine

# Decompose problem
engine = DecompositionEngine()
sub_problems = engine.decompose(problem_statement)

# Coordinate fixes
session = coordinator.coordinate_decomposition_fixes(
    problem_statement=problem_statement,
    sub_problems=[{"id": sp.id, "description": sp.description} for sp in sub_problems],
    content_items={sp.id: sp.content for sp in sub_problems},
    issues_dict={sp.id: sp.issues for sp in sub_problems}
)

# Get aggregated results
fixed_content = session.aggregated_result
```

### For Evaluator Team Integration

#### Step 1: Define Evaluators

```python
from evaluator_team import EvaluatorMember, EvaluationMetric

code_quality_evaluator = EvaluatorMember(
    evaluator_id="code_quality_expert",
    specializations=[
        EvaluationMetric.CORRECTNESS,
        EvaluationMetric.MAINTAINABILITY,
        EvaluationMetric.CLARITY
    ],
    expertise_level=9,
    evaluation_philosophy="strict"
)

security_evaluator = EvaluatorMember(
    evaluator_id="security_auditor",
    specializations=[
        EvaluationMetric.SECURITY,
        EvaluationMetric.ROBUSTNESS,
        EvaluationMetric.COMPLIANCE
    ],
    expertise_level=9,
    evaluation_philosophy="strict"
)
```

#### Step 2: Coordinate Evaluations

```python
from evaluator_team_coordinator import EvaluatorTeamCoordinator

coordinator = EvaluatorTeamCoordinator(
    evaluators=[code_quality_evaluator, security_evaluator]
)

# Coordinate parallel evaluation
integrated_eval = coordinator.coordinate_evaluations(
    content=code_to_evaluate,
    content_type="code",
    evaluator_ids=["code_quality_expert", "security_auditor"],
    evaluation_criteria=default_criteria
)

# Get consensus result
print(f"Consensus Score: {integrated_eval.consensus_score}")
print(f"Variance: {integrated_eval.variance_analysis}")
print(f"Recommendations: {integrated_eval.recommendations}")
```

---

## Usage Examples

### Example 1: Basic LLM Ensemble

```python
import asyncio
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.config import LLMModelConfig

async def main():
    # Configure ensemble
    models = [
        LLMModelConfig(
            name="gpt-4",
            api_key="sk-...",
            weight=0.7
        ),
        LLMModelConfig(
            name="claude-3-opus",
            api_key="sk-ant-...",
            weight=0.3
        )
    ]

    ensemble = LLMEnsemble(models)

    # Generate optimized code
    prompt = "Optimize this Python function for performance:"
    response = await ensemble.generate(prompt)

    print(f"Response: {response}")

asyncio.run(main())
```

### Example 2: Parallel Generation

```python
async def generate_variations():
    """Generate multiple code variations in parallel"""
    models = [...]  # Configure as above
    ensemble = LLMEnsemble(models)

    # Generate 5 variations
    variations = await ensemble.generate_multiple(
        prompt="Refactor this function to be more maintainable:",
        n=5
    )

    for i, variation in enumerate(variations, 1):
        print(f"Variation {i}:\n{variation}\n")
```

### Example 3: Blue Team Coordination

```python
from blue_team import BlueTeam, BlueTeamMember, FixType
from blue_team_coordinator import BlueTeamCoordinator

def fix_security_issues():
    # Create team
    security_expert = BlueTeamMember(
        name="Security Expert",
        specializations=[FixType.SECURITY_PATCH],
        expertise_level=9
    )

    team = BlueTeam(team_members=[security_expert])
    coordinator = BlueTeamCoordinator(blue_team=team)

    # Define sub-problems
    sub_problems = [
        {
            "id": "auth",
            "description": "Fix authentication bypass",
            "content": vulnerable_auth_code
        },
        {
            "id": "sql",
            "description": "Fix SQL injection",
            "content": vulnerable_sql_code
        }
    ]

    # Coordinate fixes
    session = coordinator.coordinate_decomposition_fixes(
        problem_statement="Fix critical security vulnerabilities",
        sub_problems=sub_problems,
        content_items={
            "auth": vulnerable_auth_code,
            "sql": vulnerable_sql_code
        },
        issues_dict={
            "auth": [auth_bypass_issue],
            "sql": [sql_injection_issue]
        }
    )

    # Get results
    print(f"Fixed {len(session.completed_tasks)} sub-problems")
    print(f"Aggregated fixes:\n{session.aggregated_result}")
```

### Example 4: Island-Based Evolution

```python
from openevolve.config import Config
from openevolve.controller import OpenEvolve

async def island_evolution():
    # Configure islands
    config = Config()
    config.database.num_islands = 4
    config.database.migration_interval = 10
    config.database.migration_rate = 0.1

    # Initialize evolution
    evolution = OpenEvolve(
        initial_program_path="initial.py",
        evaluation_file="evaluator.py",
        config=config
    )

    # Run evolution with islands
    best_program = await evolution.run(iterations=100)

    print(f"Best program from {config.database.num_islands} islands")
    print(f"Fitness: {best_program.metrics}")
```

### Example 5: Evaluator Team Consensus

```python
from evaluator_team import EvaluatorMember, EvaluationMetric
from evaluator_team_coordinator import EvaluatorTeamCoordinator

def evaluate_with_consensus():
    # Create diverse evaluators
    strict_evaluator = EvaluatorMember(
        evaluator_id="strict",
        specializations=[EvaluationMetric.CORRECTNESS, EvaluationMetric.SECURITY],
        expertise_level=9,
        evaluation_philosophy="strict"
    )

    lenient_evaluator = EvaluatorMember(
        evaluator_id="lenient",
        specializations=[EvaluationMetric.CLARITY, EvaluationMetric.COMPLETENESS],
        expertise_level=7,
        evaluation_philosophy="lenient"
    )

    coordinator = EvaluatorTeamCoordinator(
        evaluators=[strict_evaluator, lenient_evaluator]
    )

    # Get consensus evaluation
    integrated_eval = coordinator.coordinate_evaluations(
        content=code_to_evaluate,
        content_type="code",
        evaluator_ids=["strict", "lenient"]
    )

    # Check consensus
    if integrated_eval.consensus_reached:
        print(f"Consensus score: {integrated_eval.consensus_score}")
    else:
        print(f"No consensus. Variance: {integrated_eval.variance_analysis}")
        print(f"Recommendations: {integrated_eval.recommendations}")
```

---

## Best Practices

### 1. Ensemble Configuration

**DO:**
- Use 2-5 models in ensemble
- Weigh models by capability and cost
- Set different temperatures for exploration vs exploitation
- Use separate ensembles for evolution and evaluation

**DON'T:**
- Overload ensemble with too many models
- Give equal weights to models with vastly different capabilities
- Use same temperature for all use cases

### 2. Team Member Design

**DO:**
- Create members with complementary specializations
- Assign appropriate expertise levels (1-10)
- Track performance history
- Use different strategies for different problem types

**DON'T:**
- Create redundant team members
- Ignore specialization bonuses
- Set expertise too high without justification
- Use same strategy for all scenarios

### 3. Load Balancing

**DO:**
- Use LEAST_LOADED for heterogeneous tasks
- Use SPECIALIZATION_BASED for specialized teams
- Use ROUND_ROBIN for homogeneous tasks
- Monitor member utilization

**DON'T:**
- Use RANDOM for critical tasks
- Ignore member performance history
- Overload single members
- Forget to scale based on workload

### 4. Result Aggregation

**DO:**
- Use weighted averaging for numerical scores
- Use voting for categorical decisions
- Track consensus confidence
- Handle conflicts intelligently

**DON'T:**
- Simply take first result
- Ignore variance in assessments
- Forget to weight by expertise/reliability
- Overlook minority opinions

### 5. Error Handling

**DO:**
- Implement timeouts for all tasks
- Retry failed tasks with different members
- Log all failures for analysis
- Fallback to simpler strategies

**DON'T:**
- Let single failures crash coordinator
- Ignore timeout violations
- Retry indefinitely
- Hide error details

---

## Advanced Patterns

### Pattern 1: Hierarchical Teams

Create hierarchical coordinator structure:

```python
# Level 1: Domain coordinators
security_coordinator = BlueTeamCoordinator(
    team_members=[security_expert1, security_expert2]
)
performance_coordinator = BlueTeamCoordinator(
    team_members=[perf_expert1, perf_expert2]
)

# Level 2: Master coordinator
class MasterCoordinator:
    def __init__(self, sub_coordinators):
        self.sub_coordinators = sub_coordinators

    def route_task_by_domain(self, task, domain):
        coordinator = self.sub_coordinators[domain]
        return coordinator.submit_task(task)
```

### Pattern 2: Adaptive Ensemble

Dynamically adjust model weights based on performance:

```python
class AdaptiveEnsemble(LLMEnsemble):
    def update_weights_based_on_performance(self, performance_history):
        """Increase weight for better performing models"""
        for model, performance in zip(self.models, performance_history):
            # Adjust weight based on improvement rate
            model.weight *= (1.0 + performance.improvement_rate)

        # Re-normalize
        total = sum(m.weight for m in self.models)
        for model in self.models:
            model.weight /= total
```

### Pattern 3: Consensus with Confidence

Implement confidence-weighted consensus:

```python
def weighted_consensus(assessments, weights):
    """Calculate consensus with confidence intervals"""
    weighted_scores = []
    confidences = []

    for assessment, weight in zip(assessments, weights):
        weighted_scores.append(assessment.composite_score * weight)
        confidences.append(assessment.confidence_level.value)

    consensus_score = sum(weighted_scores) / sum(weights)
    confidence_interval = calculate_interval(confidences)

    return consensus_score, confidence_interval
```

### Pattern 4: Progressive Refinement

Use ensemble to progressively refine solutions:

```python
async def progressive_refinement(initial_code, num_rounds=3):
    """Progressively refine code through ensemble"""
    current_code = initial_code

    for round_num in range(num_rounds):
        # Generate variations
        variations = await ensemble.generate_multiple(
            prompt=f"Refine this code (Round {round_num + 1}):\n{current_code}",
            n=3
        )

        # Evaluate variations
        evaluations = [await evaluator.evaluate(v) for v in variations]

        # Select best
        best_idx = max(range(len(evaluations)), key=lambda i: evaluations[i].score)
        current_code = variations[best_idx]

        print(f"Round {round_num + 1} best score: {evaluations[best_idx].score}")

    return current_code
```

---

## Performance Optimization

### Parallel Execution Tuning

```python
import os

# Optimize based on workload
coordinator = BlueTeamCoordinator(
    max_concurrent_tasks=min(32, (os.cpu_count() or 1) * 4),  # 4x CPU cores
    task_timeout=300,  # 5 minutes per task
    load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE
)
```

### Caching Strategies

```python
from functools import lru_cache

class CachedCoordinator(BlueTeamCoordinator):
    @lru_cache(maxsize=128)
    def get_member_recommendation(self, task_features):
        """Cache member recommendations for similar tasks"""
        return self._calculate_best_member(task_features)
```

### Batch Processing

```python
async def batch_coordinate(tasks, batch_size=10):
    """Process tasks in batches for efficiency"""
    results = []

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_results = await coordinator.coordinate_batch(batch)
        results.extend(batch_results)

    return results
```

---

## Monitoring and Debugging

### Logging Configuration

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ensemble.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('blue_team_coordinator')
logger.setLevel(logging.DEBUG)
```

### Metrics Collection

```python
coordinator = BlueTeamCoordinator(
    ...,
    enable_metrics=True,
    metrics_export_path="./metrics.json"
)

# Export metrics
metrics = coordinator.export_metrics()
print(f"Total tasks: {metrics['total_tasks']}")
print(f"Average time: {metrics['average_task_time']}")
print(f"Team utilization: {metrics['team_utilization']}")
```

---

## Conclusion

The OpenEvolve ensemble functionality provides a robust framework for:

1. **Coordinating multiple LLMs** with weighted sampling
2. **Managing specialized teams** of agents working in parallel
3. **Aggregating results** from multiple sources intelligently
4. **Scaling computations** across islands and processes
5. **Maintaining diversity** through MAP-Elites and migration

### Key Takeaways

- **Ensemble = Quality + Reliability**: Multiple models/agents provide better results
- **Coordination = Efficiency**: Parallel processing with intelligent routing
- **Aggregation = Consensus**: Intelligent combination of diverse opinions
- **Islands = Diversity**: Separate populations prevent premature convergence
- **Monitoring = Control**: Track performance to optimize configuration

### Next Steps

1. **For Blue Team**: Integrate `BlueTeamCoordinator` into decomposition workflow
2. **For Red Team**: Implement similar coordinator for security analysis
3. **For Evaluator Team**: Use `EvaluatorTeamCoordinator` for comprehensive assessment
4. **For All Teams**: Monitor performance and optimize configurations

---

**Document End**

For additional information, refer to:
- OpenEvolve documentation: `openevolve/CLAUDE.md`
- Team coordinator implementations: `blue_team_coordinator.py`, `evaluator_team_coordinator.py`
- Configuration examples: `openevolve/config.py`
