# HYBRID MAKER INTEGRATION GUIDE

Integration guide for Hybrid MAKER strategies with existing workflows and systems.

**Version:** 1.0.0
**Paper:** arXiv:2511.09030
**Last Updated:** 2025-12-30

---

## Table of Contents

1. [Overview](#overview)
2. [Workflow Integration](#workflow-integration)
3. [Stage 3A/B/C Integration](#stage-3abc-integration)
4. [LeanAide Integration](##leanaide-integration)
5. [CrewAI Integration](#crewai-integration)
6. [Knowledge Engine Integration](#knowledge-engine-integration)
7. [API Integration](#api-integration)
8. [Testing Integration](#testing-integration)
9. [Monitoring Integration](#monitoring-integration)
10. [Deployment Integration](#deployment-integration)

---

## Overview

The Hybrid MAKER system can be integrated into various workflows and systems. This guide provides detailed instructions for integration with:

- OpenEvolve workflows (Stage 3A, 3B, 3C)
- LeanAide theorem proving
- CrewAI delegation
- Knowledge Engine
- REST APIs
- Testing frameworks
- Monitoring systems
- Deployment pipelines

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (Your Application, API, Workflow)                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Hybrid MAKER API                          │
│  (run_maker_hybrid, run_maker_evolution, etc.)              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Integration Adapters                      │
│  (LeanAide, CrewAI, Knowledge Engine, etc.)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Core MAKER Layer                          │
│  (Voting, Red Flagging, MDAP Decomposition)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Workflow Integration

### OpenEvolve Workflow Stages

The Hybrid MAKER system integrates with OpenEvolve's enhanced workflow stages:

#### Stage 3A: MAKER-Enhanced Decomposition

```python
from workflow_engine import WorkflowEngine, WorkflowStage
from hybrid_maker_integration import run_maker_evolution, MakerevolutionConfig

def create_stage_3a_maker():
    """Create Stage 3A with MAKER enhancement"""

    def maker_decomposition(problem):
        """Decompose problem using MAKER+MDAP"""
        result = run_maker_evolution(
            initial_program=problem.description,
            evaluator=lambda x: len(x.split('.')),  # Simple metric
            max_generations=10,
            config=MakerevolutionConfig(
                mode=MakerevolutionMode.DECOMPOSITION,
                enable_decomposition=True,
                decomposition_depth=3
            )
        )

        # Return decomposed subproblems
        return result['best_program'].split('\n')

    stage_3a = WorkflowStage(
        name="3A_MAKER_Decomposition",
        execute=maker_decomposition,
        dependencies=["2_Initial_Analysis"]
    )

    return stage_3a

# Add to workflow
workflow = WorkflowEngine()
workflow.add_stage(create_stage_3a_maker())
```

#### Stage 3B: MAKER-Enhanced Solving

```python
def create_stage_3b_maker():
    """Create Stage 3B with MAKER enhancement"""

    def maker_solving(subproblems):
        """Solve subproblems using MAKER voting"""
        results = []

        for subproblem in subproblems:
            result = run_maker_evolution(
                initial_program=subproblem,
                evaluator=custom_evaluator,
                max_generations=20,
                config=MakerevolutionConfig(
                    mode=MakerevolutionMode.VOTING_ONLY,
                    voting_threshold=3,
                    population_size=15
                )
            )

            results.append(result['best_program'])

        return results

    stage_3b = WorkflowStage(
        name="3B_MAKER_Solving",
        execute=maker_solving,
        dependencies=["3A_MAKER_Decomposition"]
    )

    return stage_3b
```

#### Stage 3C: MAKER-Enhanced Assembly

```python
def create_stage_3c_maker():
    """Create Stage 3C with MAKER enhancement"""

    def maker_assembly(solutions):
        """Assemble solutions using MAKER voting"""

        # Combine solutions
        combined = "\n".join(solutions)

        # Validate with MAKER
        result = run_maker_evolution(
            initial_program=combined,
            evaluator=assembly_evaluator,
            max_generations=15,
            config=MakerevolutionConfig(
                mode=MakerevolutionMode.HYBRID,
                enable_voting=True,
                enable_decomposition=True
            )
        )

        return result['best_program']

    stage_3c = WorkflowStage(
        name="3C_MAKER_Assembly",
        execute=maker_assembly,
        dependencies=["3B_MAKER_Solving"]
    )

    return stage_3c
```

### Complete Workflow Integration

```python
def create_maker_workflow():
    """Create complete workflow with MAKER enhancement"""

    workflow = WorkflowEngine(name="MAKER_Enhanced_Workflow")

    # Add stages
    workflow.add_stage(create_stage_3a_maker())
    workflow.add_stage(create_stage_3b_maker())
    workflow.add_stage(create_stage_3c_maker())

    # Configure MAKER settings
    workflow.config = {
        "maker_enabled": True,
        "voting_threshold": 3,
        "enable_checkpoints": True,
        "checkpoint_dir": "/tmp/maker_checkpoints"
    }

    return workflow

# Execute workflow
workflow = create_maker_workflow()
problem = ProblemDefinition(
    title="Prove arithmetic properties",
    description="Prove: forall n m : nat, n + m = m + n"
)

result = workflow.execute(problem)
```

---

## Stage 3A/B/C Integration

### Stage 3A: Decomposition Integration

```python
from decomposition_engine import DecompositionEngine, DecompositionStrategy
from hybrid_maker_integration import MAKERHybridConfig

class MAKERDecompositionIntegration:
    """Integrate MAKER with decomposition engine"""

    def __init__(self, openevolve_client=None):
        self.decomposition_engine = DecompositionEngine()
        self.openevolve_client = openevolve_client

    def decompose_with_maker(self, problem, strategy="hybrid"):
        """Decompose problem using MAKER-enhanced strategies"""

        # Use hybrid decomposition for complex problems
        if strategy == "hybrid":
            plan = self.decomposition_engine.decompose(
                problem=problem,
                strategy=DecompositionStrategy.HYBRID
            )

        # Validate subproblems with MAKER
        validated_subproblems = []
        for subproblem in plan.sub_problems:
            # Use MAKER to validate subproblem quality
            result = run_maker_evolution(
                initial_program=subproblem.description,
                evaluator=lambda x: self._subproblem_quality(x),
                max_generations=5,
                config=MakerevolutionConfig(
                    mode=MakerevolutionMode.VOTING_ONLY,
                    voting_threshold=2
                )
            )

            if result['best_fitness'] > 0.7:
                validated_subproblems.append(subproblem)

        # Update plan with validated subproblems
        plan.sub_problems = validated_subproblems
        return plan

    def _subproblem_quality(self, description):
        """Evaluate subproblem quality"""
        # Simple heuristic
        quality = 0.0
        if "prove" in description.lower():
            quality += 0.3
        if "show" in description.lower():
            quality += 0.2
        if "demonstrate" in description.lower():
            quality += 0.2

        # Prefer specific, actionable subproblems
        word_count = len(description.split())
        if 5 <= word_count <= 20:
            quality += 0.3

        return min(1.0, quality)
```

### Stage 3B: Solving Integration

```python
class MAKERSolvingIntegration:
    """Integrate MAKER with solving stage"""

    def __init__(self):
        self.solver_config = MAKERHybridConfig(
            voting_threshold=3,
            enable_voting=True,
            enable_decomposition=False
        )

    def solve_subproblem(self, subproblem):
        """Solve single subproblem with MAKER"""

        result = run_maker_evolution(
            initial_program=subproblem.description,
            evaluator=self._solver_evaluator,
            max_generations=25,
            config=MakerevolutionConfig(
                mode=MakerevolutionMode.HYBRID,
                voting_threshold=3,
                population_size=20
            )
        )

        return {
            "subproblem_id": subproblem.id,
            "solution": result['best_program'],
            "fitness": result['best_fitness'],
            "generations": result['generations'],
            "confidence": result['best_fitness']
        }

    def solve_all_subproblems(self, subproblems):
        """Solve all subproblems in parallel"""

        import asyncio

        async def solve_parallel():
            tasks = []
            for subproblem in subproblems:
                task = asyncio.create_task(
                    self._solve_async(subproblem)
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            return results

        return asyncio.run(solve_parallel())

    async def _solve_async(self, subproblem):
        """Async solving for parallel execution"""
        # Implementation similar to solve_subproblem
        pass

    def _solver_evaluator(self, solution):
        """Evaluate solution quality"""
        # Domain-specific evaluation
        score = 0.0

        # Check for proof structure
        if "theorem" in solution:
            score += 0.2
        if "proof" in solution or "by" in solution:
            score += 0.3

        # Check for tactics
        tactics = ["induction", "simp", "rw", "refl", "assumption"]
        found_tactics = sum(1 for t in tactics if t in solution)
        score += found_tactics * 0.1

        return min(1.0, score)
```

### Stage 3C: Assembly Integration

```python
class MAKERAssemblyIntegration:
    """Integrate MAKER with assembly stage"""

    def __init__(self):
        self.assembly_config = MAKERHybridConfig(
            enable_decomposition=True,
            enable_voting=True
        )

    def assemble_solutions(self, subproblem_solutions):
        """Assemble subproblem solutions into final answer"""

        # Combine solutions
        combined = self._combine_solutions(subproblem_solutions)

        # Validate with MAKER
        result = run_maker_evolution(
            initial_program=combined,
            evaluator=self._assembly_evaluator,
            max_generations=20,
            config=MakerevolutionConfig(
                mode=MakerevolutionMode.FULL_MAKER,
                enable_voting=True,
                enable_decomposition=True
            )
        )

        return {
            "final_solution": result['best_program'],
            "fitness": result['best_fitness'],
            "validation_passed": result['best_fitness'] > 0.8
        }

    def _combine_solutions(self, solutions):
        """Combine individual solutions"""
        parts = []
        for sol in solutions:
            parts.append(f"-- Solution for {sol['subproblem_id']}")
            parts.append(sol['solution'])
            parts.append("")  # Blank line

        return "\n".join(parts)

    def _assembly_evaluator(self, assembly):
        """Evaluate assembled solution quality"""

        score = 0.0

        # Check structure
        if "-- Solution" in assembly:
            score += 0.2

        # Check for all solutions
        solution_count = assembly.count("-- Solution")
        score += min(0.3, solution_count * 0.1)

        # Check for coherence
        lines = assembly.split('\n')
        non_empty = [l for l in lines if l.strip() and not l.strip().startswith('--')]
        if len(non_empty) >= 5:
            score += 0.3

        # Check for conclusion
        if "qed" in assembly.lower() or "QED" in assembly:
            score += 0.2

        return min(1.0, score)
```

---

## LeanAide Integration

### Basic LeanAide Integration

```python
class LeanAideMAKERIntegration:
    """Integrate MAKER with LeanAide theorem prover"""

    def __init__(self):
        try:
            from leanaide_mcts import LeanProofMCTS
            from leanaide_evolution import LeanProofEvolutionEngineMCTS
            self.leanaide_available = True
        except ImportError:
            self.leanaide_available = False
            print("LeanAide not available")

    async def prove_with_maker(self, theorem):
        """Prove theorem using MAKER + LeanAide"""

        if not self.leanaide_available:
            raise RuntimeError("LeanAide not available")

        # Configure MAKER for LeanAide
        config = MAKERHybridConfig(
            voting_threshold=3,
            mcts_simulations=100,
            evolution_generations=20,
            population_size=15
        )

        # Use MAKER hybrid to generate Lean4 proof
        result = await run_maker_hybrid(
            theorem=theorem,
            mode=MAKERHybridMode.MCTS_THEN_MAKER,
            config=config
        )

        if result.success:
            # Convert to Lean4 format
            lean4_proof = self._to_lean4(result.best_proof)

            # Validate with LeanAide
            validation = self._validate_with_leanaide(lean4_proof)

            return {
                "proof": lean4_proof,
                "fitness": result.best_fitness,
                "validation": validation
            }

        return None

    def _to_lean4(self, proof):
        """Convert MAKER proof to Lean4 format"""
        # Add Lean4 headers
        lean4_proof = f"""
import Mathlib.Tactic

theorem proved (n m : Nat) : n + m = m + n := by
  {proof}

#check proved
"""
        return lean4_proof

    def _validate_with_leanaide(self, lean4_proof):
        """Validate proof with LeanAide"""
        # This would use LeanAide's validation
        # Placeholder implementation
        return {"valid": True, "errors": []}
```

### Advanced LeanAide Integration

```python
class AdvancedLeanAideMAKER:
    """Advanced integration with LeanAide"""

    def __init__(self):
        self.basic_integration = LeanAideMAKERIntegration()
        self.tactic_library = self._load_tactics()

    def _load_tactics(self):
        """Load available Lean4 tactics"""
        return [
            "induction", "simp", "rw", "refl", "assumption",
            "apply", "exact", "have", "calc", "linarith"
        ]

    async def prove_with_tactic_selection(self, theorem):
        """Prove with MAKER-guided tactic selection"""

        # Step 1: Use MAKER to analyze theorem
        analysis_result = await run_maker_hybrid(
            theorem=theorem,
            mode=MAKERHybridMode.MCTS_THEN_MAKER,
            config=MAKERHybridConfig(
                voting_threshold=2,
                mcts_simulations=50
            )
        )

        # Step 2: Extract tactics
        tactics = self._extract_tactics(analysis_result.best_proof)

        # Step 3: Refine with MAKER-Then-Evolution
        refined_result = await run_maker_hybrid(
            theorem=theorem,
            mode=MAKERHybridMode.MAKER_THEN_EVOLUTION,
            config=MAKERHybridConfig(
                voting_threshold=3,
                evolution_generations=25,
                population_size=20
            )
        )

        return {
            "initial_proof": analysis_result.best_proof,
            "refined_proof": refined_result.best_proof,
            "tactics": tactics,
            "improvement": refined_result.best_fitness - analysis_result.best_fitness
        }

    def _extract_tactics(self, proof):
        """Extract tactics from proof"""
        import re
        tactic_pattern = r'\b(induction|simp|rw|refl|assumption|apply|exact|have|calc|linarith)\b'
        tactics = re.findall(tactic_pattern, proof)
        return list(set(tactics))
```

---

## CrewAI Integration

### Basic CrewAI Integration

```python
class CrewAIMAKERIntegration:
    """Integrate MAKER with CrewAI delegation system"""

    def __init__(self):
        try:
            from crewai_client import CrewAIClient
            self.crewai = CrewAIClient()
            self.available = True
        except ImportError:
            self.available = False
            print("CrewAI not available")

    async def delegate_to_crewai(self, problem):
        """Delegate problem solving to CrewAI with MAKER"""

        if not self.available:
            raise RuntimeError("CrewAI not available")

        # Use MAKER to decompose problem
        decomposition = run_maker_evolution(
            initial_program=problem,
            evaluator=lambda x: len(x.split()),
            max_generations=10,
            config=MakerevolutionConfig(
                mode=MakerevolutionMode.DECOMPOSITION,
                enable_decomposition=True
            )
        )

        subtasks = decomposition['best_program'].split('\n')

        # Delegate each subtask to CrewAI
        results = []
        for subtask in subtasks:
            result = await self._delegate_subtask(subtask)
            results.append(result)

        # Assemble with MAKER
        final_result = run_maker_evolution(
            initial_program="\n".join(results),
            evaluator=self._assembly_evaluator,
            max_generations=15,
            config=MakerevolutionConfig(
                mode=MakerevolutionMode.HYBRID
            )
        )

        return final_result

    async def _delegate_subtask(self, subtask):
        """Delegate single subtask to CrewAI"""
        # This would use CrewAI client to delegate
        # Placeholder implementation
        return f"Solved: {subtask}"

    def _assembly_evaluator(self, assembly):
        """Evaluate assembly quality"""
        return min(1.0, len(assembly) / 1000.0)
```

### CrewAI Workflow Integration

```python
def create_crewai_maker_workflow():
    """Create workflow integrating CrewAI and MAKER"""

    workflow = {
        "name": "CrewAI_MAKER_Workflow",
        "stages": [
            {
                "name": "MAKER_Decomposition",
                "type": "maker_decomposition",
                "config": {
                    "mode": "decomposition",
                    "decomposition_depth": 3
                }
            },
            {
                "name": "CrewAI_Delegation",
                "type": "crewai_delegation",
                "depends_on": ["MAKER_Decomposition"],
                "config": {
                    "parallel_delegation": True,
                    "timeout": 60
                }
            },
            {
                "name": "MAKER_Assembly",
                "type": "maker_assembly",
                "depends_on": ["CrewAI_Delegation"],
                "config": {
                    "mode": "hybrid",
                    "voting_threshold": 3
                }
            }
        ]
    }

    return workflow
```

---

## Knowledge Engine Integration

### Knowledge Base Query with MAKER

```python
class KnowledgeMAKERIntegration:
    """Integrate MAKER with Knowledge Engine"""

    def __init__(self, knowledge_manager=None):
        try:
            from sovereign_knowledge_manager import KnowledgeManager
            self.km = knowledge_manager or KnowledgeManager()
            self.available = True
        except ImportError:
            self.available = False
            print("Knowledge Manager not available")

    async def solve_with_knowledge(self, problem):
        """Solve problem using knowledge + MAKER"""

        if not self.available:
            raise RuntimeError("Knowledge Manager not available")

        # Step 1: Query knowledge base
        knowledge = self.km.query_similar_problems(problem)

        # Step 2: Use knowledge to inform MAKER
        config = MAKERHybridConfig(
            voting_threshold=3,
            mcts_simulations=100
        )

        # Add knowledge context to prompt
        enhanced_problem = self._enhance_with_knowledge(problem, knowledge)

        # Step 3: Solve with MAKER
        result = await run_maker_hybrid(
            theorem=enhanced_problem,
            mode=MAKERHybridMode.MAKER_THEN_EVOLUTION,
            config=config
        )

        return {
            "solution": result.best_proof,
            "fitness": result.best_fitness,
            "knowledge_used": len(knowledge),
            "sources": [k['source'] for k in knowledge]
        }

    def _enhance_with_knowledge(self, problem, knowledge):
        """Enhance problem statement with knowledge"""

        enhanced = problem
        if knowledge:
            enhanced += "\n\nRelevant Knowledge:\n"
            for k in knowledge[:3]:  # Top 3
                enhanced += f"- {k['description']}\n"

        return enhanced
```

### Knowledge Extraction with MAKER

```python
async def extract_knowledge_from_solutions():
    """Extract knowledge from MAKER-generated solutions"""

    # Generate multiple solutions
    theorems = [
        "forall n : nat, n + 0 = n",
        "forall n m : nat, n + m = m + n",
        "forall n : nat, n * 1 = n"
    ]

    knowledge_items = []

    for theorem in theorems:
        result = await run_maker_hybrid(
            theorem=theorem,
            mode=MAKERHybridMode.MCTS_THEN_MAKER,
            config=MAKERHybridConfig(voting_threshold=2, mcts_simulations=50)
        )

        if result.success:
            # Extract patterns from proof
            patterns = extract_proof_patterns(result.best_proof)

            knowledge_items.append({
                "theorem": theorem,
                "proof": result.best_proof,
                "patterns": patterns,
                "fitness": result.best_fitness
            })

    # Store in knowledge base
    # km.store_knowledge(knowledge_items)

    return knowledge_items

def extract_proof_patterns(proof):
    """Extract reusable patterns from proof"""
    import re

    patterns = []

    # Extract tactic sequences
    lines = proof.split('\n')
    for line in lines:
        if 'induction' in line:
            patterns.append({
                "type": "induction_pattern",
                "pattern": line.strip(),
                "description": "Use induction for natural number proofs"
            })

    return patterns
```

---

## API Integration

### REST API Endpoints

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

app = FastAPI(title="Hybrid MAKER API")

class TheoremRequest(BaseModel):
    theorem: str
    mode: str = "mcts_then_maker"
    voting_threshold: int = 3
    mcts_simulations: int = 100
    evolution_generations: int = 20

class TheoremResponse(BaseModel):
    success: bool
    proof: str = None
    fitness: float = 0.0
    time: float = 0.0
    error: str = None

@app.post("/api/v1/prove", response_model=TheoremResponse)
async def prove_theorem(request: TheoremRequest):
    """API endpoint to prove theorems using MAKER"""

    try:
        # Map mode string to enum
        mode_map = {
            "mcts_then_maker": MAKERHybridMode.MCTS_THEN_MAKER,
            "maker_then_evolution": MAKERHybridMode.MAKER_THEN_EVOLUTION,
            "adaptive_maker": MAKERHybridMode.ADAPTIVE_MAKER,
            "maker_adversarial": MAKERHybridMode.MAKER_ADVERSARIAL,
            "maker_mdap_parallel": MAKERHybridMode.MAKER_MDAP_PARALLEL,
            "full_maker_hybrid": MAKERHybridMode.FULL_MAKER_HYBRID
        }

        mode = mode_map.get(request.mode, MAKERHybridMode.MCTS_THEN_MAKER)

        # Configure MAKER
        config = MAKERHybridConfig(
            voting_threshold=request.voting_threshold,
            mcts_simulations=request.mcts_simulations,
            evolution_generations=request.evolution_generations
        )

        # Execute
        result = await run_maker_hybrid(
            theorem=request.theorem,
            mode=mode,
            config=config
        )

        return TheoremResponse(
            success=result.success,
            proof=result.best_proof,
            fitness=result.best_fitness,
            time=result.evolution_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/capabilities")
async def get_capabilities():
    """Get MAKER hybrid capabilities"""

    caps = get_maker_hybrid_capabilities()
    return caps

# Run with: uvicorn api:app --reload
```

### GraphQL Integration

```python
import strawberry
from strawberry.fastapi import GraphQLRouter

@strawberry.type
class ProofResult:
    success: bool
    proof: str | None
    fitness: float
    time: float

@strawberry.type
class Query:
    @strawberry.field
    async def prove_theorem(
        self,
        theorem: str,
        mode: str = "mcts_then_maker",
        voting_threshold: int = 3
    ) -> ProofResult:

        result = await run_maker_hybrid(
            theorem=theorem,
            mode=MAKERHybridMode[mode.upper()],
            config=MAKERHybridConfig(voting_threshold=voting_threshold)
        )

        return ProofResult(
            success=result.success,
            proof=result.best_proof,
            fitness=result.best_fitness,
            time=result.evolution_time
        )

    @strawberry.field
    def capabilities(self) -> dict:
        return get_maker_hybrid_capabilities()

schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)

# Add to FastAPI
# app.include_router(graphql_app, prefix="/graphql")
```

---

## Testing Integration

### Unit Tests

```python
import pytest
from hybrid_maker_integration import run_maker_hybrid, MAKERHybridConfig

@pytest.mark.asyncio
async def test_maker_hybrid_basic():
    """Test basic MAKER hybrid functionality"""

    theorem = "forall n : nat, n + 0 = n"

    result = await run_maker_hybrid(
        theorem=theorem,
        mode=MAKERHybridMode.MCTS_THEN_MAKER,
        config=MAKERHybridConfig(
            voting_threshold=2,
            mcts_simulations=10  # Low for testing
        )
    )

    assert result is not None
    assert isinstance(result.success, bool)
    assert isinstance(result.best_fitness, float)
    assert 0.0 <= result.best_fitness <= 1.0

@pytest.mark.asyncio
async def test_maker_configuration():
    """Test configuration validation"""

    config = MAKERHybridConfig(
        voting_threshold=3,
        mcts_simulations=100
    )

    assert config.voting_threshold == 3
    assert config.mcts_simulations == 100

    config_dict = config.to_dict()
    assert config_dict['voting_threshold'] == 3
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete MAKER workflow"""

    from decomposition_engine import DecompositionEngine

    # Step 1: Decompose
    decomp_engine = DecompositionEngine()
    problem = ProblemDefinition(
        title="Test",
        description="Prove: forall n : nat, n + 0 = n"
    )

    plan = decomp_engine.decompose(problem)

    # Step 2: Solve with MAKER
    results = []
    for subproblem in plan.sub_problems:
        result = run_maker_evolution(
            initial_program=subproblem.description,
            evaluator=lambda x: 0.8,  # Mock
            max_generations=5,
            config=MakerevolutionConfig(voting_threshold=2)
        )
        results.append(result)

    # Step 3: Verify
    assert len(results) == len(plan.sub_problems)
    assert all(r['success'] for r in results)
```

---

## Monitoring Integration

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
maker_requests = Counter(
    'maker_requests_total',
    'Total MAKER requests',
    ['mode', 'success']
)

maker_duration = Histogram(
    'maker_duration_seconds',
    'MAKER execution duration',
    ['mode']
)

maker_fitness = Gauge(
    'maker_fitness',
    'MAKER result fitness',
    ['mode']
)

async def monitored_maker_hybrid(theorem, mode, config):
    """MAKER hybrid with Prometheus monitoring"""

    start_time = time.time()

    try:
        result = await run_maker_hybrid(
            theorem=theorem,
            mode=mode,
            config=config
        )

        # Record metrics
        duration = time.time() - start_time
        maker_requests.labels(
            mode=mode.value,
            success=str(result.success)
        ).inc()

        maker_duration.labels(mode=mode.value).observe(duration)

        if result.success:
            maker_fitness.labels(mode=mode.value).set(result.best_fitness)

        return result

    except Exception as e:
        maker_requests.labels(
            mode=mode.value,
            success='false'
        ).inc()
        raise
```

### Logging Integration

```python
import structlog

logger = structlog.get_logger()

async def logged_maker_hybrid(theorem, mode, config):
    """MAKER hybrid with structured logging"""

    log = logger.bind(
        theorem=theorem,
        mode=mode.value,
        config=config.to_dict()
    )

    log.info("Starting MAKER hybrid")

    start_time = time.time()

    try:
        result = await run_maker_hybrid(
            theorem=theorem,
            mode=mode,
            config=config
        )

        duration = time.time() - start_time

        log.info(
            "MAKER hybrid completed",
            success=result.success,
            fitness=result.best_fitness,
            duration=duration,
            generations=result.generations_completed
        )

        return result

    except Exception as e:
        log.error("MAKER hybrid failed", error=str(e))
        raise
```

---

## Deployment Integration

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hybrid-maker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hybrid-maker
  template:
    metadata:
      labels:
        app: hybrid-maker
    spec:
      containers:
      - name: hybrid-maker
        image: hybrid-maker:latest
        ports:
        - containerPort: 8000
        env:
        - name: VOTING_THRESHOLD
          value: "3"
        - name: MCTS_SIMULATIONS
          value: "100"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: hybrid-maker-service
spec:
  selector:
    app: hybrid-maker
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

**End of Integration Guide**

For more information, see:
- Architecture: `HYBRID_MAKER_ARCHITECTURE.md`
- API Reference: `HYBRID_MAKER_API.md`
- User Guide: `HYBRID_MAKER_GUIDE.md`
- Examples: `HYBRID_MAKER_EXAMPLES.md`
