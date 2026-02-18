# Claudiomiro Integration Plan for Decomposition Workflow

**Date**: 2025-12-29
**Status**: Integration Plan
**Workflow**: Sovereign-Grade Decomposition Workflow (SGDW)

---

## Executive Summary

This document outlines the integration of **Claudiomiro** (Autonomous Development CLI) into the **Sovereign-Grade Decomposition Workflow**. Claudiomiro provides production-ready autonomous development capabilities that complement and enhance the existing workflow stages.

**Key Integration Points:**
- Stage 3A: Autonomous solution generation (Blue Team enhancement)
- Stage 3B: Automated code review (Red Team enhancement)
- Stage 3C: Test execution and auto-fixing (Gold Team enhancement)
- Stage 4: Multi-repo integration support
- Stage 5: Branch fixing and PR preparation
- Stage 6: Commit analysis and knowledge extraction

---

## Table of Contents

1. [Integration Overview](#integration-overview)
2. [Stage-by-Stage Integration Plan](#stage-by-stage-integration-plan)
3. [Architecture and Data Flow](#architecture-and-data-flow)
4. [Implementation Details](#implementation-details)
5. [Configuration and Customization](#configuration-and-customization)
6. [Cloud API Provider Mapping](#cloud-api-provider-mapping)
7. [Error Handling and Fallbacks](#error-handling-and-fallbacks)
8. [Performance Considerations](#performance-considerations)
9. [Testing Strategy](#testing-strategy)
10. [Rollout Plan](#rollout-plan)

---

## 1. Integration Overview

### 1.1 Why Integrate Claudiomiro?

**Claudiomiro's Unique Capabilities:**

| Capability | Benefit to Decomposition Workflow |
|------------|-----------------------------------|
| **Autonomous Coding** | Blue Team can generate complete implementations without manual intervention |
| **Auto-Fix Tests** | Gold Team can automatically fix failing tests (3D: Iterative Refinement) |
| **Code Review** | Red Team gets automated senior-level review capabilities |
| **Multi-Repo Support** | Handles backend/frontend/legacy repository coordination |
| **Parallel Execution** | DAG-based parallel execution aligns with MDAP principles |
| **Production Commits** | Creates tested, reviewed, documented commits automatically |
| **Cloud API Compatible** | Works with Claude, OpenAI, Gemini, DeepSeek, GLM |

### 1.2 Integration Philosophy

**Claudiomiro as an "Execution Engine":**

```
Decomposition Workflow (Planning & Orchestration)
                    ↓
        Claudiomiro (Autonomous Execution)
                    ↓
        Production-Ready Code + Commits
```

**Key Principle:**
- **Decomposition Workflow**: WHAT to do, HOW to structure it, WHEN to execute
- **Claudiomiro**: Actually DOES the work autonomously

### 1.3 Integration Scope

**In Scope:**
- ✅ Autonomous solution generation in Stage 3A
- ✅ Automated code review in Stage 3B
- ✅ Test execution and fixing in Stage 3C
- ✅ Multi-repo integration in Stage 4
- ✅ Branch preparation in Stage 5
- ✅ Commit analysis in Stage 6

**Out of Scope:**
- ❌ Replacing existing Team/Gauntlet architecture
- ❌ Changing MDAP principles
- ❌ Modifying core decomposition logic
- ❌ Altering sovereign-grade control philosophy

---

## 2. Stage-by-Stage Integration Plan

### 2.1 Stage 0: Content Analysis

**Current State:**
- AI-assisted analysis of problem statement
- Domain identification, complexity estimation
- Expertise mapping

**Claudiomiro Integration:**
- **Minimal integration needed** - Claudiomiro is not an analysis tool
- Optional: Use Claudiomiro's task decomposition for validation

```python
# Optional: Validate decomposition with Claudiomiro
if validate_with_claudiomiro:
    claudiomiro_result = decompose_task_with_claudiomiro(
        task_id="validate_decomposition",
        prompt=problem_statement,
        working_dir=project_dir,
    )
    # Compare with MDAP decomposition
    # Merge insights if beneficial
```

**Benefits:**
- Cross-validation of task breakdown
- Identifies overlooked dependencies

**When to Use:**
- Optional validation step
- Large-scale projects (>100 sub-problems)

---

### 2.2 Stage 1: AI-Assisted Decomposition

**Current State:**
- AI generates decomposition plan
- Manual review and override
- Creates `DecompositionPlan` with `SubProblem` objects

**Claudiomiro Integration:**
- **Enhancement**: Use Claudiomiro to create implementation plan

```python
# After decomposition is complete
def create_implementation_plan_with_claudiomiro(
    decomposition_plan: DecompositionPlan,
    working_dir: str,
    ai_provider: str = "claude",
):
    """
    Use Claudiomiro to create TODO.md with implementation tasks
    """
    # Build comprehensive prompt
    prompt = build_implementation_prompt(decomposition_plan)

    # Claudiomiro decomposes into actionable tasks
    result = decompose_task_with_claudiomiro(
        task_id="impl_plan",
        prompt=prompt,
        working_dir=working_dir,
        ai_provider=ai_provider,
    )

    return result["sub_tasks"]
```

**Benefits:**
- Leverages Claudiomiro's proven decomposition
- Creates parallelizable task structure
- Aligns with Claudiomiro's execution model

**Deliverables:**
- `.claudiomiro/TODO.md` - Implementation roadmap
- `.claudiomiro/BLUEPRINT.md` - Per-sub-problem blueprints

---

### 2.3 Stage 2: Manual Review & Override

**Current State:**
- Sovereign reviews and approves decomposition
- Can override AI decisions
- Final approval before execution

**Claudiomiro Integration:**
- **No integration needed** - This is a human-only stage
- Claudiomiro respects `.claudiomiro/` configuration files

**Sovereign Control:**
- User can edit `.claudiomiro/TODO.md` before execution
- User can adjust implementation priorities
- User can add custom constraints

---

### 2.4 Stage 3A: Solution Generation (Blue Team) ⭐ PRIMARY INTEGRATION

**Current State:**
```python
# Blue Team generates solutions
for sub_problem in sub_problems:
    blue_team = get_team("Blue")
    solutions = []
    for agent in blue_team.members:
        solution = generate_solution(agent, sub_problem)
        solutions.append(solution)
    best_solution = vote(solutions)
```

**With Claudiomiro Integration:**
```python
# Enhanced Blue Team with Claudiomiro
def solve_sub_problem_with_claudiomiro(
    sub_problem: SubProblem,
    working_dir: str,
    backend: Optional[str] = None,
    frontend: Optional[str] = None,
    ai_provider: str = "claude",
    use_claudiomiro: bool = True,
) -> SolutionAttempt:
    """
    Enhanced solution generation with Claudiomiro autonomous development
    """

    if use_claudiomiro and CLAUDIOMIRO_AVAILABLE:
        # Claudiomiro handles entire implementation
        result = execute_claudiomiro_task(
            task_id=f"blue_team_{sub_problem.id}",
            prompt=sub_problem.description,
            working_dir=working_dir,
            ai_provider=ai_provider,
            backend=backend,
            frontend=frontend,
            max_cycles=20,
        )

        if result["success"]:
            # Extract generated solution
            solution = extract_solution_from_claudiomiro(
                claudiomiro_result=result,
                sub_problem=sub_problem,
            )
            return solution
        else:
            # Fallback to traditional Blue Team
            logger.warning(f"Claudiomiro failed: {result.get('error')}")
            return fallback_to_blue_team(sub_problem)
    else:
        # Use traditional Blue Team
        return solve_with_blue_team_traditional(sub_problem)
```

**Key Enhancement Points:**

1. **Complete Implementation**: Claudiomiro generates full working code, not just outlines
2. **Auto-Testing**: Runs tests and fixes failures before returning
3. **Code Review**: Built-in review before submission
4. **Production-Ready**: Follows best practices, includes documentation

**MDAP Alignment:**

Claudiomiro's parallel execution aligns with MDAP principles:

```python
# Claudiomiro internally uses DAG execution
# This complements MDAP's microtask decomposition

# MDAP: Decompose into microtasks
mdap_microtasks = decompose_into_microtasks(sub_problem)

# Claudiomiro: Execute microtasks in parallel
claudiomiro_dag = build_execution_dag(mdap_microtasks)
parallel_results = claudiomiro.execute_dag(claudiomiro_dag)

# Both systems benefit from decomposition + parallelization
```

**Configuration Options:**

```python
# Per-sub-problem Claudiomiro configuration
sub_problem_claudiomiro_config = {
    "use_claudiomiro": True,  # Enable/disable per sub-problem
    "ai_provider": "claude",  # Provider per sub-problem
    "backend": "./api",  # For backend sub-problems
    "frontend": "./web",  # For frontend sub-problems
    "max_cycles": 20,  # Iteration limit
    "enable_local_llm": "qwen2.5-coder:7b",  # Optional local LLM
}
```

---

### 2.5 Stage 3B: Critique (Red Team) ⭐ SECONDARY INTEGRATION

**Current State:**
```python
# Red Team critiques solutions
critiques = []
for agent in red_team.members:
    critique = generate_critique(agent, solution, sub_problem)
    critiques.append(critique)
aggregated_critique = aggregate_critiques(critiques)
```

**With Claudiomiro Integration:**
```python
# Enhanced Red Team with Claudiomiro code review
def critique_with_claudiomiro(
    solution: SolutionAttempt,
    sub_problem: SubProblem,
    critique_criteria: List[str],
    working_dir: str,
    ai_provider: str = "claude",
) -> CritiqueReport:
    """
    Enhanced critique with Claudiomiro's automated review
    """

    if CLAUDIOMIRO_AVAILABLE:
        # Build review prompt
        review_prompt = f"""
Review the following solution for sub-problem: {sub_problem.description}

Solution Location: {solution.implementation_path}

Critique Criteria:
{chr(10).join(f'- {c}' for c in critique_criteria)}

Provide:
1. Code quality assessment
2. Security vulnerabilities
3. Performance concerns
4. Edge cases missed
5. Best practices violations
6. Specific actionable improvements
"""

        # Claudiomiro performs comprehensive review
        result = execute_claudiomiro_task(
            task_id=f"red_team_review_{solution.id}",
            prompt=review_prompt,
            working_dir=working_dir,
            ai_provider=ai_provider,
            max_cycles=5,  # Fewer cycles for review
        )

        if result["success"]:
            critique = extract_critique_from_claudiomiro(result)
            return critique

    # Fallback to traditional Red Team
    return critique_with_red_team_traditional(solution, sub_problem)
```

**Key Enhancement Points:**

1. **Senior-Level Review**: Claudiomiro applies production code review standards
2. **Multi-Aspect Analysis**: Security, performance, maintainability, testing
3. **Actionable Feedback**: Specific line-numbered recommendations
4. **Best Practices**: Industry-standard patterns and conventions

**Integration with MDAP:**

```python
# MDAP-based distributed critique
def mdap_enhanced_claudiomiro_critique(
    solution: SolutionAttempt,
    sub_problem: SubProblem,
):
    """
    Combine MDAP's distributed critique with Claudiomiro's review
    """

    # Identify solution aspects for distributed critique
    aspects = identify_solution_aspects(solution)

    aspect_critiques = {}

    for aspect in aspects:
        # Use specialized agents for each aspect (MDAP)
        specialized_agents = get_specialized_agents_for_aspect(aspect)

        # Also use Claudiomiro for comprehensive review
        claudiomiro_critique = critique_aspect_with_claudiomiro(
            aspect=aspect,
            solution=solution,
            sub_problem=sub_problem,
        )

        # Apply voting: MDAP agents + Claudiomiro
        all_critiques = specialized_agents + [claudiomiro_critique]
        best_critique = apply_voting(all_critiques, k=2)

        aspect_critiques[aspect] = best_critique

    # Aggregate aspect critiques
    return aggregate_aspect_critiques(aspect_critiques)
```

---

### 2.6 Stage 3C: Verification (Gold Team) ⭐ TERTIARY INTEGRATION

**Current State:**
```python
# Gold Team verifies solutions
verifications = []
for agent in gold_team.members:
    verification = generate_verification(agent, solution, sub_problem)
    verifications.append(verification)
```

**With Claudiomiro Integration:**
```python
# Enhanced Gold Team with Claudiomiro test execution
def verify_with_claudiomiro(
    solution: SolutionAttempt,
    sub_problem: SubProblem,
    test_command: str,
    working_dir: str,
    loop_fixes: bool = True,
    ai_provider: str = "claude",
) -> VerificationReport:
    """
    Enhanced verification with Claudiomiro's auto-fix capabilities
    """

    if CLAUDIOMIRO_AVAILABLE:
        # Claudiomiro runs tests and fixes failures automatically
        result = fix_tests_with_claudiomiro(
            task_id=f"gold_team_verify_{solution.id}",
            test_command=test_command,
            working_dir=working_dir,
            loop_fixes=loop_fixes,  # Keep fixing until tests pass
            max_iterations=10,
            ai_provider=ai_provider,
        )

        if result["success"]:
            verification = extract_verification_from_claudiomiro(result)
            return {
                "verified": True,
                "tests_passed": result["tests_fixed"],
                "iterations": result["iterations"],
                "verification_report": verification,
            }
        else:
            return {
                "verified": False,
                "error": result.get("error"),
            }

    # Fallback to traditional Gold Team
    return verify_with_gold_team_traditional(solution, sub_problem, test_command)
```

**Key Enhancement Points:**

1. **Auto-Fix**: Claudiomiro automatically fixes failing tests
2. **Iterative**: Continues until all tests pass (or max iterations)
3. **Comprehensive**: Runs unit tests, integration tests, linters
4. **Production-Ready**: Only passes when all quality gates met

**MDAP Synergy:**

```python
# MDAP ensures reliable test execution
# Claudiomiro ensures test fixes

# MDAP: Decompose testing into microtasks
test_microtasks = decompose_test_suite(test_command)

# Claudiomiro: Execute and fix each microtask
for test_microtask in test_microtasks:
    # MDAP voting on test results
    test_results = []

    for _ in range(k):  # Voting rounds
        result = claudiomiro.execute_test_and_fix(test_microtask)
        if result.passed:
            test_results.append(result)

    # Apply voting
    final_result = apply_voting(test_results, k=2)

    if not final_result.passed:
        # Escalate to human or higher-capability model
        escalate_test_failure(test_microtask)
```

---

### 2.7 Stage 3D: Iterative Refinement & Evolution

**Current State:**
- Uses OpenEvolve for evolutionary optimization
- Refines solutions through multiple generations

**With Claudiomiro Integration:**
```python
# Hybrid: OpenEvolve + Claudiomiro
def iterative_refinement_with_openevolve_and_claudiomiro(
    solution: SolutionAttempt,
    critiques: List[CritiqueReport],
    sub_problem: SubProblem,
    evolution_iterations: int = 100,
):
    """
    Combine evolutionary optimization with autonomous fixing
    """

    # Phase 1: OpenEvolve evolves the solution
    evolved_solution = evolve_solution_with_openevolve(
        initial_solution=solution,
        critiques=critiques,
        iterations=evolution_iterations,
    )

    # Phase 2: Claudiomiro implements evolutionary suggestions
    if evolved_solution.suggested_changes:
        claudiomiro_result = execute_claudiomiro_task(
            task_id=f"refinement_{sub_problem.id}",
            prompt=f"Implement these changes:\n{evolved_solution.suggested_changes}",
            working_dir=sub_problem.working_dir,
            max_cycles=10,
        )

        if claudiomiro_result["success"]:
            # Re-run tests to verify improvements
            verification = verify_with_claudiomiro(
                solution=evolved_solution,
                sub_problem=sub_problem,
                test_command=sub_problem.test_command,
            )

            return evolved_solution if verification["verified"] else solution

    return solution
```

**Synergy:**
- **OpenEvolve**: Suggests architectural improvements
- **Claudiomiro**: Implements and tests them autonomously

---

### 2.8 Stage 4: Configurable Reassembly

**Current State:**
- Integrates sub-problem solutions
- Verifies integration points

**With Claudiomiro Integration:**
```python
# Multi-repo reassembly with Claudiomiro
def reassemble_multi_repo_with_claudiomiro(
    sub_solutions: List[SolutionAttempt],
    backend: str,
    frontend: str,
    working_dir: str,
    ai_provider: str = "claude",
):
    """
    Reassemble solutions across multiple repositories
    """

    # Build integration prompt
    integration_prompt = build_integration_prompt(sub_solutions)

    # Claudiomiro handles cross-repo integration
    result = execute_multi_repo_task_with_claudiomiro(
        task_id="reassembly_multi_repo",
        prompt=integration_prompt,
        backend=backend,
        frontend=frontend,
        working_dir=working_dir,
        ai_provider=ai_provider,
    )

    if result["success"]:
        # Run integration tests
        integration_test_result = verify_with_claudiomiro(
            solution=sub_solutions[0],  # Representative solution
            test_command="npm run test:integration",
            working_dir=working_dir,
        )

        return {
            "reassembled": True,
            "integration_verified": integration_test_result["verified"],
        }

    return {"reassembled": False, "error": result.get("error")}
```

**Benefits:**
- Handles backend/frontend coordination
- Verifies API contracts between repos
- Manages legacy system integration
- Coordinated commits across repos

---

### 2.9 Stage 5: Final Verification & Self-Healing Loop

**Current State:**
- Final verification of complete solution
- Self-healing loops for failures

**With Claudiomiro Integration:**
```python
# Branch fixing before PR
def final_verification_with_claudiomiro(
    working_dir: str,
    target_branch: str = "main",
    run_tests: bool = True,
    ai_provider: str = "claude",
):
    """
    Final verification and branch preparation with Claudiomiro
    """

    # Step 1: Fix branch with Claudiomiro
    branch_fix_result = fix_branch_with_claudiomiro(
        task_id="final_branch_fix",
        working_dir=working_dir,
        target_branch=target_branch,
        ai_provider=ai_provider,
    )

    if not branch_fix_result["success"]:
        return {
            "verified": False,
            "error": "Branch fix failed",
            "details": branch_fix_result.get("error"),
        }

    # Step 2: Run final test suite
    if run_tests:
        final_test_result = fix_tests_with_claudiomiro(
            task_id="final_tests",
            test_command="npm test",  # Full test suite
            working_dir=working_dir,
            loop_fixes=True,
            max_iterations=20,  # More iterations for final
        )

        if not final_test_result["success"]:
            return {
                "verified": False,
                "error": "Final tests failed",
                "details": final_test_result.get("error"),
            }

    # Step 3: Generate final verification report
    return {
        "verified": True,
        "branch_ready": True,
        "tests_passed": final_test_result.get("tests_fixed", False),
        "commit_ready": True,
    }
```

**Benefits:**
- Production-ready branch before PR
- All tests passing
- Code reviewed
- Documentation complete

---

### 2.10 Stage 6: Knowledge Extraction & Learning

**Current State:**
- Extracts knowledge from successful executions
- Updates knowledge base

**With Claudiomiro Integration:**
```python
# Extract knowledge from Claudiomiro commits
def extract_knowledge_from_claudiomiro_commits(
    workflow_state: WorkflowState,
):
    """
    Analyze Claudiomiro commits for knowledge extraction
    """

    # Get Claudiomiro session data
    claudiomiro_dir = Path(workflow_state.working_dir) / ".claudiomiro"

    # Extract from TODO.md (planning knowledge)
    todo_file = claudiomiro_dir / "TODO.md"
    if todo_file.exists():
        planning_knowledge = extract_planning_patterns(todo_file)
        workflow_state.knowledge_artifacts.extend(planning_knowledge)

    # Extract from git commits (execution knowledge)
    commits = get_claudiomiro_commits(workflow_state.working_dir)
    for commit in commits:
        artifact = KnowledgeArtifact(
            type="execution_pattern",
            source="claudiomiro",
            content=commit["message"],
            metadata={
                "hash": commit["hash"],
                "files_changed": commit["files"],
                "sub_problem": commit.get("sub_problem_id"),
            },
        )
        workflow_state.knowledge_artifacts.append(artifact)

    # Extract from REASONING.md (if available)
    reasoning_file = claudiomiro_dir / "REASONING.md"
    if reasoning_file.exists():
        reasoning_knowledge = extract_reasoning_patterns(reasoning_file)
        workflow_state.knowledge_artifacts.extend(reasoning_knowledge)

    return workflow_state
```

**Knowledge Types Extracted:**
1. **Planning Patterns**: How complex tasks were decomposed
2. **Execution Patterns**: Successful implementation approaches
3. **Testing Patterns**: Effective test strategies
4. **Integration Patterns**: Cross-component integration techniques
5. **Anti-Patterns**: What to avoid (from failures)

---

## 3. Architecture and Data Flow

### 3.1 Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Sovereign-Grade Decomposition Workflow                    │
│                         (Planning & Orchestration)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ delegates to
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Teams & Gauntlets Layer                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                         │
│  │ Blue Team  │  │ Red Team   │  │ Gold Team  │                         │
│  │ (Solve)    │  │ (Critique)  │  │ (Verify)   │                         │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                         │
└────────┼────────────────┼────────────────┼─────────────────────────────────┘
         │                │                │
         │                │                │
         │      ┌─────────┴──────────┐     │
         │      │ Claudiomiro Layer │     │
         │      │   (Enhancement)    │     │
         │      └─────────┬──────────┘     │
         │                │                │
         ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Claudiomiro CLI                                     │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ Task Decomposition → Parallel Execution → Auto-Fix → Commit       │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Cloud Providers:                                                            │
│  ├── Anthropic Claude    (Claude's native strength)                        │
│  ├── OpenAI Codex        (Strong coding capabilities)                     │
│  ├── Google Gemini       (Good at integration)                            │
│  ├── DeepSeek            (Cost-effective)                                  │
│  └── GLM                 (Alternative option)                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow: Stage 3A Example

```python
# Step 1: Sub-problem assigned to Blue Team
sub_problem = {
    "id": "sp_001",
    "description": "Implement JWT authentication",
    "requirements": ["access_tokens", "refresh_tokens", "middleware"],
    "working_dir": "./project",
    "backend": "./api",
}

# Step 2: Blue Team delegates to Claudiomiro
def blue_team_solve(sub_problem):
    if sub_problem.config.get("use_claudiomiro", True):
        return solve_with_claudiomiro(sub_problem)
    else:
        return solve_with_traditional_blue_team(sub_problem)

# Step 3: Claudiomiro executes
claudiomiro_result = execute_claudiomiro_task(
    task_id=f"blue_team_{sub_problem['id']}",
    prompt=sub_problem["description"],
    working_dir=sub_problem["working_dir"],
    backend=sub_problem.get("backend"),
    ai_provider=sub_problem.config.get("ai_provider", "claude"),
)

# Step 4: Claudiomiro workflow (autonomous)
# - Decomposes into tasks
# - Implements JWT auth
# - Writes tests
# - Runs tests
# - Fixes failures
# - Reviews code
# - Commits

# Step 5: Extract solution
solution_attempt = SolutionAttempt(
    sub_problem_id=sub_problem["id"],
    implementation_path="./project/api/auth/jwt.py",
    test_results=claudiomiro_result["test_results"],
    commit_hash=claudiomiro_result["commit_hash"],
)

return solution_attempt
```

### 3.3 Configuration Flow

```yaml
# Sub-problem level configuration
sub_problem:
  id: "sp_001"
  description: "Implement JWT authentication"

  # Claudiomiro integration
  claudiomiro:
    enabled: true
    ai_provider: "claude"  # or "codex", "gemini", "deep-seek", "glm"
    max_cycles: 20
    backend: "./api"
    frontend: null

    # When to use Claudiomiro
    use_for:
      generation: true   # Stage 3A: Solution generation
      critique: false     # Stage 3B: Use traditional Red Team
      verification: true  # Stage 3C: Test execution and fixing

    # Local LLM integration (optional)
    local_llm:
      enabled: true
      model: "qwen2.5-coder:7b"
      use_for: "semantic_search"  # or "code_completion", "summarization"
```

---

## 4. Implementation Details

### 4.1 Enhanced Sub-Problem Schema

```python
@dataclass
class SubProblem:
    """Enhanced sub-problem with Claudiomiro integration"""
    id: str
    description: str
    requirements: List[str]
    constraints: Optional[List[str]] = None

    # Existing fields
    domain: Optional[str] = None
    problem_type: Optional[str] = None
    complexity_score: Optional[float] = None
    estimated_effort: Optional[str] = None

    # Claudiomiro configuration (NEW)
    claudiomiro_config: Optional[ClaudiomiroConfig] = None

    # Execution tracking
    working_dir: Optional[str] = None
    backend: Optional[str] = None
    frontend: Optional[str] = None
    legacy: Optional[str] = None

@dataclass
class ClaudiomiroConfig:
    """Claudiomiro-specific configuration for sub-problem"""
    enabled: bool = True
    ai_provider: str = "claude"  # claude, codex, gemini, deep-seek, glm
    max_cycles: int = 20
    use_for_generation: bool = True
    use_for_critique: bool = False
    use_for_verification: bool = True
    local_llm_model: Optional[str] = None  # e.g., "qwen2.5-coder:7b"
```

### 4.2 MCP Tool Enhancements

```python
# Enhanced decomposition MCP tool with Claudiomiro
@mcp_tool("solve_sub_problem_with_team")
def solve_sub_problem_with_team(
    sub_problem_id: str,
    sub_problem_description: str,
    team_name: str,
    context: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[str]] = None,
    requirements: Optional[List[str]] = None,
    # Claudiomiro parameters (NEW)
    use_claudiomiro: bool = True,
    claudiomiro_provider: str = "claude",
    claudiomiro_backend: Optional[str] = None,
    claudiomiro_frontend: Optional[str] = None,
    # OpenEvolve parameters
    use_evolution: bool = True,
    evolution_iterations: int = 100,
) -> Dict[str, Any]:
    """
    Enhanced sub-problem solving with Claudiomiro integration
    """

    # Load sub-problem
    sub_problem = load_sub_problem(sub_problem_id)

    # Choose execution method
    if use_claudiomiro and CLAUDIOMIRO_AVAILABLE:
        # Use Claudiomiro for autonomous execution
        return solve_sub_problem_with_claudiomiro(
            sub_problem=sub_problem,
            backend=claudiomiro_backend,
            frontend=claudiomiro_frontend,
            ai_provider=claudiomiro_provider,
        )
    else:
        # Use traditional team execution
        return solve_sub_problem_with_traditional_team(
            sub_problem=sub_problem,
            team_name=team_name,
            use_evolution=use_evolution,
            evolution_iterations=evolution_iterations,
        )
```

### 4.3 Bridge Function Updates

```python
# Enhanced decomposition_crewai_bridge.py

class DecompositionCrewAIWorkflowBridge:
    def execute_phase_3a_solution(
        self,
        sub_problems: List[SubProblem],
        context: Optional[Dict[str, Any]] = None,
        enable_claudiomiro: bool = True,  # NEW
        claudiomiro_provider: str = "claude",  # NEW
    ):
        """
        Enhanced Phase 3A with Claudiomiro integration
        """
        solutions = []

        for sub_problem in sub_problems:
            # Check if sub-problem should use Claudiomiro
            use_claudio = (
                enable_claudiomiro and
                sub_problem.claudiomiro_config.enabled if sub_problem.claudiomiro_config else True
            )

            if use_claudio:
                # Use Claudiomiro
                solution = solve_sub_problem_with_claudiomiro(
                    sub_problem=sub_problem,
                    ai_provider=claudiomiro_provider,
                    backend=sub_problem.backend,
                    frontend=sub_problem.frontend,
                )
            else:
                # Use traditional Blue Team
                solution = solve_sub_problem_with_traditional_team(
                    sub_problem=sub_problem,
                )

            solutions.append(solution)

        return solutions
```

---

## 5. Configuration and Customization

### 5.1 Provider Selection Strategy

```python
# Intelligent provider selection per sub-problem
def select_claudiomiro_provider(sub_problem: SubProblem) -> str:
    """
    Select optimal AI provider for sub-problem
    """

    # Backend tasks
    if sub_problem.backend and not sub_problem.frontend:
        # Backend tasks: Use Claude or Codex
        if sub_problem.domain == "security":
            return "claude"  # Claude best for security
        elif sub_problem.domain == "database":
            return "codex"  # Codex good at SQL
        else:
            return "claude"  # Default to Claude

    # Frontend tasks
    elif sub_problem.frontend and not sub_problem.backend:
        # Frontend tasks: Use Codex or Gemini
        if sub_problem.domain == "ui_ux":
            return "gemini"  # Gemini good at UI
        else:
            return "codex"  # Default to Codex

    # Full-stack tasks
    elif sub_problem.backend and sub_problem.frontend:
        # Multi-repo: Use Gemini or Claude
        return "gemini"  # Good at integration

    # Cost-sensitive tasks
    elif sub_problem.complexity_score < 0.3:
        return "deep-seek"  # Cost-effective

    # Default
    return "claude"
```

### 5.2 Local LLM Integration

```python
# Configure local LLM for specific tasks
def configure_local_llm_for_sub_problem(sub_problem: SubProblem):
    """
    Configure when to use local LLM vs cloud API
    """

    # Use local LLM for:
    # - Semantic search of codebase
    # - Context summarization
    # - Symbol explanation
    # - Code formatting

    # Use cloud API for:
    # - Complex logic generation
    # - Security-critical code
    # - Production implementation

    if sub_problem.task_type == "semantic_search":
        return {
            "local_llm": "qwen2.5-coder:7b",
            "cloud_llm": None,
        }

    elif sub_problem.task_type == "implementation":
        return {
            "local_llm": "qwen2.5-coder:7b",  # For search/summarization
            "cloud_llm": "claude",  # For actual implementation
        }

    return None
```

---

## 6. Cloud API Provider Mapping

### 6.1 Provider Selection Matrix

| Sub-Problem Type | Recommended Provider | Fallback | Reason |
|-----------------|----------------------|----------|---------|
| **Security** | Claude | Codex | Claude excels at security analysis |
| **Database** | Codex | Claude | Codex strong at SQL/queries |
| **API Design** | Claude | Gemini | Claude good at API contracts |
| **Frontend/UI** | Codex | Gemini | Codex strong at React/Vue |
| **Integration** | Gemini | Claude | Gemini good at connecting components |
| **Testing** | Claude | Codex | Claude thorough at test coverage |
| **Documentation** | Claude | Gemini | Claude clear explanations |
| **Performance** | Codex | Claude | Codex optimizes well |
| **Legacy Refactor** | Claude | Codex | Claude cautious with legacy |
| **Simple/Cost-sensitive** | DeepSeek | Claude | DeepSeek cost-effective |

### 6.2 Cost Optimization Strategy

```python
def optimize_cost_with_providers(sub_problem: SubProblem):
    """
    Balance quality vs cost
    """

    # High-value, high-complexity: Use best provider
    if sub_problem.complexity_score > 0.7 and sub_problem.is_critical:
        return "claude"  # Best quality, higher cost

    # Medium complexity: Use Codex
    elif 0.3 < sub_problem.complexity_score <= 0.7:
        return "codex"  # Good quality, lower cost

    # Low complexity: Use DeepSeek
    elif sub_problem.complexity_score <= 0.3 and not sub_problem.is_critical:
        return "deep-seek"  # Acceptable quality, lowest cost

    # Default
    return "claude"
```

---

## 7. Error Handling and Fallbacks

### 7.1 Claudiomiro Unavailable

```python
def solve_sub_problem_with_fallbacks(sub_problem: SubProblem):
    """
    Try Claudiomiro, fallback to traditional methods
    """

    # Attempt 1: Claudiomiro (preferred)
    if CLAUDIOMIRO_AVAILABLE and sub_problem.claudiomiro_config.enabled:
        try:
            solution = solve_with_claudiomiro(sub_problem)
            if solution["success"]:
                return solution
        except Exception as e:
            logger.warning(f"Claudiomiro failed: {e}")

    # Attempt 2: Traditional Blue Team (fallback)
    try:
        solution = solve_with_traditional_blue_team(sub_problem)
        if solution["success"]:
            return solution
    except Exception as e:
        logger.error(f"Traditional Blue Team failed: {e}")

    # Attempt 3: Basic implementation (last resort)
    return basic_implementation_fallback(sub_problem)
```

### 7.2 Provider Fallback Chain

```python
def execute_with_provider_fallback(
    sub_problem: SubProblem,
    providers: List[str],  # ["claude", "codex", "gemini"]
):
    """
    Try multiple providers until one succeeds
    """

    for provider in providers:
        try:
            result = execute_claudiomiro_task(
                sub_problem=sub_problem,
                ai_provider=provider,
            )

            if result["success"]:
                logger.info(f"Successfully executed with {provider}")
                return result

        except Exception as e:
            logger.warning(f"{provider} failed: {e}")
            continue

    # All providers failed
    raise Exception("All providers failed")
```

---

## 8. Performance Considerations

### 8.1 Parallel Execution Strategy

```python
# Parallel sub-problem solving with Claudiomiro
async def solve_sub_problems_in_parallel(
    sub_problems: List[SubProblem],
    max_parallel: int = 3,  # Limit concurrent Claudiomiro instances
):
    """
    Solve multiple sub-problems in parallel using Claudiomiro
    """

    semaphore = asyncio.Semaphore(max_parallel)

    async def solve_one(sub_problem):
        async with semaphore:
            return await asyncio.to_thread(
                solve_sub_problem_with_claudiomiro,
                sub_problem
            )

    # Run all sub-problems in parallel (with limit)
    tasks = [solve_one(sp) for sp in sub_problems]
    solutions = await asyncio.gather(*tasks)

    return solutions
```

### 8.2 Caching Strategy

```python
# Cache Claudiomiro results for similar sub-problems
from functools import lru_cache

@lru_cache(maxsize=128)
def get_claudiomiro_cached_result(
    sub_problem_hash: str,  # Hash of sub-problem description
    ai_provider: str,
):
    """
    Return cached result if available
    """
    # Check cache first
    cached_result = cache.get(sub_problem_hash)
    if cached_result:
        logger.info(f"Cache hit for {sub_problem_hash}")
        return cached_result

    # Execute with Claudiomiro
    result = execute_claudiomiro_task(...)

    # Cache successful results
    if result["success"]:
        cache.set(sub_problem_hash, result, ttl=3600)

    return result
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# Test Claudiomiro integration per stage
def test_stage_3a_claudiomiro_integration():
    """
    Test Stage 3A solution generation with Claudiomiro
    """

    # Setup
    sub_problem = create_test_sub_problem(
        description="Implement JWT auth",
        claudiomiro_config=ClaudiomiroConfig(enabled=True)
    )

    # Execute
    solution = solve_sub_problem_with_claudiomiro(sub_problem)

    # Assertions
    assert solution["success"]
    assert solution["implementation_path"] is not None
    assert solution["commit_hash"] is not None
    assert solution["tests_passed"]
```

### 9.2 Integration Tests

```python
# Test full workflow with Claudiomiro
def test_full_workflow_with_claudiomiro():
    """
    Test complete decomposition workflow with Claudiomiro
    """

    # Create decomposition plan
    plan = create_decomposition_plan(problem_statement)

    # Execute all stages with Claudiomiro
    result = execute_full_workflow(
        plan=plan,
        enable_claudiomiro=True,
        claudiomiro_provider="claude",
    )

    # Verify
    assert result["overall_success"]
    assert len(result["solutions"]) == len(plan.sub_problems)
    assert all(s["tests_passed"] for s in result["solutions"])
```

---

## 10. Rollout Plan

### 10.1 Phase 1: Pilot (Week 1-2)

**Scope:**
- Enable Claudiomiro for Stage 3A only
- Limited to 5 sub-problems
- Use Claude provider only
- Manual review before commit

**Success Criteria:**
- 80%+ autonomous success rate
- No production issues
- Time savings vs traditional method

### 10.2 Phase 2: Expansion (Week 3-4)

**Scope:**
- Enable Stage 3B (critique) and Stage 3C (verification)
- Add Codex and Gemini providers
- Implement provider fallback chain
- Auto-commit enabled (with manual override option)

**Success Criteria:**
- 75%+ autonomous success rate
- Reduced review burden
- Faster turnaround

### 10.3 Phase 3: Full Integration (Week 5-6)

**Scope:**
- Enable all stages
- Multi-repo support
- Local LLM integration
- Performance optimization

**Success Criteria:**
- 70%+ autonomous success rate
- Significant time savings
- High code quality maintained

### 10.4 Phase 4: Optimization (Week 7-8)

**Scope:**
- Fine-tune provider selection
- Optimize caching strategy
- Advanced parallel execution
- Cost optimization

**Success Criteria:**
- Maximize cost/quality ratio
- Minimize latency
- Sustained high quality

---

## Summary

**Integration Benefits:**

1. **Autonomous Development**: Reduces manual effort by 60-80%
2. **Production-Ready Code**: Built-in testing and review
3. **Multi-Repo Support**: Handles complex projects
4. **Cloud API Compatible**: Works with all major providers
5. **Synergy with MDAP**: Parallel execution + decomposition
6. **Flexibility**: Can enable/disable per sub-problem
7. **Sovereign Control**: User maintains full control

**Key Integration Points:**
- Stage 3A: Solution generation (PRIMARY)
- Stage 3B: Code review (SECONDARY)
- Stage 3C: Test fixing (TERTIARY)
- Stage 4: Multi-repo integration
- Stage 5: Branch preparation
- Stage 6: Knowledge extraction

**Implementation Priority:**
1. High: Stage 3A (autonomous coding)
2. Medium: Stage 3C (auto-fix tests)
3. Low: Stage 3B (automated review)
4. Low: Other stages (enhancement)

---

**Date**: 2025-12-29
**Status**: Integration Plan Complete
**Next Step**: Begin Phase 1 Pilot Implementation
