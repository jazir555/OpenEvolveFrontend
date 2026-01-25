# Tutorial 4: Solution Integration

**Level:** Advanced
**Time:** 50 minutes
**Prerequisites:** Tutorial 1-3

---

## Learning Objectives

After this tutorial, you will be able to:
- Solve sub-problems with different execution methods
- Integrate solutions from multiple sources
- Handle solution dependencies
- Validate and critique solutions
- Assemble final solution

---

## Overview: The Complete Workflow

```
┌─────────────────────────────────────────────────────────┐
│              1. DECOMPOSITION                           │
│  Break problem → Sub-problems                           │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              2. SOLUTION (Parallel)                     │
│  ├─ Sub-problem 1 → Traditional LLM                    │
│  ├─ Sub-problem 2 → Claudiomiro (code gen)             │
│  ├─ Sub-problem 3 → DataPizza (research)               │
│  ├─ Sub-problem 4 → ROMA (hierarchical)                │
│  └─ Sub-problem 5 → ROMA-MDAP-MAKER (zero-error)       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              3. CRITIQUE (Red Team)                     │
│  Adversarial testing → Issues found                    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              4. VERIFICATION (Gold Team)                │
│  Quality validation → Approved/Rejected                │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              5. INTEGRATION                             │
│  Assemble solutions → Final solution                   │
└─────────────────────────────────────────────────────────┘
```

---

## Step 1: Solving Sub-Problems

### Execution Method Selection

Choose execution method based on sub-problem characteristics:

```python
# example_method_selection.py
from decomposition_mcp_tools import solve_sub_problem_with_team

def select_execution_method(sub_problem) -> str:
    """Auto-select best execution method"""

    desc = sub_problem.description.lower()
    title = sub_problem.title.lower()

    # Critical zero-error tasks → ROMA-MDAP-MAKER
    if any(kw in desc for kw in ['critical', 'safety', 'zero-error', 'flawless']):
        return 'roma_mdap_maker'

    # Implementation tasks → Claudiomiro
    if any(kw in title for kw in ['implement', 'code', 'api', 'function']):
        return 'claudiomiro'

    # Research/analysis → DataPizza
    if sub_problem.type.value in ['research', 'analysis']:
        return 'datapizza'

    # Hierarchical decomposition → ROMA
    if 'decompose' in desc or 'break down' in desc:
        return 'roma'

    # Default → Traditional
    return 'traditional'

# Example: Solve each sub-problem with best method
solutions = {}
for sp in result.sub_problems:
    method = select_execution_method(sp)

    print(f"Solving {sp.title} with {method}...")

    solution = solve_sub_problem_with_team(
        sub_problem_id=sp.id,
        sub_problem_description=sp.description,
        team_name="Default-Blue",
        execution_method=method,
        constraints=[c.description for c in problem.constraints],
        requirements=[sc.description for sc in problem.success_criteria]
    )

    solutions[sp.id] = solution
```

---

## Step 2: Method-Specific Examples

### Traditional Method (LLM)

```python
# example_traditional.py
solution = solve_sub_problem_with_team(
    sub_problem_id="sub-001",
    sub_problem_description="Design database schema for user management",
    team_name="Default-Blue",
    execution_method="traditional",
    use_evolution=True,
    evolution_iterations=100
)

print(f"Status: {solution['status']}")
print(f"Solution:\n{solution['solution']}")
```

**Output:**
```
Status: completed
Solution:
# Database Schema Design

## Tables

### users
- id (UUID, PK)
- email (VARCHAR, UNIQUE, NOT NULL)
- password_hash (VARCHAR, NOT NULL)
- created_at (TIMESTAMP)
- updated_at (TIMESTAMP)

### user_profiles
- user_id (UUID, FK → users.id)
- first_name (VARCHAR)
- last_name (VARCHAR)
- avatar_url (VARCHAR)

Indexes:
- idx_users_email (email)
- idx_user_profiles_user_id (user_id)
```

---

### Claudiomiro Method (Code Generation)

```python
# example_claudiomiro.py
solution = solve_sub_problem_with_team(
    sub_problem_id="sub-002",
    sub_problem_description="Implement REST API for user authentication",
    team_name="Backend-Blue",
    execution_method="claudiomiro",
    claudiomiro_provider="claude",
    working_dir="./backend",
    max_cycles=20
)

print(f"Method: {solution['execution_method_used']}")
if solution['status'] == 'completed':
    # Claudiomiro creates actual files
    print(f"Files created in {solution['working_dir']}")
    print(solution['solution'])  # Contains file listing
```

**Output:**
```
Method: claudiomiro
Status: completed
Files created in ./backend:
✓ backend/api/auth.py
✓ backend/models/user.py
✓ backend/utils/password.py
✓ backend/tests/test_auth.py
```

---

### DataPizza Method (Multi-Agent Research)

```python
# example_datapizza.py
solution = solve_sub_problem_with_team(
    sub_problem_id="sub-003",
    sub_problem_description="Research best practices for password hashing",
    team_name="Research-Blue",
    execution_method="datapizza",
    datapizza_provider="openai",
    datapizza_model="gpt-4o",
    datapizza_tools=["filesystem", "duckduckgo"],
    datapizza_max_steps=20
)

print(f"Steps taken: {solution['steps_taken']}")
print(f"Tools used: {solution['tools_used']}")
print(f"\nResearch findings:\n{solution['solution']}")
```

**Output:**
```
Steps taken: 15
Tools used: ['duckduckgo', 'filesystem']

Research findings:
# Password Hashing Best Practices

## Recommended Algorithm: Argon2id

### Why Argon2id?
- Winner of Password Hashing Competition 2015
- Resistant to GPU/ASIC attacks
- Memory-hard computation
- Flexible parameters (time, memory, parallelism)

### Configuration (2025)
```python
import argon2

hasher = argon2.PasswordHasher(
    time_cost=3,        # Iterations
    memory_cost=262144, # 256 MB
    parallelism=4,      # Threads
    hash_len=32,        # Output hash length
    salt_len=16         # Salt length
)
```

### Security Considerations
...
```

---

### ROMA Method (Hierarchical Decomposition)

```python
# example_roma.py
solution = solve_sub_problem_with_team(
    sub_problem_id="sub-004",
    sub_problem_description="Design comprehensive system architecture",
    team_name="Architecture-Blue",
    execution_method="roma",
    roma_max_depth=2,
    roma_execution_mode="recursive",
    roma_provider="openai",
    roma_model="gpt-4o"
)

print(f"DAG tasks: {solution['dag_info']['total_tasks']}")
print(f"\nArchitecture:\n{solution['solution']}")
```

**Output:**
```
DAG tasks: 47

Architecture:
# System Architecture Design

## Layer 1: Top-Level Components

### 1.1 API Gateway
**Purpose**: Single entry point for all client requests

**Sub-components**:
- 1.1.1 Rate limiting
- 1.1.2 Authentication
- 1.1.3 Request routing
- 1.1.4 Load balancing

### 1.2 Application Services
**Purpose**: Business logic implementation

**Sub-components**:
- 1.2.1 User service
- 1.2.2 Product service
- 1.2.3 Order service
- 1.2.4 Payment service

...
```

---

### ROMA-MDAP-MAKER Method (Zero-Error)

```python
# example_roma_mdap_maker.py
solution = solve_sub_problem_with_team(
    sub_problem_id="sub-005",
    sub_problem_description="Implement safety-critical validation logic",
    team_name="Safety-Blue",
    execution_method="roma_mdap_maker",
    roma_mdap_maker_k_ahead=3,
    roma_mdap_maker_enable_red_flagging=True,
    roma_mdap_maker_enable_adaptive_k=True,
    roma_mdap_maker_provider="openai",
    roma_mdap_maker_model="gpt-4o-mini"
)

metrics = solution.get('roma_mdap_maker_metrics', {})

print(f"Zero-error solution: {solution['status']}")
print(f"ROMA levels: {metrics['roma_decomposition_levels']}")
print(f"Atomic tasks: {metrics['total_atomic_tasks']}")
print(f"Voting rounds: {metrics['total_voting_rounds']}")
print(f"Red-flags caught: {metrics['total_red_flags']}")
print(f"Final error rate: {metrics['final_error_rate']:.6f}")

if metrics['final_error_rate'] < 0.001:
    print(f"✓ Zero-error achieved! (99.9%+ accuracy)")
```

**Output:**
```
Zero-error solution: completed
ROMA levels: 3
Atomic tasks: 12
Voting rounds: 36
Red-flags caught: 3
Final error rate: 0.000234
✓ Zero-error achieved! (99.977% accuracy)

Solution:
# Safety-Critical Validation Logic

## Input Validation
```python
def validate_user_input(input_data: dict) -> tuple[bool, list[str]]:
    """
    Validate user input with 99.977% accuracy.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # Length validation
    if not isinstance(input_data.get('username'), str):
        errors.append("Username must be string")
    elif len(input_data['username']) < 3:
        errors.append("Username too short (min 3 chars)")
    elif len(input_data['username']) > 50:
        errors.append("Username too long (max 50 chars)")

    # Character validation
    if input_data.get('username'):
        if not all(c.isalnum() or c in '_-' for c in input_data['username']):
            errors.append("Username contains invalid characters")

    # ... (more validation)

    return len(errors) == 0, errors
```

This implementation was validated via 36 voting rounds with
first-to-ahead-by-k (k=3) voting, achieving 99.977% accuracy.
```

---

## Step 3: Critique (Red Team)

```python
# example_critique.py
from decomposition_mcp_tools import critique_solution_with_gauntlet

# After solving sub-problem, critique it
critique = critique_solution_with_gauntlet(
    solution=solution_text,
    sub_problem_id="sub-001",
    gauntlet_name="Security-RedTeam-Gauntlet",
    sub_problem_description=sp.description,
    use_evolution=True
)

print(f"Approved: {critique['approved']}")
print(f"Overall score: {critique['overall_score']}")
print(f"\nIssues found: {len(critique['issues_found'])}")

for issue in critique['issues_found']:
    print(f"\n[{issue['severity'].upper()}] {issue['description']}")
    print(f"  Location: {issue.get('location', 'N/A')}")
    print(f"  Fix: {issue.get('suggestion', 'N/A')}")
```

**Output:**
```
Approved: False
Overall score: 0.72

Issues found: 3

[HIGH] Password hashing uses insufficient memory cost
  Location: backend/utils/password.py, line 15
  Fix: Increase memory_cost to 262144 (256 MB)

[MEDIUM] Missing rate limiting on authentication endpoint
  Location: backend/api/auth.py, line 42
  Fix: Implement rate limiting (max 5 requests/minute)

[LOW] No input validation on email field
  Location: backend/models/user.py, line 23
  Fix: Add email format validation
```

---

## Step 4: Verification (Gold Team)

```python
# example_verification.py
from decomposition_mcp_tools import verify_solution_with_gauntlet

# After critique, verify solution
verification = verify_solution_with_gauntlet(
    solution=revised_solution_text,
    critique=critique,
    sub_problem_id="sub-001",
    gauntlet_name="Quality-GoldTeam-Gauntlet",
    requirements=[
        "Must use Argon2id with 256 MB memory",
        "Must implement rate limiting",
        "Must validate all inputs"
    ],
    use_evolution=True
)

print(f"Approved: {verification['approved']}")
print(f"Correctness: {verification['correctness_score']}")
print(f"Completeness: {verification['completeness_score']}")
print(f"Quality: {verification['quality_score']}")

print(f"\nRequirements met:")
for req, met in verification['requirements_met'].items():
    status = "✓" if met else "✗"
    print(f"{status} {req}")
```

**Output:**
```
Approved: True
Correctness: 0.95
Completeness: 0.92
Quality: 0.93

Requirements met:
✓ Must use Argon2id with 256 MB memory
✓ Must implement rate limiting
✓ Must validate all inputs
```

---

## Step 5: Integration

```python
# example_integration.py
def assemble_final_solution(sub_problems, solutions, critiques, verifications):
    """Assemble solutions into final result"""

    approved_solutions = {}
    integration_plan = []

    # Only include approved solutions
    for sp in sub_problems:
        sol = solutions[sp.id]
        crit = critiques.get(sp.id, {})
        ver = verifications.get(sp.id, {})

        if ver.get('approved', False):
            approved_solutions[sp.id] = sol['solution']
            integration_plan.append({
                'id': sp.id,
                'title': sp.title,
                'status': 'approved',
                'dependencies': sp.dependencies
            })
        else:
            integration_plan.append({
                'id': sp.id,
                'title': sp.title,
                'status': 'rejected',
                'reason': ver.get('feedback', 'Verification failed')
            })

    # Order by dependencies
    ordered = []
    executed = set()

    while len(executed) < len(integration_plan):
        ready = [
            item for item in integration_plan
            if item['id'] not in executed and
            all(dep in executed for dep in item.get('dependencies', []))
        ]

        if not ready:
            break

        ordered.extend(ready)
        for item in ready:
            executed.add(item['id'])

    return {
        'integration_plan': ordered,
        'approved_solutions': approved_solutions,
        'total_approved': len(approved_solutions),
        'total_rejected': len(sub_problems) - len(approved_solutions)
    }

# Use it
final = assemble_final_solution(
    sub_problems=result.sub_problems,
    solutions=solutions,
    critiques=critiques,
    verifications=verifications
)

print(f"\n=== Final Integration ===")
print(f"Approved: {final['total_approved']}/{len(result.sub_problems)}")

print(f"\nIntegration Plan:")
for item in final['integration_plan']:
    status_icon = "✓" if item['status'] == 'approved' else "✗"
    print(f"{status_icon} {item['title']} - {item['status'].upper()}")

    if item['status'] == 'approved':
        print(f"\n{final['approved_solutions'][item['id']][:200]}...")
        print()
```

---

## Complete Workflow Example

```python
# complete_workflow.py
from decomposition_mcp_tools import (
    analyze_problem_for_decomposition,
    decompose_problem_into_sub_problems,
    solve_sub_problem_with_team,
    critique_solution_with_gauntlet,
    verify_solution_with_gauntlet
)

def complete_workflow(problem_statement: str):
    """End-to-end decomposition workflow"""

    print("=" * 80)
    print("OPENEVOLVE DECOMPOSITION WORKFLOW")
    print("=" * 80)

    # Step 1: Analyze
    print("\n[1/5] Analyzing problem...")
    analysis = analyze_problem_for_decomposition(
        problem_statement=problem_statement,
        use_evolution=False  # Faster
    )
    print(f"✓ Domain: {analysis['domain']}")
    print(f"✓ Complexity: {analysis['complexity']['overall']}/10")

    # Step 2: Decompose
    print("\n[2/5] Decomposing problem...")
    decomp = decompose_problem_into_sub_problems(
        problem_statement=problem_statement,
        analysis=analysis,
        decomposition_strategy="semantic"
    )
    print(f"✓ Generated {decomp['total_sub_problems']} sub-problems")

    # Step 3: Solve
    print("\n[3/5] Solving sub-problems...")
    solutions = {}
    for sp in decomp['sub_problems']:
        print(f"  - {sp['title']}...", end=" ")

        solution = solve_sub_problem_with_team(
            sub_problem_id=sp['id'],
            sub_problem_description=sp['description'],
            team_name="Default-Blue",
            execution_method="traditional",
            use_evolution=False  # Faster
        )

        solutions[sp['id']] = solution
        print("✓" if solution['status'] == 'completed' else "✗")

    # Step 4: Critique & Verify
    print("\n[4/5] Validating solutions...")
    critiques = {}
    verifications = {}

    for sp in decomp['sub_problems']:
        sol = solutions[sp['id']]
        if sol['status'] != 'completed':
            continue

        print(f"  - {sp['title']}...", end=" ")

        # Critique
        critique = critique_solution_with_gauntlet(
            solution=sol['solution'],
            sub_problem_id=sp['id'],
            gauntlet_name="Default-RedTeam-Gauntlet"
        )
        critiques[sp['id']] = critique

        # Verify
        verification = verify_solution_with_gauntlet(
            solution=sol['solution'],
            critique=critique,
            sub_problem_id=sp['id'],
            gauntlet_name="Default-GoldTeam-Gauntlet"
        )
        verifications[sp['id']] = verification

        status = "✓" if verification['approved'] else "✗"
        print(status)

    # Step 5: Integrate
    print("\n[5/5] Assembling final solution...")
    final = assemble_final_solution(
        sub_problems=decomp['sub_problems'],
        solutions=solutions,
        critiques=critiques,
        verifications=verifications
    )

    print(f"\n{'='*80}")
    print(f"WORKFLOW COMPLETE")
    print(f"{'='*80}")
    print(f"Sub-Problems: {decomp['total_sub_problems']}")
    print(f"Solved: {len(solutions)}")
    print(f"Approved: {final['total_approved']}")
    print(f"Rejected: {final['total_rejected']}")

    if final['total_approved'] > 0:
        approval_rate = final['total_approved'] / decomp['total_sub_problems']
        print(f"Approval Rate: {approval_rate:.1%}")

    return final

# Run it
problem = "Build a secure authentication system with password reset, 2FA, and session management"

result = complete_workflow(problem)
```

---

## Summary

In this tutorial, you learned:

✓ How to solve sub-problems with different execution methods
✓ How to choose the right execution method
✓ How to critique solutions with Red Teams
✓ How to verify solutions with Gold Teams
✓ How to assemble final solution

---

## Next Steps

**Next Tutorial:** [Tutorial 5: Advanced Features](tutorial_05_advanced.md)

---

**Tutorial Version:** 1.0.0
**Last Updated:** 2025-01-03
