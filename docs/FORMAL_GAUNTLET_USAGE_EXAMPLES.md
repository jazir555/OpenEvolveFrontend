# Formal Gauntlet System - Usage Examples

This document provides practical examples for using the enhanced formal gauntlet system features.

---

## Table of Contents
1. [Parallel Execution](#parallel-execution)
2. [Adaptive Difficulty](#adaptive-difficulty)
3. [Human Review Queue](#human-review-queue)
4. [Complete Workflows](#complete-workflows)

---

## Parallel Execution

### Basic Setup

```python
from formal_gauntlet_system import GauntletSystem, GauntletTemplates
from sovereign_data_models import SolutionAttempt, SubProblem

# Initialize with 8 parallel workers
gauntlet_system = GauntletSystem(
    max_parallel_workers=8,
    enable_adaptive=False  # Disable adaptive for pure parallel
)

# Use a template or create custom gauntlet
gauntlet = GauntletTemplates.standard_validation_gauntlet()
gauntlet.execution_order = "parallel"  # Enable parallel execution

# Create solution and problem
solution = SolutionAttempt(
    id="sol_001",
    solution_content="def add(a, b): return a + b",
    metadata={}
)

sub_problem = SubProblem(
    id="sub_001",
    title="Implement Addition",
    description="Create a function to add two numbers"
)

# Execute gauntlet in parallel
execution = gauntlet_system.execute_gauntlet(
    gauntlet=gauntlet,
    solution=solution,
    sub_problem=sub_problem
)

print(f"Execution complete: {execution.overall_passed}")
print(f"Rounds passed: {execution.rounds_passed}/{len(gauntlet.rounds)}")
print(f"Time saved: {execution.execution_duration:.2f}s")

# Check individual round results
for result in execution.round_results:
    print(f"Round {result['round_id']}: score={result['score']:.2f}, time={result.get('execution_time', 0):.2f}s")
```

### Performance Comparison

```python
import time

# Sequential execution
gauntlet_seq = GauntletTemplates.standard_validation_gauntlet()
gauntlet_seq.execution_order = "sequential"

start = time.time()
execution_seq = gauntlet_system.execute_gauntlet(gauntlet_seq, solution, sub_problem)
seq_time = time.time() - start

# Parallel execution
gauntlet_par = GauntletTemplates.standard_validation_gauntlet()
gauntlet_par.execution_order = "parallel"

start = time.time()
execution_par = gauntlet_system.execute_gauntlet(gauntlet_par, solution, sub_problem)
par_time = time.time() - start

print(f"Sequential: {seq_time:.2f}s")
print(f"Parallel: {par_time:.2f}s")
print(f"Speedup: {seq_time/par_time:.2f}x")
```

---

## Adaptive Difficulty

### Basic Adaptive Execution

```python
# Initialize with adaptive mode enabled
gauntlet_system = GauntletSystem(
    enable_adaptive=True  # Enable adaptive difficulty
)

# Create gauntlet with adaptive execution
gauntlet = GauntletTemplates.standard_validation_gauntlet()
gauntlet.execution_order = "adaptive"

# Execute with adaptive difficulty
execution = gauntlet_system.execute_gauntlet(
    gauntlet=gauntlet,
    solution=solution,
    sub_problem=sub_problem
)

# Check adaptive metrics
metrics = gauntlet_system.adaptive_metrics
print(f"Difficulty multiplier: {metrics.current_difficulty_multiplier:.2f}")
print(f"Total adjustments: {metrics.difficulty_adjustments}")
print(f"Pass rate: {metrics.total_rounds_passed}/{metrics.total_rounds_completed}")
print(f"Recent scores: {metrics.recent_scores}")
print(f"Failure categories: {metrics.failure_categories}")
```

### Training Workflow (Easy to Hard)

```python
# Create a custom gauntlet starting easy
easy_round = GauntletRoundRule(
    rule_id="easy_validation",
    rule_type="automated",
    description="Easy validation round",
    min_score=0.5,  # Low threshold
    max_attempts=5,  # Many attempts
    can_fail_gracefully=True
)

gauntlet = GauntletDefinition(
    gauntlet_id="training_gauntlet",
    name="Training Gauntlet",
    rounds=[easy_round],
    execution_order="adaptive"
)

# Start with difficulty multiplier at 0.5 (very easy)
gauntlet_system.adaptive_metrics.current_difficulty_multiplier = 0.5

# Execute - system will increase difficulty if performing well
execution = gauntlet_system.execute_gauntlet(gauntlet, solution, sub_problem)

# If initial score > 0.9, difficulty will increase
print(f"Final difficulty: {gauntlet_system.adaptive_metrics.current_difficulty_multiplier:.2f}")
```

### Strict Mode (Hard Initial Difficulty)

```python
# Create strict gauntlet
strict_round = GauntletRoundRule(
    rule_id="strict_validation",
    rule_type="red_team",
    description="Strict adversarial review",
    min_score=0.9,  # High threshold
    max_attempts=2,  # Few attempts
    can_fail_gracefully=False
)

gauntlet = GauntletDefinition(
    gauntlet_id="strict_gauntlet",
    name="Strict Production Gauntlet",
    rounds=[strict_round],
    execution_order="adaptive"
)

# Start with high difficulty
gauntlet_system.adaptive_metrics.current_difficulty_multiplier = 1.5

# Execute - system will decrease difficulty if struggling
execution = gauntlet_system.execute_gauntlet(gauntlet, solution, sub_problem)

# If score < 0.6, difficulty will decrease with guidance
print(f"Final difficulty: {gauntlet_system.adaptive_metrics.current_difficulty_multiplier:.2f}")
```

---

## Human Review Queue

### Non-Blocking Human Review (Asynchronous)

```python
# Create human review round
human_round = GauntletRoundRule(
    rule_id="manual_security_review",
    rule_type="human",
    description="Manual security review required",
    min_score=0.0,  # N/A for human review
    max_attempts=1,
    evaluator="security-team@company.com",
    metadata={
        "wait_for_human_review": False  # Non-blocking
    }
)

gauntlet = GauntletDefinition(
    gauntlet_id="security_gauntlet",
    name="Security Review with Human",
    rounds=[human_round],
    execution_order="sequential"
)

# Execute gauntlet
execution = gauntlet_system.execute_gauntlet(gauntlet, solution, sub_problem)

# Check result - will be pending
human_result = execution.round_results[-1]
print(f"Status: {human_result['status']}")
print(f"Review ID: {human_result['review_id']}")

# Continue with other work...

# Later, assign the review
review_id = human_result['review_id']
assign_result = gauntlet_system.assign_human_review(
    review_id=review_id,
    reviewer="john.doe@company.com"
)
print(f"Assigned: {assign_result['success']}")

# Even later, complete the review
complete_result = gauntlet_system.complete_human_review(
    review_id=review_id,
    approved=True,
    feedback="Security review passed. No vulnerabilities found.",
    score=1.0
)
print(f"Completed: {complete_result['success']}")

# Check final status
status = gauntlet_system.get_review_status(review_id)
print(f"Final status: {status['status']}")
print(f"Feedback: {status['feedback']}")
```

### Blocking Human Review (Synchronous)

```python
# Create human review round with blocking enabled
human_round = GauntletRoundRule(
    rule_id="manual_approval",
    rule_type="human",
    description="Manual approval required before proceeding",
    evaluator="manager@company.com",
    metadata={
        "wait_for_human_review": True,  # Blocking mode
        "review_timeout_seconds": 600   # 10 minute timeout
    }
)

gauntlet = GauntletDefinition(
    gauntlet_id="approval_gauntlet",
    name="Approval Gauntlet",
    rounds=[human_round],
    execution_order="sequential"
)

# This will BLOCK until human completes review or timeout
# In another terminal/process, you would call:
# gauntlet_system.complete_human_review(review_id, approved=True, feedback="Approved", score=1.0)

execution = gauntlet_system.execute_gauntlet(gauntlet, solution, sub_problem)

# Result contains human's decision
human_result = execution.round_results[-1]
if human_result['status'] == 'approved':
    print("Approved! Proceeding...")
elif human_result['status'] == 'rejected':
    print(f"Rejected: {human_result['feedback']}")
elif human_result['status'] == 'timeout':
    print("Review timed out")
```

### Review Queue Management

```python
# Get all pending reviews
pending = gauntlet_system.get_pending_reviews()
print(f"Pending reviews: {len(pending)}")

for review in pending:
    print(f"  - {review['review_id']}: {review['round_type']} for solution {review['solution_id']}")

# Get reviewer workload
workload = gauntlet_system.review_queue.get_reviewer_workload("john.doe@company.com")
print(f"John's workload: {workload} reviews")

# Batch assign reviews
reviewers = ["alice@company.com", "bob@company.com", "charlie@company.com"]
for i, review in enumerate(pending):
    reviewer = reviewers[i % len(reviewers)]
    gauntlet_system.assign_human_review(review['review_id'], reviewer)
    print(f"Assigned {review['review_id']} to {reviewer}")

# Check status of specific review
review_status = gauntlet_system.get_review_status(pending[0]['review_id'])
if review_status:
    print(f"Status: {review_status['status']}")
    print(f"Assigned to: {review_status['assigned_to']}")
    print(f"Created: {review_status['created_at']}")
```

---

## Complete Workflows

### CI/CD Pipeline Integration

```python
def run_ci_gauntlet(solution_content: str) -> bool:
    """Run gauntlet in CI/CD pipeline with parallel execution."""

    # Initialize
    gauntlet_system = GauntletSystem(
        max_parallel_workers=4,
        enable_adaptive=False
    )

    # Create parallel gauntlet for fast CI
    gauntlet = GauntletTemplates.standard_validation_gauntlet()
    gauntlet.execution_order = "parallel"

    # Create solution
    solution = SolutionAttempt(
        id="ci_solution",
        solution_content=solution_content,
        metadata={"ci_run": True}
    )

    sub_problem = SubProblem(
        id="ci_check",
        title="CI/CD Validation",
        description="Automated CI/CD validation checks"
    )

    # Execute
    execution = gauntlet_system.execute_gauntlet(gauntlet, solution, sub_problem)

    # Return result
    if execution.overall_passed:
        print("✅ CI/CD gauntlet passed")
        return True
    else:
        print("❌ CI/CD gauntlet failed")
        for result in execution.round_results:
            if not result['passed']:
                print(f"  - Failed: {result['round_id']}: {result['feedback']}")
        return False

# Use in CI/CD script
if __name__ == "__main__":
    import sys
    with open(sys.argv[1], 'r') as f:
        solution_code = f.read()

    success = run_ci_gauntlet(solution_code)
    sys.exit(0 if success else 1)
```

### Multi-Stage Validation Workflow

```python
def multi_stage_validation(solution: SolutionAttempt, sub_problem: SubProblem):
    """Run multi-stage validation with adaptive difficulty."""

    gauntlet_system = GauntletSystem(
        max_parallel_workers=4,
        enable_adaptive=True
    )

    # Stage 1: Quick automated checks (parallel)
    print("Stage 1: Quick automated checks...")
    quick_gauntlet = GauntletDefinition(
        gauntlet_id="quick_checks",
        name="Quick Checks",
        rounds=[
            GauntletRoundRule(
                rule_id="syntax",
                rule_type="automated",
                description="Syntax check",
                min_score=0.8,
                max_attempts=3
            ),
            GauntletRoundRule(
                rule_id="linting",
                rule_type="automated",
                description="Code linting",
                min_score=0.8,
                max_attempts=3
            )
        ],
        execution_order="parallel"
    )

    stage1 = gauntlet_system.execute_gauntlet(quick_gauntlet, solution, sub_problem)
    if not stage1.overall_passed:
        print("❌ Stage 1 failed")
        return False

    print("✅ Stage 1 passed")

    # Stage 2: Adaptive validation (adjusts based on stage 1 performance)
    print("Stage 2: Adaptive validation...")
    adaptive_gauntlet = GauntletTemplates.standard_validation_gauntlet()
    adaptive_gauntlet.execution_order = "adaptive"

    stage2 = gauntlet_system.execute_gauntlet(adaptive_gauntlet, solution, sub_problem)

    # Show adaptation
    metrics = gauntlet_system.adaptive_metrics
    print(f"Difficulty adjusted {metrics.difficulty_adjustments} times")
    print(f"Final multiplier: {metrics.current_difficulty_multiplier:.2f}")

    if not stage2.overall_passed:
        print("❌ Stage 2 failed")
        return False

    print("✅ Stage 2 passed")

    # Stage 3: Human review for critical solutions
    print("Stage 3: Human review...")
    if solution.metadata.get("critical", False):
        human_round = GauntletRoundRule(
            rule_id="critical_review",
            rule_type="human",
            description="Critical solution requires human approval",
            evaluator="senior-engineer@company.com",
            metadata={"wait_for_human_review": False}
        )

        human_gauntlet = GauntletDefinition(
            gauntlet_id="critical_approval",
            name="Critical Approval",
            rounds=[human_round],
            execution_order="sequential"
        )

        stage3 = gauntlet_system.execute_gauntlet(human_gauntlet, solution, sub_problem)

        review_id = stage3.round_results[0]['review_id']
        print(f"⏳ Critical review queued: {review_id}")
        print("Waiting for human approval...")

        # In production, you'd poll or use webhooks here
        # For now, return pending status

        return "pending_human_review"
    else:
        print("✅ All stages passed (non-critical, no human review required)")
        return True

# Usage
solution = SolutionAttempt(
    id="prod_solution",
    solution_content="...",
    metadata={"critical": True}
)

result = multi_stage_validation(solution, sub_problem)
if result == True:
    print("✅ Solution fully validated")
elif result == "pending_human_review":
    print("⏳ Awaiting human review")
else:
    print("❌ Validation failed")
```

### Learning/Training Mode

```python
def training_mode_session(solutions: List[SolutionAttempt]):
    """Run adaptive training session."""

    # Start with low difficulty
    gauntlet_system = GauntletSystem(
        enable_adaptive=True
    )
    gauntlet_system.adaptive_metrics.current_difficulty_multiplier = 0.5

    results = []

    for i, solution in enumerate(solutions):
        print(f"\n=== Solution {i+1}/{len(solutions)} ===")

        # Create adaptive training gauntlet
        gauntlet = GauntletDefinition(
            gauntlet_id="training",
            name=f"Training Round {i+1}",
            rounds=[
                GauntletRoundRule(
                    rule_id=f"training_check_{i}",
                    rule_type="automated",
                    description=f"Training check {i+1}",
                    min_score=0.6 * gauntlet_system.adaptive_metrics.current_difficulty_multiplier,
                    max_attempts=3,
                    can_fail_gracefully=True
                )
            ],
            execution_order="adaptive"
        )

        sub_problem = SubProblem(
            id=f"train_{i}",
            title=f"Training Problem {i+1}",
            description="Training exercise"
        )

        execution = gauntlet_system.execute_gauntlet(gauntlet, solution, sub_problem)

        # Show progress
        metrics = gauntlet_system.adaptive_metrics
        print(f"Score: {execution.final_score:.2f}")
        print(f"Difficulty: {metrics.current_difficulty_multiplier:.2f}")
        print(f"Pass rate: {metrics.total_rounds_passed}/{metrics.total_rounds_completed}")

        results.append({
            "solution_id": solution.id,
            "passed": execution.overall_passed,
            "score": execution.final_score,
            "difficulty": metrics.current_difficulty_multiplier
        })

    # Show training summary
    print("\n=== Training Summary ===")
    print(f"Total solutions: {len(solutions)}")
    print(f"Passed: {sum(1 for r in results if r['passed'])}")
    print(f"Final difficulty: {gauntlet_system.adaptive_metrics.current_difficulty_multiplier:.2f}")
    print(f"Adjustments made: {gauntlet_system.adaptive_metrics.difficulty_adjustments}")

    return results
```

---

## Best Practices

### 1. Parallel Execution
- Use for independent validation rounds
- Set workers based on CPU cores (usually 4-8)
- Not suitable for rounds with dependencies
- Monitor time savings vs overhead

### 2. Adaptive Difficulty
- Enable for learning/training scenarios
- Start with appropriate base difficulty
- Monitor difficulty adjustments
- Reset metrics between sessions if needed

### 3. Human Review
- Use non-blocking mode for async workflows
- Use blocking mode only when truly necessary
- Set appropriate timeouts
- Implement notification system for production

### 4. Configuration
```python
# Development: Fast, lenient
dev_system = GauntletSystem(
    max_parallel_workers=8,
    enable_adaptive=False
)

# Production: Balanced, adaptive
prod_system = GauntletSystem(
    max_parallel_workers=4,
    enable_adaptive=True
)

# Training: Sequential, adaptive
training_system = GauntletSystem(
    max_parallel_workers=1,
    enable_adaptive=True
)
```

---

## Troubleshooting

### Issue: Parallel execution not faster
**Cause:** Round execution times are similar, or overhead is too high
**Solution:** Use sequential for <3 rounds, or check if rounds are I/O bound

### Issue: Adaptive difficulty keeps increasing
**Cause:** Solution consistently scores high
**Solution:** This is expected behavior - solution is very good

### Issue: Human review always times out
**Cause:** No one completing the review, or timeout too short
**Solution:** Increase timeout, or use non-blocking mode

### Issue: Review queue grows indefinitely
**Cause:** Not processing reviews, or not completing them
**Solution:** Implement review assignment/completion workflow

---

*Generated: 2025-01-22*
*Related: FORMAL_GAUNTLET_ENHANCEMENT_REPORT.md*
