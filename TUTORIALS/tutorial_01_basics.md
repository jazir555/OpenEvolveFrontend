# Tutorial 1: Getting Started with Decomposition

**Level:** Beginner
**Time:** 30 minutes
**Prerequisites:** Python 3.8+, basic Python knowledge

---

## Learning Objectives

After this tutorial, you will be able to:
- Understand what problem decomposition is
- Set up the decomposition engine
- Decompose a simple problem
- Analyze decomposition results

---

## What is Problem Decomposition?

Problem decomposition is the process of breaking down a complex problem into smaller, more manageable sub-problems. Each sub-problem:
- Has a clear, specific scope
- Can be solved independently
- Contributes to solving the overall problem
- Has well-defined success criteria

### Why Decompose?

1. **Manageability**: Small problems are easier to solve
2. **Parallelization**: Independent sub-problems can be solved in parallel
3. **Quality**: Focused attention on specific aspects improves quality
4. **Tracking**: Progress is easier to measure with smaller milestones
5. **Risk Mitigation**: Identify and address challenges early

---

## Installation

### Step 1: Install Dependencies

```bash
# Install OpenEvolve
pip install openevolve-client

# Install required packages
pip install pydantic dataclasses-typing python-dotenv

# Optional: Install integration dependencies
pip install claudiomiro  # For autonomous development
pip install datapizza    # For multi-agent solving
pip install roma-dspy    # For recursive decomposition
```

### Step 2: Set Up API Keys

Create a `.env` file in your project root:

```bash
# OpenAI (recommended)
OPENAI_API_KEY=sk-your-openai-key-here

# Or Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Or Google Gemini
GOOGLE_API_KEY=your-google-key-here
```

### Step 3: Verify Installation

```python
# test_installation.py
import os
from dotenv import load_dotenv

load_dotenv()

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ API key not found. Please set OPENAI_API_KEY in .env")
else:
    print("✓ API key found")

# Check decomposition engine
try:
    from decomposition_engine import DecompositionEngine
    print("✓ Decomposition engine imported successfully")
except ImportError as e:
    print(f"❌ Failed to import decomposition engine: {e}")

# Check OpenEvolve client
try:
    from openevolve_client import OpenEvolveClient
    print("✓ OpenEvolve client imported successfully")
except ImportError as e:
    print(f"❌ Failed to import OpenEvolve client: {e}")
```

Run the test:
```bash
python test_installation.py
```

---

## Your First Decomposition

### Example: Simple Software Project

Let's decompose a common software task: "Build a To-Do List Application"

#### Step 1: Define the Problem

```python
# example_01_basic_decomposition.py
from sovereign_data_models import (
    ProblemDefinition,
    ProblemType,
    DomainContext,
    ComplexityScore
)

# Create problem definition
problem = ProblemDefinition(
    id="todo-app-001",
    title="To-Do List Application",
    description="""Build a web-based to-do list application with the following features:
    - Add, edit, and delete tasks
    - Mark tasks as complete
    - Organize tasks into categories
    - Set due dates and reminders
    - User authentication
    - Responsive design for mobile and desktop
    """,
    problem_type=ProblemType.IMPLEMENTATION,
    domain_context=DomainContext(
        domain="Software Development",
        subdomain="Web Application"
    ),
    complexity_score=ComplexityScore(
        overall_complexity=6,
        cognitive_complexity=5,
        computational_complexity=4,
        domain_complexity=5,
        integration_complexity=7
    )
)

print(f"Problem: {problem.title}")
print(f"Complexity: {problem.complexity_score.overall_complexity}/10")
```

#### Step 2: Analyze the Problem

```python
from problem_analyzer import ProblemAnalyzer

# Create analyzer
analyzer = ProblemAnalyzer()

# Analyze problem
analysis = analyzer.analyze_problem(problem)

# Print analysis results
print("\n=== Problem Analysis ===")
print(f"Domain: {analysis['domain']}")
print(f"Complexity Breakdown:")
print(f"  Overall: {analysis['complexity']['overall']}/10")
print(f"  Cognitive: {analysis['complexity']['cognitive']}/10")
print(f"  Computational: {analysis['complexity']['computational']}/10")
print(f"  Domain: {analysis['complexity']['domain']}/10")
print(f"  Integration: {analysis['complexity']['integration']}/10")

print(f"\nEstimated Sub-Problems: {analysis['estimated_sub_problems']}")
print(f"\nRequired Expertise:")
for exp in analysis['required_expertise']:
    print(f"  - {exp}")

print(f"\nKey Challenges:")
for challenge in analysis['key_challenges']:
    print(f"  - {challenge}")
```

**Expected Output:**
```
=== Problem Analysis ===
Domain: Software Development
Complexity Breakdown:
  Overall: 6/10
  Cognitive: 5/10
  Computational: 4/10
  Domain: 5/10
  Integration: 7/10

Estimated Sub-Problems: 7

Required Expertise:
  - Frontend Development
  - Backend Development
  - Database Design
  - API Development
  - User Authentication

Key Challenges:
  - State management for task updates
  - Synchronization across devices
  - Responsive design implementation
```

#### Step 3: Decompose the Problem

```python
from decomposition_engine import DecompositionEngine

# Create decomposition engine
engine = DecompositionEngine()

# Decompose problem
print("\n=== Decomposing Problem ===")
result = engine.decompose(problem, strategy="semantic")

# Print results
print(f"\nGenerated {len(result.sub_problems)} sub-problems:\n")

for i, sp in enumerate(result.sub_problems, 1):
    print(f"{i}. {sp.title}")
    print(f"   Type: {sp.type.value}")
    print(f"   Priority: {sp.priority}/10")
    print(f"   Effort: {sp.estimated_effort} hours")
    print(f"   Complexity: {sp.complexity_score.overall_complexity}/10")

    if sp.dependencies:
        deps = ", ".join(sp.dependencies)
        print(f"   Dependencies: {deps}")
    else:
        print(f"   Dependencies: None")

    print(f"   Description: {sp.description[:100]}...")
    print()
```

**Expected Output:**
```
Generated 7 sub-problems:

1. Database Schema Design
   Type: implementation
   Priority: 9/10
   Effort: 8 hours
   Complexity: 6/10
   Dependencies: None
   Description: Design and implement database schema for users, tasks, categories...

2. User Authentication System
   Type: implementation
   Priority: 8/10
   Effort: 16 hours
   Complexity: 7/10
   Dependencies: sub-001
   Description: Implement secure user authentication with registration, login...

3. Task CRUD Operations
   Type: implementation
   Priority: 9/10
   Effort: 12 hours
   Complexity: 5/10
   Dependencies: sub-001
   Description: Implement create, read, update, delete operations for tasks...

4. Category Management
   Type: implementation
   Priority: 6/10
   Effort: 8 hours
   Complexity: 4/10
   Dependencies: sub-001
   Description: Implement category creation, assignment, and filtering...

5. Frontend UI Development
   Type: implementation
   Priority: 8/10
   Effort: 24 hours
   Complexity: 6/10
   Dependencies: sub-002, sub-003
   Description: Build responsive user interface with task lists, forms...

6. API Endpoints
   Type: implementation
   Priority: 7/10
   Effort: 12 hours
   Complexity: 5/10
   Dependencies: sub-003
   Description: Design and implement RESTful API endpoints...

7. Testing and Validation
   Type: validation
   Priority: 7/10
   Effort: 16 hours
   Complexity: 5/10
   Dependencies: sub-004, sub-005
   Description: Comprehensive testing including unit tests, integration tests...
```

---

## Understanding the Results

### Sub-Problem Types

Each sub-problem has a type that indicates its nature:

| Type | Description | Example |
|------|-------------|---------|
| `research` | Investigation and learning | "Research available authentication libraries" |
| `analysis` | Analysis and design | "Analyze requirements for task organization" |
| `implementation` | Building and coding | "Implement user authentication" |
| `validation` | Testing and verification | "Test authentication system" |
| `integration` | Combining components | "Integrate frontend with backend API" |

### Priority Levels

Priority ranges from 1-10:
- **9-10**: Critical path, blocks other work
- **7-8**: High priority, important for success
- **5-6**: Medium priority, can be deferred slightly
- **1-4**: Low priority, can be done later

### Complexity Scores

Each sub-problem has a complexity score (1-10):
- **1-3**: Simple, straightforward
- **4-6**: Moderate, some challenges
- **7-8**: Complex, requires expertise
- **9-10**: Very complex, high risk

### Dependencies

Dependencies indicate which sub-problems must complete first:
- **No dependencies**: Can start immediately
- **Has dependencies**: Must wait for dependencies to complete

---

## Visualizing Dependencies

```python
# visualize_dependencies.py
from collections import defaultdict

def build_dependency_graph(sub_problems):
    """Build a dependency graph from sub-problems"""
    graph = defaultdict(list)
    id_to_title = {sp.id: sp.title for sp in sub_problems}

    for sp in sub_problems:
        for dep_id in sp.dependencies:
            graph[dep_id].append(sp.id)

    return graph, id_to_title

def print_execution_order(sub_problems):
    """Print suggested execution order"""
    graph, id_to_title = build_dependency_graph(sub_problems)

    # Topological sort (simplified)
    executed = set()
    order = []

    while len(executed) < len(sub_problems):
        # Find sub-problems with no unexecuted dependencies
        ready = [
            sp for sp in sub_problems
            if sp.id not in executed and
            all(d in executed for d in sp.dependencies)
        ]

        if not ready:
            print("Warning: Circular dependency detected!")
            break

        # Execute highest priority ready tasks first
        ready.sort(key=lambda x: x.priority, reverse=True)
        for sp in ready:
            order.append(sp)
            executed.add(sp.id)

    return order

# Print execution order
print("\n=== Suggested Execution Order ===")
order = print_execution_order(result.sub_problems)

for i, sp in enumerate(order, 1):
    print(f"{i}. {sp.title} (Priority: {sp.priority}/10)")
```

---

## Common Mistakes

### Mistake 1: Problem Statement Too Vague

❌ **Bad:**
```python
problem = ProblemDefinition(
    title="Build something cool",
    description="Make an app"
)
```

✓ **Good:**
```python
problem = ProblemDefinition(
    title="To-Do List Application",
    description="""Build a web-based to-do list with:
    - Add/edit/delete tasks
    - Mark as complete
    - Organize into categories
    - User authentication
    - Responsive design
    """
)
```

### Mistake 2: Ignoring Dependencies

❌ **Bad:**
```python
# Start with frontend before defining API
frontend_sp = sub_problems[4]  # Depends on API
api_sp = sub_problems[5]  # Must come first
```

✓ **Good:**
```python
# Check dependencies before starting
for sp in order:
    if all(d in executed for d in sp.dependencies):
        execute(sp)
```

### Mistake 3: Not Reviewing Sub-Problems

❌ **Bad:**
```python
# Blindly accept all sub-problems
for sp in result.sub_problems:
    execute(sp)
```

✓ **Good:**
```python
# Review each sub-problem
for sp in result.sub_problems:
    print(f"\nReview: {sp.title}")
    print(f"Description: {sp.description}")
    print(f"Priority: {sp.priority}/10 - OK?")
    print(f"Effort: {sp.estimated_effort}h - OK?")

    # Give user chance to adjust
    if input("Accept? (y/n): ").lower() != 'y':
        # Adjust or reject
        continue
```

---

## Exercise: Decompose Your Own Problem

Now it's your turn! Choose a problem you want to solve and decompose it.

### Exercise Steps:

1. **Choose a problem**: Pick something you're familiar with
   - Example: "Build a personal blog"
   - Example: "Create a data analysis pipeline"
   - Example: "Design a home automation system"

2. **Define the problem**: Create a ProblemDefinition
   - Be specific about requirements
   - Estimate complexity honestly
   - Identify the domain

3. **Analyze it**: Use ProblemAnalyzer
   - Review the complexity breakdown
   - Check required expertise
   - Identify key challenges

4. **Decompose it**: Use DecompositionEngine
   - Review generated sub-problems
   - Check dependencies
   - Verify priority ordering

5. **Reflect**: Answer these questions
   - Are the sub-problems manageable?
   - Are dependencies correct?
   - Did the decomposition miss anything?
   - Would you adjust any sub-problems?

### Solution Template:

```python
# exercise_solution.py
from sovereign_data_models import ProblemDefinition, ProblemType, DomainContext, ComplexityScore
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine

# TODO: Replace with your problem
problem = ProblemDefinition(
    id="exercise-001",
    title="Your Problem Title",
    description="Your detailed description...",
    problem_type=ProblemType.IMPLEMENTATION,
    domain_context=DomainContext(domain="Your Domain"),
    complexity_score=ComplexityScore(
        overall_complexity=5,
        cognitive_complexity=5,
        computational_complexity=5,
        domain_complexity=5,
        integration_complexity=5
    )
)

# Analyze
analyzer = ProblemAnalyzer()
analysis = analyzer.analyze_problem(problem)

# Decompose
engine = DecompositionEngine()
result = engine.decompose(problem, strategy="semantic")

# Display results
print(f"Generated {len(result.sub_problems)} sub-problems")
for sp in result.sub_problems:
    print(f"- {sp.title} ({sp.type.value}, priority {sp.priority}/10)")
```

---

## Next Steps

Congratulations! You've completed your first decomposition. In the next tutorial, you'll learn how to:

- Use different decomposition strategies
- Customize decomposition parameters
- Handle complex dependencies
- Work with different problem types

**Next Tutorial:** [Tutorial 2: Using Strategies](tutorial_02_strategies.md)

---

## Summary

In this tutorial, you learned:

✓ What problem decomposition is and why it's useful
✓ How to install and set up the decomposition engine
✓ How to define a problem
✓ How to analyze problem characteristics
✓ How to decompose a problem into sub-problems
✓ How to interpret decomposition results
✓ Common mistakes to avoid

---

## Additional Resources

- [API Reference](../OPNEEVOLVE_DECOMPOSITION_API_REFERENCE.md)
- [Example Gallery](../EXAMPLES/)
- [Troubleshooting Guide](../TROUBLESHOOTING_GUIDE.md)
- [Best Practices](../BEST_PRACTICES.md)

---

**Tutorial Version:** 1.0.0
**Last Updated:** 2025-01-03
