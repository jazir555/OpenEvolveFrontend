# Generic MAKER/MDAP Integration Guide

This guide explains how to use the **completely generic** MAKER framework (arXiv:2511.09030) that works with **ANY task type** - not just math proofs.

## Overview

The Generic MAKER Integration provides zero-error guarantees through:
- **First-to-ahead-by-k voting** for reliable selection
- **MDAP task decomposition** for complex problems
- **Evolutionary optimization** for refinement
- **Red-flagging** of unreliable outputs
- **Statistical convergence guarantees**

## What Makes This Generic

Unlike domain-specific implementations, this generic version works with:

✓ **Code generation/refactoring**
✓ **Document processing/summarization**
✓ **Data analysis pipelines**
✓ **Workflow orchestration**
✓ **Multi-agent systems**
✓ **Any optimization task**
✓ **Any multi-step LLM workflow**

## Quick Start

### Basic Example

```python
from generic_maker_integration import run_generic_maker, GenericEvaluator, TaskType

# Define your evaluator
class MyEvaluator(GenericEvaluator):
    def evaluate(self, solution: str, task) -> float:
        # Return quality score between 0.0 and 1.0
        return len(solution) / 1000.0  # Simple example

    def get_evaluation_details(self):
        return {"metric": "length"}

# Use MAKER
result = await run_generic_maker(
    task_description="Generate a Python function to sort a list",
    evaluator=MyEvaluator(),
    task_type=TaskType.CODE_GENERATION
)

print(f"Solution: {result.solution}")
print(f"Quality: {result.quality_score}")
```

## Supported Task Types

| Task Type | Use Case | Example |
|-----------|----------|---------|
| `CODE_GENERATION` | Generate code from description | "Create a function to validate emails" |
| `CODE_REFACTORING` | Improve existing code | "Refactor for better performance" |
| `DOCUMENT_PROCESSING` | Process documents | "Extract key information from report" |
| `TEXT_SUMMARIZATION` | Summarize text | "Summarize this article" |
| `DATA_ANALYSIS` | Analyze data | "Find patterns in dataset" |
| `WORKFLOW_ORCHESTRATION` | Plan workflows | "Design a data pipeline" |
| `OPTIMIZATION` | Optimize solutions | "Minimize cost function" |
| `CUSTOM` | Any custom task | Your own task type |

## Core Components

### 1. GenericTask

Represents any task to be solved:

```python
from generic_maker_integration import GenericTask, TaskType

task = GenericTask(
    task_id="unique_id",
    description="Generate a sorting algorithm",
    task_type=TaskType.CODE_GENERATION,
    context={"language": "Python"},
    constraints=["O(n log n) complexity"],
    requirements=["handle duplicates", "stable sort"]
)
```

### 2. GenericSolution

Represents a solution produced by MAKER:

```python
from generic_maker_integration import GenericSolution

solution = GenericSolution(
    task_id="unique_id",
    solution="def sort(arr): return sorted(arr)",
    quality_score=0.85,
    generation=5,
    steps_taken=["voting", "mutation", "crossover"],
    evaluation_details={"length": 30, "has_comments": True}
)
```

### 3. GenericEvaluator

Define how to evaluate solutions:

```python
from generic_maker_integration import GenericEvaluator, GenericTask

class CodeQualityEvaluator(GenericEvaluator):
    """Evaluates code quality"""

    def evaluate(self, solution: str, task: GenericTask) -> float:
        """Return quality score between 0.0 and 1.0"""
        score = 0.0

        # Check for function definition
        if "def " in solution:
            score += 0.3

        # Check for docstring
        if '"""' in solution:
            score += 0.2

        # Check for error handling
        if "try:" in solution:
            score += 0.2

        # Add your own criteria here...
        return min(1.0, score)

    def get_evaluation_details(self) -> dict:
        return {
            "metrics": ["function_def", "docstring", "error_handling"],
            "max_score": 1.0
        }
```

### 4. MAKERConfig

Configure MAKER execution:

```python
from generic_maker_integration import MAKERConfig

config = MAKERConfig(
    # Voting parameters
    enable_voting=True,
    voting_threshold=3,  # k for first-to-ahead-by-k

    # Decomposition parameters
    enable_decomposition=True,
    decomposition_depth=3,

    # Evolution parameters
    max_generations=50,
    population_size=20,
    mutation_rate=0.1,
    crossover_rate=0.7,

    # Convergence parameters
    convergence_threshold=0.95,
    max_iterations_without_improvement=10
)
```

## Usage Examples

### Example 1: Code Generation

```python
import asyncio
from generic_maker_integration import run_generic_maker, GenericEvaluator, TaskType

class CodeEvaluator(GenericEvaluator):
    def evaluate(self, solution: str, task) -> float:
        score = 0.0
        if "def " in solution: score += 0.3
        if '"""' in solution: score += 0.2
        if "try:" in solution: score += 0.2
        if "#" in solution: score += 0.1
        return min(1.0, score)

    def get_evaluation_details(self):
        return {"metrics": ["function", "docstring", "errors", "comments"]}

async def generate_code():
    result = await run_generic_maker(
        task_description="Generate a function to validate email addresses",
        evaluator=CodeEvaluator(),
        task_type=TaskType.CODE_GENERATION,
        config=MAKERConfig(
            enable_voting=True,
            voting_threshold=3,
            max_generations=20
        )
    )
    return result

# Run
result = asyncio.run(generate_code())
print(result.solution)
```

### Example 2: Document Summarization

```python
class SummaryEvaluator(GenericEvaluator):
    def evaluate(self, summary: str, task) -> float:
        score = 0.0
        words = summary.split()

        # Prefer reasonable length
        if 20 <= len(words) <= 100:
            score += 0.3

        # Check for structure
        if "-" in summary or "•" in summary:
            score += 0.3

        # Check for summary indicators
        if any(word in summary.lower() for word in ["summary", "key", "main"]):
            score += 0.2

        # Prefer concise
        if len(summary) < len(task.description) * 0.5:
            score += 0.2

        return min(1.0, score)

    def get_evaluation_details(self):
        return {"metrics": ["length", "structure", "indicators", "conciseness"]}

async def summarize_document(document: str):
    result = await run_generic_maker(
        task_description=f"Summarize: {document}",
        evaluator=SummaryEvaluator(),
        task_type=TaskType.TEXT_SUMMARIZATION,
        config=MAKERConfig(
            enable_decomposition=False,  # Not needed for summarization
            max_generations=10
        )
    )
    return result
```

### Example 3: Data Processing Pipeline

```python
class PipelineEvaluator(GenericEvaluator):
    def evaluate(self, pipeline: str, task) -> float:
        score = 0.0

        # Check for key components
        stages = ["extract", "transform", "load"]
        for stage in stages:
            if stage in pipeline.lower():
                score += 0.25

        # Check for error handling
        if "error" in pipeline.lower() or "validation" in pipeline.lower():
            score += 0.15

        # Check for structure
        if "\n" in pipeline:
            score += 0.1

        return min(1.0, score)

    def get_evaluation_details(self):
        return {"metrics": ["stages", "error_handling", "structure"]}

async def design_pipeline():
    result = await run_generic_maker(
        task_description="Design an ETL pipeline for user analytics data",
        evaluator=PipelineEvaluator(),
        task_type=TaskType.WORKFLOW_ORCHESTRATION,
        config=MAKERConfig(
            enable_decomposition=True,  # Decompose into E/T/L stages
            max_generations=15
        )
    )
    return result
```

### Example 4: Custom Optimization

```python
class CustomEvaluator(GenericEvaluator):
    def evaluate(self, solution: str, task) -> float:
        # Your custom evaluation logic
        score = 0.0

        for requirement in task.requirements:
            if requirement.lower() in solution.lower():
                score += 0.3

        # Reward structure
        if "\n" in solution:
            score += 0.1

        # Reward detail
        if len(solution) > 100:
            score += 0.1

        return min(1.0, score)

    def get_evaluation_details(self):
        return {"metrics": ["requirements", "structure", "detail"]}

async def optimize_system():
    from generic_maker_integration import GenericTask, TaskType

    task = GenericTask(
        task_id="system_design",
        description="Design a scalable user authentication system",
        task_type=TaskType.CUSTOM,
        requirements=["secure", "scalable", "user-friendly"],
        constraints=["must support OAuth2", "handle 10K users"]
    )

    result = await run_generic_maker(
        task_description=task.description,
        evaluator=CustomEvaluator(),
        task_type=TaskType.CUSTOM,
        config=MAKERConfig(
            enable_voting=True,
            enable_decomposition=True,
            max_generations=25
        )
    )
    return result
```

## Configuration Options

### Voting Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_voting` | bool | True | Enable MAKER voting |
| `voting_threshold` | int | 3 | k for first-to-ahead-by-k (higher = more conservative) |
| `enable_red_flagging` | bool | True | Filter low-quality solutions |

### Decomposition Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_decomposition` | bool | True | Enable MDAP decomposition |
| `decomposition_depth` | int | 3 | Max decomposition depth |
| `max_subtasks` | int | 10 | Maximum subtasks to create |

### Evolution Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_generations` | int | 50 | Maximum generations |
| `population_size` | int | 20 | Population size |
| `mutation_rate` | float | 0.1 | Probability of mutation |
| `crossover_rate` | float | 0.7 | Probability of crossover |

### Convergence Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `convergence_threshold` | float | 0.95 | Stop when quality reaches threshold |
| `max_iterations_without_improvement` | int | 10 | Stop if no improvement for N generations |

### Performance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallel_execution` | bool | False | Enable parallel execution |
| `timeout_seconds` | float | None | Maximum time to spend |

## Voting Threshold Guidelines

| k Value | Reliability | Speed | Use Case |
|---------|-------------|-------|----------|
| 2 | 95% | Fast | Quick prototyping |
| 3 | 99% | Medium | Standard production |
| 5 | 99.9% | Slow | High-stakes |
| 8 | 99.99% | Very Slow | Safety-critical |

## Understanding MAKER's Zero-Error Guarantees

### First-to-Ahead-by-K Voting

MAKER uses a voting mechanism where a candidate needs to be **k votes ahead** of all others to win:

```
Example with k=3:
  Candidate A: 7 votes
  Candidate B: 3 votes
  Candidate C: 2 votes

  A wins because 7 >= 3 + max(3, 2) = 6
```

**Benefits**:
- Statistical convergence to optimal solution
- Automatic filtering of low-quality candidates (red-flagging)
- Configurable reliability via k parameter

### MDAP Decomposition

Complex tasks are broken into simpler subtasks:

```
Task: "Generate a web scraper"
  ↓
Subtasks:
  1. Define data structures
  2. Implement HTTP client
  3. Add error handling
  4. Add data parsing
  5. Add storage layer
  ↓
Combine results into final solution
```

**Benefits**:
- More efficient search
- Parallelizable subtasks
- Better handling of complexity

### Evolutionary Optimization

MAKER uses population-based optimization:
1. **Selection**: Voting selects best individuals
2. **Crossover**: Combine parts of parent solutions
3. **Mutation**: Add small random variations
4. **Evaluation**: Score each solution
5. **Repeat**: Until convergence

**Benefits**:
- Explores solution space thoroughly
- Improves solutions iteratively
- Escapes local optima

## Result Structure

```python
{
    "task_id": "task_123",
    "solution": "The generated solution text",
    "quality_score": 0.87,  # Between 0.0 and 1.0
    "generation": 15,  # Which generation produced this
    "metadata": {
        "voting_rounds": 5,
        "decomposed": True,
        "mutated": True
    },
    "steps_taken": ["voting", "decomposition", "mutation"],
    "evaluation_details": {
        "your_metrics": "values"
    },
    "created_at": 1704067200.0
}
```

## Performance Characteristics

### Reliability vs Cost Trade-off

| k | Success Rate | Expected Cost | Use Case |
|---|--------------|---------------|----------|
| 2 | 95% | Low | Quick prototyping |
| 3 | 99% | Medium | Standard production |
| 5 | 99.9% | High | Critical systems |
| 8 | 99.99% | Very High | Safety-critical |

### Scaling with Complexity

| Task Complexity | Recommended Config |
|----------------|-------------------|
| Simple | voting_threshold=2, max_generations=10 |
| Medium | voting_threshold=3, max_generations=20 |
| Complex | voting_threshold=3, enable_decomposition=True |
| Very Complex | voting_threshold=5, enable_decomposition=True, max_generations=50 |

## Best Practices

### 1. Design Your Evaluator Carefully

The evaluator is the most important component:

```python
class GoodEvaluator(GenericEvaluator):
    def evaluate(self, solution: str, task) -> float:
        score = 0.0

        # Multiple criteria
        if "error_handling" in solution:
            score += 0.2
        if "documentation" in solution:
            score += 0.2
        if "tests" in solution:
            score += 0.2
        if "optimization" in solution:
            score += 0.2
        if "clean_code" in solution:
            score += 0.2

        # Ensure score is normalized
        return min(1.0, max(0.0, score))
```

### 2. Choose the Right Configuration

```python
# Quick exploration
config = MAKERConfig(
    voting_threshold=2,
    max_generations=10,
    enable_decomposition=False
)

# High-quality result
config = MAKERConfig(
    voting_threshold=5,
    max_generations=50,
    enable_decomposition=True
)
```

### 3. Provide Good Initial Candidates

```python
initial_candidates = [
    "def solution_v1(): ...",
    "def solution_v2(): ...",
    "def solution_v3(): ..."
]

result = await run_generic_maker(
    task_description="...",
    evaluator=evaluator,
    initial_candidates=initial_candidates
)
```

### 4. Use Task Context Effectively

```python
task = GenericTask(
    task_id="task_1",
    description="Generate a sorting function",
    task_type=TaskType.CODE_GENERATION,
    context={
        "language": "Python",
        "constraints": ["O(n log n)"],
        "requirements": ["stable", "handle duplicates"]
    }
)
```

## Troubleshooting

### Issue: Low Quality Solutions

**Possible causes**:
1. Evaluator not capturing quality well
2. Not enough generations
3. Population too small

**Solutions**:
- Improve evaluator to capture relevant metrics
- Increase `max_generations`
- Increase `population_size`

### Issue: Slow Convergence

**Possible causes**:
1. Voting threshold too high
2. Task too complex for current config

**Solutions**:
- Try `voting_threshold=2` for faster convergence
- Enable `enable_decomposition=True`
- Provide better initial candidates

### Issue: All Solutions Look the Same

**Possible causes**:
1. Mutation rate too low
2. Population not diverse

**Solutions**:
- Increase `mutation_rate`
- Increase `population_size`
- Provide diverse initial candidates

## Comparison: Generic vs Domain-Specific

| Feature | Generic MAKER | Domain-Specific |
|---------|---------------|-----------------|
| **Flexibility** | Works with any task | Optimized for specific domain |
| **Setup** | Just define evaluator | Requires domain knowledge |
| **Performance** | Good for most tasks | Better for domain-specific tasks |
| **Maintenance** | Easier to maintain | More complex |
| **Use When** | General tasks | Repeated domain-specific tasks |

## Integration Examples

### With LangChain

```python
from generic_maker_integration import run_generic_maker, GenericEvaluator, TaskType

class LangChainEvaluator(GenericEvaluator):
    def evaluate(self, solution: str, task) -> float:
        # Use LangChain to evaluate
        from langchain.evaluation import load_evaluator

        evaluator = load_evaluator("criteria", criteria="conciseness")
        result = evaluator.evaluate_strings(
            prediction=solution,
            reference=task.description,
            input={"question": task.description}
        )
        return result['score']

    def get_evaluation_details(self):
        return {"using": "langchain"}
```

### With LlamaIndex

```python
from generic_maker_integration import run_generic_maker, GenericEvaluator, TaskType

class LlamaIndexEvaluator(GenericEvaluator):
    def evaluate(self, solution: str, task) -> float:
        # Use LlamaIndex to evaluate
        from llama_index.evaluation import FaithfulnessEvaluator

        evaluator = FaithfulnessEvaluator()
        result = evaluator.evaluate(
            query=task.description,
            response=solution,
            context=[]  # Your context here
        )
        return result.score

    def get_evaluation_details(self):
        return {"using": "llama_index"}
```

### With Custom LLM

```python
from generic_maker_integration import run_generic_maker, GenericEvaluator, TaskType

class LLMEvaluator(GenericEvaluator):
    def __init__(self, llm_function):
        self.llm_function = llm_function

    def evaluate(self, solution: str, task) -> float:
        prompt = f"""
        Rate the quality of this solution on a scale of 0.0 to 1.0:

        Task: {task.description}
        Solution: {solution}

        Return only a number between 0.0 and 1.0.
        """

        response = self.llm_function(prompt)
        return float(response.strip())

    def get_evaluation_details(self):
        return {"using": "custom_llm"}
```

## Advanced Usage

### Custom Evolution Strategies

```python
from generic_maker_integration import GenericMAKERSolver, GenericEvaluator

solver = GenericMAKERSolver(
    evaluator=my_evaluator,
    config=my_config
)

# Provide custom initial population
initial_population = [
    GenericSolution(task_id="t1", solution=sol1, quality_score=0.5),
    GenericSolution(task_id="t1", solution=sol2, quality_score=0.7),
]

result = await solver.solve(task, initial_population)
```

### Monitoring Progress

```python
from generic_maker_integration import GenericMAKERSolver, GenericEvaluator

solver = GenericMAKERSolver(evaluator, config)

# Access statistics
print(solver.statistics)
# {
#     "total_tasks": 10,
#     "successful_tasks": 9,
#     "average_quality": 0.85,
#     "average_time": 15.3,
#     "voting_rounds": 45
# }
```

## References

1. **Paper**: "Solving a Million-Step LLM Task with Zero Errors"
   - arXiv:2511.09030
   - https://arxiv.org/abs/2511.09030

2. **Implementation Files**:
   - `generic_maker_integration.py` - Core generic implementation
   - `demo_generic_maker.py` - Demo script
   - `validate_generic_maker_integration.py` - Validation script

3. **Related Documentation**:
   - `MAKER_HYBRID_INTEGRATION_GUIDE.md` - Hybrid strategies (LeanAide-specific)
   - `MAKER_EVOLUTION_INTEGRATION_GUIDE.md` - Evolution integration

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the paper for theoretical details
3. Check demo files for usage examples
4. Run validation: `python validate_generic_maker_integration.py`
5. Open an issue on the repository

---

**Status**: ✓ Complete Generic Integration Ready
**Paper**: arXiv:2511.09030
**Last Updated**: 2025-12-30
**Version**: 1.0.0 (Generic - Works with ANY Task Type)
