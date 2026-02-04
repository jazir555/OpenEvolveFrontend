# DSPy Integration Guide

## Overview

The DSPy integration provides advanced program-of-thought prompting capabilities to the Knowledge Engine. DSPy is a framework for algorithmically optimizing LM prompts and weights, enabling sophisticated reasoning chains and multi-step problem solving.

### Key Features
- Chain-of-thought reasoning
- Program-of-thought execution
- Multi-step problem decomposition
- Reasoning trace extraction
- Automatic prompt optimization (teleprompters)
- Self-consistency and ensembling

### Use Cases
- Complex problem solving requiring step-by-step reasoning
- Mathematical proof generation
- Multi-hop question answering
- Code generation with explanation
- Scientific reasoning tasks

## Installation

```bash
# Basic installation
pip install dspy-ai

# With specific LM backend
pip install dspy-ai[openai]  # For OpenAI
pip install dspy-ai[anthropic]  # For Anthropic
pip install dspy-ai[local]  # For local LLMs

# With Knowledge Engine
pip install knowledge-engine[dspy]
```

### Configuration

Set up environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
export DSPY_MODEL="gpt-4o"
```

## Quick Start

### Basic Usage

```python
from knowledge_engine.integrations import DSPyIntegration

# Initialize with default configuration
integration = DSPyIntegration()

# Or with custom configuration
integration = DSPyIntegration(config={
    "model": "gpt-4o",
    "api_key": "your-api-key",
    "temperature": 0.7,
    "max_tokens": 4096
})

# Perform chain-of-thought reasoning
result = await integration.chain_of_thought(
    query="If I have 5 apples and eat 2, then buy 3 more, how many do I have?",
    context={}
)

if result.success:
    print(f"Answer: {result.output}")
    print(f"Reasoning: {result.reasoning}")
```

### Advanced Program-of-Thought

```python
# Define a multi-step program
program_steps = [
    "Understand the problem",
    "Break down into sub-problems",
    "Solve each sub-problem",
    "Synthesize final answer"
]

result = await integration.program_of_thought(
    query="Design a sorting algorithm with O(n log n) complexity",
    steps=program_steps,
    context={"language": "Python"}
)
```

## Configuration Options

### Full Configuration Schema

```python
config = {
    # LLM Configuration
    "model": "gpt-4o",  # Model name
    "api_key": None,  # API key (or set via environment)
    "api_base": None,  # Custom API endpoint
    "temperature": 0.7,  # Sampling temperature
    "max_tokens": 4096,  # Maximum output tokens
    "top_p": 1.0,  # Nucleus sampling
    "frequency_penalty": 0.0,  # Frequency penalty
    "presence_penalty": 0.0,  # Presence penalty

    # Retry Configuration
    "max_retries": 3,  # Maximum retry attempts
    "backoff_factor": 1,  # Exponential backoff factor

    # Teleprompter Configuration
    "teleprompter": {
        "type": "BootstrapFewShot",  # Optimization strategy
        "k": 8,  # Number of examples to bootstrap
        "max_bootstrapped_demos": 8,  # Max bootstrapped demonstrations
        "max_labeled_demos": 8  # Max labeled demonstrations
    },

    # Chain-of-Thought Configuration
    "cot_config": {
        "max_iters": 3,  # Maximum reasoning iterations
        "verbose": False,  # Enable verbose logging
    },

    # Advanced Features
    "enable_self_consistency": False,  # Use self-consistency
    "num_consistency_samples": 5,  # Number of samples for SC
    "enable_ensembling": False,  # Use ensembling
    "ensemble_size": 3,  # Number of models to ensemble
}
```

### Teleprompter Types

DSPy supports several optimization strategies:

1. **BootstrapFewShot**: Automatically generates few-shot examples
   ```python
   teleprompter = {"type": "BootstrapFewShot", "k": 8}
   ```

2. **COPRO**: Coordinate Prompt Optimization
   ```python
   teleprompter = {"type": "COPRO", "iterations": 10}
   ```

3. **Ensemble**: Combines multiple programs
   ```python
   teleprompter = {"type": "Ensemble", "size": 3}
   ```

4. **KNN**: K-nearest neighbors for example selection
   ```python
   teleprompter = {"type": "KNN", "k": 5}
   ```

## API Reference

### Core Methods

#### `chain_of_thought(query, context, options)`

Perform chain-of-thought reasoning.

**Parameters:**
- `query` (str): The problem to solve
- `context` (dict): Additional context for reasoning
- `options` (dict, optional): Override options

**Returns:** `DSPyResult` object with:
- `success` (bool): Whether reasoning succeeded
- `output` (Any): The final answer
- `reasoning` (str): Step-by-step reasoning trace
- `processing_time_ms` (float): Processing time
- `error` (str, optional): Error message if failed

**Example:**
```python
result = await integration.chain_of_thought(
    query="What is the capital of France?",
    context={"domain": "geography"}
)
```

#### `program_of_thought(query, steps, context, options)`

Execute a custom program-of-thought with defined steps.

**Parameters:**
- `query` (str): The problem to solve
- `steps` (List[str]): List of reasoning steps
- `context` (dict): Additional context
- `options` (dict, optional): Override options

**Returns:** `DSPyResult` object

**Example:**
```python
steps = [
    "Identify key variables",
    "Formulate equations",
    "Solve equations",
    "Verify solution"
]
result = await integration.program_of_thought(
    query="Solve: 2x + 5 = 15",
    steps=steps
)
```

#### `multi_step_reasoning(tasks, context, options)`

Execute multiple reasoning tasks in sequence.

**Parameters:**
- `tasks` (List[dict]): List of tasks with 'query' and 'step' fields
- `context` (dict): Shared context
- `options` (dict, optional): Override options

**Returns:** List of `DSPyResult` objects

**Example:**
```python
tasks = [
    {"step": 1, "query": "Analyze the problem"},
    {"step": 2, "query": "Generate solution"},
    {"step": 3, "query": "Verify solution"}
]
results = await integration.multi_step_reasoning(tasks)
```

#### `optimize_prompt(program, training_data, metric)`

Optimize a DSPy program using training data.

**Parameters:**
- `program`: DSPy program to optimize
- `training_data` (List): Training examples
- `metric` (callable): Evaluation metric

**Returns:** Optimized program

**Example:**
```python
training_data = [
    {"question": "1+1=?", "answer": "2"},
    {"question": "2+2=?", "answer": "4"},
]

def exact_match(example, pred, trace=None):
    return example.answer == pred.output

optimized = await integration.optimize_prompt(
    program=my_program,
    training_data=training_data,
    metric=exact_match
)
```

## Advanced Usage

### Self-Consistency

Generate multiple reasoning paths and select the most consistent answer:

```python
config = {
    "enable_self_consistency": True,
    "num_consistency_samples": 5
}
integration = DSPyIntegration(config=config)

result = await integration.chain_of_thought(
    query="What is 15 * 23?",
    context={}
)
# Samples 5 reasoning paths and returns most common answer
```

### Ensembling

Combine predictions from multiple models:

```python
config = {
    "enable_ensembling": True,
    "ensemble_size": 3,
    "models": ["gpt-4o", "gpt-4-turbo", "claude-3-opus"]
}
integration = DSPyIntegration(config=config)

result = await integration.chain_of_thought(
    query="Solve this complex problem",
    context={}
)
# Combines predictions from all 3 models
```

### Custom Programs

Define custom DSPy programs:

```python
import dspy

class MathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.thinker = dspy.ChainOfThought("question -> reasoning -> answer")
        self.verifier = dspy.ChainOfThought("answer, reasoning -> verified")

    def forward(self, question):
        # Generate solution
        result = self.thinker(question=question)
        # Verify solution
        verified = self.verifier(
            answer=result.answer,
            reasoning=result.reasoning
        )
        return dspy.Prediction(
            answer=verified.answer,
            reasoning=result.reasoning,
            verified=verified.verified
        )

# Use custom program
solver = MathSolver()
result = await integration.execute_program(
    program=solver,
    input_data={"question": "Solve: x^2 = 16"}
)
```

## Integration with Knowledge Engine

### Using with Entity Knowledge Graph

```python
from knowledge_engine.integrations import DSPyIntegration, ROMAEntityExtractor

# Extract entities with ROMA
roma = ROMAEntityExtractor()
entities = await roma.extract_entities(
    text="Apple is a technology company founded by Steve Jobs."
)

# Reason about entities with DSPy
dspy = DSPyIntegration()
result = await dspy.chain_of_thought(
    query="What is the relationship between Apple and Steve Jobs?",
    context={"entities": entities}
)
```

### Using with ROMA Pipeline

```python
from knowledge_engine.integrations import ROMADSPyIntegration

# Use ROMA-DSPy integration
roma_dspy = ROMADSPyIntegration()

# Decompose and reason
result = await roma_dspy.decompose_and_reason(
    problem="Design a scalable microservices architecture",
    domain="software_architecture"
)
```

## Performance Considerations

### Optimization Tips

1. **Caching**: Enable caching for repeated queries
   ```python
   config = {"cache_enabled": True, "cache_ttl": 3600}
   ```

2. **Batch Processing**: Process multiple queries together
   ```python
   results = await integration.batch_chain_of_thought(
       queries=["q1", "q2", "q3"],
       batch_size=5
   )
   ```

3. **Token Management**: Monitor token usage
   ```python
   result = await integration.chain_of_thought(query)
   print(f"Tokens used: {result.metadata.get('tokens_used', 0)}")
   ```

4. **Temperature Tuning**: Lower temperature for more deterministic reasoning
   ```python
   config = {"temperature": 0.3}  # More focused reasoning
   ```

### Resource Management

```python
# Configure timeouts
config = {
    "request_timeout": 30,  # seconds
    "max_concurrent_requests": 10
}

# Monitor usage
result = await integration.chain_of_thought(query)
print(f"Processing time: {result.processing_time_ms}ms")
print(f"Tokens: {result.metadata.get('tokens_used')}")
```

## Error Handling

### Common Errors and Solutions

1. **API Key Invalid**
   ```python
   # Error: "Authentication failed"
   # Solution: Check OPENAI_API_KEY environment variable
   import os
   assert os.getenv("OPENAI_API_KEY"), "API key not set"
   ```

2. **Rate Limiting**
   ```python
   # Error: "Rate limit exceeded"
   # Solution: Implement retry with backoff
   config = {
       "max_retries": 5,
       "backoff_factor": 2,
       "retry_on_rate_limit": True
   }
   ```

3. **Timeout**
   ```python
   # Error: "Request timeout"
   # Solution: Increase timeout or reduce complexity
   config = {"request_timeout": 60}
   ```

### Graceful Degradation

```python
try:
    result = await integration.chain_of_thought(query)
    if not result.success:
        # Fallback to simpler reasoning
        result = await integration.simple_answer(query)
except Exception as e:
    logger.error(f"DSPy failed: {e}")
    # Use fallback system
    result = await fallback_system.answer(query)
```

## Troubleshooting

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = {"verbose": True}
integration = DSPyIntegration(config=config)
```

### Common Issues

1. **Import Error**: `ImportError: No module named 'dspy'`
   - Solution: `pip install dspy-ai`

2. **AIOHTTP Compatibility**: Issues with litellm
   - Solution: The integration handles this automatically via `aiohttp_compat`

3. **Memory Issues**: Large responses causing memory problems
   - Solution: Reduce `max_tokens` or use streaming

### Validation

```python
# Test configuration
integration = DSPyIntegration()
assert integration.lm is not None, "LM not initialized"
assert integration.teleprompter is not None, "Teleprompter not initialized"

# Test basic functionality
test_result = await integration.chain_of_thought("2 + 2 = ?")
assert test_result.success, "Basic chain-of-thought failed"
```

## Examples

See the DSPy examples in `examples/dspy/`:
- `basic_cot.py` - Basic chain-of-thought
- `math_solving.py` - Mathematical problem solving
- `code_generation.py` - Code generation with reasoning
- `optimization.py` - Prompt optimization with teleprompters
- `integration_example.py` - Integration with other systems

## References

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Paper: "DSPy: Compiling Self-Improving Language Programs"](https://arxiv.org/abs/2310.03714)

---

**Last Updated**: 2025-02-03
**Integration Version**: 1.0.0
