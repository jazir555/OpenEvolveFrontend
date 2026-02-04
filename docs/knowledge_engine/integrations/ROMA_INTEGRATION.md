# ROMA Integration Guide

## Overview

The ROMA (Recursive Omniscient Meta-Agent) integration provides advanced meta-agent orchestration capabilities to the Knowledge Engine. ROMA is a sophisticated system for decomposing complex problems, coordinating multiple specialist agents, and synthesizing solutions.

### Key Features
- Hierarchical problem decomposition
- Meta-agent orchestration
- Specialist agent coordination
- Solution synthesis and validation
- Adaptive problem solving
- Multi-modal reasoning
- Self-improving strategies

### Use Cases
- Complex multi-step problem solving
- Research and analysis tasks
- Code generation and review
- Mathematical theorem proving
- Scientific hypothesis generation
- System design and architecture

## Installation

```bash
# ROMA is typically in the core-projects directory
# No additional installation required if using Knowledge Engine

# Or install as standalone
cd core-projects/roma
pip install -e .
```

### Configuration

Set up environment variables:

```bash
export ROMA_CONFIG_PATH="/path/to/roma/config"
export ROMA_LOG_LEVEL="INFO"
export ROMA_MAX_AGENTS="10"
export ROMA_DECOMPOSITION_DEPTH="3"
```

## Quick Start

### Basic Usage

```python
from knowledge_engine.integrations import ROMAIntegration

# Initialize ROMA
roma = ROMAIntegration()

# Solve a problem
result = await roma.solve(
    problem="Design a scalable microservices architecture for an e-commerce platform",
    domain="software_architecture",
    context={
        "requirements": ["high availability", "low latency", "fault tolerance"],
        "constraints": {"budget": "moderate", "timeline": "3 months"}
    }
)

if result.success:
    print(f"Solution: {result.solution}")
    print(f"Decomposition: {result.decomposition}")
    print(f"Agents used: {result.agents}")
```

### Problem Decomposition

```python
# Decompose a problem
decomposition = await roma.decompose(
    problem="Prove that the sum of two even numbers is even",
    domain="mathematics"
)

print("Sub-problems:")
for i, subproblem in enumerate(decomposition.subproblems, 1):
    print(f"{i}. {subproblem.description}")
    print(f"   Type: {subproblem.type}")
    print(f"   Dependencies: {subproblem.dependencies}")
```

### Meta-Agent Coordination

```python
# Create meta-agent
meta_agent = ROMAMetaAgent(
    role="Research Coordinator",
    capabilities=["planning", "coordination", "synthesis"],
    max_depth=3
)

# Execute with meta-agent
result = await meta_agent.execute(
    problem="Investigate the latest advances in quantum computing",
    specialist_agents=[
        "literature_reviewer",
        "technical_analyst",
        "synthesis_specialist"
    ]
)
```

## Configuration Options

### Full Configuration Schema

```python
config = {
    # Meta-Agent Configuration
    "meta_agent": {
        "max_depth": 3,  # Maximum decomposition depth
        "max_branching": 5,  # Maximum branches per decomposition
        "timeout": 300,  # Timeout per subproblem (seconds)
        "strategy": "recursive",  # recursive, iterative, parallel
        "validation": True  # Validate solutions
    },

    # Decomposition Configuration
    "decomposition": {
        "method": "hierarchical",  # hierarchical, flat, hybrid
        "granularity": "medium",  # fine, medium, coarse
        "dependency_tracking": True,
        "parallel_execution": True,
        "max_parallel_tasks": 5
    },

    # Agent Configuration
    "agents": {
        "max_agents": 10,
        "agent_types": [
            "researcher",
            "analyst",
            "solver",
            "validator",
            "synthesizer"
        ],
        "agent_selection": "auto",  # auto, manual
        "load_balancing": True
    },

    # Knowledge Configuration
    "knowledge": {
        "use_knowledge_graph": True,
        "embeddings": True,
        "similarity_threshold": 0.75,
        "cache_solutions": True,
        "learn_from_feedback": True
    },

    # Synthesis Configuration
    "synthesis": {
        "method": "weighted",  # weighted, voting, consensus
        "validation": True,
        "consensus_threshold": 0.7,
        "conflict_resolution": "merit_based"
    },

    # Logging Configuration
    "logging": {
        "level": "INFO",
        "trace_decomposition": True,
        "log_agent_communication": True,
        "output_file": None
    }
}
```

## API Reference

### Core Methods

#### `solve(problem, domain, context, options)`

Solve a complex problem using ROMA.

**Parameters:**
- `problem` (str): The problem to solve
- `domain` (str): Problem domain
- `context` (dict): Additional context and constraints
- `options` (dict, optional): Override options

**Returns:** `ROMAResult` object
- `success` (bool): Solution success
- `solution` (Any): The solution
- `decomposition` (dict): Problem decomposition
- `agents` (List[str]): Agents involved
- `metadata` (dict): Execution metadata
- `processing_time_ms` (float): Processing time

**Example:**
```python
result = await roma.solve(
    problem="Design a neural network for image classification",
    domain="machine_learning",
    context={"dataset": "ImageNet", "accuracy_target": 0.95}
)
```

#### `decompose(problem, domain, options)`

Decompose a problem into sub-problems.

**Parameters:**
- `problem` (str): The problem to decompose
- `domain` (str): Problem domain
- `options` (dict, optional): Decomposition options

**Returns:** `ROMADecomposition` object
- `subproblems` (List[dict]): List of sub-problems
- `dependencies` (List[tuple]): Dependencies between sub-problems
- `execution_order` (List[str]): Optimal execution order

**Example:**
```python
decomposition = await roma.decompose(
    problem="Build a web application",
    domain="software_engineering"
)

for subproblem in decomposition.subproblems:
    print(f"Subproblem: {subproblem['description']}")
    print(f"Dependencies: {subproblem['dependencies']}")
```

#### `create_meta_agent(role, capabilities, config)`

Create a meta-agent for coordination.

**Parameters:**
- `role` (str): Meta-agent role
- `capabilities` (List[str]): Meta-agent capabilities
- `config` (dict): Meta-agent configuration

**Returns:** `ROMAMetaAgent` object

**Example:**
```python
meta_agent = await roma.create_meta_agent(
    role="Project Manager",
    capabilities=["planning", "coordination", "monitoring"],
    config={
        "max_depth": 3,
        "timeout": 600
    }
)
```

#### `execute_agent(agent_type, task, context)`

Execute a specialist agent.

**Parameters:**
- `agent_type` (str): Type of specialist agent
- `task` (dict): Task to execute
- `context` (dict): Execution context

**Returns:** Agent execution result

**Example:**
```python
result = await roma.execute_agent(
    agent_type="researcher",
    task={"query": "Latest developments in LLMs"},
    context={"domain": "ai_research"}
)
```

## Advanced Usage

### Custom Decomposition Strategy

```python
# Define custom decomposition strategy
async def custom_decomposition(problem, domain):
    # Analyze problem structure
    if "design" in problem.lower():
        return decompose_design_problem(problem)
    elif "prove" in problem.lower():
        return decompose_proof_problem(problem)
    else:
        return decompose_generic(problem)

# Use custom strategy
roma = ROMAIntegration(config={
    "decomposition": {
        "strategy": "custom",
        "custom_function": custom_decomposition
    }
})
```

### Parallel Execution

```python
# Enable parallel execution of independent sub-problems
config = {
    "decomposition": {
        "parallel_execution": True,
        "max_parallel_tasks": 10
    }
}
roma = ROMAIntegration(config=config)

result = await roma.solve(
    problem="Large problem with many independent sub-problems"
)
```

### Knowledge-Augmented Solving

```python
# Use knowledge graph to guide decomposition
roma = ROMAIntegration(config={
    "knowledge": {
        "use_knowledge_graph": True,
        "embeddings": True,
        "retrieve_similar": True
    }
})

result = await roma.solve(
    problem="Similar to previous problems we've solved"
)
# ROMA will retrieve similar problems and their solutions
```

### Solution Validation

```python
# Enable automatic solution validation
config = {
    "meta_agent": {
        "validation": True,
        "validation_agents": ["validator", "tester"]
    }
}
roma = ROMAIntegration(config=config)

result = await roma.solve(problem="...")
# Solution will be automatically validated before returning
```

### Learning from Feedback

```python
# Enable learning from execution feedback
config = {
    "knowledge": {
        "learn_from_feedback": True,
        "feedback_storage": "knowledge_base"
    }
}
roma = ROMAIntegration(config=config)

# Provide feedback
await roma.provide_feedback(
    problem_id=result.problem_id,
    feedback="The solution was correct and complete",
    rating=5
)
# ROMA will learn to use similar strategies for future problems
```

## Integration with Knowledge Engine

### Using with DSPy

```python
from knowledge_engine.integrations import ROMAIntegration, DSPyIntegration

# Use DSPy for reasoning within ROMA agents
dspy = DSPyIntegration()

# ROMA will use DSPy for chain-of-thought reasoning
roma = ROMAIntegration(config={
    "agents": {
        "reasoning_backend": "dspy",
        "dspy_config": {
            "model": "gpt-4o",
            "enable_self_consistency": True
        }
    }
})
```

### Using with DeepKE

```python
from knowledge_engine.integrations import ROMAIntegration, DeepKEIntegration

# Extract knowledge with DeepKE
deepke = DeepKEIntegration()

# Use in ROMA for knowledge extraction
roma = ROMAIntegration()

# ROMA can extract knowledge from solutions
result = await roma.solve(problem="...")

# Extract entities and relations from solution
knowledge = await deepke.extract_entities_relations(
    text=str(result.solution)
)
```

### Using with ROMA Entity Knowledge Graph

```python
from knowledge_engine.integrations import ROMAIntegration, ROMAEntityExtractor

# Store solutions in knowledge graph
roma = ROMAIntegration()
ekg = ROMAEntityExtractor()

result = await roma.solve(problem="...")

# Extract and store entities
await ekg.store_solution(
    problem=result.problem,
    solution=result.solution,
    decomposition=result.decomposition
)
```

### Cross-Integration: ROMA-DSPy

```python
from knowledge_engine.integrations import ROMADSPyIntegration

# Use combined ROMA-DSPy integration
roma_dspy = ROMADSPyIntegration()

# Decompose and reason with DSPy
result = await roma_dspy.decompose_and_reason(
    problem="Complex reasoning problem",
    reasoning_method="chain_of_thought"
)
```

### Cross-Integration: ROMA-DeepKE

```python
from knowledge_engine.integrations import ROMADeepKEIntegration

# Use combined ROMA-DeepKE integration
roma_deepke = ROMADeepKEIntegration()

# Solve and extract knowledge
result = await roma_deepke.solve_and_extract(
    problem="Knowledge-intensive problem",
    extraction_config={
        "extract_entities": True,
        "extract_relations": True
    }
)
```

### Cross-Integration: ROMA-Ragbits

```python
from knowledge_engine.integrations import ROMARagbitsIntegration

# Use combined ROMA-Ragbits integration
roma_ragbits = ROMARagbitsIntegration()

# Solve with retrieval
result = await roma_ragbits.solve_with_retrieval(
    problem="Problem requiring external knowledge",
    retrieval_config={
        "top_k": 10,
        "similarity_threshold": 0.7
    }
)
```

## Performance Considerations

### Decomposition Depth

```python
# Balance between depth and performance
config = {
    "meta_agent": {
        "max_depth": 3  # Increase for more complex problems
    }
}
```

### Parallel Execution

```python
# Configure parallel execution
config = {
    "decomposition": {
        "parallel_execution": True,
        "max_parallel_tasks": min(10, os.cpu_count())
    }
}
```

### Caching

```python
# Cache solutions and decompositions
config = {
    "knowledge": {
        "cache_solutions": True,
        "cache_decompositions": True,
        "cache_ttl": 3600
    }
}
```

## Error Handling

### Common Errors

1. **Decomposition Timeout**
   ```python
   # Solution: Increase timeout or reduce max_depth
   config = {
       "meta_agent": {
           "timeout": 600,
           "max_depth": 2
       }
   }
   ```

2. **Agent Unavailable**
   ```python
   # Solution: Check agent availability
   available_agents = await roma.list_agents()
   assert required_agent in available_agents
   ```

3. **Solution Validation Failed**
   ```python
   # Solution: Adjust validation threshold
   config = {
       "synthesis": {
           "validation": True,
           "validation_threshold": 0.6  # Lower threshold
       }
   }
   ```

## Troubleshooting

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = {
    "logging": {
        "level": "DEBUG",
        "trace_decomposition": True,
        "log_agent_communication": True
    }
}
```

### Common Issues

1. **Slow Decomposition**
    - Reduce `max_depth`
    - Enable `parallel_execution`
    - Use caching

2. **Poor Solutions**
    - Adjust `granularity`
    - Enable `validation`
    - Provide better `context`

3. **Agent Coordination Issues**
    - Check agent configurations
    - Verify agent availability
    - Review agent communication logs

## Examples

See the ROMA examples in `examples/roma/`:
- `basic_solving.py` - Basic problem solving
- `decomposition.py` - Problem decomposition
- `meta_agent.py` - Meta-agent coordination
- `parallel_execution.py` - Parallel task execution
- `integration_dspy.py` - Integration with DSPy
- `integration_deepke.py` - Integration with DeepKE

## References

- [ROMA Documentation](../ROMA/)
- [Meta-Agent Systems](../architecture/meta_agents.md)
- [Problem Decomposition](../algorithms/decomposition.md)

---

**Last Updated**: 2025-02-03
**Integration Version**: 1.0.0
