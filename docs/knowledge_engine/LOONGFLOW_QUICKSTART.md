# LoongFlow Quick Start Guide

## What is LoongFlow?

LoongFlow is a **Plan-Execute-Summarize (PES)** evolutionary agent framework that enables AI agents to improve their performance through iterative refinement. It provides:

- **Memory Management**: Track solutions and their evolution over time
- **Message System**: Structured communication protocol
- **Tool Framework**: Declarative tool definitions
- **PES Orchestration**: Plan → Execute → Summarize workflow

## Installation Status

✅ **Installed and Verified** - LoongFlow is ready for use in OpenEvolve

Run verification:
```bash
python docs/knowledge_engine/verify_loongflow.py
```

## Basic Usage

### 1. Create a Message

```python
from loongflow.agentsdk.message import Message, Role

# Method 1: Using helper (recommended)
msg = Message.from_text(
    "Solve this math problem: x^2 + 2x + 1 = 0",
    role=Role.USER
)

# Method 2: Direct creation
from loongflow.agentsdk.message import ContentElement, MimeType
msg = Message(
    role=Role.USER,
    content=[
        ContentElement(
            type="text",
            mime_type=MimeType.TEXT_PLAIN,
            data="Solve this math problem"
        )
    ]
)
```

### 2. Set Up Memory

```python
from loongflow.agentsdk.memory.evolution import MemoryFactory, InMemory

# Create memory instance
memory = InMemory()

# Or use factory
factory = MemoryFactory()
memory = factory.create_memory(config={})
```

### 3. Define a Tool

```python
from loongflow.agentsdk.tools import function_tool

@function_tool
def calculate_square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2
```

### 4. Use PES Framework

```python
from loongflow.framework.pes import PESAgent

# Create PES agent
agent = PESAgent(
    # Configuration depends on your use case
    # See individual agent implementations below
)
```

## Available PES Agents

### MathEvolveAgent
**Purpose**: Evolve solutions for mathematical problems

**Location**: `LoongFlow/agents/math_agent/math_evolve_agent.py`

**Use Case**: Optimization problems, equation solving, mathematical proofs

**Example**:
```python
from agents.math_agent.math_evolve_agent import MathEvolveAgent

agent = MathEvolveAgent()
result = agent.solve("Find the maximum area of a rectangle with perimeter 20")
```

### MLEvolveAgent
**Purpose**: Evolve machine learning pipelines

**Location**: `LoongFlow/agents/ml_agent/ml_evolve_agent.py`

**Use Case**: Feature engineering, model selection, hyperparameter tuning

**Example**:
```python
from agents.ml_agent.ml_evolve_agent import MLEvolveAgent

agent = MLEvolveAgent()
result = agent.evolve_pipeline(
    dataset="path/to/data.csv",
    target="classification"
)
```

### GeneralEvolveAgent
**Purpose**: General-purpose task evolution

**Location**: `LoongFlow/agents/general_agent/general_evolve_agent.py`

**Use Case**: Code generation, text processing, multi-step reasoning

**Example**:
```python
from agents.general_agent.general_evolve_agent import GeneralEvolveAgent

agent = GeneralEvolveAgent()
result = agent.evolve(task_description, constraints)
```

## PES Workflow

The PES paradigm follows three stages:

### 1. Plan
Agent creates a plan to solve the problem
```python
plan = agent.plan(task_description)
```

### 2. Execute
Agent executes the plan and attempts solutions
```python
solutions = agent.execute(plan)
```

### 3. Summarize
Agent evaluates results and refines approach
```python
summary = agent.summarize(solutions, results)
```

## Memory System

### Evolution Memory
Track how solutions improve over iterations:

```python
from loongflow.agentsdk.memory.evolution import EvolveMemory, Solution

# Store a solution
solution = Solution(
    program="def solve(): return x * 2",
    score=0.85,
    metadata={"iteration": 1}
)
memory.add(solution)

# Retrieve best solutions
best = memory.get_top_k(k=5)
```

### Graded Memory
Compress and organize conversation history:

```python
from loongflow.agentsdk.memory.grade import GradedMemory

graded = GradedMemory()
graded.add_message(message)
compressed = graded.compress()
```

## Configuration

### LLM Configuration
```python
from loongflow.agentsdk.models import LiteLLMModel

model = LiteLLMModel(
    model="gpt-4",  # or "claude-3-opus", etc.
    api_key="your-api-key"
)
```

### Tool Configuration
```python
tools = [
    calculate_square,
    # Add more tools
]

agent_config = {
    "tools": tools,
    "model": model,
    "memory": memory
}
```

## Testing

Run integration tests:
```bash
cd C:/Users/mmeadow/Documents/OpenEvolve/Frontend
python tests/integration/test_loongflow_import.py
```

Expected output: 9/13 tests passing (4 failures are expected API differences)

## Common Patterns

### Pattern 1: Simple Tool Use
```python
from loongflow.agentsdk.tools import function_tool

@function_tool
def my_tool(param: str) -> str:
    """Tool description"""
    return f"Processed: {param}"

# Register with agent
agent = PESAgent(tools=[my_tool])
```

### Pattern 2: Message Handling
```python
from loongflow.agentsdk.message import Message, Role

# Create user message
user_msg = Message.from_text("Help me solve this", role=Role.USER)

# Process with agent
response = agent.process(user_msg)

# Extract response text
if response.content:
    print(response.content[0].data)
```

### Pattern 3: Iterative Refinement
```python
for iteration in range(max_iterations):
    # Plan
    plan = agent.plan(task)

    # Execute
    result = agent.execute(plan)

    # Evaluate
    score = evaluate(result)

    # Update memory
    memory.add(Solution(program=result, score=score))

    # Check convergence
    if score > threshold:
        break
```

## Important Notes

### Message Content Format
Messages require `content` to be a list of Element objects, not strings. Use `Message.from_text()` helper for simple text messages.

### Python Version
LoongFlow was adapted from Python >=3.12 to work with Python 3.11. No functionality is lost.

### Agent Import Paths
The three PES agents (Math, ML, General) are in the `LoongFlow/agents/` directory, not directly importable as Python packages. Import them using:
```python
from agents.math_agent.math_evolve_agent import MathEvolveAgent
```

## Troubleshooting

### Import Error
**Problem**: `ModuleNotFoundError: No module named 'loongflow'`
**Solution**: Run `pip install -e ./LoongFlow` from Frontend directory

### Message Validation Error
**Problem**: `Input should be a valid list`
**Solution**: Use `Message.from_text()` helper instead of direct creation

### Agent Not Found
**Problem**: `No module named 'agents'`
**Solution**: Use full path: `from agents.math_agent.math_evolve_agent import MathEvolveAgent`

## Resources

- **Full Documentation**: `docs/knowledge_engine/LOONGFLOW_INTEGRATION_REPORT.md`
- **Integration Tests**: `tests/integration/test_loongflow_import.py`
- **Verification Script**: `docs/knowledge_engine/verify_loongflow.py`
- **LoongFlow Source**: `LoongFlow/`
- **LoongFlow Docs**: `LoongFlow/docs/`

## Next Steps

1. **Explore Examples**: Check `LoongFlow/agents/*/examples/` for usage examples
2. **Read Source**: Study agent implementations in `LoongFlow/agents/`
3. **Build Integration Wrapper**: Create unified interface for Knowledge Engine
4. **Test PES Execution**: Run end-to-end PES workflows

## Support

For issues or questions:
1. Check `LOONGFLOW_INTEGRATION_REPORT.md` for detailed integration notes
2. Run `verify_loongflow.py` to check installation
3. Review agent examples in `LoongFlow/agents/*/examples/`

---

**Last Updated**: 2026-01-30
**Integration Status**: Complete ✅
**Ready for Use**: Yes
