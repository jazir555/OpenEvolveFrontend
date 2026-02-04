# CrewAI Integration Guide

## Overview

The CrewAI integration provides multi-agent orchestration capabilities to the Knowledge Engine. CrewAI enables collaborative AI agents to work together on complex tasks, combining their strengths to achieve better results than any single agent could alone.

### Key Features
- Multi-agent collaboration
- Role-based agent definition
- Sequential and hierarchical process flows
- Task delegation and execution
- Agent memory and context sharing
- Tool integration for agents
- Crew management and monitoring

### Use Cases
- Complex multi-step workflows
- Collaborative problem solving
- Parallel task processing
- Research and analysis tasks
- Content creation workflows
- Code review and generation

## Installation

```bash
# Basic installation
pip install crewai

# With Knowledge Engine
pip install knowledge-engine[crewai]

# Optional: Additional tools
pip install crewai-tools  # For built-in tools
```

### Configuration

Set up environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export SERPER_API_KEY="your-serper-key"  # For search tools
export CREWAI_VERBOSE="true"  # Enable verbose logging
```

## Quick Start

### Basic Usage

```python
from knowledge_engine.integrations import CrewAIIntegration

# Initialize integration
integration = CrewAIIntegration()

# Define agents
researcher = {
    "role": "Senior Research Analyst",
    "goal": "Discover cutting-edge developments in AI and data science",
    "backstory": "You are an experienced researcher with a passion for discovering trends",
    "verbose": True,
    "allow_delegation": False
}

writer = {
    "role": "Technical Writer",
    "goal": "Write compelling blog posts about AI developments",
    "backstory": "You are a technical writer with a knack for explaining complex topics",
    "verbose": True,
    "allow_delegation": False
}

# Define tasks
research_task = {
    "description": "Research the latest developments in large language models",
    "expected_output": "A comprehensive report on LLM developments",
    "agent": "researcher"
}

writing_task = {
    "description": "Write a blog post based on the research findings",
    "expected_output": "A well-written blog post",
    "agent": "writer"
}

# Create and run crew
crew_id = "research_crew"
success = await integration.create_crew(
    crew_id=crew_id,
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process="sequential"
)

if success:
    # Execute the crew
    result = await integration.execute_crew(crew_id=crew_id)
    print(f"Output: {result.output}")
```

### Advanced Usage with Tools

```python
# Define agents with tools
agent_with_tools = {
    "role": "Data Analyst",
    "goal": "Analyze data and provide insights",
    "backstory": "You are an expert data analyst",
    "tools": [
        "search_tool",  # Web search
        "calculator",   # Calculator
        "file_read"     # File reading
    ],
    "llm": "gpt-4o"  # Use specific model
}

# Create crew with tool-using agents
await integration.create_crew(
    crew_id="analyst_crew",
    agents=[agent_with_tools],
    tasks=[...],
    process="sequential"
)
```

## Configuration Options

### Full Configuration Schema

```python
config = {
    # LLM Configuration
    "default_llm": "gpt-4o",
    "max_rpm": 100,  # Rate limit (requests per minute)
    "temperature": 0.7,
    "max_tokens": 8192,

    # Process Configuration
    "process": "sequential",  # sequential, hierarchical
    "memory": False,  # Enable crew memory
    "cache": True,  # Enable caching
    "max_iter": 25,  # Maximum iterations for hierarchical process
    "verbose": False,  # Verbose logging

    # Crew Configuration
    "share_crew": False,  # Share crew across executions
    "manager_llm": "gpt-4o",  # LLM for manager in hierarchical process

    # Agent Configuration
    "agent_config": {
        "allow_delegation": True,  # Allow agents to delegate tasks
        "async_execution": False,  # Execute agents asynchronously
        "max_execution_time": 300,  # Maximum execution time per agent (seconds)
        "human_handoff": False  # Allow human intervention
    },

    # Logging Configuration
    "crew_logging": {
        "enabled": True,
        "level": "INFO",
        "output_file": None  # Log to file
    },

    # Callback Configuration
    "callbacks": {
        "on_task_start": None,
        "on_task_end": None,
        "on_crew_start": None,
        "on_crew_end": None
    }
}
```

## API Reference

### Core Methods

#### `create_crew(crew_id, agents, tasks, process, options)`

Create a new crew with agents and tasks.

**Parameters:**
- `crew_id` (str): Unique identifier for the crew
- `agents` (List[dict]): List of agent configurations
- `tasks` (List[dict]): List of task configurations
- `process` (str): Process type ("sequential" or "hierarchical")
- `options` (dict, optional): Additional options

**Agent Configuration:**
```python
{
    "role": str,  # Agent role
    "goal": str,  # Agent goal
    "backstory": str,  # Agent backstory
    "tools": List[str],  # List of tool names
    "llm": str,  # LLM to use
    "verbose": bool,  # Enable verbose logging
    "allow_delegation": bool,  # Allow task delegation
    "max_iter": int,  # Maximum iterations
    "max_execution_time": int  # Maximum execution time (seconds)
}
```

**Task Configuration:**
```python
{
    "description": str,  # Task description
    "expected_output": str,  # Expected output format
    "agent": str,  # Agent to assign to
    "context": List[dict],  # Context from other tasks
    "tools": List[str],  # Tools for this task
    "async_execution": bool  # Execute asynchronously
}
```

**Returns:** bool indicating success

**Example:**
```python
await integration.create_crew(
    crew_id="my_crew",
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process="sequential"
)
```

#### `execute_crew(crew_id, inputs, options)`

Execute a crew.

**Parameters:**
- `crew_id` (str): Crew identifier
- `inputs` (dict): Input data for the crew
- `options` (dict, optional): Execution options

**Returns:** `CrewAIResult` object
- `success` (bool): Execution success
- `output` (Any): Crew output
- `token_usage` (dict): Token usage statistics
- `execution_time_ms` (float): Execution time
- `error` (str, optional): Error message

**Example:**
```python
result = await integration.execute_crew(
    crew_id="research_crew",
    inputs={"topic": "artificial intelligence"}
)
```

#### `add_agent(crew_id, agent_config)`

Add an agent to an existing crew.

**Parameters:**
- `crew_id` (str): Crew identifier
- `agent_config` (dict): Agent configuration

**Returns:** bool indicating success

#### `add_task(crew_id, task_config)`

Add a task to an existing crew.

**Parameters:**
- `crew_id` (str): Crew identifier
- `task_config` (dict): Task configuration

**Returns:** bool indicating success

#### `delete_crew(crew_id)`

Delete a crew.

**Parameters:**
- `crew_id` (str): Crew identifier

**Returns:** bool indicating success

#### `list_crews()`

List all crews.

**Returns:** List of crew identifiers

## Advanced Usage

### Hierarchical Process

Use a manager to coordinate agents:

```python
# Create hierarchical crew
await integration.create_crew(
    crew_id="hierarchical_crew",
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    process="hierarchical"  # Manager coordinates agents
)

result = await integration.execute_crew(
    crew_id="hierarchical_crew",
    inputs={"query": "..."}
)
```

### Task Context

Pass context between tasks:

```python
# Task 2 uses output from Task 1
task1 = {
    "description": "Research the topic",
    "expected_output": "Research findings",
    "agent": "researcher"
}

task2 = {
    "description": "Write based on research",
    "expected_output": "Written article",
    "agent": "writer",
    "context": [task1]  # Receives context from task1
}
```

### Agent Tools

Provide tools to agents:

```python
# Built-in tools
from crewai_tools import (
    SerperDevTool,
    FileReadTool,
    DirectoryReadTool,
    CodeInterpreterTool
)

search_tool = SerperDevTool()
file_tool = FileReadTool()

agent_with_tools = {
    "role": "Researcher",
    "goal": "Find and analyze information",
    "tools": [search_tool, file_tool],
    "verbose": True
}
```

### Custom Tools

Create custom tools:

```python
from crewai_tools import BaseTool
from pydantic import Field

class CustomSearchTool(BaseTool):
    name: str = "custom_search"
    description: str = "Search custom database"

    def _run(self, query: str) -> str:
        # Custom search logic
        return f"Results for: {query}"

# Use in agent
agent = {
    "role": "Specialist",
    "tools": [CustomSearchTool()],
    ...
}
```

### Crew Memory

Enable memory for context retention:

```python
config = {
    "memory": True,
    "memory_config": {
        "type": "short_term",  # short_term, long_term, shared
        "embeddings": "text-embedding-3-small",
        "storage": "redis"  # redis, local
    }
}
integration = CrewAIIntegration(config=config)

# Crew will remember previous interactions
result1 = await integration.execute_crew(
    crew_id="memory_crew",
    inputs={"query": "First query"}
)

result2 = await integration.execute_crew(
    crew_id="memory_crew",
    inputs={"query": "Follow up query"}  # Has context from result1
)
```

### Callbacks

Add callbacks for monitoring:

```python
def on_task_start(task):
    print(f"Task started: {task.description}")

def on_task_end(task, output):
    print(f"Task completed: {task.description}")
    print(f"Output: {output}")

config = {
    "callbacks": {
        "on_task_start": on_task_start,
        "on_task_end": on_task_end
    }
}
integration = CrewAIIntegration(config=config)
```

## Integration with Knowledge Engine

### Using with DSPy

```python
from knowledge_engine.integrations import CrewAIIntegration, DSPyIntegration

# Use DSPy for reasoning within agents
dspy = DSPyIntegration()

agent = {
    "role": "Reasoning Agent",
    "goal": "Solve complex problems",
    "tools": ["dspy_reasoning"],  # Custom DSPy tool
    ...
}

# In the tool implementation
async def dspy_reasoning_tool(query):
    result = await dspy.chain_of_thought(query=query)
    return result.reasoning
```

### Using with DeepKE

```python
from knowledge_engine.integrations import CrewAIIntegration, DeepKEIntegration

# Extract knowledge with DeepKE
deepke = DeepKEIntegration()

agent = {
    "role": "Knowledge Extractor",
    "goal": "Extract structured knowledge",
    "tools": ["deepke_extract"],  # Custom DeepKE tool
    ...
}

# In the tool implementation
async def deepke_extract_tool(text):
    result = await deepke.extract_entities_relations(text)
    return result.triples
```

### Using with ROMA

```python
from knowledge_engine.integrations import CrewAIIntegration, ROMAIntegration

# Coordinate ROMA agents
crewai = CrewAIIntegration()
roma = ROMAIntegration()

# Create ROMA meta-agents via CrewAI
meta_agent = {
    "role": "ROMA Coordinator",
    "goal": "Coordinate ROMA problem solving",
    "tools": ["roma_solve"],
    ...
}
```

## Performance Considerations

### Parallel Execution

Execute agents in parallel when possible:

```python
# Enable async execution
agent = {
    "role": "Parallel Agent",
    "async_execution": True,
    ...
}

task = {
    "description": "Independent task",
    "async_execution": True,
    ...
}
```

### Caching

Enable caching to avoid redundant work:

```python
config = {
    "cache": True,
    "cache_config": {
        "type": "redis",  # redis, memory, file
        "ttl": 3600  # Cache for 1 hour
    }
}
```

### Rate Limiting

Configure rate limits:

```python
config = {
    "max_rpm": 100,  # 100 requests per minute
    "rate_limit_strategy": "exponential_backoff"
}
```

## Error Handling

### Common Errors

1. **API Key Invalid**
   ```python
   # Solution: Check OPENAI_API_KEY
   import os
   assert os.getenv("OPENAI_API_KEY"), "API key not set"
   ```

2. **Agent Timeout**
   ```python
   # Solution: Increase max_execution_time
   agent = {
       "max_execution_time": 600,  # 10 minutes
       ...
   }
   ```

3. **Memory Issues**
   ```python
   # Solution: Disable memory or reduce context
   config = {"memory": False}
   ```

### Retry Logic

```python
# Configure retries
config = {
    "max_retries": 3,
    "retry_delay": 1.0,
    "retry_on_timeout": True
}
```

## Troubleshooting

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = {
    "verbose": True,
    "crew_logging": {
        "enabled": True,
        "level": "DEBUG"
    }
}
```

### Common Issues

1. **Agents Not Starting**
   - Check agent configurations
   - Verify LLM API keys
   - Enable verbose logging

2. **Tasks Hanging**
   - Check if tasks have clear descriptions
   - Verify agent capabilities
   - Set appropriate timeouts

3. **Memory Not Working**
   - Check memory configuration
   - Verify storage backend
   - Check embeddings configuration

## Examples

See the CrewAI examples in `examples/crewai/`:
- `basic_crew.py` - Basic crew creation and execution
- `hierarchical.py` - Hierarchical process
- `with_tools.py` - Using tools with agents
- `memory_demo.py` - Crew memory demo
- `integration_example.py` - Integration with other systems

## References

- [CrewAI Documentation](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)
- [CrewAI Tools](https://github.com/joaomdmoura/crewAI-tools)

---

**Last Updated**: 2025-02-03
**Integration Version**: 1.0.0
