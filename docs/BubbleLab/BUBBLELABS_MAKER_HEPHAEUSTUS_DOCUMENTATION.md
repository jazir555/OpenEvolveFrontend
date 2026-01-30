# BubbleLabs + Maker + Hephaestus Integration Documentation

## Overview

This integration combines three powerful systems:

- **Maker Engine**: Zero-error task solving using voting-based consensus (arXiv:2511.09030)
- **Hephaestus**: Project management and ticket tracking system
- **BubbleLabs**: Visual workflow interface for OpenEvolve

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [UI Components](#ui-components)
6. [Advanced Usage](#advanced-usage)
7. [Examples](#examples)

---

## Installation

### Prerequisites

```bash
# Ensure you have the required dependencies
pip install streamlit pandas requests

# Optional: For full Hephaestus integration
pip install requests-threading
```

### Setup

1. **Import the Integration**

```python
from bubblelabs_maker_integration import (
    BubbleLabsMakerUI,
    ToolRepository,
    HephaestusDelegationManager,
    MakerWorkflowManager,
    create_bubblelabs_maker_integration
)
```

2. **Initialize with Hephaestus (Optional)**

```python
from hephaestus_integration import HephaestusIntegrationManager

# Create Hephaestus manager
hephaestus_manager = HephaestusIntegrationManager(
    api_base="https://your-hephaestus-api.com",
    api_key="your-api-key",
    project_id="your-project-id"
)

# Create integration
maker_ui = create_bubblelabs_maker_integration(hephaestus_manager)
```

---

## Quick Start

### Creating Your First Tool

```python
from bubblelabs_maker_integration import MakerWorkflowManager

# Initialize manager
workflow_manager = MakerWorkflowManager()

# Create a tool
tool, error = workflow_manager.create_tool_workflow(
    name="Data Analyzer",
    description="Analyzes datasets and provides insights",
    task="Analyze the provided dataset and summarize key findings",
    maker_mode="recursive",
    k_ahead=3,
    max_depth=5
)

if tool:
    print(f"Tool created: {tool.tool_id}")
else:
    print(f"Error: {error}")
```

### Executing a Tool

```python
# Execute the tool
result, error = workflow_manager.execute_tool_workflow(
    tool_id=tool.tool_id,
    input_data={
        "task": "Analyze this sales data",
        "context": {"data_format": "csv"}
    },
    delegate_to_hephaestus=True
)

if result:
    print(f"Execution completed in {result.execution_time:.2f}s")
    print(f"Output: {result.output_data}")
else:
    print(f"Execution failed: {error}")
```

---

## Core Concepts

### 1. Tool Repository

The **ToolRepository** manages tools created by the Maker Engine.

**Key Features:**
- Tool registration and versioning
- Tool discovery and search
- Usage tracking
- Status management (draft → testing → validated → deployed)

**Example:**

```python
from bubblelabs_maker_integration import ToolRepository, ToolStatus

# Create repository
repo = ToolRepository(storage_path="./tools.json")

# Register a new tool
tool = repo.register_tool(
    name="Code Generator",
    description="Generates code from natural language",
    maker_mode="recursive",
    config={"k_ahead": 3, "max_depth": 5},
    prompt_template="Generate code for: {task}",
    system_prompt="You are an expert programmer."
)

# Update tool status
repo.update_tool(tool.tool_id, status=ToolStatus.VALIDATED)

# Search tools
tools = repo.search_tools("code")
```

### 2. Hephaestus Delegation

The **HephaestusDelegationManager** handles task delegation to Hephaestus.

**Key Features:**
- Task delegation to Hephaestus tickets
- Status synchronization
- Progress tracking
- Result management

**Example:**

```python
from bubblelabs_maker_integration import HephaestusDelegationManager
from hephaestus_integration import HephaestusIntegrationManager

# Setup
hephaestus_manager = HephaestusIntegrationManager(...)
delegation_manager = HephaestusDelegationManager(hephaestus_manager)

# Delegate a MAKER run
delegation = delegation_manager.delegate_maker_run(
    run_id="run_123",
    title="Solve Complex Problem",
    description="Use MAKER to solve this multi-step problem",
    initial_state={"problem": "..."},
    maker_config=maker_config,
    workflow_epic_id="epic_456"
)

# Update delegation status
delegation_manager.update_delegation_status(
    delegation.delegation_id,
    DelegationStatus.COMPLETE,
    result={"solution": "..."}
)

# Sync from Hephaestus
synced = delegation_manager.sync_from_hephaestus()
```

### 3. Maker Workflow Manager

The **MakerWorkflowManager** orchestrates tool creation and execution.

**Key Features:**
- Tool creation workflows
- Tool execution with progress tracking
- Hephaestus integration
- Result storage and retrieval

**Example:**

```python
from bubblelabs_maker_integration import MakerWorkflowManager

# Initialize
manager = MakerWorkflowManager()

# Create tool
tool, error = manager.create_tool_workflow(
    name="Document Summarizer",
    description="Summarizes long documents",
    task="Summarize the following document",
    maker_mode="recursive",
    k_ahead=3
)

# Execute tool
result, error = manager.execute_tool_workflow(
    tool_id=tool.tool_id,
    input_data={
        "task": "Summarize this report",
        "context": {"max_length": 500}
    }
)

# Get status
status = manager.get_workflow_status(tool.metadata["workflow_id"])
```

---

## API Reference

### ToolRepository

#### Methods

##### `register_tool(...)`

Register a new tool in the repository.

```python
def register_tool(
    self,
    name: str,
    description: str,
    maker_mode: str,
    config: Dict[str, Any],
    prompt_template: Optional[str] = None,
    system_prompt: Optional[str] = None,
    expected_schema: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ToolDefinition
```

**Parameters:**
- `name` (str): Tool name
- `description` (str): Tool description
- `maker_mode` (str): "sequential", "recursive", or "hybrid"
- `config` (dict): MAKER configuration
- `prompt_template` (str, optional): Prompt template
- `system_prompt` (str, optional): System prompt
- `expected_schema` (dict, optional): Expected output schema
- `metadata` (dict, optional): Additional metadata

**Returns:** `ToolDefinition`

##### `update_tool(tool_id, status=None, test_results=None, metadata=None)`

Update an existing tool.

**Parameters:**
- `tool_id` (str): Tool ID
- `status` (ToolStatus, optional): New status
- `test_results` (dict, optional): Test results
- `metadata` (dict, optional): Additional metadata

**Returns:** `bool` - True if successful

##### `list_tools(status_filter=None, maker_mode_filter=None)`

List tools with optional filters.

**Parameters:**
- `status_filter` (ToolStatus, optional): Filter by status
- `maker_mode_filter` (str, optional): Filter by maker mode

**Returns:** `List[ToolDefinition]`

##### `search_tools(query)`

Search tools by name or description.

**Parameters:**
- `query` (str): Search query

**Returns:** `List[ToolDefinition]`

---

### HephaestusDelegationManager

#### Methods

##### `delegate_maker_run(...)`

Delegate a MAKER run to Hephaestus.

```python
def delegate_maker_run(
    self,
    run_id: str,
    title: str,
    description: str,
    initial_state: Any,
    maker_config: MakerConfig,
    workflow_epic_id: Optional[str] = None
) -> Optional[HephaestusDelegation]
```

**Parameters:**
- `run_id` (str): MAKER run ID
- `title` (str): Task title
- `description` (str): Task description
- `initial_state` (Any): Initial state
- `maker_config` (MakerConfig): MAKER configuration
- `workflow_epic_id` (str, optional): Parent workflow epic ID

**Returns:** `HephaestusDelegation` or None

##### `delegate_tool_execution(tool_id, tool_name, input_data, workflow_epic_id=None)`

Delegate a tool execution to Hephaestus.

**Parameters:**
- `tool_id` (str): Tool ID
- `tool_name` (str): Tool name
- `input_data` (dict): Input data
- `workflow_epic_id` (str, optional): Parent workflow epic ID

**Returns:** `HephaestusDelegation` or None

##### `update_delegation_status(delegation_id, status, result=None)`

Update delegation status.

**Parameters:**
- `delegation_id` (str): Delegation ID
- `status` (DelegationStatus): New status
- `result` (dict, optional): Result data

**Returns:** `bool` - True if successful

##### `sync_from_hephaestus()`

Sync delegation statuses from Hephaestus.

**Returns:** `int` - Number of delegations synced

---

### MakerWorkflowManager

#### Methods

##### `create_tool_workflow(...)`

Create a new tool using MAKER workflow.

```python
def create_tool_workflow(
    self,
    name: str,
    description: str,
    task: str,
    maker_mode: str = "recursive",
    k_ahead: int = 3,
    max_depth: int = 5,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[ToolDefinition], Optional[str]]
```

**Parameters:**
- `name` (str): Tool name
- `description` (str): Tool description
- `task` (str): Task to solve
- `maker_mode` (str): "sequential", "recursive", or "hybrid"
- `k_ahead` (int): Voting threshold
- `max_depth` (int): Max decomposition depth
- `context` (dict, optional): Additional context

**Returns:** `Tuple[ToolDefinition, error_message]`

##### `execute_tool_workflow(tool_id, input_data, delegate_to_hephaestus=False)`

Execute a tool workflow.

```python
def execute_tool_workflow(
    self,
    tool_id: str,
    input_data: Dict[str, Any],
    delegate_to_hephaestus: bool = False
) -> Tuple[Optional[ToolExecutionResult], Optional[str]]
```

**Parameters:**
- `tool_id` (str): Tool ID
- `input_data` (dict): Input data
- `delegate_to_hephaestus` (bool): Whether to delegate to Hephaestus

**Returns:** `Tuple[ToolExecutionResult, error_message]`

---

## UI Components

### BubbleLabsMakerUI

Main UI class for Streamlit integration.

#### Usage

```python
import streamlit as st
from bubblelabs_maker_integration import BubbleLabsMakerUI

# Initialize
ui = BubbleLabsMakerUI()

# Render in Streamlit
ui.render_maker_studio()
```

#### Tabs

1. **Tool Creator** - Create new tools with MAKER
2. **Tool Repository** - Browse and manage tools
3. **Tool Executor** - Execute tools
4. **Hephaestus Tracker** - Track delegated tasks
5. **Workflow Analytics** - View analytics

---

## Advanced Usage

### Custom Tool Creation with Full Config

```python
from bubblelabs_maker_integration import MakerWorkflowManager
from maker_integration_bridge import create_maker_config

# Create custom MAKER config
config = create_maker_config(
    mode="hybrid",
    k_ahead=5,
    max_depth=8,
    enable_red_flagging=True,
    enable_caching=True
)

# Create tool with custom config
manager = MakerWorkflowManager()

tool, error = manager.create_tool_workflow(
    name="Advanced Problem Solver",
    description="Solves complex multi-step problems",
    task="Solve this problem using decomposition and voting",
    maker_mode="hybrid",
    k_ahead=5,
    max_depth=8,
    context={
        "enable_caching": True,
        "cache_ttl_seconds": 3600
    }
)
```

### Batch Tool Execution

```python
# Execute multiple tools in batch
tasks = [
    {"task": "Analyze data A", "context": {"source": "db1"}},
    {"task": "Analyze data B", "context": {"source": "db2"}},
    {"task": "Analyze data C", "context": {"source": "db3"}}
]

results = []
for task in tasks:
    result, error = manager.execute_tool_workflow(
        tool_id=tool_id,
        input_data=task,
        delegate_to_hephaestus=True
    )
    if result:
        results.append(result)

print(f"Completed {len(results)} executions")
```

### Hephaestus Integration with Workflows

```python
from bubblelabs_maker_integration import (
    MakerWorkflowManager,
    HephaestusDelegationManager
)
from hephaestus_integration import HephaestusIntegrationManager

# Setup
hephaestus_manager = HephaestusIntegrationManager(...)
delegation_manager = HephaestusDelegationManager(hephaestus_manager)
workflow_manager = MakerWorkflowManager(
    delegation_manager=delegation_manager
)

# Create workflow epic
workflow_epic_id = hephaestus_manager.initialize_workflow_sync(workflow_state)

# Create and delegate tools
for task in tasks:
    tool, _ = workflow_manager.create_tool_workflow(...)

    # Delegate to Hephaestus
    delegation = delegation_manager.delegate_tool_execution(
        tool_id=tool.tool_id,
        tool_name=tool.name,
        input_data={"task": task},
        workflow_epic_id=workflow_epic_id
    )

# Sync status
synced = delegation_manager.sync_from_hephaestus()
```

---

## Examples

### Example 1: Creating a Data Analysis Tool

```python
from bubblelabs_maker_integration import MakerWorkflowManager

# Initialize
manager = MakerWorkflowManager()

# Create data analysis tool
tool, error = manager.create_tool_workflow(
    name="Data Analyzer Pro",
    description="Advanced data analysis with statistical insights",
    task="Analyze the provided dataset and generate insights",
    maker_mode="recursive",
    k_ahead=3,
    max_depth=5,
    context={
        "include_statistics": True,
        "include_visualizations": True
    }
)

if tool:
    print(f"✅ Tool created: {tool.tool_id}")

    # Execute tool
    result, error = manager.execute_tool_workflow(
        tool_id=tool.tool_id,
        input_data={
            "task": "Analyze sales data from Q1 2024",
            "context": {
                "data_format": "csv",
                "metrics": ["revenue", "growth", "churn"]
            }
        }
    )

    if result:
        print(f"✅ Analysis completed in {result.execution_time:.2f}s")
        print(f"Output: {result.output_data}")
```

### Example 2: Multi-Tool Workflow with Hephaestus

```python
from bubblelabs_maker_integration import (
    MakerWorkflowManager,
    HephaestusDelegationManager
)
from hephaestus_integration import HephaestusIntegrationManager

# Setup integrations
hephaestus_manager = HephaestusIntegrationManager(
    api_base="https://hephaestus.example.com",
    api_key="your-key",
    project_id="data-pipeline-2024"
)

delegation_manager = HephaestusDelegationManager(hephaestus_manager)
workflow_manager = MakerWorkflowManager(
    delegation_manager=delegation_manager
)

# Define pipeline
pipeline_tools = []

# Tool 1: Data Ingestion
tool1, _ = workflow_manager.create_tool_workflow(
    name="Data Ingestor",
    description="Ingest data from various sources",
    task="Ingest data from the provided source",
    maker_mode="sequential"
)
pipeline_tools.append(("ingest", tool1))

# Tool 2: Data Transformation
tool2, _ = workflow_manager.create_tool_workflow(
    name="Data Transformer",
    description="Transform and clean data",
    task="Transform the data according to specifications",
    maker_mode="recursive"
)
pipeline_tools.append(("transform", tool2))

# Tool 3: Data Analysis
tool3, _ = workflow_manager.create_tool_workflow(
    name="Data Analyzer",
    description="Analyze transformed data",
    task="Perform analysis on the data",
    maker_mode="recursive"
)
pipeline_tools.append(("analyze", tool3))

# Execute pipeline
data_flow = {"raw_data": "..."}

for stage, tool in pipeline_tools:
    result, error = workflow_manager.execute_tool_workflow(
        tool_id=tool.tool_id,
        input_data={"task": f"Execute {stage} stage", "context": data_flow},
        delegate_to_hephaestus=True
    )

    if result:
        print(f"✅ {stage} completed")
        # Update data flow for next stage
        data_flow[stage] = result.output_data
    else:
        print(f"❌ {stage} failed: {error}")
        break
```

### Example 3: Tool Repository Management

```python
from bubblelabs_maker_integration import ToolRepository, ToolStatus

# Initialize repository
repo = ToolRepository(storage_path="./my_tools.json")

# Register multiple tools
tools = []
for i in range(5):
    tool = repo.register_tool(
        name=f"Calculator Tool {i}",
        description=f"Performs calculations type {i}",
        maker_mode="sequential",
        config={"k_ahead": 2},
        metadata={"category": "calculator"}
    )
    tools.append(tool)

# List all draft tools
draft_tools = repo.list_tools(status_filter=ToolStatus.DRAFT)
print(f"Draft tools: {len(draft_tools)}")

# Validate tools
for tool in draft_tools[:3]:
    repo.update_tool(tool.tool_id, status=ToolStatus.VALIDATED)

# Search for calculator tools
calc_tools = repo.search_tools("calculator")
print(f"Calculator tools: {len(calc_tools)}")

# Get tool details
tool = repo.get_tool(tools[0].tool_id)
print(f"Tool: {tool.name}, Usage: {tool.usage_count}")
```

---

## Troubleshooting

### Issue: "Maker Engine not available"

**Solution:** Ensure Maker Engine is properly installed and imported.

```python
# Check availability
from maker_integration_bridge import MAKER_AVAILABLE

if not MAKER_AVAILABLE:
    print("Maker Engine not available. Check dependencies.")
```

### Issue: "Hephaestus delegation fails"

**Solution:** Verify Hephaestus API credentials and connectivity.

```python
# Test connection
from hephaestus_integration import HephaestusClient

client = HephaestusClient(api_base, api_key, project_id)
# Try creating a test ticket
ticket_id = client.create_ticket("Test", "Test ticket")
if ticket_id:
    print("Hephaestus connection OK")
else:
    print("Check API credentials")
```

### Issue: "Tool execution timeout"

**Solution:** Increase timeout in MAKER config.

```python
config = create_maker_config(
    mode="recursive",
    timeout_seconds=600  # 10 minutes
)
```

---

## Performance Tips

1. **Use Caching**: Enable caching for repeated operations
   ```python
   config = create_maker_config(enable_caching=True)
   ```

2. **Adjust K-Ahead**: Lower k_ahead for faster consensus (less accuracy)
   ```python
   config = create_maker_config(k_ahead=2)  # Faster
   ```

3. **Limit Depth**: Reduce max_depth for simpler tasks
   ```python
   config = create_maker_config(max_depth=3)  # Faster
   ```

4. **Batch Execution**: Execute multiple tools in parallel
   ```python
   from concurrent.futures import ThreadPoolExecutor

   with ThreadPoolExecutor(max_workers=3) as executor:
       futures = [
           executor.submit(execute_tool, tool_id, data)
           for tool_id, data in tasks
       ]
       results = [f.result() for f in futures]
   ```

---

## Best Practices

1. **Tool Design**
   - Keep tools focused on specific tasks
   - Provide clear descriptions
   - Include example inputs/outputs in metadata

2. **Hephaestus Integration**
   - Always delegate long-running tasks
   - Sync status regularly
   - Use workflow epics to group related tasks

3. **Error Handling**
   - Always check error returns
   - Implement retry logic
   - Log failures for debugging

4. **Performance**
   - Use appropriate MAKER modes
   - Adjust k_ahead based on complexity
   - Enable caching when possible

---

## Support

For issues, questions, or contributions:
- GitHub: [Your Repository]
- Documentation: [Link to docs]
- Issues: [Link to issue tracker]

---

## License

[Your License Here]

## Changelog

### Version 1.0.0 (2025-01-03)
- Initial release
- Maker Engine integration
- Hephaestus delegation support
- Tool repository management
- Streamlit UI components
- Comprehensive documentation
