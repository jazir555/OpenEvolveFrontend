# BubbleLabs + Maker + Hephaestus Quick Reference

## 🚀 Quick Start

### Basic Setup (5 lines)

```python
from bubblelabs_maker_integration import create_bubblelabs_maker_integration
import BubbleLab UI as st

# Create UI
ui = create_bubblelabs_maker_integration()

# Render in BubbleLab UI
ui.render_maker_studio()
```

### Initialize with Hephaestus

```python
from hephaestus_integration import HephaestusIntegrationManager
from bubblelabs_maker_integration import create_bubblelabs_maker_integration

# Setup Hephaestus
heph_manager = HephaestusIntegrationManager(
    api_base="https://api.hephaestus.com",
    api_key="your-key",
    project_id="your-project"
)

# Create integration
ui = create_bubblelabs_maker_integration(heph_manager)
```

---

## 🛠️ Core Operations

### Create Tool (3 steps)

```python
from bubblelabs_maker_integration import MakerWorkflowManager

# 1. Initialize
manager = MakerWorkflowManager()

# 2. Create tool
tool, error = manager.create_tool_workflow(
    name="My Tool",
    description="Does something cool",
    task="The task to solve",
    maker_mode="recursive",
    k_ahead=3
)

# 3. Check result
if tool:
    print(f"✅ Created: {tool.tool_id}")
else:
    print(f"❌ Error: {error}")
```

### Execute Tool (2 steps)

```python
# 1. Execute
result, error = manager.execute_tool_workflow(
    tool_id=tool.tool_id,
    input_data={"task": "Do this", "context": {}}
)

# 2. Use result
if result:
    print(f"✅ Success: {result.output_data}")
    print(f"⏱️ Time: {result.execution_time:.2f}s")
else:
    print(f"❌ Error: {error}")
```

### List Tools

```python
from bubblelabs_maker_integration import ToolRepository, ToolStatus

# Initialize
repo = ToolRepository()

# Get all tools
all_tools = repo.list_tools()

# Filter by status
validated = repo.list_tools(status_filter=ToolStatus.VALIDATED)

# Search
results = repo.search_tools("analysis")
```

---

## 📋 Hephaestus Operations

### Delegate Task

```python
from bubblelabs_maker_integration import HephaestusDelegationManager

# Initialize
delegation_mgr = HephaestusDelegationManager(hephaestus_manager)

# Delegate
delegation = delegation_mgr.delegate_tool_execution(
    tool_id="tool_123",
    tool_name="My Tool",
    input_data={"task": "Execute this"}
)

# Track
print(f"Ticket ID: {delegation.task_id}")
print(f"Status: {delegation.status}")
```

### Sync Status

```python
# Sync from Hephaestus
synced = delegation_mgr.sync_from_hephaestus()
print(f"Synced {synced} delegation(s)")

# List delegations
pending = delegation_mgr.list_delegations(
    status_filter=DelegationStatus.PENDING
)
```

---

## 🎨 Maker Modes

### Sequential (Step-by-Step)

```python
config = create_maker_config(
    mode="sequential",
    k_ahead=3,
    max_steps=1000
)
```

**Use for:**
- Predetermined workflows
- Linear processes
- Known step sequences

### Recursive (Decomposition)

```python
config = create_maker_config(
    mode="recursive",
    k_ahead=3,
    max_depth=5
)
```

**Use for:**
- Complex problems
- Unknown solutions
- Hierarchical tasks

### Hybrid (Combined)

```python
config = create_maker_config(
    mode="hybrid",
    k_ahead=3,
    max_depth=5,
    enable_roma=True
)
```

**Use for:**
- Very complex tasks
- ROMA decomposition
- Maximum accuracy

---

## 📊 Data Structures

### ToolDefinition

```python
{
    "tool_id": "tool_123",
    "name": "My Tool",
    "description": "Does something",
    "version": "1.0.0",
    "status": "validated",
    "maker_mode": "recursive",
    "config": {...},
    "created_at": "2025-01-03T...",
    "usage_count": 5
}
```

### HephaestusDelegation

```python
{
    "delegation_id": "del_456",
    "task_id": "ticket_789",
    "title": "Execute Tool",
    "status": "in_progress",
    "delegation_type": "custom_tool",
    "created_at": "2025-01-03T...",
    "result": {...}
}
```

### ToolExecutionResult

```python
{
    "tool_id": "tool_123",
    "execution_id": "exec_999",
    "input_data": {...},
    "output_data": {...},
    "execution_time": 12.34,
    "success": true,
    "metrics": {...},
    "hephaestus_ticket_id": "ticket_789"
}
```

---

## 🎯 Common Patterns

### Pattern 1: Tool Factory

```python
def create_analysis_tool(name, data_type):
    """Create a data analysis tool"""
    manager = MakerWorkflowManager()

    return manager.create_tool_workflow(
        name=name,
        description=f"Analyzes {data_type} data",
        task=f"Analyze {data_type} dataset",
        maker_mode="recursive",
        k_ahead=3,
        context={"data_type": data_type}
    )

# Use
sales_tool, _ = create_analysis_tool("Sales Analyzer", "sales")
hr_tool, _ = create_analysis_tool("HR Analyzer", "hr")
```

### Pattern 2: Batch Execution

```python
def execute_batch(tool_id, tasks):
    """Execute tool on multiple tasks"""
    manager = MakerWorkflowManager()
    results = []

    for task in tasks:
        result, _ = manager.execute_tool_workflow(
            tool_id=tool_id,
            input_data={"task": task}
        )
        if result:
            results.append(result)

    return results

# Use
tasks = ["Analyze Q1", "Analyze Q2", "Analyze Q3"]
results = execute_batch(tool_id, tasks)
```

### Pattern 3: Pipeline Execution

```python
def execute_pipeline(tools_and_data):
    """Execute tools in sequence"""
    manager = MakerWorkflowManager()
    data = tools_and_data

    for stage_name, tool_id in data["pipeline"]:
        result, _ = manager.execute_tool_workflow(
            tool_id=tool_id,
            input_data={"task": stage_name, "context": data}
        )
        if result:
            data[stage_name] = result.output_data
        else:
            break

    return data

# Use
pipeline_data = {
    "pipeline": [
        ("ingest", tool_id_1),
        ("transform", tool_id_2),
        ("analyze", tool_id_3)
    ],
    "source": "database"
}

result = execute_pipeline(pipeline_data)
```

---

## 🔧 Configuration Parameters

### MAKER Config

```python
config = create_maker_config(
    # Core parameters
    mode="recursive",           # "sequential", "recursive", "hybrid"
    k_ahead=3,                  # Voting threshold (1-10)
    max_depth=5,                # Decomposition depth (1-10)

    # Red-flagging
    enable_red_flagging=True,   # Enable filtering
    max_token_length=750,       # Max response length

    # Execution
    max_steps=1000,             # For sequential mode
    timeout_seconds=300,        # Execution timeout

    # ROMA integration
    enable_roma=False,          # Use ROMA decomposition
    roma_max_depth=3,           # ROMA depth

    # Caching
    enable_caching=True,        # Cache results
    cache_ttl_seconds=3600,     # Cache TTL

    # Provider
    provider="openai",          # LLM provider
    model="gpt-4o-mini",        # Model name
    temperature_first=0.0,      # First vote temp
    temperature_subsequent=0.1  # Subsequent vote temp
)
```

### Hephaestus Config

```python
heph_manager = HephaestusIntegrationManager(
    api_base="https://api.hephaestus.com",  # API endpoint
    api_key="your-key",                       # Authentication
    project_id="your-project"                 # Project ID
)
```

---

## 📈 Status Enums

### ToolStatus

```python
ToolStatus.DRAFT        # Initial state
ToolStatus.TESTING      # Under testing
ToolStatus.VALIDATED    # Ready for use
ToolStatus.DEPLOYED     # Production ready
ToolStatus.DEPRECATED   # Deprecated
```

### DelegationStatus

```python
DelegationStatus.PENDING      # Waiting to start
DelegationStatus.ASSIGNED     # Assigned to worker
DelegationStatus.IN_PROGRESS  # Currently executing
DelegationStatus.REVIEW       # Under review
DelegationStatus.COMPLETE     # Successfully completed
DelegationStatus.FAILED       # Failed
```

---

## 🐛 Troubleshooting

### Problem: Tool creation fails

```python
# Check MAKER availability
from maker_integration_bridge import MAKER_AVAILABLE

if not MAKER_AVAILABLE:
    print("❌ Maker Engine not installed")
else:
    print("✅ Maker Engine available")
```

### Problem: Hephaestus connection fails

```python
# Test Hephaestus connection
from hephaestus_integration import HephaestusClient

client = HephaestusClient(api_base, api_key, project_id)
test_ticket = client.create_ticket("Test", "Test connection")

if test_ticket:
    print("✅ Hephaestus connection OK")
else:
    print("❌ Check Hephaestus credentials")
```

### Problem: Tool execution times out

```python
# Increase timeout
config = create_maker_config(
    timeout_seconds=600  # 10 minutes
)
```

---

## 💡 Tips

### Performance

1. **Lower k_ahead** for faster execution (less accuracy)
   ```python
   config = create_maker_config(k_ahead=2)  # Fast
   ```

2. **Enable caching** for repeated operations
   ```python
   config = create_maker_config(enable_caching=True)
   ```

3. **Reduce depth** for simpler tasks
   ```python
   config = create_maker_config(max_depth=3)  # Faster
   ```

### Quality

1. **Higher k_ahead** for better consensus
   ```python
   config = create_maker_config(k_ahead=5)  # More accurate
   ```

2. **Enable red-flagging** for reliability
   ```python
   config = create_maker_config(enable_red_flagging=True)
   ```

3. **Use recursive mode** for complex tasks
   ```python
   config = create_maker_config(mode="recursive")
   ```

---

## 📝 Examples Repository

### Example 1: Simple Tool

```python
# Create and execute a simple text analyzer
manager = MakerWorkflowManager()

tool, _ = manager.create_tool_workflow(
    name="Sentiment Analyzer",
    description="Analyzes sentiment in text",
    task="Analyze sentiment of the provided text",
    maker_mode="sequential"
)

result, _ = manager.execute_tool_workflow(
    tool_id=tool.tool_id,
    input_data={"task": "Analyze: I love this product!"}
)

print(result.output_data)
```

### Example 2: Complex Tool

```python
# Create a complex problem solver
manager = MakerWorkflowManager()

tool, _ = manager.create_tool_workflow(
    name="Optimization Solver",
    description="Solves complex optimization problems",
    task="Optimize the provided problem",
    maker_mode="recursive",
    k_ahead=5,
    max_depth=7,
    context={
        "objective": "minimize_cost",
        "constraints": ["time", "resources"]
    }
)

result, _ = manager.execute_tool_workflow(
    tool_id=tool.tool_id,
    input_data={
        "task": "Optimize delivery routes",
        "context": {"vehicles": 10, "locations": 50}
    }
)
```

### Example 3: Hephaestus Workflow

```python
# Create and delegate tool with Hephaestus tracking
heph_manager = HephaestusIntegrationManager(...)
delegation_mgr = HephaestusDelegationManager(heph_manager)

workflow_manager = MakerWorkflowManager(
    delegation_manager=delegation_mgr
)

tool, _ = workflow_manager.create_tool_workflow(
    name="Report Generator",
    description="Generate reports from data",
    task="Generate a report",
    maker_mode="recursive"
)

# Execute with Hephaestus delegation
result, _ = workflow_manager.execute_tool_workflow(
    tool_id=tool.tool_id,
    input_data={"task": "Generate Q1 report"},
    delegate_to_hephaestus=True
)

# Check delegation status
if result.hephaestus_ticket_id:
    print(f"Tracking in Hephaestus: {result.hephaestus_ticket_id}")
```

---

## 🔗 Related Documentation

- **Full API Documentation**: `BUBBLELABS_MAKER_HEPHAEUSTUS_DOCUMENTATION.md`
- **Maker Engine Paper**: arXiv:2511.09030
- **Hephaestus Integration**: `hephaestus_integration.py`
- **ROMA MDAP**: `mdap_maker_complete.py`

---

## 🆘 Getting Help

1. **Check Logs**: Look at `openevolve.log` for errors
2. **Verify Dependencies**: Ensure all imports are available
3. **Test Connection**: Verify Hephaestus API access
4. **Review Config**: Check MAKER configuration parameters

---

## ✅ Checklist

### Before Using

- [ ] Maker Engine imported successfully
- [ ] Hephaestus configured (if using)
- [ ] Tool repository initialized
- [ ] API credentials set (if needed)

### Creating Tools

- [ ] Descriptive tool name
- [ ] Clear task description
- [ ] Appropriate MAKER mode
- [ ] Reasonable k_ahead value
- [ ] Suitable max_depth

### Executing Tools

- [ ] Valid tool ID
- [ ] Proper input format
- [ ] Hephaestus enabled (if tracking)
- [ ] Timeout configured

---

**Quick Reference Version**: 1.0.0
**Last Updated**: 2025-01-03

