# BubbleLabs-LeanAide Quick Reference Card

## 🚀 Quick Start (30 seconds)

```python
# 1. Import
from bubblelabs_leanaide_ui import render_leanaide_in_bubblelabs

# 2. Add to BubbleLabs
render_leanaide_in_bubblelabs()

# Done! 🎉
```

## 📦 Files Created

| File | Purpose |
|------|---------|
| `bubblelabs_leanaide_integration.py` | Core integration bridge |
| `bubblelabs_leanaide_ui.py` | BubbleLab UI UI components |
| `bubblelabs_leanaide_examples.py` | Example workflows |
| `BUBBLELABS_LEANAIDE_INTEGRATION_GUIDE.md` | Complete documentation |
| `BUBBLELABS_LEANAIDE_IMPLEMENTATION_SUMMARY.md` | Implementation summary |
| `BUBBLELABS_LEANAIDE_QUICK_REFERENCE.md` | This file |

## 🔧 Key Classes

### LeanAideIntegrationBridge
```python
from bubblelabs_leanaide_integration import get_leanaide_bridge

bridge = get_leanaide_bridge()
result = bridge.execute_task(task_type, **kwargs)
```

### LeanAideUIComponent
```python
from bubblelabs_leanaide_ui import LeanAideUIComponent

ui = LeanAideUIComponent()
ui.render_leanaide_control_panel()
```

## 🎯 Task Types

```python
from bubblelabs_leanaide_integration import LeanAideTaskType

LeanAideTaskType.TRANSLATE_THEOREM    # Natural language → Lean
LeanAideTaskType.GENERATE_PROOF       # Generate proof
LeanAideTaskType.VERIFY_SOLUTION      # Verify Lean code
LeanAideTaskType.MATH_QUERY           # Math Q&A
LeanAideTaskType.ELABORATE_CODE       # Type check code
LeanAideTaskType.MCTS_SEARCH          # MCTS proof search
```

## 💡 Common Patterns

### Translate Theorem
```python
result = bridge.execute_task(
    LeanAideTaskType.TRANSLATE_THEOREM,
    theorem_text="There are infinitely many primes",
    theorem_name="inf_primes"
)
lean_code = result.data['lean_code']
```

### Generate Proof
```python
result = bridge.execute_task(
    LeanAideTaskType.GENERATE_PROOF,
    theorem_text="sqrt(2) is irrational"
)
proof = result.data['proof_document']
```

### MCTS Search
```python
result = bridge.execute_task(
    LeanAideTaskType.MCTS_SEARCH,
    theorem="forall (n m : Nat), n + m = m + n",
    max_iterations=1000,
    time_budget=60.0
)
tree = bridge.get_tree(result.visualization_data['tree_id'])
```

### Verify Code
```python
result = bridge.execute_task(
    LeanAideTaskType.VERIFY_SOLUTION,
    code="theorem add_comm (a b : Nat) : a + b = b + a := by rfl"
)
is_valid = result.data['is_valid']
```

### Math Query
```python
result = bridge.execute_task(
    LeanAideTaskType.MATH_QUERY,
    query="What is the fundamental theorem of calculus?",
    n=3
)
answers = result.data['answers']
```

## 🌳 MCTS Visualization

```python
# Get tree
tree_id = result.visualization_data['tree_id']
tree = bridge.get_tree(tree_id)

# Access data
print(f"Nodes: {len(tree.nodes)}")
print(f"Win rate: {tree.statistics['win_rate']:.2%}")
print(f"Best path: {tree.best_path}")

# Export
import json
with open('tree.json', 'w') as f:
    json.dump(tree.to_dict(), f, indent=2)
```

## ✅ Lean4 Proof Tracking

```python
# Get proof
proof_id = result.visualization_data['proof_id']
proof = bridge.get_proof(proof_id)

# Access steps
for step in proof.steps:
    print(f"Step {step.step_number}: {step.tactic}")
    print(f"  Goals: {step.goals_after}")
    print(f"  Valid: {step.is_valid}")
```

## 🔍 Status Check

```python
# System status
status = bridge.get_status()
print(f"MCTS available: {status['mcts_available']}")
print(f"Server: {status['server']}")

# Execution history
history = bridge.get_execution_history(limit=10)
for result in history:
    print(f"{result.task_type}: {result.success}")
```

## ⚙️ Configuration

```python
# Environment variables
export LEANAIDE_HOST="localhost"
export LEANAIDE_PORT="7654"

# Programmatic
bridge = get_leanaide_bridge()
bridge.leanaide_host = "localhost"
bridge.leanaide_port = 7654
bridge.enable_mcts = True
```

## 🎨 UI Integration

```python
import BubbleLab UI as st
from bubblelabs_leanaide_ui import LeanAideUIComponent

# In BubbleLabs app
def main():
    st.title("My Workflow")

    # Add LeanAide tab
    tabs = st.tabs(["Workflows", "LeanAide"])

    with tabs[1]:
        ui = LeanAideUIComponent()
        ui.render_leanaide_control_panel()

if __name__ == "__main__":
    main()
```

## 📊 Visualization Data Structures

### MCTSNodeVisualization
```python
{
    "node_id": "str",
    "parent_id": "str | None",
    "action": "str",
    "visits": "int",
    "value": "float",
    "win_rate": "float",
    "depth": "int",
    "children": ["str"],  # List of child IDs
    "agent_votes": ["dict"],
    "red_flagged": "bool"
}
```

### Lean4ProofStep
```python
{
    "step_id": "str",
    "step_number": "int",
    "tactic": "str",
    "goals_before": ["str"],
    "goals_after": ["str"],
    "is_valid": "bool",
    "error_message": "str | None"
}
```

## 🧪 Running Examples

```bash
# All examples
python bubblelabs_leanaide_examples.py

# Specific example
python bubblelabs_leanaide_examples.py basic      # Basic theorem proving
python bubblelabs_leanaide_examples.py mcts       # MCTS search
python bubblelabs_leanaide_examples.py verify     # Verification
python bubblelabs_leanaide_examples.py math       # Math queries
python bubblelabs_leanaide_examples.py batch      # Batch processing
python bubblelabs_leanaide_examples.py complete   # Full workflow
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "LeanAide not available" | `pip install leanaide-client` |
| "Connection refused" | Start LeanAide server |
| "MCTS not available" | `pip install leanaide-mcts-mdap` |
| Import error | Check Python path |
| Timeout | Increase `time_budget` parameter |

## 📈 Performance

| Task | Time |
|------|------|
| Translate | 2-10s |
| Prove | 10-60s |
| Verify | 1-5s |
| Query | 3-15s |
| MCTS (1000) | 60-300s |

## 🔗 Links

- **Full Guide**: `BUBBLELABS_LEANAIDE_INTEGRATION_GUIDE.md`
- **Implementation**: `BUBBLELABS_LEANAIDE_IMPLEMENTATION_SUMMARY.md`
- **Examples**: `bubblelabs_leanaide_examples.py`

## ✅ Requirements

```
BubbleLab UI>=1.20.0
pandas>=1.5.0
asyncio
aiohttp>=3.8.0
leanaide-client (optional)
leanaide-mcts-mdap (optional)
```

## 🎓 Best Practices

1. **Always check availability** before using features
2. **Handle errors gracefully** with try-except
3. **Use appropriate timeouts** for tasks
4. **Clean up resources** when done: `bridge.cleanup()`
5. **Monitor history** for debugging

## 🚀 Production Checklist

- [ ] Install dependencies
- [ ] Start LeanAide server
- [ ] Configure environment variables
- [ ] Test with examples
- [ ] Integrate into BubbleLabs
- [ ] Monitor performance
- [ ] Set up error handling
- [ ] Configure logging

## 📞 Support

```python
# Get help
status = bridge.get_status()
print(status)

# Check logs
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

**Version**: 1.0.0
**Date**: 2025-01-03
**Author**: OpenEvolve
**License**: See main project LICENSE

