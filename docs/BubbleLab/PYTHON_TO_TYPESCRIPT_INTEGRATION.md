# Python Nodes to TypeScript Bubbles Integration Guide

## Overview

This document explains how the **50+ Python backend nodes** integrate with **BubbleLabs TypeScript** to form a coherent system.

---

## Architecture

### The Pattern: Backend Services as "Bubbles"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BUBBLELABS UI (TypeScript)                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Canvas (React)          │  Workflow Engine   │  Node Registry        │  │
│  │  • Drag/drop bubbles     │  • Execution graph │  • Type definitions   │  │
│  │  • Visual connections    │  • State management│  • UI components      │  │
│  └─────────────────────────┬┴───────────────────┴──────────────────────┘  │
│                            │                                              │
│                            │  HTTP / MCP / WebSocket                      │
└────────────────────────────┼──────────────────────────────────────────────┘
                             │
┌────────────────────────────┼──────────────────────────────────────────────┐
│                    PYTHON BACKEND (OpenEvolve + Math)                      │
│                            │                                               │
│  ┌─────────────────────────┴───────────────────────┐                       │
│  │         MCP Server / REST API Gateway           │                       │
│  │  • Exposes Python nodes as remote procedures    │                       │
│  │  • Handles TypeScript → Python calls            │                       │
│  │  • Returns JSON-serializable results            │                       │
│  └─────────────────────────┬───────────────────────┘                       │
│                            │                                               │
│  ┌─────────────────────────┴───────────────────────┐                       │
│  │         BubbleLabsNode Classes (50+)            │                       │
│  │  • LeanAutoformalizationNode                    │                       │
│  │  • MathWorkflowOrchestratorNode                 │                       │
│  │  • OpenEvolveMathBridgeNode                     │                       │
│  │  • KnowledgeExtractionNode                      │                       │
│  │  • ... and 46 more                              │                       │
│  └─────────────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Pattern?

| TypeScript Bubble | Python Node | Purpose |
|-------------------|-------------|---------|
| UI/UX layer | Backend intelligence | Separation of concerns |
| State management | Algorithm execution | Performance optimization |
| Type safety | Rich Python ecosystem | Best of both worlds |
| Visual workflow | Complex computation | User-friendly + powerful |

---

## Key Insight

**The Python files in `bubblelabs_nodes/` are NOT TypeScript bubbles directly.**

Instead, they are:

1. **Backend service implementations** that follow a BubbleLabs-compatible interface
2. **Exposed via MCP/REST** as callable tools from TypeScript
3. **Wrapped by thin TypeScript adapters** that integrate with the BubbleLabs UI

---

## How Integration Works

### Step 1: Python Node Implementation

```python
# bubblelabs_nodes/lean_autoformalization_node.py

class LeanAutoformalizationNode(BubbleLabsNode):
    """Python backend implementation"""
    
    DISPLAY_NAME = "Lean Autoformalization"
    CATEGORY = "mathematical_verification"
    
    def execute(self, inputs: Dict, context) -> Dict:
        text = inputs.get("text", "")
        # Complex Python logic here
        lean_code = self._autoformalize(text)
        return {"lean_code": lean_code, "success": True}
    
    def get_parameter_schema(self) -> Dict:
        # This schema is used by TypeScript to generate UI
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "strategy": {"type": "string", "enum": ["direct", "mdap"]}
            }
        }
```

### Step 2: MCP Server Exposure

```python
# mcp_server.py - Exposes Python nodes to TypeScript

from mcp.server import Server
from bubblelabs_nodes.lean_autoformalization_node import LeanAutoformalizationNode

server = Server("bubblelab-python-nodes")

@server.tool()
async def lean_autoformalize(text: str, strategy: str = "adaptive") -> dict:
    """Autoformalize natural language to Lean 4"""
    node = LeanAutoformalizationNode(config={"strategy": strategy})
    return node.execute({"text": text}, context=MockContext())

@server.tool()
async def math_classify_problem(problem: str) -> dict:
    """Classify mathematical problem"""
    from bubblelabs_nodes.math_problem_classification_node import MathProblemClassificationNode
    node = MathProblemClassificationNode(config={})
    return node.execute({"problem": problem, "operation": "classify"}, context=MockContext())
```

### Step 3: TypeScript Bubble Wrapper

```typescript
// bubbles/MathBubbles.ts

export const LeanAutoformalizationBubble = defineBubble({
  id: 'lean_autoformalization',
  name: 'Lean Autoformalization',
  category: 'Mathematical Verification',
  
  // Generated from Python node's get_parameter_schema()
  parameters: {
    text: {
      type: 'string',
      label: 'Natural Language Text',
      description: 'Mathematical statement to formalize'
    },
    strategy: {
      type: 'select',
      label: 'Strategy',
      options: ['direct', 'mdap', 'maker', 'hybrid', 'adaptive'],
      default: 'adaptive'
    }
  },
  
  // Input/output ports
  inputs: {
    text: { type: 'string', label: 'Text to formalize' }
  },
  outputs: {
    lean_code: { type: 'string', label: 'Lean 4 Code' },
    success: { type: 'boolean', label: 'Success' }
  },
  
  // Execution calls Python backend
  async execute({ inputs, config }) {
    const result = await callMcpTool('lean_autoformalize', {
      text: inputs.text,
      strategy: config.strategy
    });
    return result;
  }
});
```

### Step 4: Registration

```typescript
// registry.ts
import { LeanAutoformalizationBubble } from './bubbles/MathBubbles';

bubbleRegistry.register(LeanAutoformalizationBubble);
```

---

## Communication Protocol

### MCP (Model Context Protocol) - Recommended

```yaml
# mcp_config.json
{
  "mcpServers": {
    "bubblelab-python": {
      "command": "python",
      "args": ["-m", "bubblelab_mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/openevolve"
      }
    }
  }
}
```

### REST API - Alternative

```python
# api_server.py
from fastapi import FastAPI
from bubblelabs_nodes.lean_autoformalization_node import LeanAutoformalizationNode

app = FastAPI()

@app.post("/api/nodes/lean_autoformalization")
async def lean_autoformalization_endpoint(request: dict):
    node = LeanAutoformalizationNode(config=request.get("config", {}))
    return node.execute(request.get("inputs", {}), context=MockContext())
```

```typescript
// TypeScript client
const result = await fetch('/api/nodes/lean_autoformalization', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    inputs: { text: "n + 0 = n" },
    config: { strategy: "adaptive" }
  })
});
```

---

## Why Not Direct TypeScript?

| Requirement | Python Solution | TypeScript Alternative |
|-------------|-----------------|------------------------|
| Lean 4 integration | `leanaide_client` | Would need port |
| Z3 SMT solver | `z3-solver` Python lib | `z3` npm (limited) |
| Math libraries | SciPy, NumPy, SymPy | Limited equivalents |
| ML/AI models | PyTorch, TensorFlow | TensorFlow.js (limited) |
| Knowledge graphs | NetworkX, custom | Would need rewrite |

**Benefits of Python backend:**
- ✅ Rich mathematical ecosystem
- ✅ Existing Lean 4 integration
- ✅ Z3 Python bindings
- ✅ ML/AI libraries
- ✅ Can be reused outside BubbleLabs

---

## Data Flow Example

```
User Action:
  User drops "Lean Autoformalization" bubble onto canvas
       │
       ▼
TypeScript Layer:
  BubbleLabs renders configuration panel
  (generated from JSON schema sent by Python)
       │
       ▼
User Configuration:
  User enters: "For all n, n + 0 = n"
  Selects: strategy = "adaptive"
       │
       ▼
Execution Trigger:
  User clicks "Run" or workflow executes
       │
       ▼
MCP Call:
  TypeScript → MCP Server → Python Node
  {
    "tool": "lean_autoformalize",
    "params": {
      "text": "For all n, n + 0 = n",
      "strategy": "adaptive"
    }
  }
       │
       ▼
Python Processing:
  LeanAutoformalizationNode.execute()
  → Calls LeanAide server
  → Generates Lean code
  → Returns result
       │
       ▼
Response:
  Python → MCP → TypeScript
  {
    "lean_code": "theorem add_zero : ∀ n : Nat, n + 0 = n := by...",
    "success": true
  }
       │
       ▼
TypeScript Display:
  Shows result in bubble output port
  User can connect to next bubble
```

---

## Configuration Mapping

### Python → TypeScript Schema Mapping

| Python Schema | TypeScript UI | Example |
|---------------|---------------|---------|
| `{"type": "string"}` | Text input | `"text": "hello"` |
| `{"type": "number"}` | Number input | `"timeout": 30` |
| `{"type": "boolean"}` | Checkbox | `"verbose": true` |
| `{"type": "string", "enum": [...]}` | Dropdown | `"strategy": "adaptive"` |
| `{"type": "array"}` | List input | `"tags": ["a", "b"]` |
| `{"type": "object"}` | Nested form | `"config": {...}` |

### Example Schema Conversion

**Python (source of truth):**
```python
def get_parameter_schema(self) -> Dict:
    return {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["translate_theorem", "translate_definition", "elaborate"],
                "default": "translate_theorem"
            },
            "text": {"type": "string"},
            "confidence_threshold": {
                "type": "number",
                "default": 0.8,
                "minimum": 0.0,
                "maximum": 1.0
            }
        }
    }
```

**TypeScript (generated):**
```typescript
parameters: {
  operation: {
    type: 'select',
    label: 'Operation',
    options: ['translate_theorem', 'translate_definition', 'elaborate'],
    default: 'translate_theorem'
  },
  text: {
    type: 'string',
    label: 'Text',
    multiline: true
  },
  confidence_threshold: {
    type: 'number',
    label: 'Confidence Threshold',
    default: 0.8,
    min: 0,
    max: 1,
    step: 0.1
  }
}
```

---

## Development Workflow

### Adding a New Node

1. **Create Python Node:**
   ```bash
   # bubblelabs_nodes/my_new_node.py
   ```

2. **Add to MCP Server:**
   ```python
   # mcp_server.py
   @server.tool()
   async def my_new_node_func(...): ...
   ```

3. **Generate TypeScript Wrapper:**
   ```bash
   npm run generate:bubbles
   # or manually create TypeScript bubble
   ```

4. **Register in BubbleLabs:**
   ```typescript
   // registry.ts
   bubbleRegistry.register(MyNewBubble);
   ```

---

## Summary

| Component | Language | Purpose |
|-----------|----------|---------|
| BubbleLabs UI | TypeScript | Visual workflow editor, canvas, state management |
| Bubble Wrappers | TypeScript | Thin adapters that call Python backend |
| Node Logic | Python | Complex algorithms, math, ML, verification |
| MCP Server | Python | Bridge between TypeScript and Python |

**The Python "nodes" are backend services** that implement the BubbleLabs node interface pattern, allowing them to be seamlessly integrated into the TypeScript-based BubbleLabs workflow system.

---

## Next Steps

See related documentation:
- `MCP_INTEGRATION_GUIDE.md` - Setting up MCP server
- `TYPESCRIPT_WRAPPER_EXAMPLES.md` - TypeScript bubble examples
- `JSON_SCHEMA_TO_TYPESCRIPT.md` - Schema mapping details
- `NODE_ARCHITECTURE_EXPLAINED.md` - Deep dive into node pattern
