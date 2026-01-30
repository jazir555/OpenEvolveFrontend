# BubbleLabs Nodes - Quick Reference Guide for Agents

**Purpose**: Quick reference for implementing OpenEvolve nodes for BubbleLabs integration

---

## 🎯 Quick Start

### What You're Doing
Wrapping 8 OpenEvolve components as standardized BubbleLabs nodes with:
- Consistent interface
- Error handling
- Progress reporting
- State management
- Parameter configuration

### Where to Start
1. Read the main task document: `BUBBLELABS_INTEGRATION_TASKS.md`
2. Review existing code in the referenced files
3. Implement nodes one by one following the pattern
4. Test each node as you go
5. Update this checklist

---

## 📦 The 8 Nodes - At a Glance

| # | Node Name | Purpose | Key File | Complexity |
|---|-----------|---------|----------|------------|
| 1 | DecompositionNode | Break down problems | `decomposition_engine.py` | ⭐⭐⭐⭐ |
| 2 | SubProblemNode | Manage sub-problems | `workflow_structures.py` | ⭐⭐⭐ |
| 3 | GauntletNode | Quality control | `gauntlet_manager.py` | ⭐⭐⭐⭐⭐ |
| 4 | SolutionNode | Generate solutions | `solution_orchestration.py` | ⭐⭐⭐⭐ |
| 5 | VerificationNode | Verify correctness | `verification_engine.py` | ⭐⭐⭐⭐⭐ |
| 6 | AssemblyNode | Merge solutions | `solution_assembly.py` | ⭐⭐⭐ |
| 7 | OutputNode | Generate SOPs | `sop_generator.py` | ⭐⭐ |
| 8 | KnowledgeExtractionNode | Extract artifacts | `workflow_knowledge_extractor.py` | ⭐⭐⭐ |

---

## 🏗️ Implementation Pattern

### Step 1: Create Base Class
```python
# File: bubblelabs_nodes/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from workflow_structures import WorkflowState

class BubbleLabsNode(ABC):
    """Base class for all BubbleLabs nodes"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.status = "initialized"

    @abstractmethod
    def execute(self, inputs: Dict, context: WorkflowState) -> Dict:
        """Main execution method"""
        pass

    @abstractmethod
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate inputs, return empty list if valid"""
        pass

    @abstractmethod
    def get_parameter_schema(self) -> Dict:
        """Return JSON schema of parameters"""
        pass

    def get_display_name(self) -> str:
        """Human-readable name"""
        return self.__class__.__name__

    def get_icon(self) -> str:
        """Icon name for UI"""
        return "default-node"

    def get_category(self) -> str:
        """Category for organization"""
        return "general"
```

### Step 2: Implement Each Node
```python
# File: bubblelabs_nodes/decomposition_node.py

from .base import BubbleLabsNode
from decomposition_engine import DecompositionEngine
from typing import Dict, Any, List

class DecompositionNode(BubbleLabsNode):
    """Node for decomposing complex problems"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.engine = DecompositionEngine()

    def execute(self, inputs: Dict, context: WorkflowState) -> Dict:
        """
        Execute decomposition

        Args:
            inputs: Must contain 'problem_statement'
            context: Workflow state for tracking

        Returns:
            Dict with sub_problems, decomposition_tree, etc.
        """
        # 1. Validate inputs
        errors = self.validate_inputs(inputs)
        if errors:
            raise ValueError(f"Invalid inputs: {errors}")

        # 2. Update context
        context.update_progress(0, "Starting decomposition")

        # 3. Execute core logic
        result = self.engine.decompose(
            problem_statement=inputs['problem_statement'],
            method=inputs.get('method', 'roma'),
            constraints=inputs.get('constraints'),
            max_depth=inputs.get('max_depth', 3)
        )

        # 4. Update context with progress
        context.update_progress(100, "Decomposition complete")
        context.add_artifact('decomposition_result', result)

        # 5. Return standardized output
        return {
            'sub_problems': result.sub_problems,
            'decomposition_tree': result.tree,
            'complexity_metrics': result.metrics,
            'estimated_time': result.estimated_time
        }

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate required inputs"""
        errors = []

        if 'problem_statement' not in inputs:
            errors.append("Missing required field: problem_statement")

        if not isinstance(inputs.get('problem_statement'), str):
            errors.append("problem_statement must be a string")

        if 'method' in inputs and inputs['method'] not in ['roma', 'maker', 'mdap']:
            errors.append("method must be one of: roma, maker, mdap")

        return errors

    def get_parameter_schema(self) -> Dict:
        """JSON schema for configuration UI"""
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["roma", "maker", "mdap"],
                    "default": "roma",
                    "description": "Decomposition method to use"
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                    "description": "Maximum depth of decomposition tree"
                },
                "parallel": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable parallel processing"
                }
            }
        }

    def get_display_name(self) -> str:
        return "Problem Decomposition"

    def get_icon(self) -> str:
        return "decomposition-icon"

    def get_category(self) -> str:
        return "analysis"
```

### Step 3: Register Node
```python
# File: bubblelabs_nodes/registry.py

from .decomposition_node import DecompositionNode
from .subproblem_node import SubProblemNode
# ... import other nodes

NODE_REGISTRY = {
    'decomposition': DecompositionNode,
    'subproblem': SubProblemNode,
    'gauntlet': GauntletNode,
    'solution': SolutionNode,
    'verification': VerificationNode,
    'assembly': AssemblyNode,
    'output': OutputNode,
    'knowledge_extraction': KnowledgeExtractionNode
}

def get_node(node_type: str, config: Dict = None) -> BubbleLabsNode:
    """Factory function to create node instances"""
    node_class = NODE_REGISTRY.get(node_type)
    if not node_class:
        raise ValueError(f"Unknown node type: {node_type}")
    return node_class(config)
```

---

## 🧪 Testing Pattern

### Unit Test
```python
# File: tests/test_decomposition_node.py

import pytest
from bubblelabs_nodes.decomposition_node import DecompositionNode
from workflow_structures import WorkflowState

def test_decomposition_node_execute():
    """Test basic execution"""
    node = DecompositionNode()
    context = WorkflowState()

    inputs = {
        'problem_statement': 'Solve climate change',
        'method': 'roma',
        'max_depth': 2
    }

    result = node.execute(inputs, context)

    assert 'sub_problems' in result
    assert 'decomposition_tree' in result
    assert len(result['sub_problems']) > 0
    assert context.progress == 100

def test_decomposition_node_validation():
    """Test input validation"""
    node = DecompositionNode()

    # Missing required field
    result = node.validate_inputs({})
    assert len(result) > 0
    assert 'problem_statement' in result[0]

    # Valid inputs
    result = node.validate_inputs({
        'problem_statement': 'Test problem',
        'method': 'roma'
    })
    assert len(result) == 0

def test_decomposition_node_error_handling():
    """Test error handling"""
    node = DecompositionNode()
    context = WorkflowState()

    with pytest.raises(ValueError):
        node.execute({}, context)
```

### Integration Test
```python
# File: tests/test_node_integration.py

from bubblelabs_nodes.registry import get_node
from workflow_structures import WorkflowState

def test_decomposition_to_subproblem_flow():
    """Test connecting decomposition to subproblem nodes"""
    # Create nodes
    decomp_node = get_node('decomposition')
    subprob_node = get_node('subproblem')

    context = WorkflowState()

    # Execute decomposition
    decomp_result = decomp_node.execute({
        'problem_statement': 'Build a house'
    }, context)

    # Execute subproblem with first result
    subprob_result = subprob_node.execute({
        'sub_problem': decomp_result['sub_problems'][0]
    }, context)

    assert 'solution' in subprob_result
```

---

## ✅ Verification Checklist for Each Node

Copy this checklist for each node you implement:

### [Node Name] Implementation Checklist

**Interface**
- [ ] Extends `BubbleLabsNode`
- [ ] Implements `execute()` method
- [ ] Implements `validate_inputs()` method
- [ ] Implements `get_parameter_schema()` method
- [ ] Returns consistent data structure

**Error Handling**
- [ ] Validates all inputs
- [ ] Wraps exceptions appropriately
- [ ] Provides helpful error messages
- [ ] Updates context with errors

**State Management**
- [ ] Updates progress in context
- [ ] Stores results in context
- [ ] Adds artifacts to context
- [ ] Handles context restoration

**Testing**
- [ ] Unit test for execute()
- [ ] Unit test for validate_inputs()
- [ ] Unit test for error handling
- [ ] Integration test with other nodes
- [ ] All tests passing

**Documentation**
- [ ] Class docstring
- [ ] Method docstrings
- [ ] Parameter descriptions
- [ ] Return value descriptions
- [ ] Examples in docstrings
- [ ] UI tooltip text

**Performance**
- [ ] Executes in reasonable time
- [ ] Memory usage acceptable
- [ ] No memory leaks
- [ ] Caching implemented (if applicable)

---

## 🎨 UI Integration

### Node Configuration Panel
Each node needs a React component in BubbleLabs:

```typescript
// File: BubbleLab/apps/bubble-studio/src/nodes/DecompositionNode.tsx

import React from 'react';
import { NodeConfigPanel } from './NodeConfigPanel';

interface DecompositionNodeProps {
  nodeConfig: any;
  onConfigChange: (config: any) => void;
}

export const DecompositionNodePanel: React.FC<DecompositionNodeProps> = ({
  nodeConfig,
  onConfigChange
}) => {
  return (
    <NodeConfigPanel title="Decomposition Configuration">
      <label>
        Method:
        <select
          value={nodeConfig.method || 'roma'}
          onChange={(e) => onConfigChange({
            ...nodeConfig,
            method: e.target.value
          })}
        >
          <option value="roma">ROMA</option>
          <option value="maker">MAKER</option>
          <option value="mdap">MDAP</option>
        </select>
      </label>

      <label>
        Max Depth:
        <input
          type="number"
          min="1"
          max="10"
          value={nodeConfig.max_depth || 3}
          onChange={(e) => onConfigChange({
            ...nodeConfig,
            max_depth: parseInt(e.target.value)
          })}
        />
      </label>

      <label>
        <input
          type="checkbox"
          checked={nodeConfig.parallel !== false}
          onChange={(e) => onConfigChange({
            ...nodeConfig,
            parallel: e.target.checked
          })}
        />
        Enable Parallel Processing
      </label>
    </NodeConfigPanel>
  );
};
```

### Progress Display
```typescript
// Real-time progress updates
useEffect(() => {
  const subscription = nodeProgress$.subscribe((update) => {
    if (update.nodeId === currentNode.id) {
      setProgress(update.progress);
      setStatus(update.status);
    }
  });

  return () => subscription.unsubscribe();
}, [currentNode]);
```

---

## 📝 Common Patterns

### Error Handling Pattern
```python
try:
    # Do work
    result = self.engine.process(inputs)
except SpecificException as e:
    # Wrap in standard error
    raise NodeExecutionError(
        node_name=self.get_display_name(),
        message=f"Processing failed: {str(e)}",
        details={'original_error': str(e)}
    ) from e
```

### Progress Reporting Pattern
```python
def execute(self, inputs: Dict, context: WorkflowState) -> Dict:
    total_steps = 5
    context.update_progress(0, "Initializing")

    for i, step in enumerate(steps):
        context.update_progress(
            (i / total_steps) * 100,
            f"Processing step {i+1}: {step.name}"
        )
        step.execute()

    context.update_progress(100, "Complete")
    return result
```

### Caching Pattern
```python
def execute(self, inputs: Dict, context: WorkflowState) -> Dict:
    # Generate cache key
    cache_key = self._generate_cache_key(inputs)

    # Check cache
    if cached := context.cache.get(cache_key):
        return cached

    # Execute
    result = self._do_work(inputs)

    # Store in cache
    context.cache.set(cache_key, result)

    return result
```

---

## 🚀 Quick Commands

### Create new node from template
```bash
# Create node file
cat > bubblelabs_nodes/${NODE_NAME}_node.py << 'EOF'
# Use the template from Step 2 above
EOF

# Create test file
cat > tests/test_${NODE_NAME}_node.py << 'EOF'
# Use the test template from Testing Pattern above
EOF
```

### Run tests
```bash
# Run all node tests
pytest tests/test_*_node.py -v

# Run specific node tests
pytest tests/test_decomposition_node.py -v

# Run with coverage
pytest tests/test_*_node.py --cov=bubblelabs_nodes --cov-report=html
```

### Check code quality
```bash
# Linting
pylint bubblelabs_nodes/

# Type checking
mypy bubblelabs_nodes/

# Format code
black bubblelabs_nodes/
```

---

## 📚 Reference Materials

### Key Files to Review
- `workflow_structures.py` - Core data structures
- `decomposition_engine.py` - Decomposition implementation
- `gauntlet_manager.py` - Gauntlet system
- `solution_orchestration.py` - Solution generation
- `verification_engine.py` - Verification logic
- `sop_generator.py` - Output generation
- `workflow_knowledge_extractor.py` - Knowledge extraction

### Related Documentation
- `BUBBLELABS_INTEGRATION_TASKS.md` - Main task document
- `BubbleLab/apps/bubble-studio/README.md` - BubbleLabs UI docs
- `integrated_workflow.py` - Workflow orchestration

---

## 🆘 Troubleshooting

### Common Issues

**Issue**: Node doesn't show up in BubbleLabs UI
**Solution**: Check node is registered in `registry.py`

**Issue**: Input validation fails
**Solution**: Check input types match schema, use logging

**Issue**: Context not updating
**Solution**: Ensure `context.update_progress()` is called

**Issue**: Tests fail with import errors
**Solution**: Check Python path and virtual environment

**Issue**: Progress not visible in UI
**Solution**: Verify WebSocket connection and event emission

---

## 📊 Progress Tracking

### Overall Progress
- [ ] Phase 1: Node Wrappers (0/8 complete)
- [ ] Phase 2: Integration Testing (0/8 tested)
- [ ] Phase 3: UI Components (0/8 UI panels)
- [ ] Phase 4: Performance (0/8 optimized)
- [ ] Phase 5: Documentation (0/8 documented)

### Individual Node Status
| Node | Wrapper | Tests | UI | Docs | Complete |
|------|---------|-------|-------|------|----------|
| Decomposition | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| SubProblem | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Gauntlet | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Solution | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Verification | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Assembly | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Output | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| KnowledgeExtraction | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

---

**Quick Reference Last Updated**: 2025-01-03
**For detailed tasks, see**: `BUBBLELABS_INTEGRATION_TASKS.md`
