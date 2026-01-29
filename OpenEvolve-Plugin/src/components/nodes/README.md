# OpenEvolve React Flow Nodes

Comprehensive React Flow node components for visualizing OpenEvolve workflows in BubbleLab's workflow editor.

## Overview

These components provide a rich, interactive visual interface for OpenEvolve's core functionality:
- **Problem Decomposition** - Break down complex problems
- **Solution Generation** - Evolve and optimize solutions
- **Verification** - Validate against requirements

## Features

### ✨ All Node Components Include
- 🎨 **Dark Mode Support** - Beautiful purple/indigo OpenEvolve theme
- 📊 **Real-time Status Updates** - Visual feedback for execution state
- ⚡ **Interactive Controls** - Edit parameters, trigger execution
- 🔌 **Input/Output Handles** - Connect nodes in workflows
- 📱 **Responsive Design** - Smooth animations and transitions
- ♿ **Accessible** - Proper contrast ratios and ARIA labels
- 🎯 **Type-Safe** - Full TypeScript support

## Component Structure

```
src/components/nodes/
├── OpenEvolveNode.tsx              # Base node component
├── DecompositionNodeComponent.tsx  # Problem decomposition
├── SolutionNodeComponent.tsx       # Solution generation
├── VerificationNodeComponent.tsx   # Verification & validation
├── index.ts                        # Exports and registry
└── README.md                       # This file
```

## Installation

The components are already part of the OpenEvolve BubbleLab plugin. No additional installation needed.

## Quick Start

### 1. Import the Components

```typescript
import {
  DecompositionNodeComponent,
  SolutionNodeComponent,
  VerificationNodeComponent,
  OPENEVOLVE_NODE_TYPES
} from '@openevolve/bubblelab-plugin/src/components/nodes';
```

### 2. Register with React Flow

```typescript
import { ReactFlow } from '@xyflow/react';
import { openEvolveNodeComponents } from '@openevolve/bubblelab-plugin/src/components/nodes';

function WorkflowEditor() {
  return (
    <ReactFlow
      nodeTypes={openEvolveNodeComponents}
      // ... other props
    />
  );
}
```

### 3. Create Nodes

```typescript
import { createFlowNode } from '@openevolve/bubblelab-plugin/src/components/nodes';

// Create a decomposition node
const decompositionNode = createFlowNode('decomposition', { x: 0, y: 0 }, {
  displayName: 'Decompose Problem',
  description: 'Break down the main problem',
  parameters: {
    maxSubProblems: 10,
    strategy: 'hierarchical'
  },
  onExecute: async () => {
    // Handle execution
  }
});

// Create a solution node
const solutionNode = createFlowNode('solution', { x: 300, y: 0 }, {
  displayName: 'Generate Solution',
  currentStrategy: 'genetic_algorithm',
  qualityScore: 0.85,
  confidence: 0.92
});

// Create a verification node
const verificationNode = createFlowNode('verification', { x: 600, y: 0 }, {
  displayName: 'Verify Solution',
  verificationStatus: 'pass',
  verificationScore: 0.88
});
```

## Component Details

### OpenEvolveNode (Base)

The foundational component that provides:
- Common UI structure
- Status indicators (idle, running, completed, error)
- Input/output handles
- Collapsible details panel
- Parameter quick-edit interface
- Error display
- Progress indicator

**Usage:**
```typescript
import { OpenEvolveNode } from '@openevolve/bubblelab-plugin/src/components/nodes';

const nodeData = {
  id: 'base-node',
  type: 'openevolve',
  displayName: 'My OpenEvolve Node',
  description: 'Does something awesome',
  status: 'idle',
  parameters: {
    param1: 'value1',
    param2: 42
  },
  onParameterChange: (name, value) => {
    console.log(`Updated ${name}:`, value);
  },
  onExecute: () => {
    console.log('Executing...');
  }
};
```

### DecompositionNodeComponent

Specialized for problem decomposition with:
- 📊 **Sub-problem List** - Expandable list with status indicators
- 🔗 **Dependency Graph Preview** - Visualize dependencies
- 📈 **Quality Metrics** - Quality score, complexity, completeness
- 📊 **Progress Tracking** - Track decomposition progress
- 🎯 **Strategy Selection** - Choose decomposition strategy

**Extended Data Interface:**
```typescript
interface DecompositionNodeData extends OpenEvolveNodeData {
  subProblems?: SubProblem[];
  dependencyGraph?: DependencyInfo;
  qualityScore?: number;
  complexity?: number;
  completeness?: number;
}
```

**Example:**
```typescript
const decompositionNode = {
  ...createFlowNode('decomposition', { x: 0, y: 0 }),
  data: {
    ...createFlowNode('decomposition', { x: 0, y: 0 }).data,
    subProblems: [
      {
        id: 'sp1',
        title: 'Sub-problem 1',
        description: 'First part of the problem',
        status: 'completed',
        complexity: 0.7,
        dependencies: []
      },
      // ... more sub-problems
    ],
    dependencyGraph: {
      totalDependencies: 5,
      criticalPath: 3,
      circularDeps: 0
    },
    qualityScore: 0.85,
    complexity: 0.65,
    completeness: 0.90
  }
};
```

### SolutionNodeComponent

For solution generation with:
- 🎛️ **Strategy Selector** - Dropdown to choose strategies
- 📊 **Quality Gauge** - Visual circular gauge for quality score
- 📈 **Confidence Meter** - Linear progress for confidence
- 🔁 **Iteration Counter** - Track evolution iterations
- 💡 **Alternative Solutions** - View and compare alternatives
- 📊 **Metrics Dashboard** - Convergence, diversity, efficiency

**Extended Data Interface:**
```typescript
interface SolutionNodeData extends OpenEvolveNodeData {
  currentStrategy?: string;
  availableStrategies?: string[];
  qualityScore?: number;
  confidence?: number;
  iterations?: number;
  alternativeSolutions?: AlternativeSolution[];
  metrics?: SolutionMetrics;
}
```

**Example:**
```typescript
const solutionNode = {
  ...createFlowNode('solution', { x: 300, y: 0 }),
  data: {
    ...createFlowNode('solution', { x: 300, y: 0 }).data,
    currentStrategy: 'genetic_algorithm',
    availableStrategies: [
      'genetic_algorithm',
      'quality_diversity',
      'novelty_search'
    ],
    qualityScore: 0.87,
    confidence: 0.92,
    iterations: 15,
    alternativeSolutions: [
      {
        id: 'alt1',
        name: 'Alternative 1',
        score: 0.91,
        confidence: 0.88,
        strategy: 'quality_diversity'
      }
    ],
    metrics: {
      executionTime: 5200,
      convergence: 0.85,
      diversity: 0.78,
      efficiency: 0.92
    }
  }
};
```

### VerificationNodeComponent

For verification and validation:
- ✅ **Pass/Fail Badge** - Large, clear status indicator
- 📊 **Quality Metrics** - 5 quality dimensions with bars
- ✅ **Requirement Checklist** - Expandable with categories
- 📈 **Verification Score** - Overall score display
- 🏷️ **Category Filter** - Filter requirements by category
- 📊 **Statistics** - Quick stats (total, pass, fail, warning)

**Extended Data Interface:**
```typescript
interface VerificationNodeData extends OpenEvolveNodeData {
  verificationStatus?: 'pass' | 'fail' | 'warning' | 'pending';
  verificationScore?: number;
  qualityMetrics?: QualityMetrics;
  requirements?: Requirement[];
  checksPerformed?: number;
  checksPassed?: number;
  checksFailed?: number;
}
```

**Example:**
```typescript
const verificationNode = {
  ...createFlowNode('verification', { x: 600, y: 0 }),
  data: {
    ...createFlowNode('verification', { x: 600, y: 0 }).data,
    verificationStatus: 'pass',
    verificationScore: 0.88,
    qualityMetrics: {
      accuracy: 0.92,
      completeness: 0.88,
      consistency: 0.95,
      performance: 0.85,
      security: 0.90
    },
    requirements: [
      {
        id: 'req1',
        name: 'Functional Requirement 1',
        status: 'pass',
        description: 'Must handle edge cases correctly',
        category: 'Functional'
      },
      {
        id: 'req2',
        name: 'Performance Requirement',
        status: 'warning',
        description: 'Response time should be < 100ms',
        category: 'Performance'
      }
    ]
  }
};
```

## Styling

### Theme Colors

All components use OpenEvolve's purple/indigo theme:

- **Primary**: Purple (`#9333ea`, `#7c3aed`)
- **Secondary**: Indigo (`#6366f1`, `#4f46e5`)
- **Success**: Green (`#22c55e`, `#10b981`)
- **Warning**: Yellow (`#eab308`, `#f59e0b`)
- **Error**: Red (`#ef4444`, `#dc2626`)
- **Neutral**: Gray scale (`#404040` to `#fafafa`)

### Customization

Components use Tailwind CSS classes. To customize:

1. **Override styles** using standard CSS cascade
2. **Extend Tailwind** configuration in your project
3. **Use style props** if needed (not recommended)

## TypeScript Support

Full TypeScript definitions are included:

```typescript
import type {
  OpenEvolveNodeData,
  NodeStatus,
  NodeType,
  SubProblem,
  AlternativeSolution,
  Requirement,
  // ... and more
} from '@openevolve/bubblelab-plugin/src/components/nodes';
```

## State Management

Nodes are controlled components. Update their data to trigger re-renders:

```typescript
// Using React Flow's updateNode
const updateNode = useStoreState(state => state.updateNode);

// Update node data
updateNode(nodeId, (node) => ({
  ...node.data,
  status: 'running',
  progress: 45
}));
```

## Best Practices

### 1. **Always Provide Callbacks**
```typescript
const nodeData = {
  // ... other props
  onParameterChange: (name, value) => {
    // Persist changes
    saveParameter(nodeId, name, value);
  },
  onExecute: async () => {
    // Handle execution
    await executeNode(nodeId);
  }
};
```

### 2. **Update Progress During Execution**
```typescript
// Start execution
updateNode(nodeId, { status: 'running', progress: 0 });

// Update progress
updateNode(nodeId, { progress: 50 });

// Complete
updateNode(nodeId, { status: 'completed', progress: 100 });
```

### 3. **Handle Errors Gracefully**
```typescript
try {
  const result = await execute();
  updateNode(nodeId, {
    status: 'completed',
    results: { success: true, ...result }
  });
} catch (error) {
  updateNode(nodeId, {
    status: 'error',
    results: { error: error.message }
  });
}
```

### 4. **Use Unique IDs**
```typescript
// Good
const id = `decomp-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

// Avoid
const id = 'decomposition-node'; // Will conflict if multiple
```

## Performance

Components are optimized for performance:
- ✅ **Memoized** with `React.memo`
- ✅ **Efficient re-renders** using hooks
- ✅ **Lazy loading** support via React.lazy
- ✅ **Minimal DOM** - only renders visible elements

For large workflows:
1. Use React Flow's built-in virtualization
2. Lazy load node components
3. Debounce rapid updates

## Integration with OpenEvolve Backend

Connect nodes to OpenEvolve backend:

```typescript
import { OpenEvolveClient } from '@openevolve/client';

const client = new OpenEvolveClient({ apiUrl: '/api/openevolve' });

const nodeData = {
  // ... other props
  onExecute: async () => {
    try {
      const result = await client.executeDecomposition({
        problem: parameters.problem,
        maxSubProblems: parameters.maxSubProblems
      });

      updateNode(nodeId, {
        status: 'completed',
        results: result,
        subProblems: result.subProblems,
        dependencyGraph: result.dependencies
      });
    } catch (error) {
      updateNode(nodeId, {
        status: 'error',
        results: { error: error.message }
      });
    }
  }
};
```

## Troubleshooting

### Nodes Not Rendering
- Ensure node types are registered with React Flow
- Check component exports in `index.ts`
- Verify TypeScript compilation

### Styles Not Applying
- Ensure Tailwind CSS is configured
- Check for style conflicts with parent components
- Verify dark mode is enabled

### Performance Issues
- Reduce number of rendered nodes
- Use React Flow's zoom/pan
- Enable virtualization for large workflows

## Contributing

When adding new node types:
1. Create component in `nodes/` directory
2. Extend `OpenEvolveNodeData` interface
3. Add to `OPENEVOLVE_NODE_TYPES` registry
4. Update this README
5. Add TypeScript definitions

## License

MIT License - see project root for details.

## Support

For issues or questions:
- GitHub Issues: [OpenEvolve Issues](https://github.com/openevolve/openevolve-bubblelab-plugin/issues)
- Documentation: [OpenEvolve Docs](https://docs.openevolve.ai)

---

**Built with ❤️ for the OpenEvolve and BubbleLab communities**
