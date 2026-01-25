# OpenEvolve React Flow Nodes - Quick Reference

## 🚀 Quick Start

```typescript
// 1. Import
import {
  DecompositionNodeComponent,
  SolutionNodeComponent,
  VerificationNodeComponent,
  createFlowNode
} from '@openevolve/bubblelab-plugin/src/components/nodes';

// 2. Create node
const node = createFlowNode('decomposition', { x: 0, y: 0 }, {
  displayName: 'My Node',
  status: 'idle'
});

// 3. Use in React Flow
<ReactFlow nodeTypes={{ decomposition: DecompositionNodeComponent }} />
```

## 📦 Node Types

| Node Type | Component | Purpose | Key Features |
|-----------|-----------|---------|--------------|
| **Decomposition** | `DecompositionNodeComponent` | Break down problems | Sub-problem list, dependency graph, quality metrics |
| **Solution** | `SolutionNodeComponent` | Generate solutions | Strategy selector, quality gauge, alternatives |
| **Verification** | `VerificationNodeComponent` | Validate solutions | Pass/fail badge, requirement checklist, metrics |

## 🎨 Common Props

All nodes accept these base props:

```typescript
interface OpenEvolveNodeData {
  // Required
  id: string;
  type: NodeType;
  displayName: string;
  status: NodeStatus;

  // Optional
  description?: string;
  progress?: number; // 0-100
  config?: Record<string, any>;
  parameters?: Record<string, any>;
  results?: NodeResult;

  // Callbacks
  onParameterChange?: (name: string, value: any) => void;
  onExecute?: () => void;
}
```

## 📊 Status Values

```typescript
type NodeStatus =
  | 'idle'       // Ready to execute
  | 'running'    // Currently executing
  | 'completed'  // Finished successfully
  | 'error'      // Failed with error
  | 'paused';    // Paused by user
```

## 🎯 Node-Specific Props

### Decomposition Node

```typescript
interface DecompositionNodeData extends OpenEvolveNodeData {
  subProblems?: SubProblem[];
  dependencyGraph?: DependencyInfo;
  qualityScore?: number;      // 0-1
  complexity?: number;        // 0-1
  completeness?: number;      // 0-1
}

interface SubProblem {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'blocked';
  complexity: number;
  dependencies: string[];
}
```

### Solution Node

```typescript
interface SolutionNodeData extends OpenEvolveNodeData {
  currentStrategy?: string;
  availableStrategies?: string[];
  qualityScore?: number;       // 0-1
  confidence?: number;         // 0-1
  iterations?: number;
  alternativeSolutions?: AlternativeSolution[];
  metrics?: SolutionMetrics;
}

interface AlternativeSolution {
  id: string;
  name: string;
  score: number;
  confidence: number;
  strategy: string;
}
```

### Verification Node

```typescript
interface VerificationNodeData extends OpenEvolveNodeData {
  verificationStatus?: 'pass' | 'fail' | 'warning' | 'pending';
  verificationScore?: number;  // 0-1
  qualityMetrics?: QualityMetrics;
  requirements?: Requirement[];
}

interface Requirement {
  id: string;
  name: string;
  status: 'pass' | 'fail' | 'warning' | 'skipped';
  description: string;
  category: string;
}
```

## 🔄 Common Patterns

### Update Node Status

```typescript
updateNode(nodeId, (node) => ({
  ...node,
  data: {
    ...node.data,
    status: 'running',
    progress: 50
  }
}));
```

### Handle Execution

```typescript
const nodeData = {
  ...otherProps,
  onExecute: async () => {
    try {
      const result = await api.execute();
      updateNode(nodeId, {
        status: 'completed',
        results: result
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

### Update Parameters

```typescript
const nodeData = {
  ...otherProps,
  parameters: {
    maxSubProblems: 10,
    strategy: 'hierarchical'
  },
  onParameterChange: (name, value) => {
    // Save to backend
    await api.updateParameter(nodeId, name, value);
    // Update local state
    updateNode(nodeId, {
      parameters: {
        ...node.data.parameters,
        [name]: value
      }
    });
  }
};
```

## 🎨 Styling

### Colors

```typescript
// Primary theme colors
const colors = {
  purple: {
    50:  '#faf5ff',
    500: '#a855f7',
    600: '#9333ea',
    700: '#7e22ce',
    900: '#581c87',
  },
  indigo: {
    500: '#6366f1',
    600: '#4f46e5',
    700: '#4338ca',
  },
  green: {
    400: '#4ade80',
    500: '#22c55e',
    600: '#16a34a',
  },
  red: {
    400: '#f87171',
    500: '#ef4444',
    600: '#dc2626',
  },
  yellow: {
    400: '#facc15',
    500: '#eab308',
    600: '#ca8a04',
  }
};
```

### Tailwind Classes Used

```typescript
// Backgrounds
bg-neutral-800/90   // Default node background
bg-purple-950/50    // Purple tinted background
bg-green-950/50     // Green for success
bg-red-950/50       // Red for error

// Borders
border-neutral-600  // Default border
border-purple-500   // Selected state
border-green-500    // Success state
border-red-500      // Error state

// Text
text-neutral-100    // Primary text
text-neutral-400    // Secondary text
text-purple-300     // Accent text
```

## ⚡ Performance Tips

1. **Memoize callbacks**
   ```typescript
   const onExecute = useCallback(async () => {
     // ...
   }, [dependencies]);
   ```

2. **Use React.memo for custom components**
   ```typescript
   export const MyCustomNode = memo(({ data }) => {
     // ...
   });
   ```

3. **Debounce rapid updates**
   ```typescript
   const debouncedUpdate = debounce(updateNode, 300);
   ```

4. **Lazy load for large workflows**
   ```typescript
   const NodeComponent = lazy(() => import('./Node'));
   ```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Nodes not rendering | Check nodeTypes registration |
| Styles not applying | Verify Tailwind CSS is loaded |
| Props not updating | Ensure proper React state usage |
| Type errors | Check TypeScript imports |
| Performance issues | Reduce node count, enable virtualization |

## 📚 Resources

- **Full Documentation**: `./README.md`
- **Examples**: `./example.tsx`
- **TypeScript Definitions**: `../types/nodeTypes.ts`
- **React Flow Docs**: https://reactflow.dev
- **OpenEvolve Docs**: https://docs.openevolve.ai

## 🎓 Key Concepts

### Node Lifecycle

```
idle → running → completed
        ↓
      error
```

### Data Flow

```
Input Parameters → Execute → Results → Update Display
                       ↓
                   Progress Updates
```

### State Management

```typescript
// Local component state (fast)
const [localState, setLocalState] = useState();

// React Flow state (managed)
const [nodes, setNodes] = useNodesState();

// Backend state (persistent)
const backendState = await api.getState();
```

## 🔧 Helper Functions

### Create Node

```typescript
const node = createFlowNode('decomposition', { x: 0, y: 0 }, {
  displayName: 'My Node'
});
```

### Update Node Data

```typescript
setNodes((nds) =>
  nds.map((node) =>
    node.id === targetId
      ? { ...node, data: { ...node.data, key: newValue } }
      : node
  )
);
```

### Find Node

```typescript
const node = nodes.find((n) => n.id === targetId);
```

### Filter by Type

```typescript
const decompositionNodes = nodes.filter(
  (n) => n.data.type === 'decomposition'
);
```

---

**Need More Help?**
- Check the full README: `./README.md`
- View examples: `./example.tsx`
- Open an issue on GitHub
