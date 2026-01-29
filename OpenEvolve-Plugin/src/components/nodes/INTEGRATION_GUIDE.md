# OpenEvolve Nodes Integration Guide

Complete guide for integrating OpenEvolve React Flow nodes into BubbleLab.

## Table of Contents

1. [Installation](#installation)
2. [Basic Setup](#basic-setup)
3. [Backend Integration](#backend-integration)
4. [State Management](#state-management)
5. [Event Handling](#event-handling)
6. [Real-time Updates](#real-time-updates)
7. [Error Handling](#error-handling)
8. [Testing](#testing)
9. [Production Checklist](#production-checklist)

## Installation

### Step 1: Install Dependencies

The nodes require these peer dependencies:

```bash
npm install react react-dom @xyflow/react
npm install -D @types/react @types/react-dom
```

### Step 2: Copy Component Files

Ensure these files are in place:

```
src/components/nodes/
├── OpenEvolveNode.tsx
├── DecompositionNodeComponent.tsx
├── SolutionNodeComponent.tsx
├── VerificationNodeComponent.tsx
├── index.ts
├── types/
│   └── nodeTypes.ts
├── README.md
├── QUICK_REFERENCE.md
└── INTEGRATION_GUIDE.md (this file)
```

### Step 3: Install Tailwind CSS

Nodes use Tailwind CSS. Ensure it's configured:

```javascript
// tailwind.config.js
module.exports = {
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
    './openevolve-bubblelab-plugin/src/**/*.{js,jsx,ts,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        purple: {
          50: '#faf5ff',
          500: '#a855f7',
          600: '#9333ea',
          700: '#7e22ce',
          900: '#581c87',
          950: '#3b0764',
        }
      }
    }
  }
};
```

## Basic Setup

### Option 1: Direct Integration

```typescript
// src/App.tsx
import React, { useCallback } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  addEdge
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import {
  DecompositionNodeComponent,
  SolutionNodeComponent,
  VerificationNodeComponent
} from '@openevolve/bubblelab-plugin/src/components/nodes';

const nodeTypes = {
  decomposition: DecompositionNodeComponent,
  solution: SolutionNodeComponent,
  verification: VerificationNodeComponent,
};

function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
}

export default App;
```

### Option 2: Plugin Integration (BubbleLab)

```typescript
// src/plugins/openevolve.ts
import { registerBubblePlugin } from '@bubblelab/core';
import {
  openEvolveNodeComponents,
  OPENEVOLVE_NODE_TYPES
} from '@openevolve/bubblelab-plugin/src/components/nodes';

export function registerOpenEvolvePlugin() {
  registerBubblePlugin({
    id: 'openevolve',
    name: 'OpenEvolve',
    version: '1.0.0',
    nodeTypes: OPENEVOLVE_NODE_TYPES,
    nodeComponents: openEvolveNodeComponents,
    bubbles: [
      {
        type: 'decomposition',
        name: 'Problem Decomposition',
        category: 'OpenEvolve',
        description: 'Break down complex problems into sub-problems'
      },
      {
        type: 'solution',
        name: 'Solution Generator',
        category: 'OpenEvolve',
        description: 'Generate and optimize solutions'
      },
      {
        type: 'verification',
        name: 'Verification',
        category: 'OpenEvolve',
        description: 'Validate solutions against requirements'
      }
    ]
  });
}
```

## Backend Integration

### API Client

Create a client to communicate with OpenEvolve backend:

```typescript
// src/api/openevolve.ts
import axios from 'axios';

export class OpenEvolveClient {
  private baseUrl: string;

  constructor(config: { baseUrl: string }) {
    this.baseUrl = config.baseUrl;
  }

  async executeDecomposition(params: {
    problem: string;
    maxSubProblems: number;
    strategy: string;
  }) {
    const response = await axios.post(
      `${this.baseUrl}/decomposition/execute`,
      params
    );
    return response.data;
  }

  async executeSolution(params: {
    problem: string;
    strategy: string;
    parameters: Record<string, any>;
  }) {
    const response = await axios.post(
      `${this.baseUrl}/solution/execute`,
      params
    );
    return response.data;
  }

  async executeVerification(params: {
    solution: any;
    requirements: any[];
  }) {
    const response = await axios.post(
      `${this.baseUrl}/verification/execute`,
      params
    );
    return response.data;
  }

  async subscribeExecution(executionId: string, callback: (update: any) => void) {
    const ws = new WebSocket(`${this.baseUrl.replace('http', 'ws')}/ws/${executionId}`);

    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      callback(update);
    };

    return ws;
  }
}
```

### Wire Up Execution

```typescript
// src/hooks/useOpenEvolveExecution.ts
import { useCallback } from 'react';
import { useNodesState } from '@xyflow/react';
import { OpenEvolveClient } from '../api/openevolve';

export function useOpenEvolveExecution() {
  const [nodes, setNodes] = useNodesState([]);
  const client = new OpenEvolveClient({ baseUrl: '/api/openevolve' });

  const executeNode = useCallback(async (nodeId: string) => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;

    // Update to running
    setNodes((nds) =>
      nds.map((n) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, status: 'running', progress: 0 } }
          : n
      )
    );

    try {
      let result;

      // Execute based on node type
      switch (node.data.type) {
        case 'decomposition':
          result = await client.executeDecomposition(node.data.parameters);
          break;
        case 'solution':
          result = await client.executeSolution(node.data.parameters);
          break;
        case 'verification':
          result = await client.executeVerification(node.data.parameters);
          break;
      }

      // Update with results
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId
            ? {
                ...n,
                data: {
                  ...n.data,
                  status: 'completed',
                  progress: 100,
                  results: result
                }
              }
            : n
        )
      );
    } catch (error) {
      // Update with error
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId
            ? {
                ...n,
                data: {
                  ...n.data,
                  status: 'error',
                  results: { error: error.message }
                }
              }
            : n
        )
      );
    }
  }, [nodes, setNodes, client]);

  return { nodes, setNodes, executeNode };
}
```

## State Management

### Zustand Store

```typescript
// src/stores/openevolveStore.ts
import { create } from 'zustand';
import { Node, Edge } from '@xyflow/react';
import type { OpenEvolveNodeData } from '@openevolve/bubblelab-plugin/src/components/nodes';

interface OpenEvolveStore {
  nodes: Node<OpenEvolveNodeData>[];
  edges: Edge[];
  executions: Map<string, any>;

  updateNode: (nodeId: string, updates: Partial<OpenEvolveNodeData>) => void;
  updateNodeProgress: (nodeId: string, progress: number) => void;
  setNodeStatus: (nodeId: string, status: NodeStatus) => void;
  addExecution: (nodeId: string, execution: any) => void;
}

export const useOpenEvolveStore = create<OpenEvolveStore>((set) => ({
  nodes: [],
  edges: [],
  executions: new Map(),

  updateNode: (nodeId, updates) =>
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...updates } }
          : node
      ),
    })),

  updateNodeProgress: (nodeId, progress) =>
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, progress } }
          : node
      ),
    })),

  setNodeStatus: (nodeId, status) =>
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, status } }
          : node
      ),
    })),

  addExecution: (nodeId, execution) =>
    set((state) => ({
      executions: new Map(state.executions).set(nodeId, execution),
    })),
}));
```

### Using the Store

```typescript
function WorkflowEditor() {
  const { nodes, updateNode, setNodeStatus } = useOpenEvolveStore();

  const handleExecute = async (nodeId: string) => {
    setNodeStatus(nodeId, 'running');

    // ... execution logic

    updateNode(nodeId, {
      status: 'completed',
      results: { success: true, score: 0.9 }
    });
  };

  return <ReactFlow nodes={nodes} /* ... */ />;
}
```

## Event Handling

### Node Click Events

```typescript
const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
  console.log('Clicked:', node.data.displayName);

  // Show details panel
  setSelectedNode(node);

  // Or trigger execution
  if (node.data.status === 'idle') {
    executeNode(node.id);
  }
}, [executeNode]);
```

### Parameter Changes

```typescript
const handleParameterChange = useCallback((
  nodeId: string,
  paramName: string,
  value: any
) => {
  // Validate
  if (paramName === 'maxIterations' && value < 1) {
    console.error('Iterations must be >= 1');
    return;
  }

  // Update node
  updateNode(nodeId, {
    parameters: {
      ...getNode(nodeId).data.parameters,
      [paramName]: value
    }
  });

  // Persist to backend
  api.updateNodeParameters(nodeId, { [paramName]: value });
}, [updateNode]);
```

### Completion Events

```typescript
const handleExecutionComplete = useCallback((
  nodeId: string,
  results: any
) => {
  // Update node
  updateNode(nodeId, {
    status: 'completed',
    results
  });

  // Trigger dependent nodes
  const dependentEdges = edges.filter(e => e.source === nodeId);
  dependentEdges.forEach(edge => {
    if (shouldAutoExecute(edge.target)) {
      executeNode(edge.target);
    }
  });

  // Show notification
  toast.success(`Execution complete: ${getNode(nodeId).data.displayName}`);
}, [updateNode, edges, executeNode]);
```

## Real-time Updates

### WebSocket Integration

```typescript
function useRealtimeExecution(nodeId: string) {
  const { updateNode, updateNodeProgress } = useOpenEvolveStore();
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/execution/${nodeId}`);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'progress':
          updateNodeProgress(nodeId, data.progress);
          break;
        case 'status':
          updateNode(nodeId, { status: data.status });
          break;
        case 'result':
          updateNode(nodeId, {
            status: 'completed',
            results: data.result
          });
          break;
        case 'error':
          updateNode(nodeId, {
            status: 'error',
            results: { error: data.error }
          });
          break;
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [nodeId, updateNode, updateNodeProgress]);

  return wsRef.current;
}
```

### Polling Fallback

```typescript
function usePollingExecution(nodeId: string, interval = 1000) {
  const { updateNode } = useOpenEvolveStore();

  useEffect(() => {
    const poll = async () => {
      try {
        const status = await api.getExecutionStatus(nodeId);
        updateNode(nodeId, {
          progress: status.progress,
          results: status.partialResults
        });

        if (status.completed) {
          updateNode(nodeId, {
            status: 'completed',
            results: status.results
          });
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    };

    const timer = setInterval(poll, interval);
    return () => clearInterval(timer);
  }, [nodeId, interval, updateNode]);
}
```

## Error Handling

### Error Boundaries

```typescript
// src/components/ErrorBoundary.tsx
class NodeErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('Node error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 bg-red-950 border border-red-700 rounded-lg">
          <h3 className="text-red-300 font-semibold">Something went wrong</h3>
          <p className="text-red-400 text-sm mt-2">
            {this.state.error?.message}
          </p>
        </div>
      );
    }

    return this.props.children;
  }
}
```

### Retry Logic

```typescript
async function executeWithRetry(
  execute: () => Promise<any>,
  maxRetries = 3,
  delay = 1000
) {
  let lastError;

  for (let i = 0; i < maxRetries; i++) {
    try {
      return await execute();
    } catch (error) {
      lastError = error;
      console.warn(`Execution attempt ${i + 1} failed:`, error);

      if (i < maxRetries - 1) {
        await new Promise(resolve => setTimeout(resolve, delay * (i + 1)));
      }
    }
  }

  throw lastError;
}
```

## Testing

### Unit Tests

```typescript
// src/components/nodes/__tests__/DecompositionNode.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { ReactFlowProvider } from '@xyflow/react';
import { DecompositionNodeComponent } from '../DecompositionNodeComponent';

const mockNode = {
  id: 'test-node',
  type: 'decomposition',
  data: {
    id: 'test-node',
    type: 'decomposition',
    displayName: 'Test Decomposition',
    status: 'idle',
    subProblems: [
      {
        id: 'sp1',
        title: 'Test Sub-Problem',
        description: 'Test description',
        status: 'pending',
        complexity: 0.5,
        dependencies: []
      }
    ]
  }
};

test('renders decomposition node', () => {
  render(
    <ReactFlowProvider>
      <DecompositionNodeComponent data={mockNode.data} selected={false} />
    </ReactFlowProvider>
  );

  expect(screen.getByText('Test Decomposition')).toBeInTheDocument();
});

test('expands sub-problem on click', () => {
  render(
    <ReactFlowProvider>
      <DecompositionNodeComponent data={mockNode.data} selected={false} />
    </ReactFlowProvider>
  );

  const subProblem = screen.getByText('Test Sub-Problem');
  fireEvent.click(subProblem);

  expect(screen.getByText('Test description')).toBeInTheDocument();
});
```

### Integration Tests

```typescript
// src/__tests__/workflow.integration.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import { ReactFlowProvider } from '@xyflow/react';
import { OpenEvolveWorkflow } from '../OpenEvolveWorkflow';

test('executes workflow end-to-end', async () => {
  render(
    <ReactFlowProvider>
      <OpenEvolveWorkflow />
    </ReactFlowProvider>
  );

  // Start execution
  const executeButton = screen.getByText('Execute Decomposition');
  fireEvent.click(executeButton);

  // Wait for completion
  await waitFor(() => {
    expect(screen.getByText('Completed')).toBeInTheDocument();
  });

  // Verify results
  expect(screen.getByText('Quality Score: 88%')).toBeInTheDocument();
});
```

## Production Checklist

### Performance
- [ ] Enable React Flow virtualization for large workflows
- [ ] Memoize all callbacks with `useCallback`
- [ ] Use `React.memo` for custom components
- [ ] Implement pagination for long lists
- [ ] Lazy load node components

### Security
- [ ] Sanitize all user inputs
- [ ] Use HTTPS for all API calls
- [ ] Implement authentication
- [ ] Validate all parameters
- [ ] Sanitize error messages

### Error Handling
- [ ] Implement error boundaries
- [ ] Add retry logic for failed requests
- [ ] Log errors to monitoring service
- [ ] Show user-friendly error messages
- [ ] Implement circuit breakers

### User Experience
- [ ] Add loading states
- [ ] Show progress indicators
- [ ] Provide keyboard shortcuts
- [ ] Add tooltips
- [ ] Implement undo/redo
- [ ] Save workflow state

### Monitoring
- [ ] Track execution metrics
- [ ] Monitor performance
- [ ] Log errors and warnings
- [ ] Track user interactions
- [ ] Set up alerts

### Documentation
- [ ] Update README
- [ ] Add API documentation
- [ ] Create usage examples
- [ ] Document configuration options
- [ ] Provide troubleshooting guide

---

**Ready for Production?** ✨

This integration guide covers everything needed to successfully integrate OpenEvolve nodes into BubbleLab. For additional support, refer to the other documentation files or open an issue on GitHub.
