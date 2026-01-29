/**
 * Complete Example: OpenEvolve Workflow
 *
 * This example demonstrates how to use all OpenEvolve node components
 * together in a complete workflow.
 */

import React, { useCallback, useState } from 'react';
import { BubbleButton } from '../bubblelab';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

// Import OpenEvolve nodes
import {
  DecompositionNodeComponent,
  SolutionNodeComponent,
  VerificationNodeComponent,
  createFlowNode,
  type OpenEvolveNodeData,
} from './index';

/**
 * Example 1: Basic Node Creation
 */
function createExampleNodes(): Node<OpenEvolveNodeData>[] {
  return [
    // Decomposition Node
    createFlowNode('decomposition', { x: 0, y: 0 }, {
      displayName: 'Problem Decomposition',
      description: 'Break down complex AI system design',
      status: 'completed',
      progress: 100,
      parameters: {
        maxSubProblems: 8,
        strategy: 'hierarchical',
        granularityLevel: 'medium'
      },
      subProblems: [
        {
          id: 'sp-1',
          title: 'Data Pipeline Design',
          description: 'Design efficient data processing pipeline',
          status: 'completed',
          complexity: 0.7,
          dependencies: []
        },
        {
          id: 'sp-2',
          title: 'Model Architecture',
          description: 'Design neural network architecture',
          status: 'completed',
          complexity: 0.85,
          dependencies: ['sp-1']
        },
        {
          id: 'sp-3',
          title: 'Training Strategy',
          description: 'Plan training approach and hyperparameters',
          status: 'in_progress',
          complexity: 0.75,
          dependencies: ['sp-2']
        },
        {
          id: 'sp-4',
          title: 'Evaluation Framework',
          description: 'Create comprehensive testing framework',
          status: 'pending',
          complexity: 0.6,
          dependencies: ['sp-2', 'sp-3']
        }
      ],
      dependencyGraph: {
        totalDependencies: 3,
        criticalPath: 3,
        circularDeps: 0
      },
      qualityScore: 0.88,
      complexity: 0.72,
      completeness: 0.90
    }),

    // Solution Node
    createFlowNode('solution', { x: 400, y: 0 }, {
      displayName: 'Solution Generator',
      description: 'Evolve optimal AI system architecture',
      status: 'running',
      progress: 65,
      currentStrategy: 'genetic_algorithm',
      availableStrategies: [
        'genetic_algorithm',
        'quality_diversity',
        'novelty_search',
        'multi_objective'
      ],
      qualityScore: 0.84,
      confidence: 0.89,
      iterations: 23,
      parameters: {
        populationSize: 50,
        mutationRate: 0.15,
        crossoverRate: 0.8
      },
      alternativeSolutions: [
        {
          id: 'alt-1',
          name: 'Transformer-based',
          score: 0.91,
          confidence: 0.87,
          strategy: 'quality_diversity'
        },
        {
          id: 'alt-2',
          name: 'Hybrid CNN-RNN',
          score: 0.86,
          confidence: 0.82,
          strategy: 'genetic_algorithm'
        },
        {
          id: 'alt-3',
          name: 'Graph Neural Network',
          score: 0.79,
          confidence: 0.75,
          strategy: 'novelty_search'
        }
      ],
      metrics: {
        executionTime: 8450,
        convergence: 0.82,
        diversity: 0.76,
        efficiency: 0.91
      }
    }),

    // Verification Node
    createFlowNode('verification', { x: 800, y: 0 }, {
      displayName: 'Quality Verification',
      description: 'Validate against requirements and quality standards',
      status: 'idle',
      verificationStatus: 'warning',
      verificationScore: 0.78,
      parameters: {
        strictness: 'medium',
        checkSecurity: true,
        checkPerformance: true
      },
      qualityMetrics: {
        accuracy: 0.92,
        completeness: 0.78,
        consistency: 0.85,
        performance: 0.71,
        security: 0.88
      },
      requirements: [
        {
          id: 'req-1',
          name: 'Functional Correctness',
          status: 'pass',
          description: 'System produces correct outputs for all test cases',
          category: 'Functional'
        },
        {
          id: 'req-2',
          name: 'Response Time',
          status: 'warning',
          description: 'Average response time under 100ms (currently: 125ms)',
          category: 'Performance'
        },
        {
          id: 'req-3',
          name: 'Data Privacy',
          status: 'pass',
          description: 'All sensitive data is encrypted at rest and in transit',
          category: 'Security'
        },
        {
          id: 'req-4',
          name: 'Scalability',
          status: 'fail',
          description: 'System should handle 10K concurrent requests',
          category: 'Performance'
        },
        {
          id: 'req-5',
          name: 'API Compatibility',
          status: 'pass',
          description: 'Compatible with REST API v2.0 specification',
          category: 'Integration'
        }
      ],
      checksPerformed: 15,
      checksPassed: 12,
      checksFailed: 1
    })
  ];
}

/**
 * Example 2: Create edges connecting the nodes
 */
function createExampleEdges(): Edge[] {
  return [
    {
      id: 'e1-2',
      source: 'decomposition', // Will be set dynamically
      target: 'solution',
      animated: true,
      style: { stroke: '#9333ea', strokeWidth: 2 }
    },
    {
      id: 'e2-3',
      source: 'solution',
      target: 'verification',
      animated: true,
      style: { stroke: '#9333ea', strokeWidth: 2 }
    }
  ];
}

/**
 * Example 3: Complete Workflow Component
 */
function OpenEvolveWorkflowExampleBase() {
  // Initialize nodes and edges
  const [nodes, setNodes, onNodesChange] = useNodesState(createExampleNodes());
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Handle new connections
  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge({
        ...connection,
        animated: true,
        style: { stroke: '#9333ea', strokeWidth: 2 }
      }, eds));
    },
    [setEdges]
  );

  // Handle node clicks
  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    console.log('Clicked node:', node.data.displayName);
    // Could open a details panel, trigger execution, etc.
  }, []);

  // Register node types
  const nodeTypes = {
    decomposition: DecompositionNodeComponent,
    solution: SolutionNodeComponent,
    verification: VerificationNodeComponent,
  };

  return (
    <div className="w-full h-screen bg-neutral-950">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        className="bg-neutral-950"
      >
        <Background color="#374151" gap={16} />
        <Controls className="bg-neutral-900 border border-neutral-700" />
        <MiniMap
          nodeColor={(node) => {
            switch (node.type) {
              case 'decomposition': return '#4f46e5';
              case 'solution': return '#9333ea';
              case 'verification': return '#059669';
              default: return '#6b7280';
            }
          }}
          className="bg-neutral-900"
        />
      </ReactFlow>
    </div>
  );
}

/**
 * Example 4: Dynamic Node Updates
 */
function DynamicNodeUpdatesExampleBase() {
  const [nodes, setNodes] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Simulate execution progress
  const simulateExecution = useCallback(async () => {
    const nodeId = nodes[0]?.id;
    if (!nodeId) return;

    // Start execution
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, status: 'running', progress: 0 } }
          : node
      )
    );

    // Simulate progress
    for (let progress = 0; progress <= 100; progress += 10) {
      await new Promise(resolve => setTimeout(resolve, 500));
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodeId
            ? { ...node, data: { ...node.data, progress } }
            : node
        )
      );
    }

    // Complete execution
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              data: {
                ...node.data,
                status: 'completed',
                progress: 100,
                results: {
                  success: true,
                  score: 0.92,
                  iterations: 15,
                  duration: 5000
                }
              }
            }
          : node
      )
    );
  }, [nodes, setNodes]);

  return (
    <div className="p-4 space-y-4">
      <BubbleButton
        onClick={simulateExecution}
        className="px-4 py-2"
      >
        Simulate Execution
      </BubbleButton>

      <div className="h-[600px] border border-neutral-700 rounded-lg">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onEdgesChange={onEdgesChange}
          nodeTypes={{
            decomposition: DecompositionNodeComponent,
            solution: SolutionNodeComponent,
            verification: VerificationNodeComponent,
          }}
        />
      </div>
    </div>
  );
}

/**
 * Example 5: Interactive Parameter Updates
 */
function InteractiveParametersExampleBase() {
  const [nodes, setNodes] = useNodesState([
    createFlowNode('solution', { x: 0, y: 0 }, {
      displayName: 'Interactive Solution',
      description: 'Try changing the parameters below',
      status: 'idle',
      parameters: {
        populationSize: 50,
        mutationRate: 0.15,
        crossoverRate: 0.8,
        maxIterations: 100
      },
      onParameterChange: (name, value) => {
        console.log(`Parameter changed: ${name} = ${value}`);
        // Parameter changes are handled by the node component
      },
      onExecute: () => {
        console.log('Executing with current parameters...');
      }
    })
  ]);

  return (
    <div className="h-screen">
      <ReactFlow
        nodes={nodes}
        edges={[]}
        onNodesChange={useCallback((changes) => {
          setNodes((nds) => {
            // Apply changes
            const updated = nds.map(node => {
              // Update node when parameters change
              return node;
            });
            return updated;
          });
        }, [setNodes])}
        nodeTypes={{
          solution: SolutionNodeComponent
        }}
      />
    </div>
  );
}

/**
 * Example 6: Error Handling
 */
function ErrorHandlingExampleBase() {
  const [nodes, setNodes] = useNodesState([
    createFlowNode('decomposition', { x: 0, y: 0 }, {
      displayName: 'Error Example',
      description: 'This node demonstrates error states',
      status: 'error',
      results: {
        error: 'Failed to decompose problem: Maximum sub-problems exceeded'
      },
      onExecute: async () => {
        try {
          // Simulate execution that might fail
          await new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Execution failed')), 1000);
          });
        } catch (error) {
          // Update node with error
          setNodes((nds) =>
            nds.map((node) =>
              node.id === nodes[0].id
                ? {
                    ...node,
                    data: {
                      ...node.data,
                      status: 'error',
                      results: { error: (error as Error).message }
                    }
                  }
                : node
            )
          );
        }
      }
    })
  ]);

  return (
    <div className="h-screen">
      <ReactFlow
        nodes={nodes}
        edges={[]}
        nodeTypes={{
          decomposition: DecompositionNodeComponent
        }}
      />
    </div>
  );
}

export const OpenEvolveWorkflowExample = withComponentBoundary(
  OpenEvolveWorkflowExampleBase,
  'OpenEvolveWorkflowExample'
);
export const DynamicNodeUpdatesExample = withComponentBoundary(
  DynamicNodeUpdatesExampleBase,
  'DynamicNodeUpdatesExample'
);
export const InteractiveParametersExample = withComponentBoundary(
  InteractiveParametersExampleBase,
  'InteractiveParametersExample'
);
export const ErrorHandlingExample = withComponentBoundary(
  ErrorHandlingExampleBase,
  'ErrorHandlingExample'
);

// Export all examples
export default {
  OpenEvolveWorkflowExample,
  DynamicNodeUpdatesExample,
  InteractiveParametersExample,
  ErrorHandlingExample,
};
