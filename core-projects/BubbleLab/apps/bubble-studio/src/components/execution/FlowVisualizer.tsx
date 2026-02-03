/**
 * FlowVisualizer Component
 * Visual representation of workflow execution flow
 */

import { useState, useEffect } from 'react';
import { Card } from '../common/Card';

interface FlowNode {
  id: string;
  label: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  position: { x: number; y: number };
  result?: string;
}

interface FlowEdge {
  from: string;
  to: string;
  status: 'pending' | 'active' | 'completed';
}

interface FlowVisualizerProps {
  nodes: FlowNode[];
  edges: FlowEdge[];
  onNodeClick?: (nodeId: string) => void;
  className?: string;
}

export function FlowVisualizer({
  nodes,
  edges,
  onNodeClick,
  className = '',
}: FlowVisualizerProps) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const statusColors = {
    pending: 'bg-gray-200 dark:bg-gray-700 border-gray-400 dark:border-gray-600',
    running: 'bg-blue-100 dark:bg-blue-900/30 border-blue-500 dark:border-blue-400 animate-pulse',
    completed: 'bg-green-100 dark:bg-green-900/30 border-green-500 dark:border-green-400',
    failed: 'bg-red-100 dark:bg-red-900/30 border-red-500 dark:border-red-400',
  };

  const statusIcons = {
    pending: '○',
    running: '◌',
    completed: '●',
    failed: '✕',
  };

  return (
    <div className={`relative bg-gray-50 dark:bg-gray-900 rounded-lg ${className}`}>
      {/* SVG for edges */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        {edges.map((edge, index) => {
          const fromNode = nodes.find((n) => n.id === edge.from);
          const toNode = nodes.find((n) => n.id === edge.to);

          if (!fromNode || !toNode) return null;

          return (
            <g key={index}>
              <line
                x1={fromNode.position.x + 80}
                y1={fromNode.position.y + 30}
                x2={toNode.position.x}
                y2={toNode.position.y + 30}
                className={`transition-colors ${
                  edge.status === 'completed'
                    ? 'stroke-green-500'
                    : edge.status === 'active'
                    ? 'stroke-blue-500 stroke-2'
                    : 'stroke-gray-400 dark:stroke-gray-600'
                }`}
                strokeWidth="2"
                strokeDasharray={edge.status === 'pending' ? '4' : '0'}
              />
              <polygon
                points={`${toNode.position.x},${toNode.position.y + 30} ${toNode.position.x - 8},${toNode.position.y + 25} ${toNode.position.x - 8},${toNode.position.y + 35}`}
                className={`fill-current ${
                  edge.status === 'completed'
                    ? 'text-green-500'
                    : edge.status === 'active'
                    ? 'text-blue-500'
                    : 'text-gray-400 dark:text-gray-600'
                }`}
              />
            </g>
          );
        })}
      </svg>

      {/* Nodes */}
      {nodes.map((node) => (
        <div
          key={node.id}
          className={`absolute w-40 p-3 rounded-lg border-2 cursor-pointer transition-all hover:shadow-lg ${statusColors[node.status]} ${
            selectedNode === node.id ? 'ring-4 ring-blue-300 dark:ring-blue-600' : ''
          }`}
          style={{
            left: `${node.position.x}px`,
            top: `${node.position.y}px`,
          }}
          onClick={() => {
            setSelectedNode(node.id);
            onNodeClick?.(node.id);
          }}
        >
          <div className="flex items-center gap-2">
            <span className="text-lg" style={{ color: getStatusColor(node.status) }}>
              {statusIcons[node.status]}
            </span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {node.label}
            </span>
          </div>

          {node.result && (
            <div className="mt-2 text-xs text-gray-600 dark:text-gray-400 truncate">
              {node.result}
            </div>
          )}
        </div>
      ))}

      {/* Selected Node Details */}
      {selectedNode && (
        <Card className="absolute bottom-4 right-4 p-4 max-w-sm">
          {(() => {
            const node = nodes.find((n) => n.id === selectedNode);
            if (!node) return null;

            return (
              <div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                  {node.label}
                </h3>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Status:</span>
                    <span className={`font-medium ${
                      node.status === 'completed' ? 'text-green-600 dark:text-green-400' :
                      node.status === 'running' ? 'text-blue-600 dark:text-blue-400' :
                      node.status === 'failed' ? 'text-red-600 dark:text-red-400' :
                      'text-gray-600 dark:text-gray-400'
                    }`}>
                      {node.status.charAt(0).toUpperCase() + node.status.slice(1)}
                    </span>
                  </div>
                  {node.result && (
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Result:</span>
                      <p className="text-gray-900 dark:text-white mt-1">{node.result}</p>
                    </div>
                  )}
                </div>
              </div>
            );
          })()}
        </Card>
      )}
    </div>
  );
}

function getStatusColor(status: FlowNode['status']): string {
  switch (status) {
    case 'pending':
      return '#9CA3AF';
    case 'running':
      return '#3B82F6';
    case 'completed':
      return '#10B981';
    case 'failed':
      return '#EF4444';
    default:
      return '#9CA3AF';
  }
}
