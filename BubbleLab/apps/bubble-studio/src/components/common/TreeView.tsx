/**
 * TreeView Component
 * Hierarchical data display with expand/collapse
 */

import { useState } from 'react';

interface TreeNode {
  id: string;
  label: string;
  children?: TreeNode[];
  icon?: React.ReactNode;
  data?: unknown;
}

interface TreeViewProps {
  data: TreeNode[];
  selectable?: boolean;
  defaultExpanded?: string[];
  onSelect?: (nodeId: string, node: TreeNode) => void;
  className?: string;
}

export function TreeView({
  data,
  selectable = true,
  defaultExpanded = [],
  onSelect,
  className = '',
}: TreeViewProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(
    new Set(defaultExpanded)
  );
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const toggleNode = (nodeId: string) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(nodeId)) {
      newExpanded.delete(nodeId);
    } else {
      newExpanded.add(nodeId);
    }
    setExpandedNodes(newExpanded);
  };

  const handleSelect = (nodeId: string, node: TreeNode) => {
    if (!selectable) return;
    setSelectedNode(nodeId);
    onSelect?.(nodeId, node);
  };

  return (
    <div className={className}>
      {data.map((node) => (
        <TreeNodeItem
          key={node.id}
          node={node}
          level={0}
          expandedNodes={expandedNodes}
          selectedNode={selectedNode}
          onToggle={toggleNode}
          onSelect={handleSelect}
          selectable={selectable}
        />
      ))}
    </div>
  );
}

interface TreeNodeItemProps {
  node: TreeNode;
  level: number;
  expandedNodes: Set<string>;
  selectedNode: string | null;
  onToggle: (nodeId: string) => void;
  onSelect: (nodeId: string, node: TreeNode) => void;
  selectable: boolean;
}

function TreeNodeItem({
  node,
  level,
  expandedNodes,
  selectedNode,
  onToggle,
  onSelect,
  selectable,
}: TreeNodeItemProps) {
  const hasChildren = node.children && node.children.length > 0;
  const isExpanded = expandedNodes.has(node.id);
  const isSelected = selectedNode === node.id;

  return (
    <div>
      <div
        className={`
          flex items-center py-1.5 px-2 cursor-pointer
          ${isSelected ? 'bg-blue-100 dark:bg-blue-900/30' : 'hover:bg-gray-100 dark:hover:bg-gray-800'}
          ${selectable ? '' : 'cursor-default'}
        `}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
        onClick={() => {
          if (hasChildren) {
            onToggle(node.id);
          }
          if (selectable) {
            onSelect(node.id, node);
          }
        }}
      >
        {/* Expand/Collapse Icon */}
        <span
          className={`w-4 h-4 flex items-center justify-center mr-1 ${
            hasChildren ? 'cursor-pointer' : ''
          }`}
        >
          {hasChildren ? (
            <svg
              className={`w-3 h-3 text-gray-500 transition-transform ${
                isExpanded ? 'transform rotate-90' : ''
              }`}
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                clipRule="evenodd"
              />
            </svg>
          ) : null}
        </span>

        {/* Node Icon */}
        {node.icon && (
          <span className="w-4 h-4 mr-2 flex items-center justify-center text-gray-500">
            {node.icon}
          </span>
        )}

        {/* Label */}
        <span className="text-sm text-gray-900 dark:text-white">
          {node.label}
        </span>
      </div>

      {/* Children */}
      {hasChildren && isExpanded && node.children && (
        <div>
          {node.children.map((child) => (
            <TreeNodeItem
              key={child.id}
              node={child}
              level={level + 1}
              expandedNodes={expandedNodes}
              selectedNode={selectedNode}
              onToggle={onToggle}
              onSelect={onSelect}
              selectable={selectable}
            />
          ))}
        </div>
      )}
    </div>
  );
}
