/**
 * KnowledgeGraphView
 * Dependency-free SVG rendering of the knowledge graph returned by
 * `getKnowledgeGraph`. Nodes are laid out on a circle and edges are drawn as
 * lines between them. Hovering a node highlights its connections.
 */

import { useMemo, useState } from 'react';
import type { KnowledgeGraph, KnowledgeGraphNode } from '@/types/openevolve';

interface KnowledgeGraphViewProps {
  graph: KnowledgeGraph;
}

const WIDTH = 800;
const HEIGHT = 560;
const RADIUS = 220;
const CENTER_X = WIDTH / 2;
const CENTER_Y = HEIGHT / 2 + 10;

const NODE_COLORS = [
  '#3b82f6',
  '#10b981',
  '#8b5cf6',
  '#f59e0b',
  '#f43f5e',
  '#06b6d4',
  '#d946ef',
];

export function KnowledgeGraphView({ graph }: KnowledgeGraphViewProps) {
  const [hovered, setHovered] = useState<string | null>(null);

  const { positions, nodeColor } = useMemo(() => {
    const nodes = graph.nodes ?? [];
    const typeIndex = new Map<string, number>();
    const positions = new Map<string, { x: number; y: number }>();

    nodes.forEach((node, index) => {
      const angle = (index / Math.max(1, nodes.length)) * 2 * Math.PI - Math.PI / 2;
      positions.set(node.id, {
        x: CENTER_X + RADIUS * Math.cos(angle),
        y: CENTER_Y + RADIUS * Math.sin(angle),
      });
      if (node.type && !typeIndex.has(node.type)) {
        typeIndex.set(node.type, typeIndex.size);
      }
    });

    const nodeColor = (node: KnowledgeGraphNode): string => {
      const key = node.type ?? node.id;
      const idx = typeIndex.get(key) ?? 0;
      return NODE_COLORS[idx % NODE_COLORS.length];
    };

    return { positions, nodeColor };
  }, [graph.nodes]);

  const edges = graph.edges ?? [];

  if (graph.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center py-10 text-sm text-gray-500 dark:text-gray-400">
        The knowledge graph is empty.
      </div>
    );
  }

  const connected = useMemo(() => {
    if (!hovered) return null;
    const set = new Set<string>([hovered]);
    edges.forEach((e) => {
      if (e.source === hovered) set.add(e.target);
      if (e.target === hovered) set.add(e.source);
    });
    return set;
  }, [hovered, edges]);

  return (
    <div className="thin-scrollbar overflow-auto rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-auto w-full"
        role="img"
        aria-label="Knowledge graph"
      >
        {edges.map((edge, index) => {
          const from = positions.get(edge.source);
          const to = positions.get(edge.target);
          if (!from || !to) return null;
          const isActive =
            !connected || (connected.has(edge.source) && connected.has(edge.target));
          return (
            <line
              key={`${edge.source}-${edge.target}-${index}`}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke={isActive ? '#94a3b8' : '#475569'}
              strokeWidth={isActive ? 1.5 : 1}
              strokeOpacity={isActive ? 0.6 : 0.2}
            />
          );
        })}

        {graph.nodes.map((node) => {
          const pos = positions.get(node.id);
          if (!pos) return null;
          const color = nodeColor(node);
          const isDimmed = connected ? !connected.has(node.id) : false;
          const size = 6 + Math.min(10, (node.usage ?? 0) / 2);
          return (
            <g
              key={node.id}
              transform={`translate(${pos.x}, ${pos.y})`}
              onMouseEnter={() => setHovered(node.id)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: 'pointer', opacity: isDimmed ? 0.25 : 1 }}
            >
              <circle
                r={size}
                fill={color}
                stroke="#0f172a"
                strokeWidth={1}
              />
              <text
                x={0}
                y={size + 12}
                textAnchor="middle"
                className="fill-gray-600 text-[10px] dark:fill-gray-300"
              >
                {truncate(node.id, 16)}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="mt-3 flex flex-wrap gap-2">
        {Array.from(
          new Set(graph.nodes.map((n) => n.type).filter(Boolean) as string[])
        ).map((type, index) => (
          <span
            key={type}
            className="inline-flex items-center gap-1 text-xs text-gray-600 dark:text-gray-300"
          >
            <span
              className="inline-block h-3 w-3 rounded-full"
              style={{ backgroundColor: NODE_COLORS[index % NODE_COLORS.length] }}
            />
            {type}
          </span>
        ))}
      </div>
    </div>
  );
}

function truncate(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max)}…` : value;
}
