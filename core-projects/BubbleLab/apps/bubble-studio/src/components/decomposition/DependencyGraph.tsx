import { useMemo } from 'react';

export interface DependencyGraphProps {
  /** Node identifiers (typically `WorkflowSubProblem.id` values). */
  nodes: string[];
  /** Dependency adjacency: `edges[id]` lists the sub-problem ids `id` depends on. */
  edges: Record<string, string[]>;
  /** Optional topological execution order (e.g. from `WorkflowDependencyGraph.execution_order`). */
  executionOrder?: string[];
}

const NODE_W = 180;
const NODE_H = 40;
const GAP_X = 70;
const GAP_Y = 26;
const MARGIN = 24;

/**
 * Dependency-free SVG visualization of a `WorkflowDependencyGraph`.
 *
 * Mirrors the canonical wire shape: `edges` is `Record<string, string[]>` and there
 * is no explicit `nodes` field, so the panel derives the node set from the plan's
 * sub-problem ids and passes it in. Nodes are laid out in topological layers
 * (longest-path from roots) and connected with directed arrows. Cycles are tolerated
 * via a bounded relaxation pass.
 */
export function DependencyGraph({ nodes, edges, executionOrder }: DependencyGraphProps) {
  const { positioned, width, height } = useMemo(
    () => computeLayout(nodes, edges),
    [nodes, edges]
  );

  const edgeList = useMemo(() => {
    const list: Array<{ from: string; to: string }> = [];
    for (const [target, deps] of Object.entries(edges)) {
      for (const dep of deps) {
        if (nodes.includes(dep)) {
          list.push({ from: dep, to: target });
        }
      }
    }
    return list;
  }, [edges, nodes]);

  if (nodes.length === 0) {
    return (
      <div className="rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4 text-sm text-gray-500">
        No sub-problems to visualize yet.
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-[#2b2b2b] bg-[#0d0d0d] p-4">
      <h3 className="text-base font-medium text-white">Dependency Graph</h3>

      {executionOrder && executionOrder.length > 0 && (
        <div className="mt-2">
          <span className="text-xs text-gray-400">Computed execution order:</span>
          <div className="mt-1 flex flex-wrap items-center gap-1">
            {executionOrder.map((id, i) => (
              <span key={`${id}-${i}`} className="flex items-center gap-1">
                <span className="rounded bg-[#1c1c1c] px-2 py-0.5 font-mono text-xs text-emerald-300">
                  {id}
                </span>
                {i < executionOrder.length - 1 && (
                  <span className="text-gray-600">→</span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="thin-scrollbar mt-3 overflow-auto">
        <svg
          width={Math.max(width, NODE_W + MARGIN * 2)}
          height={Math.max(height, NODE_H + MARGIN * 2)}
          className="min-w-full"
          role="img"
          aria-label="Workflow sub-problem dependency graph"
        >
          <defs>
            <marker
              id="decomp-arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280" />
            </marker>
          </defs>

          {edgeList.map(({ from, to }, i) => {
            const a = positioned[from];
            const b = positioned[to];
            if (!a || !b) return null;
            const x1 = a.x + NODE_W;
            const y1 = a.y + NODE_H / 2;
            const x2 = b.x;
            const y2 = b.y + NODE_H / 2;
            const midX = (x1 + x2) / 2;
            return (
              <path
                key={`edge-${i}`}
                d={`M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`}
                fill="none"
                stroke="#4b5563"
                strokeWidth={1.5}
                markerEnd="url(#decomp-arrow)"
              />
            );
          })}

          {Object.entries(positioned).map(([id, pos]) => (
              <g key={id}>
                <rect
                  x={pos.x}
                  y={pos.y}
                  width={NODE_W}
                  height={NODE_H}
                  rx={6}
                  fill="#161616"
                  stroke="#3a3a3a"
                  strokeWidth={1}
                />
                <text
                  x={pos.x + NODE_W / 2}
                  y={pos.y + NODE_H / 2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize={11}
                  fill="#e5e7eb"
                  fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
                >
                  {truncate(id, 22)}
                </text>
              </g>
            ))}
        </svg>
      </div>

      {edgeList.length === 0 && (
        <p className="mt-2 text-xs text-gray-600">
          No dependency edges defined — sub-problems are independent.
        </p>
      )}
    </div>
  );
}

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function computeLayout(
  nodes: string[],
  edges: Record<string, string[]>
): { positioned: Record<string, { x: number; y: number }>; width: number; height: number } {
  const byId = new Set(nodes);
  const depsOf = (id: string): string[] =>
    (edges[id] ?? []).filter((d) => byId.has(d) && d !== id);

  // Longest-path layering with a bounded relaxation pass to tolerate cycles.
  const layer: Record<string, number> = {};
  const queue = [...nodes];
  let guard = 0;
  const maxIterations = nodes.length * nodes.length + nodes.length + 10;
  while (queue.length > 0 && guard < maxIterations) {
    guard += 1;
    const id = queue.shift() as string;
    const parentLayer = Math.max(-1, ...depsOf(id).map((d) => layer[d] ?? -1));
    const desired = parentLayer + 1;
    if (layer[id] === undefined || desired > layer[id]) {
      layer[id] = desired;
      for (const [other, deps] of Object.entries(edges)) {
        if (deps.includes(id) && byId.has(other)) {
          queue.push(other);
        }
      }
    }
  }
  for (const id of nodes) {
    if (layer[id] === undefined) layer[id] = 0;
  }

  const layers = new Map<number, string[]>();
  for (const id of nodes) {
    const l = layer[id];
    if (!layers.has(l)) layers.set(l, []);
    layers.get(l)?.push(id);
  }

  const positioned: Record<string, { x: number; y: number }> = {};
  let maxLayer = 0;
  let maxRows = 0;
  for (const [l, ids] of layers) {
    maxLayer = Math.max(maxLayer, l);
    maxRows = Math.max(maxRows, ids.length);
    ids.forEach((id, row) => {
      positioned[id] = {
        x: MARGIN + l * (NODE_W + GAP_X),
        y: MARGIN + row * (NODE_H + GAP_Y),
      };
    });
  }

  const width = MARGIN * 2 + (maxLayer + 1) * (NODE_W + GAP_X) - GAP_X;
  const height = MARGIN * 2 + maxRows * (NODE_H + GAP_Y) - GAP_Y;
  return { positioned, width, height };
}
