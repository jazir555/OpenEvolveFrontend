import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { openevolveApi } from "../../../lib/openevolveApi";
import type {
  SovereignPlan,
  WorkflowSummary,
  WorkflowPlanResponse,
  WorkflowSubProblem,
  SovereignSubProblem,
} from "../../../lib/types";

interface GraphNode {
  id: string;
  label: string;
  status?: string;
  complexity?: number | null;
  x: number;
  y: number;
  layer: number;
}

interface GraphEdge {
  from: string;
  to: string;
}

const computeLayers = (nodes: string[], edges: GraphEdge[]) => {
  const inDegree: Record<string, number> = {};
  const outgoing: Record<string, string[]> = {};
  nodes.forEach((node) => {
    inDegree[node] = 0;
    outgoing[node] = [];
  });
  edges.forEach((edge) => {
    if (!outgoing[edge.from]) {
      outgoing[edge.from] = [];
    }
    outgoing[edge.from].push(edge.to);
    inDegree[edge.to] = (inDegree[edge.to] ?? 0) + 1;
  });

  const queue = nodes.filter((node) => (inDegree[node] ?? 0) === 0);
  const layers: Record<string, number> = {};
  queue.forEach((node) => {
    layers[node] = 0;
  });

  while (queue.length) {
    const current = queue.shift() as string;
    const currentLayer = layers[current] ?? 0;
    (outgoing[current] || []).forEach((neighbor) => {
      layers[neighbor] = Math.max(layers[neighbor] ?? 0, currentLayer + 1);
      inDegree[neighbor] -= 1;
      if (inDegree[neighbor] === 0) {
        queue.push(neighbor);
      }
    });
  }

  const maxLayer = Object.values(layers).reduce((acc, value) => Math.max(acc, value), 0);
  nodes.forEach((node) => {
    if (layers[node] === undefined) {
      layers[node] = maxLayer + 1;
    }
  });

  return layers;
};

const detectCycles = (nodes: string[], edges: GraphEdge[]) => {
  const adjacency: Record<string, string[]> = {};
  nodes.forEach((node) => {
    adjacency[node] = [];
  });
  edges.forEach((edge) => {
    adjacency[edge.from] = adjacency[edge.from] || [];
    adjacency[edge.from].push(edge.to);
  });

  const visited = new Set<string>();
  const stack = new Set<string>();
  const cycles: string[][] = [];

  const dfs = (node: string, path: string[]) => {
    visited.add(node);
    stack.add(node);
    path.push(node);

    for (const neighbor of adjacency[node] || []) {
      if (!visited.has(neighbor)) {
        dfs(neighbor, path);
      } else if (stack.has(neighbor)) {
        const idx = path.indexOf(neighbor);
        if (idx >= 0) {
          cycles.push(path.slice(idx));
        }
      }
    }

    stack.delete(node);
    path.pop();
  };

  nodes.forEach((node) => {
    if (!visited.has(node)) {
      dfs(node, []);
    }
  });

  return cycles;
};

const topologicalSort = (nodes: string[], edges: GraphEdge[]) => {
  const inDegree: Record<string, number> = {};
  const adjacency: Record<string, string[]> = {};
  nodes.forEach((node) => {
    inDegree[node] = 0;
    adjacency[node] = [];
  });
  edges.forEach((edge) => {
    adjacency[edge.from] = adjacency[edge.from] || [];
    adjacency[edge.from].push(edge.to);
    inDegree[edge.to] = (inDegree[edge.to] ?? 0) + 1;
  });

  const queue = nodes.filter((node) => (inDegree[node] ?? 0) === 0);
  const order: string[] = [];

  while (queue.length) {
    const node = queue.shift() as string;
    order.push(node);
    (adjacency[node] || []).forEach((neighbor) => {
      inDegree[neighbor] -= 1;
      if (inDegree[neighbor] === 0) {
        queue.push(neighbor);
      }
    });
  }

  return { order, hasCycle: order.length !== nodes.length };
};

const computeLongestPath = (nodes: string[], edges: GraphEdge[]) => {
  const { order, hasCycle } = topologicalSort(nodes, edges);
  if (hasCycle) {
    return null;
  }

  const adjacency: Record<string, string[]> = {};
  nodes.forEach((node) => {
    adjacency[node] = [];
  });
  edges.forEach((edge) => {
    adjacency[edge.from].push(edge.to);
  });

  const distances: Record<string, number> = {};
  nodes.forEach((node) => {
    distances[node] = 0;
  });

  order.forEach((node) => {
    (adjacency[node] || []).forEach((neighbor) => {
      distances[neighbor] = Math.max(distances[neighbor] ?? 0, (distances[node] ?? 0) + 1);
    });
  });

  return Math.max(...Object.values(distances));
};

const buildGraphLayout = (
  nodes: Array<{ id: string; label: string; status?: string; complexity?: number | null }>,
  edges: GraphEdge[],
) => {
  const nodeIds = nodes.map((node) => node.id);
  const layers = computeLayers(nodeIds, edges);
  const maxLayer = Math.max(...Object.values(layers));
  const width = 900;
  const height = 520;

  const grouped: Record<number, string[]> = {};
  nodeIds.forEach((nodeId) => {
    const layer = layers[nodeId] ?? 0;
    if (!grouped[layer]) {
      grouped[layer] = [];
    }
    grouped[layer].push(nodeId);
  });

  const positionMap: Record<string, { x: number; y: number; layer: number }> = {};
  Object.entries(grouped).forEach(([layerStr, ids]) => {
    const layer = Number(layerStr);
    const x = (width / (maxLayer + 2)) * (layer + 1);
    ids.forEach((id, index) => {
      const y = (height / (ids.length + 1)) * (index + 1);
      positionMap[id] = { x, y, layer };
    });
  });

  const layoutNodes: GraphNode[] = nodes.map((node) => ({
    ...node,
    x: positionMap[node.id]?.x ?? 0,
    y: positionMap[node.id]?.y ?? 0,
    layer: positionMap[node.id]?.layer ?? 0,
  }));

  return { width, height, nodes: layoutNodes, edges };
};

export const DependencyGraphTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [sovereignPlans, setSovereignPlans] = useState<SovereignPlan[]>([]);
  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [dataSource, setDataSource] = useState<"sovereign" | "workflow">("sovereign");
  const [selectedPlanId, setSelectedPlanId] = useState<string>("");
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>("");
  const [graphNodes, setGraphNodes] = useState<GraphNode[]>([]);
  const [graphEdges, setGraphEdges] = useState<GraphEdge[]>([]);
  const [cycles, setCycles] = useState<string[][]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const loadPlans = async () => {
    setErrorMessage(null);
    try {
      const [plansRes, workflowsRes] = await Promise.all([
        openevolveApi.listSovereignPlans(apiConfig),
        openevolveApi.listWorkflows(apiConfig),
      ]);
      setSovereignPlans(plansRes.plans || []);
      setWorkflows(workflowsRes.workflows || []);
      if (!selectedPlanId && plansRes.plans?.length) {
        setSelectedPlanId(plansRes.plans[0].id);
      }
      if (!selectedWorkflowId && workflowsRes.workflows?.length) {
        setSelectedWorkflowId(workflowsRes.workflows[0].workflow_id);
      }
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load dependency data.");
    }
  };

  useEffect(() => {
    loadPlans();
  }, [apiConfig.apiKey]);

  useEffect(() => {
    const buildFromSovereign = (plan: SovereignPlan) => {
      const nodes = plan.sub_problems || [];
      const edges: GraphEdge[] = [];
      if (plan.dependency_graph?.edges) {
        Object.entries(plan.dependency_graph.edges).forEach(([nodeId, deps]) => {
          (deps || []).forEach((dep) => {
            edges.push({ from: dep, to: nodeId });
          });
        });
      } else {
        nodes.forEach((node) => {
          (node.dependencies || []).forEach((dep) => {
            edges.push({ from: dep, to: node.id });
          });
        });
      }

      const layout = buildGraphLayout(
        nodes.map((node: SovereignSubProblem) => ({
          id: node.id,
          label: node.title || node.id,
          status: node.status,
          complexity: Number((node.complexity_score as any)?.value ?? (node.complexity_score as any)?.score ?? 0),
        })),
        edges,
      );
      setGraphNodes(layout.nodes);
      setGraphEdges(layout.edges);
      setCycles(detectCycles(nodes.map((node) => node.id), edges));
    };

    const buildFromWorkflow = async (workflowId: string) => {
      if (!workflowId) return;
      try {
        const res: WorkflowPlanResponse = await openevolveApi.getWorkflowPlan(workflowId, apiConfig);
        const nodes = res.plan.sub_problems;
        const edges: GraphEdge[] = [];
        Object.entries(res.dependency_graph.edges || {}).forEach(([nodeId, deps]) => {
          (deps || []).forEach((dep) => edges.push({ from: dep, to: nodeId }));
        });

        const layout = buildGraphLayout(
          nodes.map((node: WorkflowSubProblem) => ({
            id: node.id,
            label: node.description.slice(0, 40) || node.id,
            status: node.status,
            complexity: node.ai_suggested_complexity_score ?? null,
          })),
          edges,
        );
        setGraphNodes(layout.nodes);
        setGraphEdges(layout.edges);
        setCycles(detectCycles(nodes.map((node) => node.id), edges));
      } catch (error: any) {
        setErrorMessage(error?.message ?? "Failed to load workflow plan.");
      }
    };

    if (dataSource === "sovereign" && selectedPlanId) {
      const plan = sovereignPlans.find((p) => p.id === selectedPlanId);
      if (plan) {
        buildFromSovereign(plan);
      }
    }
    if (dataSource === "workflow" && selectedWorkflowId) {
      buildFromWorkflow(selectedWorkflowId);
    }
  }, [dataSource, selectedPlanId, selectedWorkflowId, sovereignPlans, apiConfig]);

  const layout = useMemo(() => buildGraphLayout(graphNodes, graphEdges), [graphNodes, graphEdges]);
  const nodeIds = graphNodes.map((node) => node.id);
  const topo = useMemo(() => topologicalSort(nodeIds, graphEdges), [nodeIds, graphEdges]);
  const longestChain = useMemo(() => computeLongestPath(nodeIds, graphEdges), [nodeIds, graphEdges]);
  const dependencyMatrix = useMemo(() => {
    const matrix: Record<string, Record<string, boolean>> = {};
    nodeIds.forEach((row) => {
      matrix[row] = {};
      nodeIds.forEach((col) => {
        matrix[row][col] = false;
      });
    });
    graphEdges.forEach((edge) => {
      if (matrix[edge.to]) {
        matrix[edge.to][edge.from] = true;
      }
    });
    return matrix;
  }, [nodeIds, graphEdges]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Dependency Visualizer</CardTitle>
          <CardDescription>Inspect dependency graphs for decomposition plans.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key</label>
              <input
                className="h-9 w-full rounded border px-3 text-sm"
                type="password"
                value={apiKey}
                onChange={(event) => {
                  const value = event.target.value;
                  setApiKey(value);
                  try {
                    globalThis.localStorage?.setItem("openevolve_api_key", value);
                  } catch {
                    // ignore storage errors
                  }
                }}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Data Source</label>
              <Select value={dataSource} onValueChange={(value) => setDataSource(value as any)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select source" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="sovereign">Sovereign Plans</SelectItem>
                  <SelectItem value="workflow">Active Workflows</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">
                {dataSource === "sovereign" ? "Plan" : "Workflow"}
              </label>
              <Select
                value={dataSource === "sovereign" ? selectedPlanId : selectedWorkflowId}
                onValueChange={(value) =>
                  dataSource === "sovereign" ? setSelectedPlanId(value) : setSelectedWorkflowId(value)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  {(dataSource === "sovereign" ? sovereignPlans : workflows).map((item) => (
                    <SelectItem
                      key={dataSource === "sovereign" ? (item as SovereignPlan).id : (item as WorkflowSummary).workflow_id}
                      value={
                        dataSource === "sovereign"
                          ? (item as SovereignPlan).id
                          : (item as WorkflowSummary).workflow_id
                      }
                    >
                      {dataSource === "sovereign"
                        ? (item as SovereignPlan).id
                        : (item as WorkflowSummary).workflow_id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <Tabs defaultValue="graph">
            <TabsList className="flex flex-wrap gap-2">
              <TabsTrigger value="graph">Graph View</TabsTrigger>
              <TabsTrigger value="analysis">Analysis</TabsTrigger>
              <TabsTrigger value="matrix">Matrix</TabsTrigger>
            </TabsList>

            <TabsContent value="graph" className="mt-4 space-y-4">
              <div className="rounded border p-4">
                <svg
                  viewBox={`0 0 ${layout.width} ${layout.height}`}
                  className="h-[520px] w-full"
                >
                  <defs>
                    <marker
                      id="arrow"
                      markerWidth="10"
                      markerHeight="10"
                      refX="6"
                      refY="3"
                      orient="auto"
                    >
                      <path d="M0,0 L0,6 L9,3 z" fill="#888" />
                    </marker>
                  </defs>
                  {layout.edges.map((edge, index) => {
                    const from = layout.nodes.find((node) => node.id === edge.from);
                    const to = layout.nodes.find((node) => node.id === edge.to);
                    if (!from || !to) return null;
                    return (
                      <line
                        key={`edge-${index}`}
                        x1={from.x}
                        y1={from.y}
                        x2={to.x}
                        y2={to.y}
                        stroke="#888"
                        strokeWidth={2}
                        markerEnd="url(#arrow)"
                      />
                    );
                  })}
                  {layout.nodes.map((node) => (
                    <g key={node.id}>
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r={18}
                        fill={
                          node.status === "solved"
                            ? "#22c55e"
                            : node.status === "failed"
                            ? "#ef4444"
                            : node.status === "in_progress"
                            ? "#3b82f6"
                            : "#f59e0b"
                        }
                      />
                      <text x={node.x} y={node.y - 24} fontSize="10" textAnchor="middle">
                        {node.label}
                      </text>
                    </g>
                  ))}
                </svg>
              </div>

              {cycles.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Circular Dependencies</CardTitle>
                    <CardDescription>Cycles detected in the dependency graph.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {cycles.map((cycle, index) => (
                      <div key={`cycle-${index}`} className="rounded border p-2">
                        Cycle {index + 1}: {cycle.join(" → ")}
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}

              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Node Summary</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {graphNodes.length === 0 ? (
                      <div className="text-muted-foreground">No nodes loaded.</div>
                    ) : (
                      graphNodes.map((node) => (
                        <div key={node.id} className="flex items-center justify-between">
                          <span>{node.label}</span>
                          <Badge variant="secondary">{node.status ?? "pending"}</Badge>
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Edge Summary</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {graphEdges.length === 0 ? (
                      <div className="text-muted-foreground">No edges loaded.</div>
                    ) : (
                      graphEdges.slice(0, 10).map((edge, index) => (
                        <div key={`edge-row-${index}`}>
                          {edge.from} → {edge.to}
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="analysis" className="mt-4 space-y-4">
              <div className="grid gap-4 md:grid-cols-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Total Sub-Problems</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm">{graphNodes.length}</CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Dependencies</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm">{graphEdges.length}</CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Longest Chain</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm">
                    {longestChain !== null ? longestChain : "cycle detected"}
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Cycles</CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm">{cycles.length}</CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Execution Order</CardTitle>
                  <CardDescription>Topological order based on dependencies.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {topo.hasCycle ? (
                    <div className="text-red-500">
                      Cycles detected. Resolve dependencies to compute execution order.
                    </div>
                  ) : (
                    topo.order.map((nodeId, index) => (
                      <div key={nodeId} className="flex items-center gap-2">
                        <Badge variant="secondary">{index + 1}</Badge>
                        <span>{nodeId}</span>
                      </div>
                    ))
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="matrix" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Dependency Matrix</CardTitle>
                  <CardDescription>Row depends on column.</CardDescription>
                </CardHeader>
                <CardContent className="overflow-auto">
                  {nodeIds.length === 0 ? (
                    <div className="text-muted-foreground">No dependency data loaded.</div>
                  ) : (
                    <table className="w-full text-xs border-collapse">
                      <thead>
                        <tr>
                          <th className="border px-2 py-1 text-left">Sub-problem</th>
                          {nodeIds.map((id) => (
                            <th key={id} className="border px-2 py-1 text-left">
                              {id}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {nodeIds.map((row) => (
                          <tr key={row}>
                            <td className="border px-2 py-1 font-semibold">{row}</td>
                            {nodeIds.map((col) => (
                              <td key={`${row}-${col}`} className="border px-2 py-1 text-center">
                                {dependencyMatrix[row]?.[col] ? "✔" : ""}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          <Button variant="outline" onClick={loadPlans}>
            Refresh Data
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};
