"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.DependencyGraphTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const select_1 = require("@/components/ui/select");
const badge_1 = require("@/components/ui/badge");
const tabs_1 = require("@/components/ui/tabs");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const computeLayers = (nodes, edges) => {
    const inDegree = {};
    const outgoing = {};
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
    const layers = {};
    queue.forEach((node) => {
        layers[node] = 0;
    });
    while (queue.length) {
        const current = queue.shift();
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
const detectCycles = (nodes, edges) => {
    const adjacency = {};
    nodes.forEach((node) => {
        adjacency[node] = [];
    });
    edges.forEach((edge) => {
        adjacency[edge.from] = adjacency[edge.from] || [];
        adjacency[edge.from].push(edge.to);
    });
    const visited = new Set();
    const stack = new Set();
    const cycles = [];
    const dfs = (node, path) => {
        visited.add(node);
        stack.add(node);
        path.push(node);
        for (const neighbor of adjacency[node] || []) {
            if (!visited.has(neighbor)) {
                dfs(neighbor, path);
            }
            else if (stack.has(neighbor)) {
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
const topologicalSort = (nodes, edges) => {
    const inDegree = {};
    const adjacency = {};
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
    const order = [];
    while (queue.length) {
        const node = queue.shift();
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
const computeLongestPath = (nodes, edges) => {
    const { order, hasCycle } = topologicalSort(nodes, edges);
    if (hasCycle) {
        return null;
    }
    const adjacency = {};
    nodes.forEach((node) => {
        adjacency[node] = [];
    });
    edges.forEach((edge) => {
        adjacency[edge.from].push(edge.to);
    });
    const distances = {};
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
const buildGraphLayout = (nodes, edges) => {
    const nodeIds = nodes.map((node) => node.id);
    const layers = computeLayers(nodeIds, edges);
    const maxLayer = Math.max(...Object.values(layers));
    const width = 900;
    const height = 520;
    const grouped = {};
    nodeIds.forEach((nodeId) => {
        const layer = layers[nodeId] ?? 0;
        if (!grouped[layer]) {
            grouped[layer] = [];
        }
        grouped[layer].push(nodeId);
    });
    const positionMap = {};
    Object.entries(grouped).forEach(([layerStr, ids]) => {
        const layer = Number(layerStr);
        const x = (width / (maxLayer + 2)) * (layer + 1);
        ids.forEach((id, index) => {
            const y = (height / (ids.length + 1)) * (index + 1);
            positionMap[id] = { x, y, layer };
        });
    });
    const layoutNodes = nodes.map((node) => ({
        ...node,
        x: positionMap[node.id]?.x ?? 0,
        y: positionMap[node.id]?.y ?? 0,
        layer: positionMap[node.id]?.layer ?? 0,
    }));
    return { width, height, nodes: layoutNodes, edges };
};
const DependencyGraphTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [sovereignPlans, setSovereignPlans] = (0, react_1.useState)([]);
    const [workflows, setWorkflows] = (0, react_1.useState)([]);
    const [dataSource, setDataSource] = (0, react_1.useState)("sovereign");
    const [selectedPlanId, setSelectedPlanId] = (0, react_1.useState)("");
    const [selectedWorkflowId, setSelectedWorkflowId] = (0, react_1.useState)("");
    const [graphNodes, setGraphNodes] = (0, react_1.useState)([]);
    const [graphEdges, setGraphEdges] = (0, react_1.useState)([]);
    const [cycles, setCycles] = (0, react_1.useState)([]);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const loadPlans = async () => {
        setErrorMessage(null);
        try {
            const [plansRes, workflowsRes] = await Promise.all([
                openevolveApi_1.openevolveApi.listSovereignPlans(apiConfig),
                openevolveApi_1.openevolveApi.listWorkflows(apiConfig),
            ]);
            setSovereignPlans(plansRes.plans || []);
            setWorkflows(workflowsRes.workflows || []);
            if (!selectedPlanId && plansRes.plans?.length) {
                setSelectedPlanId(plansRes.plans[0].id);
            }
            if (!selectedWorkflowId && workflowsRes.workflows?.length) {
                setSelectedWorkflowId(workflowsRes.workflows[0].workflow_id);
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load dependency data.");
        }
    };
    (0, react_1.useEffect)(() => {
        loadPlans();
    }, [apiConfig.apiKey]);
    (0, react_1.useEffect)(() => {
        const buildFromSovereign = (plan) => {
            const nodes = plan.sub_problems || [];
            const edges = [];
            if (plan.dependency_graph?.edges) {
                Object.entries(plan.dependency_graph.edges).forEach(([nodeId, deps]) => {
                    (deps || []).forEach((dep) => {
                        edges.push({ from: dep, to: nodeId });
                    });
                });
            }
            else {
                nodes.forEach((node) => {
                    (node.dependencies || []).forEach((dep) => {
                        edges.push({ from: dep, to: node.id });
                    });
                });
            }
            const layout = buildGraphLayout(nodes.map((node) => ({
                id: node.id,
                label: node.title || node.id,
                status: node.status,
                complexity: Number(node.complexity_score?.value ?? node.complexity_score?.score ?? 0),
            })), edges);
            setGraphNodes(layout.nodes);
            setGraphEdges(layout.edges);
            setCycles(detectCycles(nodes.map((node) => node.id), edges));
        };
        const buildFromWorkflow = async (workflowId) => {
            if (!workflowId)
                return;
            try {
                const res = await openevolveApi_1.openevolveApi.getWorkflowPlan(workflowId, apiConfig);
                const nodes = res.plan.sub_problems;
                const edges = [];
                Object.entries(res.dependency_graph.edges || {}).forEach(([nodeId, deps]) => {
                    (deps || []).forEach((dep) => edges.push({ from: dep, to: nodeId }));
                });
                const layout = buildGraphLayout(nodes.map((node) => ({
                    id: node.id,
                    label: node.description.slice(0, 40) || node.id,
                    status: node.status,
                    complexity: node.ai_suggested_complexity_score ?? null,
                })), edges);
                setGraphNodes(layout.nodes);
                setGraphEdges(layout.edges);
                setCycles(detectCycles(nodes.map((node) => node.id), edges));
            }
            catch (error) {
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
    const layout = (0, react_1.useMemo)(() => buildGraphLayout(graphNodes, graphEdges), [graphNodes, graphEdges]);
    const nodeIds = graphNodes.map((node) => node.id);
    const topo = (0, react_1.useMemo)(() => topologicalSort(nodeIds, graphEdges), [nodeIds, graphEdges]);
    const longestChain = (0, react_1.useMemo)(() => computeLongestPath(nodeIds, graphEdges), [nodeIds, graphEdges]);
    const dependencyMatrix = (0, react_1.useMemo)(() => {
        const matrix = {};
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
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Dependency Visualizer</card_1.CardTitle>
          <card_1.CardDescription>Inspect dependency graphs for decomposition plans.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key</label>
              <input className="h-9 w-full rounded border px-3 text-sm" type="password" value={apiKey} onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            try {
                globalThis.localStorage?.setItem("openevolve_api_key", value);
            }
            catch {
                // ignore storage errors
            }
        }}/>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Data Source</label>
              <select_1.Select value={dataSource} onValueChange={(value) => setDataSource(value)}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue placeholder="Select source"/>
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  <select_1.SelectItem value="sovereign">Sovereign Plans</select_1.SelectItem>
                  <select_1.SelectItem value="workflow">Active Workflows</select_1.SelectItem>
                </select_1.SelectContent>
              </select_1.Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">
                {dataSource === "sovereign" ? "Plan" : "Workflow"}
              </label>
              <select_1.Select value={dataSource === "sovereign" ? selectedPlanId : selectedWorkflowId} onValueChange={(value) => dataSource === "sovereign" ? setSelectedPlanId(value) : setSelectedWorkflowId(value)}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue placeholder="Select"/>
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  {(dataSource === "sovereign" ? sovereignPlans : workflows).map((item) => (<select_1.SelectItem key={dataSource === "sovereign" ? item.id : item.workflow_id} value={dataSource === "sovereign"
                ? item.id
                : item.workflow_id}>
                      {dataSource === "sovereign"
                ? item.id
                : item.workflow_id}
                    </select_1.SelectItem>))}
                </select_1.SelectContent>
              </select_1.Select>
            </div>
          </div>

          <tabs_1.Tabs defaultValue="graph">
            <tabs_1.TabsList className="flex flex-wrap gap-2">
              <tabs_1.TabsTrigger value="graph">Graph View</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="analysis">Analysis</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="matrix">Matrix</tabs_1.TabsTrigger>
            </tabs_1.TabsList>

            <tabs_1.TabsContent value="graph" className="mt-4 space-y-4">
              <div className="rounded border p-4">
                <svg viewBox={`0 0 ${layout.width} ${layout.height}`} className="h-[520px] w-full">
                  <defs>
                    <marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto">
                      <path d="M0,0 L0,6 L9,3 z" fill="#888"/>
                    </marker>
                  </defs>
                  {layout.edges.map((edge, index) => {
            const from = layout.nodes.find((node) => node.id === edge.from);
            const to = layout.nodes.find((node) => node.id === edge.to);
            if (!from || !to)
                return null;
            return (<line key={`edge-${index}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y} stroke="#888" strokeWidth={2} markerEnd="url(#arrow)"/>);
        })}
                  {layout.nodes.map((node) => (<g key={node.id}>
                      <circle cx={node.x} cy={node.y} r={18} fill={node.status === "solved"
                ? "#22c55e"
                : node.status === "failed"
                    ? "#ef4444"
                    : node.status === "in_progress"
                        ? "#3b82f6"
                        : "#f59e0b"}/>
                      <text x={node.x} y={node.y - 24} fontSize="10" textAnchor="middle">
                        {node.label}
                      </text>
                    </g>))}
                </svg>
              </div>

              {cycles.length > 0 && (<card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Circular Dependencies</card_1.CardTitle>
                    <card_1.CardDescription>Cycles detected in the dependency graph.</card_1.CardDescription>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2 text-sm">
                    {cycles.map((cycle, index) => (<div key={`cycle-${index}`} className="rounded border p-2">
                        Cycle {index + 1}: {cycle.join(" → ")}
                      </div>))}
                  </card_1.CardContent>
                </card_1.Card>)}

              <div className="grid gap-4 md:grid-cols-2">
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Node Summary</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2 text-sm">
                    {graphNodes.length === 0 ? (<div className="text-muted-foreground">No nodes loaded.</div>) : (graphNodes.map((node) => (<div key={node.id} className="flex items-center justify-between">
                          <span>{node.label}</span>
                          <badge_1.Badge variant="secondary">{node.status ?? "pending"}</badge_1.Badge>
                        </div>)))}
                  </card_1.CardContent>
                </card_1.Card>
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Edge Summary</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2 text-sm">
                    {graphEdges.length === 0 ? (<div className="text-muted-foreground">No edges loaded.</div>) : (graphEdges.slice(0, 10).map((edge, index) => (<div key={`edge-row-${index}`}>
                          {edge.from} → {edge.to}
                        </div>)))}
                  </card_1.CardContent>
                </card_1.Card>
              </div>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="analysis" className="mt-4 space-y-4">
              <div className="grid gap-4 md:grid-cols-4">
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Total Sub-Problems</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="text-sm">{graphNodes.length}</card_1.CardContent>
                </card_1.Card>
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Dependencies</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="text-sm">{graphEdges.length}</card_1.CardContent>
                </card_1.Card>
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Longest Chain</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="text-sm">
                    {longestChain !== null ? longestChain : "cycle detected"}
                  </card_1.CardContent>
                </card_1.Card>
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Cycles</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="text-sm">{cycles.length}</card_1.CardContent>
                </card_1.Card>
              </div>

              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Execution Order</card_1.CardTitle>
                  <card_1.CardDescription>Topological order based on dependencies.</card_1.CardDescription>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  {topo.hasCycle ? (<div className="text-red-500">
                      Cycles detected. Resolve dependencies to compute execution order.
                    </div>) : (topo.order.map((nodeId, index) => (<div key={nodeId} className="flex items-center gap-2">
                        <badge_1.Badge variant="secondary">{index + 1}</badge_1.Badge>
                        <span>{nodeId}</span>
                      </div>)))}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="matrix" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Dependency Matrix</card_1.CardTitle>
                  <card_1.CardDescription>Row depends on column.</card_1.CardDescription>
                </card_1.CardHeader>
                <card_1.CardContent className="overflow-auto">
                  {nodeIds.length === 0 ? (<div className="text-muted-foreground">No dependency data loaded.</div>) : (<table className="w-full text-xs border-collapse">
                      <thead>
                        <tr>
                          <th className="border px-2 py-1 text-left">Sub-problem</th>
                          {nodeIds.map((id) => (<th key={id} className="border px-2 py-1 text-left">
                              {id}
                            </th>))}
                        </tr>
                      </thead>
                      <tbody>
                        {nodeIds.map((row) => (<tr key={row}>
                            <td className="border px-2 py-1 font-semibold">{row}</td>
                            {nodeIds.map((col) => (<td key={`${row}-${col}`} className="border px-2 py-1 text-center">
                                {dependencyMatrix[row]?.[col] ? "✔" : ""}
                              </td>))}
                          </tr>))}
                      </tbody>
                    </table>)}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>
          </tabs_1.Tabs>

          <button_1.Button variant="outline" onClick={loadPlans}>
            Refresh Data
          </button_1.Button>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.DependencyGraphTab = DependencyGraphTab;
//# sourceMappingURL=DependencyGraphTab.js.map