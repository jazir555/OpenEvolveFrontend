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
exports.KnowledgeExplorerTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const tabs_1 = require("@/components/ui/tabs");
const checkbox_1 = require("@/components/ui/checkbox");
const select_1 = require("@/components/ui/select");
const badge_1 = require("@/components/ui/badge");
const separator_1 = require("@/components/ui/separator");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const readApiKey = () => {
    try {
        return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    }
    catch {
        return "";
    }
};
const buildGraphLayout = (entities, relationships) => {
    const nodes = entities.map((entity, index) => ({
        id: entity.name || `entity-${index}`,
        label: entity.name || `Entity ${index + 1}`,
    }));
    const edges = relationships
        .filter((rel) => rel.source && rel.target)
        .map((rel, index) => ({
        id: `edge-${index}`,
        source: rel.source,
        target: rel.target,
        label: rel.relation,
    }));
    return { nodes, edges };
};
const computeGraphStats = (nodes, edges) => {
    const nodeCount = nodes.length;
    const edgeCount = edges.length;
    const density = nodeCount > 1 ? edgeCount / (nodeCount * (nodeCount - 1)) : 0;
    const adjacency = {};
    nodes.forEach((node) => {
        adjacency[node] = [];
    });
    edges.forEach((edge) => {
        if (!adjacency[edge.source])
            adjacency[edge.source] = [];
        adjacency[edge.source].push(edge.target);
    });
    const visited = new Set();
    const stack = nodes.length ? [nodes[0]] : [];
    while (stack.length) {
        const node = stack.pop();
        if (visited.has(node))
            continue;
        visited.add(node);
        (adjacency[node] || []).forEach((neighbor) => {
            if (!visited.has(neighbor))
                stack.push(neighbor);
        });
    }
    return {
        nodeCount,
        edgeCount,
        density,
        isConnected: nodeCount === 0 ? true : visited.size === nodeCount,
    };
};
const KnowledgeExplorerTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(readApiKey);
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [queryText, setQueryText] = (0, react_1.useState)("");
    const [sources, setSources] = (0, react_1.useState)({ bedrock: true, graphiti: true, local: false });
    const [bedrockKbId, setBedrockKbId] = (0, react_1.useState)("");
    const [indexPath, setIndexPath] = (0, react_1.useState)("");
    const [queryResults, setQueryResults] = (0, react_1.useState)(null);
    const [queryHistory, setQueryHistory] = (0, react_1.useState)([]);
    const [extractSourceType, setExtractSourceType] = (0, react_1.useState)("text");
    const [extractSourceValue, setExtractSourceValue] = (0, react_1.useState)("");
    const [extractFile, setExtractFile] = (0, react_1.useState)(null);
    const [extractResults, setExtractResults] = (0, react_1.useState)(null);
    const [entities, setEntities] = (0, react_1.useState)([]);
    const [relationships, setRelationships] = (0, react_1.useState)([]);
    const [selectedEntity, setSelectedEntity] = (0, react_1.useState)("");
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const graphLayout = (0, react_1.useMemo)(() => buildGraphLayout(entities, relationships), [entities, relationships]);
    const graphStats = (0, react_1.useMemo)(() => computeGraphStats(graphLayout.nodes.map((node) => node.id), graphLayout.edges), [graphLayout]);
    const handleQuery = async () => {
        setErrorMessage(null);
        if (!queryText.trim()) {
            setErrorMessage("Enter a query to search.");
            return;
        }
        setLoading(true);
        try {
            const response = await openevolveApi_1.openevolveApi.bubblelabsKnowledgeQueryAdvanced({
                query: queryText,
                sources: Object.entries(sources)
                    .filter(([, enabled]) => enabled)
                    .map(([key]) => key),
                bedrock_kb_id: bedrockKbId || undefined,
                index_path: indexPath || undefined,
            }, apiConfig);
            setQueryResults(response.results ?? null);
            setQueryHistory(response.history ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Query failed.");
        }
        finally {
            setLoading(false);
        }
    };
    const handleExtract = async () => {
        setErrorMessage(null);
        if (extractSourceType === "file" && !extractFile) {
            setErrorMessage("Select a file to extract.");
            return;
        }
        if (extractSourceType !== "file" && !extractSourceValue.trim()) {
            setErrorMessage("Provide a source value to extract.");
            return;
        }
        setLoading(true);
        try {
            const response = extractSourceType === "file" && extractFile
                ? await openevolveApi_1.openevolveApi.bubblelabsKnowledgeExtractFile(extractFile, undefined, apiConfig)
                : await openevolveApi_1.openevolveApi.bubblelabsKnowledgeExtract({
                    source_type: extractSourceType,
                    source_value: extractSourceValue,
                }, apiConfig);
            setExtractResults(response.results ?? null);
            const nextEntities = response.results?.entities || [];
            const nextRelationships = response.results?.relationships || [];
            setEntities(nextEntities);
            setRelationships(nextRelationships);
            if (nextEntities.length && !selectedEntity) {
                setSelectedEntity(nextEntities[0].name);
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Extraction failed.");
        }
        finally {
            setLoading(false);
        }
    };
    const loadHistory = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.bubblelabsKnowledgeQueryHistory(apiConfig);
            setQueryHistory(response.history ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load history.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        loadHistory();
    }, [apiConfig.apiKey]);
    const selectedEntityNeighbors = (0, react_1.useMemo)(() => {
        if (!selectedEntity)
            return { predecessors: [], successors: [] };
        const predecessors = relationships
            .filter((rel) => rel.target === selectedEntity)
            .map((rel) => rel.source);
        const successors = relationships
            .filter((rel) => rel.source === selectedEntity)
            .map((rel) => rel.target);
        return { predecessors, successors };
    }, [relationships, selectedEntity]);
    const renderGraph = () => {
        const width = 720;
        const height = 420;
        if (!graphLayout.nodes.length) {
            return <div className="text-sm text-muted-foreground">No graph data available.</div>;
        }
        const radius = Math.min(width, height) / 2 - 40;
        const positions = {};
        graphLayout.nodes.forEach((node, index) => {
            const angle = (2 * Math.PI * index) / graphLayout.nodes.length;
            positions[node.id] = {
                x: width / 2 + radius * Math.cos(angle),
                y: height / 2 + radius * Math.sin(angle),
            };
        });
        return (<svg width={width} height={height} className="border rounded-md bg-muted">
        {graphLayout.edges.map((edge) => {
                const source = positions[edge.source];
                const target = positions[edge.target];
                if (!source || !target)
                    return null;
                return (<g key={edge.id}>
              <line x1={source.x} y1={source.y} x2={target.x} y2={target.y} stroke="#94a3b8" strokeWidth={1}/>
              <text x={(source.x + target.x) / 2} y={(source.y + target.y) / 2} fontSize={10} fill="#64748b">
                {edge.label}
              </text>
            </g>);
            })}
        {graphLayout.nodes.map((node) => {
                const position = positions[node.id];
                const isSelected = node.id === selectedEntity;
                return (<g key={node.id}>
              <circle cx={position.x} cy={position.y} r={isSelected ? 14 : 10} fill={isSelected ? "#0f172a" : "#1d4ed8"}/>
              <text x={position.x + 12} y={position.y + 4} fontSize={11} fill="#0f172a">
                {node.label}
              </text>
            </g>);
            })}
      </svg>);
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Knowledge Explorer</card_1.CardTitle>
          <card_1.CardDescription>Query, extract, and visualize knowledge graphs across sources.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label_1.Label>API Key</label_1.Label>
              <input_1.Input type="password" value={apiKey} onChange={(event) => {
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
            <button_1.Button variant="outline" onClick={loadHistory} disabled={loading}>
              Refresh History
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <tabs_1.Tabs defaultValue="query">
            <tabs_1.TabsList className="flex flex-wrap gap-2">
              <tabs_1.TabsTrigger value="query">Query</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="graph">Graph</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="extract">Extraction</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="stats">Stats</tabs_1.TabsTrigger>
            </tabs_1.TabsList>

            <tabs_1.TabsContent value="query" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">Query Knowledge Sources</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <textarea_1.Textarea value={queryText} onChange={(event) => setQueryText(event.target.value)} placeholder="Ask a question or search for concepts"/>
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="flex items-center gap-2">
                      <checkbox_1.Checkbox checked={sources.bedrock} onCheckedChange={(checked) => setSources((prev) => ({ ...prev, bedrock: Boolean(checked) }))}/>
                      <label_1.Label>Bedrock KB</label_1.Label>
                    </div>
                    <div className="flex items-center gap-2">
                      <checkbox_1.Checkbox checked={sources.graphiti} onCheckedChange={(checked) => setSources((prev) => ({ ...prev, graphiti: Boolean(checked) }))}/>
                      <label_1.Label>Graphiti</label_1.Label>
                    </div>
                    <div className="flex items-center gap-2">
                      <checkbox_1.Checkbox checked={sources.local} onCheckedChange={(checked) => setSources((prev) => ({ ...prev, local: Boolean(checked) }))}/>
                      <label_1.Label>Local Index</label_1.Label>
                    </div>
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label_1.Label>Bedrock KB ID</label_1.Label>
                      <input_1.Input value={bedrockKbId} onChange={(event) => setBedrockKbId(event.target.value)}/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Local Index Path</label_1.Label>
                      <input_1.Input value={indexPath} onChange={(event) => setIndexPath(event.target.value)}/>
                    </div>
                  </div>
                  <button_1.Button onClick={handleQuery} disabled={loading}>
                    Execute Query
                  </button_1.Button>
                </card_1.CardContent>
              </card_1.Card>

              {queryHistory.length ? (<card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-base">Query History</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2">
                    {queryHistory.slice(-5).reverse().map((entry, index) => (<div key={`${entry.timestamp}-${index}`} className="rounded border p-2 text-xs">
                        <div className="font-semibold">{entry.query ?? "Query"}</div>
                        <div className="text-muted-foreground">{entry.timestamp}</div>
                      </div>))}
                  </card_1.CardContent>
                </card_1.Card>) : null}

              {queryResults ? (<card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-base">Query Results</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent>
                    <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(queryResults, null, 2)}</pre>
                  </card_1.CardContent>
                </card_1.Card>) : null}
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="graph" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">Knowledge Graph</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  {renderGraph()}
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="space-y-2">
                      <label_1.Label>Select Entity</label_1.Label>
                      <select_1.Select value={selectedEntity} onValueChange={setSelectedEntity}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue placeholder="Select entity"/>
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          {entities.map((entity) => (<select_1.SelectItem key={entity.name} value={entity.name}>
                              {entity.name}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                    <div className="space-y-1 text-sm">
                      <div className="font-semibold">Predecessors</div>
                      {selectedEntityNeighbors.predecessors.map((pred) => (<div key={pred} className="text-muted-foreground">
                          {pred}
                        </div>))}
                    </div>
                    <div className="space-y-1 text-sm">
                      <div className="font-semibold">Successors</div>
                      {selectedEntityNeighbors.successors.map((succ) => (<div key={succ} className="text-muted-foreground">
                          {succ}
                        </div>))}
                    </div>
                  </div>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="extract" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">Knowledge Extraction</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label_1.Label>Source Type</label_1.Label>
                    <select_1.Select value={extractSourceType} onValueChange={setExtractSourceType}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue />
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                      <select_1.SelectItem value="text">Text</select_1.SelectItem>
                      <select_1.SelectItem value="url">URL</select_1.SelectItem>
                      <select_1.SelectItem value="path">File Path</select_1.SelectItem>
                      <select_1.SelectItem value="file">Upload File</select_1.SelectItem>
                    </select_1.SelectContent>
                  </select_1.Select>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Source</label_1.Label>
                  {extractSourceType === "file" ? (<input_1.Input type="file" onChange={(event) => setExtractFile(event.target.files?.[0] ?? null)}/>) : (<textarea_1.Textarea value={extractSourceValue} onChange={(event) => setExtractSourceValue(event.target.value)} placeholder="Paste text, URL, or file path" className="min-h-[140px]"/>)}
                </div>
                  <button_1.Button onClick={handleExtract} disabled={loading}>
                    Extract Knowledge
                  </button_1.Button>
                </card_1.CardContent>
              </card_1.Card>

              {extractResults ? (<card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-base">Extraction Results</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="rounded border p-3 text-sm">
                        <div className="font-semibold">Entities</div>
                        {entities.map((entity) => (<div key={entity.name} className="text-muted-foreground">
                            {entity.name} {entity.type ? `(${entity.type})` : ""}
                          </div>))}
                      </div>
                      <div className="rounded border p-3 text-sm">
                        <div className="font-semibold">Relationships</div>
                        {relationships.map((rel, index) => (<div key={`${rel.source}-${rel.target}-${index}`} className="text-muted-foreground">
                            {rel.source} → {rel.relation} → {rel.target}
                          </div>))}
                      </div>
                    </div>
                    <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(extractResults, null, 2)}</pre>
                  </card_1.CardContent>
                </card_1.Card>) : null}
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="stats" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">Statistics Dashboard</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-4">
                    <card_1.Card>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-sm">Entities</card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-2xl font-semibold">{graphStats.nodeCount}</card_1.CardContent>
                    </card_1.Card>
                    <card_1.Card>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-sm">Relationships</card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-2xl font-semibold">{graphStats.edgeCount}</card_1.CardContent>
                    </card_1.Card>
                    <card_1.Card>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-sm">Graph Density</card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-2xl font-semibold">
                        {graphStats.density.toFixed(3)}
                      </card_1.CardContent>
                    </card_1.Card>
                    <card_1.Card>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-sm">Connected</card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-2xl font-semibold">
                        {graphStats.isConnected ? "Yes" : "No"}
                      </card_1.CardContent>
                    </card_1.Card>
                  </div>

                  <separator_1.Separator />

                  <div className="space-y-2">
                    <div className="font-semibold text-sm">Recent Queries</div>
                    <div className="flex flex-wrap gap-2">
                      {queryHistory.slice(-6).map((entry, index) => (<badge_1.Badge key={`${entry.timestamp}-${index}`} variant="secondary">
                          {entry.query ?? "Query"}
                        </badge_1.Badge>))}
                    </div>
                  </div>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>
          </tabs_1.Tabs>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.KnowledgeExplorerTab = KnowledgeExplorerTab;
//# sourceMappingURL=KnowledgeExplorerTab.js.map