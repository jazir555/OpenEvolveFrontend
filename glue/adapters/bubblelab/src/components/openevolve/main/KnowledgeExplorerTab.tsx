import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { openevolveApi } from "../../../lib/openevolveApi";

interface GraphEntity {
  name: string;
  type?: string;
  attributes?: Record<string, unknown>;
}

interface GraphRelationship {
  source: string;
  relation: string;
  target: string;
  attributes?: Record<string, unknown>;
}

const readApiKey = () => {
  try {
    return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
  } catch {
    return "";
  }
};

const buildGraphLayout = (entities: GraphEntity[], relationships: GraphRelationship[]) => {
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

const computeGraphStats = (nodes: string[], edges: Array<{ source: string; target: string }>) => {
  const nodeCount = nodes.length;
  const edgeCount = edges.length;
  const density = nodeCount > 1 ? edgeCount / (nodeCount * (nodeCount - 1)) : 0;
  const adjacency: Record<string, string[]> = {};
  nodes.forEach((node) => {
    adjacency[node] = [];
  });
  edges.forEach((edge) => {
    if (!adjacency[edge.source]) adjacency[edge.source] = [];
    adjacency[edge.source].push(edge.target);
  });
  const visited = new Set<string>();
  const stack = nodes.length ? [nodes[0]] : [];
  while (stack.length) {
    const node = stack.pop() as string;
    if (visited.has(node)) continue;
    visited.add(node);
    (adjacency[node] || []).forEach((neighbor) => {
      if (!visited.has(neighbor)) stack.push(neighbor);
    });
  }
  return {
    nodeCount,
    edgeCount,
    density,
    isConnected: nodeCount === 0 ? true : visited.size === nodeCount,
  };
};

export const KnowledgeExplorerTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(readApiKey);
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [queryText, setQueryText] = useState("");
  const [sources, setSources] = useState({ bedrock: true, graphiti: true, local: false });
  const [bedrockKbId, setBedrockKbId] = useState("");
  const [indexPath, setIndexPath] = useState("");

  const [queryResults, setQueryResults] = useState<Record<string, unknown> | null>(null);
  const [queryHistory, setQueryHistory] = useState<Array<Record<string, unknown>>>([]);

  const [extractSourceType, setExtractSourceType] = useState("text");
  const [extractSourceValue, setExtractSourceValue] = useState("");
  const [extractFile, setExtractFile] = useState<File | null>(null);
  const [extractResults, setExtractResults] = useState<Record<string, unknown> | null>(null);
  const [entities, setEntities] = useState<GraphEntity[]>([]);
  const [relationships, setRelationships] = useState<GraphRelationship[]>([]);
  const [selectedEntity, setSelectedEntity] = useState<string>("");

  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const graphLayout = useMemo(() => buildGraphLayout(entities, relationships), [entities, relationships]);
  const graphStats = useMemo(
    () => computeGraphStats(graphLayout.nodes.map((node) => node.id), graphLayout.edges),
    [graphLayout],
  );

  const handleQuery = async () => {
    setErrorMessage(null);
    if (!queryText.trim()) {
      setErrorMessage("Enter a query to search.");
      return;
    }
    setLoading(true);
    try {
      const response = await openevolveApi.bubblelabsKnowledgeQueryAdvanced(
        {
          query: queryText,
          sources: Object.entries(sources)
            .filter(([, enabled]) => enabled)
            .map(([key]) => key),
          bedrock_kb_id: bedrockKbId || undefined,
          index_path: indexPath || undefined,
        },
        apiConfig,
      );
      setQueryResults(response.results ?? null);
      setQueryHistory(response.history ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Query failed.");
    } finally {
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
      const response =
        extractSourceType === "file" && extractFile
          ? await openevolveApi.bubblelabsKnowledgeExtractFile(extractFile, undefined, apiConfig)
          : await openevolveApi.bubblelabsKnowledgeExtract(
              {
                source_type: extractSourceType,
                source_value: extractSourceValue,
              },
              apiConfig,
            );
      setExtractResults(response.results ?? null);
      const nextEntities = (response.results?.entities as GraphEntity[]) || [];
      const nextRelationships = (response.results?.relationships as GraphRelationship[]) || [];
      setEntities(nextEntities);
      setRelationships(nextRelationships);
      if (nextEntities.length && !selectedEntity) {
        setSelectedEntity(nextEntities[0].name);
      }
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Extraction failed.");
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.bubblelabsKnowledgeQueryHistory(apiConfig);
      setQueryHistory(response.history ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load history.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, [apiConfig.apiKey]);

  const selectedEntityNeighbors = useMemo(() => {
    if (!selectedEntity) return { predecessors: [], successors: [] };
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
    const positions: Record<string, { x: number; y: number }> = {};
    graphLayout.nodes.forEach((node, index) => {
      const angle = (2 * Math.PI * index) / graphLayout.nodes.length;
      positions[node.id] = {
        x: width / 2 + radius * Math.cos(angle),
        y: height / 2 + radius * Math.sin(angle),
      };
    });

    return (
      <svg width={width} height={height} className="border rounded-md bg-muted">
        {graphLayout.edges.map((edge) => {
          const source = positions[edge.source];
          const target = positions[edge.target];
          if (!source || !target) return null;
          return (
            <g key={edge.id}>
              <line
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                stroke="#94a3b8"
                strokeWidth={1}
              />
              <text
                x={(source.x + target.x) / 2}
                y={(source.y + target.y) / 2}
                fontSize={10}
                fill="#64748b"
              >
                {edge.label}
              </text>
            </g>
          );
        })}
        {graphLayout.nodes.map((node) => {
          const position = positions[node.id];
          const isSelected = node.id === selectedEntity;
          return (
            <g key={node.id}>
              <circle
                cx={position.x}
                cy={position.y}
                r={isSelected ? 14 : 10}
                fill={isSelected ? "#0f172a" : "#1d4ed8"}
              />
              <text x={position.x + 12} y={position.y + 4} fontSize={11} fill="#0f172a">
                {node.label}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Knowledge Explorer</CardTitle>
          <CardDescription>Query, extract, and visualize knowledge graphs across sources.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
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
            <Button variant="outline" onClick={loadHistory} disabled={loading}>
              Refresh History
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <Tabs defaultValue="query">
            <TabsList className="flex flex-wrap gap-2">
              <TabsTrigger value="query">Query</TabsTrigger>
              <TabsTrigger value="graph">Graph</TabsTrigger>
              <TabsTrigger value="extract">Extraction</TabsTrigger>
              <TabsTrigger value="stats">Stats</TabsTrigger>
            </TabsList>

            <TabsContent value="query" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Query Knowledge Sources</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Textarea
                    value={queryText}
                    onChange={(event) => setQueryText(event.target.value)}
                    placeholder="Ask a question or search for concepts"
                  />
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="flex items-center gap-2">
                      <Checkbox
                        checked={sources.bedrock}
                        onCheckedChange={(checked) =>
                          setSources((prev) => ({ ...prev, bedrock: Boolean(checked) }))
                        }
                      />
                      <Label>Bedrock KB</Label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Checkbox
                        checked={sources.graphiti}
                        onCheckedChange={(checked) =>
                          setSources((prev) => ({ ...prev, graphiti: Boolean(checked) }))
                        }
                      />
                      <Label>Graphiti</Label>
                    </div>
                    <div className="flex items-center gap-2">
                      <Checkbox
                        checked={sources.local}
                        onCheckedChange={(checked) =>
                          setSources((prev) => ({ ...prev, local: Boolean(checked) }))
                        }
                      />
                      <Label>Local Index</Label>
                    </div>
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Bedrock KB ID</Label>
                      <Input value={bedrockKbId} onChange={(event) => setBedrockKbId(event.target.value)} />
                    </div>
                    <div className="space-y-2">
                      <Label>Local Index Path</Label>
                      <Input value={indexPath} onChange={(event) => setIndexPath(event.target.value)} />
                    </div>
                  </div>
                  <Button onClick={handleQuery} disabled={loading}>
                    Execute Query
                  </Button>
                </CardContent>
              </Card>

              {queryHistory.length ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Query History</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {queryHistory.slice(-5).reverse().map((entry, index) => (
                      <div key={`${entry.timestamp}-${index}`} className="rounded border p-2 text-xs">
                        <div className="font-semibold">{entry.query ?? "Query"}</div>
                        <div className="text-muted-foreground">{entry.timestamp}</div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              ) : null}

              {queryResults ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Query Results</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(queryResults, null, 2)}</pre>
                  </CardContent>
                </Card>
              ) : null}
            </TabsContent>

            <TabsContent value="graph" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Knowledge Graph</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {renderGraph()}
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="space-y-2">
                      <Label>Select Entity</Label>
                      <Select value={selectedEntity} onValueChange={setSelectedEntity}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select entity" />
                        </SelectTrigger>
                        <SelectContent>
                          {entities.map((entity) => (
                            <SelectItem key={entity.name} value={entity.name}>
                              {entity.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-1 text-sm">
                      <div className="font-semibold">Predecessors</div>
                      {selectedEntityNeighbors.predecessors.map((pred) => (
                        <div key={pred} className="text-muted-foreground">
                          {pred}
                        </div>
                      ))}
                    </div>
                    <div className="space-y-1 text-sm">
                      <div className="font-semibold">Successors</div>
                      {selectedEntityNeighbors.successors.map((succ) => (
                        <div key={succ} className="text-muted-foreground">
                          {succ}
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="extract" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Knowledge Extraction</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Source Type</Label>
                    <Select value={extractSourceType} onValueChange={setExtractSourceType}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                      <SelectItem value="text">Text</SelectItem>
                      <SelectItem value="url">URL</SelectItem>
                      <SelectItem value="path">File Path</SelectItem>
                      <SelectItem value="file">Upload File</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Source</Label>
                  {extractSourceType === "file" ? (
                    <Input
                      type="file"
                      onChange={(event) => setExtractFile(event.target.files?.[0] ?? null)}
                    />
                  ) : (
                    <Textarea
                      value={extractSourceValue}
                      onChange={(event) => setExtractSourceValue(event.target.value)}
                      placeholder="Paste text, URL, or file path"
                      className="min-h-[140px]"
                    />
                  )}
                </div>
                  <Button onClick={handleExtract} disabled={loading}>
                    Extract Knowledge
                  </Button>
                </CardContent>
              </Card>

              {extractResults ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Extraction Results</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="rounded border p-3 text-sm">
                        <div className="font-semibold">Entities</div>
                        {entities.map((entity) => (
                          <div key={entity.name} className="text-muted-foreground">
                            {entity.name} {entity.type ? `(${entity.type})` : ""}
                          </div>
                        ))}
                      </div>
                      <div className="rounded border p-3 text-sm">
                        <div className="font-semibold">Relationships</div>
                        {relationships.map((rel, index) => (
                          <div key={`${rel.source}-${rel.target}-${index}`} className="text-muted-foreground">
                            {rel.source} → {rel.relation} → {rel.target}
                          </div>
                        ))}
                      </div>
                    </div>
                    <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(extractResults, null, 2)}</pre>
                  </CardContent>
                </Card>
              ) : null}
            </TabsContent>

            <TabsContent value="stats" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Statistics Dashboard</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-4">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Entities</CardTitle>
                      </CardHeader>
                      <CardContent className="text-2xl font-semibold">{graphStats.nodeCount}</CardContent>
                    </Card>
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Relationships</CardTitle>
                      </CardHeader>
                      <CardContent className="text-2xl font-semibold">{graphStats.edgeCount}</CardContent>
                    </Card>
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Graph Density</CardTitle>
                      </CardHeader>
                      <CardContent className="text-2xl font-semibold">
                        {graphStats.density.toFixed(3)}
                      </CardContent>
                    </Card>
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Connected</CardTitle>
                      </CardHeader>
                      <CardContent className="text-2xl font-semibold">
                        {graphStats.isConnected ? "Yes" : "No"}
                      </CardContent>
                    </Card>
                  </div>

                  <Separator />

                  <div className="space-y-2">
                    <div className="font-semibold text-sm">Recent Queries</div>
                    <div className="flex flex-wrap gap-2">
                      {queryHistory.slice(-6).map((entry, index) => (
                        <Badge key={`${entry.timestamp}-${index}`} variant="secondary">
                          {entry.query ?? "Query"}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};
