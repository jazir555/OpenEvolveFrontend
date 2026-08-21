import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { openevolveApi } from "@/lib/openevolveApi";
import type {
  KnowledgeArtifact,
  KnowledgeGraph,
  KnowledgeStats,
  KnowledgeRecommendations,
} from "@/lib/types";

const ARTIFACT_TYPES = ["pattern", "solution", "error", "best_practice", "solution_pattern"];

const readApiKey = () => {
  try {
    return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
  } catch {
    return "";
  }
};

const downloadJson = (filename: string, payload: unknown) => {
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

export const KnowledgeBaseTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(readApiKey);
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [artifacts, setArtifacts] = useState<KnowledgeArtifact[]>([]);
  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [recommendations, setRecommendations] = useState<KnowledgeRecommendations | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [searchDomain, setSearchDomain] = useState("");
  const [searchType, setSearchType] = useState("All");
  const [searchResults, setSearchResults] = useState<KnowledgeArtifact[]>([]);

  const [newArtifact, setNewArtifact] = useState({
    artifact_type: "pattern",
    content: "",
    domain: "",
    problem_type: "",
    source_workflow_id: "manual",
    related_artifacts: "",
  });

  const [recommendationInput, setRecommendationInput] = useState({
    problem_statement: "",
    domain: "",
  });

  const [importPayload, setImportPayload] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadKnowledge = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const [artifactResponse, statsResponse, graphResponse] = await Promise.all([
        openevolveApi.listKnowledgeArtifacts(apiConfig),
        openevolveApi.getKnowledgeStats(apiConfig),
        openevolveApi.getKnowledgeGraph(apiConfig),
      ]);
      setArtifacts(artifactResponse.artifacts ?? []);
      setStats(statsResponse);
      setGraph(graphResponse);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load knowledge base.");
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.searchKnowledge(
        {
          query: searchQuery,
          domain: searchDomain || undefined,
          artifact_types: searchType === "All" ? undefined : [searchType],
          limit: 50,
        },
        apiConfig,
      );
      setSearchResults(response.results ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Search failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleCreateArtifact = async () => {
    setErrorMessage(null);
    setStatusMessage(null);
    if (!newArtifact.content.trim()) {
      setErrorMessage("Content is required.");
      return;
    }
    try {
      await openevolveApi.createKnowledgeArtifact(
        {
          artifact_type: newArtifact.artifact_type,
          content: newArtifact.content,
          domain: newArtifact.domain || undefined,
          problem_type: newArtifact.problem_type || undefined,
          source_workflow_id: newArtifact.source_workflow_id,
          related_artifacts: newArtifact.related_artifacts
            ? newArtifact.related_artifacts.split(",").map((item) => item.trim())
            : [],
        },
        apiConfig,
      );
      setStatusMessage("Artifact created.");
      setNewArtifact({
        artifact_type: "pattern",
        content: "",
        domain: "",
        problem_type: "",
        source_workflow_id: "manual",
        related_artifacts: "",
      });
      await loadKnowledge();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to create artifact.");
    }
  };

  const handleDeleteArtifact = async (artifactId: string) => {
    setErrorMessage(null);
    try {
      await openevolveApi.deleteKnowledgeArtifact(artifactId, apiConfig);
      await loadKnowledge();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to delete artifact.");
    }
  };

  const handleRecommendations = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.getKnowledgeRecommendations(
        recommendationInput,
        apiConfig,
      );
      setRecommendations(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to fetch recommendations.");
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    try {
      const data = await openevolveApi.exportKnowledgeBase(apiConfig);
      downloadJson("knowledge_base_export.json", data);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to export knowledge base.");
    }
  };

  const handleImport = async () => {
    if (!importPayload.trim()) {
      setErrorMessage("Paste JSON payload to import.");
      return;
    }
    try {
      const parsed = JSON.parse(importPayload);
      await openevolveApi.importKnowledgeBase({ artifacts: parsed }, apiConfig);
      setStatusMessage("Knowledge base imported.");
      setImportPayload("");
      await loadKnowledge();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to import knowledge base.");
    }
  };

  useEffect(() => {
    loadKnowledge();
  }, [apiConfig.apiKey]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Knowledge Base</CardTitle>
          <CardDescription>Explore, curate, and reuse knowledge artifacts.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
                value={apiKey}
                type="password"
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
            <Button variant="outline" onClick={loadKnowledge} disabled={loading}>
              Refresh
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-3 text-sm">
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Total Artifacts</div>
              <div className="text-lg font-semibold">{stats?.total_artifacts ?? 0}</div>
            </div>
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Total Usage</div>
              <div className="text-lg font-semibold">{stats?.total_usage ?? 0}</div>
            </div>
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Avg Effectiveness</div>
              <div className="text-lg font-semibold">
                {(stats?.average_effectiveness ?? 0).toFixed(2)}
              </div>
            </div>
          </div>

          <Tabs defaultValue="browse" className="w-full">
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="browse">Browse</TabsTrigger>
              <TabsTrigger value="search">Search</TabsTrigger>
              <TabsTrigger value="graph">Graph</TabsTrigger>
              <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
              <TabsTrigger value="import">Import/Export</TabsTrigger>
            </TabsList>

            <TabsContent value="browse" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Create Artifact</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Type</Label>
                      <Select
                        value={newArtifact.artifact_type}
                        onValueChange={(value) =>
                          setNewArtifact((prev) => ({ ...prev, artifact_type: value }))
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {ARTIFACT_TYPES.map((type) => (
                            <SelectItem key={type} value={type}>
                              {type}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Source Workflow</Label>
                      <Input
                        value={newArtifact.source_workflow_id}
                        onChange={(event) =>
                          setNewArtifact((prev) => ({
                            ...prev,
                            source_workflow_id: event.target.value,
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Domain</Label>
                      <Input
                        value={newArtifact.domain}
                        onChange={(event) =>
                          setNewArtifact((prev) => ({ ...prev, domain: event.target.value }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Problem Type</Label>
                      <Input
                        value={newArtifact.problem_type}
                        onChange={(event) =>
                          setNewArtifact((prev) => ({ ...prev, problem_type: event.target.value }))
                        }
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Content</Label>
                    <Textarea
                      value={newArtifact.content}
                      onChange={(event) =>
                        setNewArtifact((prev) => ({ ...prev, content: event.target.value }))
                      }
                      className="min-h-[120px]"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Related Artifacts (comma separated)</Label>
                    <Input
                      value={newArtifact.related_artifacts}
                      onChange={(event) =>
                        setNewArtifact((prev) => ({
                          ...prev,
                          related_artifacts: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <Button onClick={handleCreateArtifact}>Create Artifact</Button>
                </CardContent>
              </Card>

              <Separator />

              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                {artifacts.map((artifact) => (
                  <Card key={artifact.id}>
                    <CardHeader>
                      <CardTitle className="text-sm">{artifact.artifact_type}</CardTitle>
                      <CardDescription>ID: {artifact.id}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <div>Domain: {artifact.domain || "n/a"}</div>
                      <div>Usage: {artifact.usage_count}</div>
                      <div>Effectiveness: {artifact.effectiveness_score.toFixed(2)}</div>
                      <Textarea
                        readOnly
                        value={
                          typeof artifact.content === "string"
                            ? artifact.content
                            : JSON.stringify(artifact.content, null, 2)
                        }
                        className="min-h-[120px]"
                      />
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => downloadJson(`artifact_${artifact.id}.json`, artifact)}
                        >
                          Export
                        </Button>
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() => handleDeleteArtifact(artifact.id)}
                        >
                          Delete
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
                {artifacts.length === 0 && (
                  <div className="text-sm text-muted-foreground">No artifacts stored yet.</div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="search" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Search Artifacts</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="space-y-2">
                      <Label>Query</Label>
                      <Input value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} />
                    </div>
                    <div className="space-y-2">
                      <Label>Domain</Label>
                      <Input value={searchDomain} onChange={(event) => setSearchDomain(event.target.value)} />
                    </div>
                    <div className="space-y-2">
                      <Label>Type</Label>
                      <Select value={searchType} onValueChange={setSearchType}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="All">All</SelectItem>
                          {ARTIFACT_TYPES.map((type) => (
                            <SelectItem key={type} value={type}>
                              {type}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <Button onClick={handleSearch}>Search</Button>
                </CardContent>
              </Card>

              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                {searchResults.map((artifact) => (
                  <Card key={artifact.id}>
                    <CardHeader>
                      <CardTitle className="text-sm">{artifact.artifact_type}</CardTitle>
                      <CardDescription>ID: {artifact.id}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <div>Domain: {artifact.domain || "n/a"}</div>
                      <div>Usage: {artifact.usage_count}</div>
                      <Textarea
                        readOnly
                        value={
                          typeof artifact.content === "string"
                            ? artifact.content
                            : JSON.stringify(artifact.content, null, 2)
                        }
                        className="min-h-[120px]"
                      />
                    </CardContent>
                  </Card>
                ))}
                {searchResults.length === 0 && (
                  <div className="text-sm text-muted-foreground">No search results yet.</div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="graph" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Knowledge Graph Snapshot</CardTitle>
                  <CardDescription>Nodes and edges derived from related artifacts.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <div>Nodes: {graph?.nodes.length ?? 0}</div>
                  <div>Edges: {graph?.edges.length ?? 0}</div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div>
                      <Label>Nodes</Label>
                      <Textarea
                        readOnly
                        value={graph ? JSON.stringify(graph.nodes.slice(0, 50), null, 2) : ""}
                        className="min-h-[140px]"
                      />
                    </div>
                    <div>
                      <Label>Edges</Label>
                      <Textarea
                        readOnly
                        value={graph ? JSON.stringify(graph.edges.slice(0, 50), null, 2) : ""}
                        className="min-h-[140px]"
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="recommendations" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Recommendations</CardTitle>
                  <CardDescription>Apply learned patterns to new problems.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label>Problem Statement</Label>
                    <Textarea
                      value={recommendationInput.problem_statement}
                      onChange={(event) =>
                        setRecommendationInput((prev) => ({
                          ...prev,
                          problem_statement: event.target.value,
                        }))
                      }
                      className="min-h-[120px]"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Domain</Label>
                    <Input
                      value={recommendationInput.domain}
                      onChange={(event) =>
                        setRecommendationInput((prev) => ({ ...prev, domain: event.target.value }))
                      }
                    />
                  </div>
                  <Button onClick={handleRecommendations} disabled={loading}>
                    Get Recommendations
                  </Button>
                </CardContent>
              </Card>

              {recommendations ? (
                <div className="grid gap-4 md:grid-cols-2">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">Recommended Approaches</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      {recommendations.recommended_approaches.length === 0 && (
                        <div className="text-muted-foreground">No approaches found.</div>
                      )}
                      {recommendations.recommended_approaches.map((approach, index) => (
                        <div key={index} className="rounded border p-2">
                          <div className="font-semibold">{String(approach.approach ?? "Approach")}</div>
                          <div className="text-xs text-muted-foreground">
                            Effectiveness: {String(approach.effectiveness ?? "n/a")}
                          </div>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">Similar Problems</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      {recommendations.similar_problems.length === 0 && (
                        <div className="text-muted-foreground">No similar problems found.</div>
                      )}
                      {recommendations.similar_problems.map((similar, index) => (
                        <div key={index} className="rounded border p-2">
                          <div className="font-semibold">Problem {index + 1}</div>
                          <div className="text-xs text-muted-foreground">
                            {String(similar.problem ?? "").slice(0, 160)}
                          </div>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">Team Recommendations</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      {recommendations.team_recommendations.length === 0 && (
                        <div className="text-muted-foreground">No team recommendations.</div>
                      )}
                      {recommendations.team_recommendations.map((team, index) => (
                        <div key={index} className="rounded border p-2">
                          <div className="font-semibold">{String(team.team_name ?? "Team")}</div>
                          <div className="text-xs text-muted-foreground">
                            Effectiveness: {String(team.effectiveness ?? "n/a")}
                          </div>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">Gauntlet Recommendations</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      {recommendations.gauntlet_recommendations.length === 0 && (
                        <div className="text-muted-foreground">No gauntlets recommended.</div>
                      )}
                      {recommendations.gauntlet_recommendations.map((gauntlet, index) => (
                        <div key={index} className="rounded border p-2">
                          <div className="font-semibold">
                            {String(gauntlet.gauntlet_name ?? "Gauntlet")}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Score: {String(gauntlet.score ?? "n/a")}
                          </div>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                </div>
              ) : null}
            </TabsContent>

            <TabsContent value="import" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Export Knowledge Base</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <Button onClick={handleExport}>Download JSON</Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Import Knowledge Base</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Textarea
                    value={importPayload}
                    onChange={(event) => setImportPayload(event.target.value)}
                    placeholder="Paste knowledge base JSON"
                    className="min-h-[160px]"
                  />
                  <Button onClick={handleImport}>Import</Button>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};
