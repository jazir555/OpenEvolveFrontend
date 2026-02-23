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
exports.KnowledgeBaseTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const separator_1 = require("@/components/ui/separator");
const tabs_1 = require("@/components/ui/tabs");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const ARTIFACT_TYPES = ["pattern", "solution", "error", "best_practice", "solution_pattern"];
const readApiKey = () => {
    try {
        return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    }
    catch {
        return "";
    }
};
const downloadJson = (filename, payload) => {
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
const KnowledgeBaseTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(readApiKey);
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [artifacts, setArtifacts] = (0, react_1.useState)([]);
    const [stats, setStats] = (0, react_1.useState)(null);
    const [graph, setGraph] = (0, react_1.useState)(null);
    const [recommendations, setRecommendations] = (0, react_1.useState)(null);
    const [searchQuery, setSearchQuery] = (0, react_1.useState)("");
    const [searchDomain, setSearchDomain] = (0, react_1.useState)("");
    const [searchType, setSearchType] = (0, react_1.useState)("All");
    const [searchResults, setSearchResults] = (0, react_1.useState)([]);
    const [newArtifact, setNewArtifact] = (0, react_1.useState)({
        artifact_type: "pattern",
        content: "",
        domain: "",
        problem_type: "",
        source_workflow_id: "manual",
        related_artifacts: "",
    });
    const [recommendationInput, setRecommendationInput] = (0, react_1.useState)({
        problem_statement: "",
        domain: "",
    });
    const [importPayload, setImportPayload] = (0, react_1.useState)("");
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const loadKnowledge = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const [artifactResponse, statsResponse, graphResponse] = await Promise.all([
                openevolveApi_1.openevolveApi.listKnowledgeArtifacts(apiConfig),
                openevolveApi_1.openevolveApi.getKnowledgeStats(apiConfig),
                openevolveApi_1.openevolveApi.getKnowledgeGraph(apiConfig),
            ]);
            setArtifacts(artifactResponse.artifacts ?? []);
            setStats(statsResponse);
            setGraph(graphResponse);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load knowledge base.");
        }
        finally {
            setLoading(false);
        }
    };
    const handleSearch = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.searchKnowledge({
                query: searchQuery,
                domain: searchDomain || undefined,
                artifact_types: searchType === "All" ? undefined : [searchType],
                limit: 50,
            }, apiConfig);
            setSearchResults(response.results ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Search failed.");
        }
        finally {
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
            await openevolveApi_1.openevolveApi.createKnowledgeArtifact({
                artifact_type: newArtifact.artifact_type,
                content: newArtifact.content,
                domain: newArtifact.domain || undefined,
                problem_type: newArtifact.problem_type || undefined,
                source_workflow_id: newArtifact.source_workflow_id,
                related_artifacts: newArtifact.related_artifacts
                    ? newArtifact.related_artifacts.split(",").map((item) => item.trim())
                    : [],
            }, apiConfig);
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
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to create artifact.");
        }
    };
    const handleDeleteArtifact = async (artifactId) => {
        setErrorMessage(null);
        try {
            await openevolveApi_1.openevolveApi.deleteKnowledgeArtifact(artifactId, apiConfig);
            await loadKnowledge();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to delete artifact.");
        }
    };
    const handleRecommendations = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.getKnowledgeRecommendations(recommendationInput, apiConfig);
            setRecommendations(response);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to fetch recommendations.");
        }
        finally {
            setLoading(false);
        }
    };
    const handleExport = async () => {
        try {
            const data = await openevolveApi_1.openevolveApi.exportKnowledgeBase(apiConfig);
            downloadJson("knowledge_base_export.json", data);
        }
        catch (error) {
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
            await openevolveApi_1.openevolveApi.importKnowledgeBase({ artifacts: parsed }, apiConfig);
            setStatusMessage("Knowledge base imported.");
            setImportPayload("");
            await loadKnowledge();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to import knowledge base.");
        }
    };
    (0, react_1.useEffect)(() => {
        loadKnowledge();
    }, [apiConfig.apiKey]);
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Knowledge Base</card_1.CardTitle>
          <card_1.CardDescription>Explore, curate, and reuse knowledge artifacts.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label_1.Label>API Key</label_1.Label>
              <input_1.Input value={apiKey} type="password" onChange={(event) => {
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
            <button_1.Button variant="outline" onClick={loadKnowledge} disabled={loading}>
              Refresh
            </button_1.Button>
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

          <tabs_1.Tabs defaultValue="browse" className="w-full">
            <tabs_1.TabsList className="grid w-full grid-cols-5">
              <tabs_1.TabsTrigger value="browse">Browse</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="search">Search</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="graph">Graph</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="recommendations">Recommendations</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="import">Import/Export</tabs_1.TabsTrigger>
            </tabs_1.TabsList>

            <tabs_1.TabsContent value="browse" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Create Artifact</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3">
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <label_1.Label>Type</label_1.Label>
                      <select_1.Select value={newArtifact.artifact_type} onValueChange={(value) => setNewArtifact((prev) => ({ ...prev, artifact_type: value }))}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue />
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          {ARTIFACT_TYPES.map((type) => (<select_1.SelectItem key={type} value={type}>
                              {type}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Source Workflow</label_1.Label>
                      <input_1.Input value={newArtifact.source_workflow_id} onChange={(event) => setNewArtifact((prev) => ({
            ...prev,
            source_workflow_id: event.target.value,
        }))}/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Domain</label_1.Label>
                      <input_1.Input value={newArtifact.domain} onChange={(event) => setNewArtifact((prev) => ({ ...prev, domain: event.target.value }))}/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Problem Type</label_1.Label>
                      <input_1.Input value={newArtifact.problem_type} onChange={(event) => setNewArtifact((prev) => ({ ...prev, problem_type: event.target.value }))}/>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Content</label_1.Label>
                    <textarea_1.Textarea value={newArtifact.content} onChange={(event) => setNewArtifact((prev) => ({ ...prev, content: event.target.value }))} className="min-h-[120px]"/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Related Artifacts (comma separated)</label_1.Label>
                    <input_1.Input value={newArtifact.related_artifacts} onChange={(event) => setNewArtifact((prev) => ({
            ...prev,
            related_artifacts: event.target.value,
        }))}/>
                  </div>
                  <button_1.Button onClick={handleCreateArtifact}>Create Artifact</button_1.Button>
                </card_1.CardContent>
              </card_1.Card>

              <separator_1.Separator />

              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                {artifacts.map((artifact) => (<card_1.Card key={artifact.id}>
                    <card_1.CardHeader>
                      <card_1.CardTitle className="text-sm">{artifact.artifact_type}</card_1.CardTitle>
                      <card_1.CardDescription>ID: {artifact.id}</card_1.CardDescription>
                    </card_1.CardHeader>
                    <card_1.CardContent className="space-y-2 text-sm">
                      <div>Domain: {artifact.domain || "n/a"}</div>
                      <div>Usage: {artifact.usage_count}</div>
                      <div>Effectiveness: {artifact.effectiveness_score.toFixed(2)}</div>
                      <textarea_1.Textarea readOnly value={typeof artifact.content === "string"
                ? artifact.content
                : JSON.stringify(artifact.content, null, 2)} className="min-h-[120px]"/>
                      <div className="flex gap-2">
                        <button_1.Button size="sm" variant="outline" onClick={() => downloadJson(`artifact_${artifact.id}.json`, artifact)}>
                          Export
                        </button_1.Button>
                        <button_1.Button size="sm" variant="destructive" onClick={() => handleDeleteArtifact(artifact.id)}>
                          Delete
                        </button_1.Button>
                      </div>
                    </card_1.CardContent>
                  </card_1.Card>))}
                {artifacts.length === 0 && (<div className="text-sm text-muted-foreground">No artifacts stored yet.</div>)}
              </div>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="search" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Search Artifacts</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3">
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="space-y-2">
                      <label_1.Label>Query</label_1.Label>
                      <input_1.Input value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)}/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Domain</label_1.Label>
                      <input_1.Input value={searchDomain} onChange={(event) => setSearchDomain(event.target.value)}/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Type</label_1.Label>
                      <select_1.Select value={searchType} onValueChange={setSearchType}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue />
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          <select_1.SelectItem value="All">All</select_1.SelectItem>
                          {ARTIFACT_TYPES.map((type) => (<select_1.SelectItem key={type} value={type}>
                              {type}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                  </div>
                  <button_1.Button onClick={handleSearch}>Search</button_1.Button>
                </card_1.CardContent>
              </card_1.Card>

              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                {searchResults.map((artifact) => (<card_1.Card key={artifact.id}>
                    <card_1.CardHeader>
                      <card_1.CardTitle className="text-sm">{artifact.artifact_type}</card_1.CardTitle>
                      <card_1.CardDescription>ID: {artifact.id}</card_1.CardDescription>
                    </card_1.CardHeader>
                    <card_1.CardContent className="space-y-2 text-sm">
                      <div>Domain: {artifact.domain || "n/a"}</div>
                      <div>Usage: {artifact.usage_count}</div>
                      <textarea_1.Textarea readOnly value={typeof artifact.content === "string"
                ? artifact.content
                : JSON.stringify(artifact.content, null, 2)} className="min-h-[120px]"/>
                    </card_1.CardContent>
                  </card_1.Card>))}
                {searchResults.length === 0 && (<div className="text-sm text-muted-foreground">No search results yet.</div>)}
              </div>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="graph" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Knowledge Graph Snapshot</card_1.CardTitle>
                  <card_1.CardDescription>Nodes and edges derived from related artifacts.</card_1.CardDescription>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3 text-sm">
                  <div>Nodes: {graph?.nodes.length ?? 0}</div>
                  <div>Edges: {graph?.edges.length ?? 0}</div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div>
                      <label_1.Label>Nodes</label_1.Label>
                      <textarea_1.Textarea readOnly value={graph ? JSON.stringify(graph.nodes.slice(0, 50), null, 2) : ""} className="min-h-[140px]"/>
                    </div>
                    <div>
                      <label_1.Label>Edges</label_1.Label>
                      <textarea_1.Textarea readOnly value={graph ? JSON.stringify(graph.edges.slice(0, 50), null, 2) : ""} className="min-h-[140px]"/>
                    </div>
                  </div>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="recommendations" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Recommendations</card_1.CardTitle>
                  <card_1.CardDescription>Apply learned patterns to new problems.</card_1.CardDescription>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3">
                  <div className="space-y-2">
                    <label_1.Label>Problem Statement</label_1.Label>
                    <textarea_1.Textarea value={recommendationInput.problem_statement} onChange={(event) => setRecommendationInput((prev) => ({
            ...prev,
            problem_statement: event.target.value,
        }))} className="min-h-[120px]"/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Domain</label_1.Label>
                    <input_1.Input value={recommendationInput.domain} onChange={(event) => setRecommendationInput((prev) => ({ ...prev, domain: event.target.value }))}/>
                  </div>
                  <button_1.Button onClick={handleRecommendations} disabled={loading}>
                    Get Recommendations
                  </button_1.Button>
                </card_1.CardContent>
              </card_1.Card>

              {recommendations ? (<div className="grid gap-4 md:grid-cols-2">
                  <card_1.Card>
                    <card_1.CardHeader>
                      <card_1.CardTitle className="text-sm">Recommended Approaches</card_1.CardTitle>
                    </card_1.CardHeader>
                    <card_1.CardContent className="space-y-2 text-sm">
                      {recommendations.recommended_approaches.length === 0 && (<div className="text-muted-foreground">No approaches found.</div>)}
                      {recommendations.recommended_approaches.map((approach, index) => (<div key={index} className="rounded border p-2">
                          <div className="font-semibold">{String(approach.approach ?? "Approach")}</div>
                          <div className="text-xs text-muted-foreground">
                            Effectiveness: {String(approach.effectiveness ?? "n/a")}
                          </div>
                        </div>))}
                    </card_1.CardContent>
                  </card_1.Card>
                  <card_1.Card>
                    <card_1.CardHeader>
                      <card_1.CardTitle className="text-sm">Similar Problems</card_1.CardTitle>
                    </card_1.CardHeader>
                    <card_1.CardContent className="space-y-2 text-sm">
                      {recommendations.similar_problems.length === 0 && (<div className="text-muted-foreground">No similar problems found.</div>)}
                      {recommendations.similar_problems.map((similar, index) => (<div key={index} className="rounded border p-2">
                          <div className="font-semibold">Problem {index + 1}</div>
                          <div className="text-xs text-muted-foreground">
                            {String(similar.problem ?? "").slice(0, 160)}
                          </div>
                        </div>))}
                    </card_1.CardContent>
                  </card_1.Card>
                  <card_1.Card>
                    <card_1.CardHeader>
                      <card_1.CardTitle className="text-sm">Team Recommendations</card_1.CardTitle>
                    </card_1.CardHeader>
                    <card_1.CardContent className="space-y-2 text-sm">
                      {recommendations.team_recommendations.length === 0 && (<div className="text-muted-foreground">No team recommendations.</div>)}
                      {recommendations.team_recommendations.map((team, index) => (<div key={index} className="rounded border p-2">
                          <div className="font-semibold">{String(team.team_name ?? "Team")}</div>
                          <div className="text-xs text-muted-foreground">
                            Effectiveness: {String(team.effectiveness ?? "n/a")}
                          </div>
                        </div>))}
                    </card_1.CardContent>
                  </card_1.Card>
                  <card_1.Card>
                    <card_1.CardHeader>
                      <card_1.CardTitle className="text-sm">Gauntlet Recommendations</card_1.CardTitle>
                    </card_1.CardHeader>
                    <card_1.CardContent className="space-y-2 text-sm">
                      {recommendations.gauntlet_recommendations.length === 0 && (<div className="text-muted-foreground">No gauntlets recommended.</div>)}
                      {recommendations.gauntlet_recommendations.map((gauntlet, index) => (<div key={index} className="rounded border p-2">
                          <div className="font-semibold">
                            {String(gauntlet.gauntlet_name ?? "Gauntlet")}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Score: {String(gauntlet.score ?? "n/a")}
                          </div>
                        </div>))}
                    </card_1.CardContent>
                  </card_1.Card>
                </div>) : null}
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="import" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Export Knowledge Base</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  <button_1.Button onClick={handleExport}>Download JSON</button_1.Button>
                </card_1.CardContent>
              </card_1.Card>

              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Import Knowledge Base</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2">
                  <textarea_1.Textarea value={importPayload} onChange={(event) => setImportPayload(event.target.value)} placeholder="Paste knowledge base JSON" className="min-h-[160px]"/>
                  <button_1.Button onClick={handleImport}>Import</button_1.Button>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>
          </tabs_1.Tabs>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.KnowledgeBaseTab = KnowledgeBaseTab;
//# sourceMappingURL=KnowledgeBaseTab.js.map