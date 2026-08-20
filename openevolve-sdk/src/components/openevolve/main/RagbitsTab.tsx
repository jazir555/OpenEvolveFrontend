import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Database, Search, Upload } from "lucide-react";
import { openevolveApi } from "@/lib/openevolveApi";
import type {
  RagbitsSearchRequest,
  RagbitsSearchResponse,
  RagbitsIngestRequest,
  RagbitsIngestResponse,
  RagbitsStats,
} from "@/lib/types";

const readApiKey = () => {
  try {
    return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
  } catch {
    return "";
  }
};

export const RagbitsTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(readApiKey);
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [stats, setStats] = useState<RagbitsStats | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [searchTopK, setSearchTopK] = useState("5");
  const [searchMinScore, setSearchMinScore] = useState("0");
  const [searchResults, setSearchResults] = useState<RagbitsSearchResponse | null>(null);

  const [ingestContent, setIngestContent] = useState("");
  const [ingestSource, setIngestSource] = useState("manual");
  const [ingestMetadata, setIngestMetadata] = useState("");
  const [ingestResult, setIngestResult] = useState<RagbitsIngestResponse | null>(null);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadStats = async () => {
    setErrorMessage(null);
    try {
      const response = await openevolveApi.getRagbitsStats(apiConfig);
      setStats(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load RAGBits stats.");
    }
  };

  const handleSearch = async () => {
    setLoading(true);
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const payload: RagbitsSearchRequest = {
        query: searchQuery,
        top_k: parseInt(searchTopK, 10) || 5,
        min_score: parseFloat(searchMinScore) || 0,
      };
      const response = await openevolveApi.ragbitsSearch(payload, apiConfig);
      setSearchResults(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "RAGBits search failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleIngest = async () => {
    setLoading(true);
    setErrorMessage(null);
    setStatusMessage(null);
    if (!ingestContent.trim()) {
      setErrorMessage("Content is required.");
      setLoading(false);
      return;
    }
    try {
      let metadata: Record<string, unknown> | undefined;
      if (ingestMetadata.trim()) {
        try {
          metadata = JSON.parse(ingestMetadata);
        } catch {
          setErrorMessage("Metadata must be valid JSON.");
          setLoading(false);
          return;
        }
      }
      const payload: RagbitsIngestRequest = {
        content: ingestContent,
        source: ingestSource || "manual",
        metadata,
      };
      const response = await openevolveApi.ragbitsIngest(payload, apiConfig);
      setIngestResult(response);
      setStatusMessage(response.status === "success" ? "Document ingested." : "Ingest returned an error.");
      await loadStats();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "RAGBits ingest failed.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStats();
  }, [apiConfig.apiKey]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            RAGBits Knowledge Retrieval
          </CardTitle>
          <CardDescription>
            Semantic search, document ingest, and system statistics backed by RAGBits.
          </CardDescription>
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
            <Button variant="outline" onClick={loadStats} disabled={loading}>
              Refresh Stats
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-3 text-sm">
            <div className="rounded border border-[#30363d] bg-[#0d1117] p-3">
              <div className="text-xs text-muted-foreground">Status</div>
              <div className="text-lg font-semibold">{stats?.status ?? "—"}</div>
            </div>
            <div className="rounded border border-[#30363d] bg-[#0d1117] p-3">
              <div className="text-xs text-muted-foreground">Processor</div>
              <div className="text-lg font-semibold">
                {stats?.processor ? Object.keys(stats.processor).length : 0} fields
              </div>
            </div>
            <div className="rounded border border-[#30363d] bg-[#0d1117] p-3">
              <div className="text-xs text-muted-foreground">Retriever</div>
              <div className="text-lg font-semibold">
                {stats?.retriever ? Object.keys(stats.retriever).length : 0} fields
              </div>
            </div>
          </div>

          <Tabs defaultValue="search" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="search">Search</TabsTrigger>
              <TabsTrigger value="ingest">Ingest</TabsTrigger>
            </TabsList>

            <TabsContent value="search" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Search className="h-4 w-4" />
                    Semantic Search
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label>Query</Label>
                    <Input
                      value={searchQuery}
                      onChange={(event) => setSearchQuery(event.target.value)}
                      placeholder="Search the knowledge corpus"
                    />
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Top K</Label>
                      <Input
                        value={searchTopK}
                        onChange={(event) => setSearchTopK(event.target.value)}
                        inputMode="numeric"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Min Score</Label>
                      <Input
                        value={searchMinScore}
                        onChange={(event) => setSearchMinScore(event.target.value)}
                        inputMode="decimal"
                      />
                    </div>
                  </div>
                  <Button onClick={handleSearch} disabled={loading || !searchQuery.trim()}>
                    Search
                  </Button>
                </CardContent>
              </Card>

              <div className="space-y-2">
                <div className="text-sm text-gray-300">
                  Results: {searchResults?.total_results ?? 0}
                </div>
                {searchResults?.results.map((result, index) => (
                  <div
                    key={index}
                    className="rounded border border-[#30363d] bg-[#0d1117] p-3 text-sm text-gray-300"
                  >
                    <pre className="whitespace-pre-wrap break-words">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </div>
                ))}
                {searchResults && searchResults.results.length === 0 && (
                  <div className="text-sm text-muted-foreground">No results returned.</div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="ingest" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Upload className="h-4 w-4" />
                    Ingest Document
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label>Content</Label>
                    <Textarea
                      value={ingestContent}
                      onChange={(event) => setIngestContent(event.target.value)}
                      className="min-h-[120px]"
                      placeholder="Document text to ingest"
                    />
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Source</Label>
                      <Input
                        value={ingestSource}
                        onChange={(event) => setIngestSource(event.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Metadata (JSON)</Label>
                      <Input
                        value={ingestMetadata}
                        onChange={(event) => setIngestMetadata(event.target.value)}
                        placeholder='{"doc_id": "abc"}'
                      />
                    </div>
                  </div>
                  <Button onClick={handleIngest} disabled={loading || !ingestContent.trim()}>
                    Ingest
                  </Button>
                </CardContent>
              </Card>

              {ingestResult && (
                <div className="rounded border border-[#30363d] bg-[#0d1117] p-3 text-sm text-gray-300">
                  <div className="mb-2 flex items-center gap-2">
                    <Badge variant="outline" className="border-[#30363d]">
                      {ingestResult.status}
                    </Badge>
                    {ingestResult.document_id && (
                      <span className="text-xs text-muted-foreground">
                        id: {ingestResult.document_id}
                      </span>
                    )}
                  </div>
                  <pre className="whitespace-pre-wrap break-words">
                    {JSON.stringify(ingestResult, null, 2)}
                  </pre>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};
