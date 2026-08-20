import React, { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Sparkles, Wrench, Network } from "lucide-react";
import { openevolveApi } from "@/lib/openevolveApi";
import type {
  DspyAssessmentRequest,
  DspyAssessmentResponse,
  DspyFixRequest,
  DspyFixResponse,
  PygraphistryVisualizeRequest,
  PygraphistryVisualizeResponse,
} from "@/lib/types";

const readApiKey = () => {
  try {
    return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
  } catch {
    return "";
  }
};

const DEFAULT_GRAPH = {
  nodes: [
    { id: "a", label: "Alpha" },
    { id: "b", label: "Beta" },
  ],
  edges: [{ source: "a", target: "b" }],
};

export const DspyGraphistryTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(readApiKey);
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  // Assess
  const [assessContent, setAssessContent] = useState("");
  const [assessType, setAssessType] = useState("general");
  const [assessKind, setAssessKind] = useState("comprehensive");
  const [assessment, setAssessment] = useState<DspyAssessmentResponse | null>(null);

  // Fix
  const [fixContent, setFixContent] = useState("");
  const [fixType, setFixType] = useState("general");
  const [fixResult, setFixResult] = useState<DspyFixResponse | null>(null);

  // Visualize
  const [graphPayload, setGraphPayload] = useState(JSON.stringify(DEFAULT_GRAPH, null, 2));
  const [vizResult, setVizResult] = useState<PygraphistryVisualizeResponse | null>(null);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const runAssess = async () => {
    setLoading(true);
    setErrorMessage(null);
    setStatusMessage(null);
    if (!assessContent.trim()) {
      setErrorMessage("Content is required.");
      setLoading(false);
      return;
    }
    try {
      const payload: DspyAssessmentRequest = {
        content: assessContent,
        content_type: assessType,
        assessment_type: assessKind,
      };
      const response = await openevolveApi.assessDspy(payload, apiConfig);
      setAssessment(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "DSPy assessment failed.");
    } finally {
      setLoading(false);
    }
  };

  const runFix = async () => {
    setLoading(true);
    setErrorMessage(null);
    setStatusMessage(null);
    if (!fixContent.trim()) {
      setErrorMessage("Content is required.");
      setLoading(false);
      return;
    }
    try {
      const payload: DspyFixRequest = {
        content: fixContent,
        content_type: fixType,
      };
      const response = await openevolveApi.fixDspy(payload, apiConfig);
      setFixResult(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "DSPy fix generation failed.");
    } finally {
      setLoading(false);
    }
  };

  const runVisualize = async () => {
    setLoading(true);
    setErrorMessage(null);
    setStatusMessage(null);
    let parsed: { nodes: Array<Record<string, unknown>>; edges: Array<Record<string, unknown>> };
    try {
      const raw = JSON.parse(graphPayload);
      if (!Array.isArray(raw.nodes) || !Array.isArray(raw.edges)) {
        throw new Error("Payload must have `nodes` and `edges` arrays.");
      }
      parsed = raw;
    } catch (error: any) {
      setErrorMessage(`Invalid graph JSON: ${error?.message ?? "parse error"}`);
      setLoading(false);
      return;
    }
    try {
      const payload: PygraphistryVisualizeRequest = {
        nodes: parsed.nodes,
        edges: parsed.edges,
      };
      const response = await openevolveApi.visualizePygraphistry(payload, apiConfig);
      setVizResult(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "PyGraphistry visualization failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            DSPy Assessment &amp; PyGraphistry
          </CardTitle>
          <CardDescription>
            DSPy-enhanced assessment and fix generation, plus graph visualization.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>API Key</Label>
            <input
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
              className="w-full rounded border border-[#30363d] bg-[#0d1117] px-3 py-2 text-sm text-gray-300"
            />
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <Tabs defaultValue="assess" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="assess">Assess</TabsTrigger>
              <TabsTrigger value="fix">Fix</TabsTrigger>
              <TabsTrigger value="visualize">Visualize</TabsTrigger>
            </TabsList>

            <TabsContent value="assess" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Sparkles className="h-4 w-4" />
                    Assess Content
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Content Type</Label>
                      <input
                        value={assessType}
                        onChange={(event) => setAssessType(event.target.value)}
                        className="w-full rounded border border-[#30363d] bg-[#0d1117] px-3 py-2 text-sm text-gray-300"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Assessment Type</Label>
                      <input
                        value={assessKind}
                        onChange={(event) => setAssessKind(event.target.value)}
                        className="w-full rounded border border-[#30363d] bg-[#0d1117] px-3 py-2 text-sm text-gray-300"
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Content</Label>
                    <Textarea
                      value={assessContent}
                      onChange={(event) => setAssessContent(event.target.value)}
                      className="min-h-[120px]"
                    />
                  </div>
                  <Button onClick={runAssess} disabled={loading || !assessContent.trim()}>
                    Assess
                  </Button>
                </CardContent>
              </Card>

              {assessment && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="border-[#30363d]">
                      {assessment.status}
                    </Badge>
                    {assessment.confidence_score !== undefined && assessment.confidence_score !== null && (
                      <Badge variant="outline" className="border-[#30363d]">
                        confidence {assessment.confidence_score.toFixed(2)}
                      </Badge>
                    )}
                    {assessment.issues_found !== undefined && assessment.issues_found !== null && (
                      <Badge variant="outline" className="border-[#30363d]">
                        issues {assessment.issues_found}
                      </Badge>
                    )}
                  </div>
                  {assessment.message && (
                    <div className="text-sm text-muted-foreground">{assessment.message}</div>
                  )}
                  {assessment.recommendations && assessment.recommendations.length > 0 && (
                    <ul className="list-disc pl-5 text-sm text-gray-300">
                      {assessment.recommendations.map((rec, index) => (
                        <li key={index}>{rec}</li>
                      ))}
                    </ul>
                  )}
                  <div className="rounded border border-[#30363d] bg-[#0d1117] p-3 text-sm text-gray-300">
                    <pre className="whitespace-pre-wrap break-words">
                      {JSON.stringify(assessment.assessment_result ?? {}, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="fix" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Wrench className="h-4 w-4" />
                    Generate Fixes
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label>Content Type</Label>
                    <input
                      value={fixType}
                      onChange={(event) => setFixType(event.target.value)}
                      className="w-full rounded border border-[#30363d] bg-[#0d1117] px-3 py-2 text-sm text-gray-300"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Content</Label>
                    <Textarea
                      value={fixContent}
                      onChange={(event) => setFixContent(event.target.value)}
                      className="min-h-[120px]"
                    />
                  </div>
                  <Button onClick={runFix} disabled={loading || !fixContent.trim()}>
                    Generate Fix
                  </Button>
                </CardContent>
              </Card>

              {fixResult && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="border-[#30363d]">
                      {fixResult.status}
                    </Badge>
                    {fixResult.fixes_applied !== undefined && fixResult.fixes_applied !== null && (
                      <Badge variant="outline" className="border-[#30363d]">
                        fixes {fixResult.fixes_applied}
                      </Badge>
                    )}
                  </div>
                  {fixResult.message && (
                    <div className="text-sm text-muted-foreground">{fixResult.message}</div>
                  )}
                  {fixResult.fixed_content && (
                    <div className="space-y-2">
                      <Label>Fixed Content</Label>
                      <Textarea
                        readOnly
                        value={fixResult.fixed_content}
                        className="min-h-[120px]"
                      />
                    </div>
                  )}
                  {fixResult.suggested_fixes && fixResult.suggested_fixes.length > 0 && (
                    <div className="rounded border border-[#30363d] bg-[#0d1117] p-3 text-sm text-gray-300">
                      <pre className="whitespace-pre-wrap break-words">
                        {JSON.stringify(fixResult.suggested_fixes, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </TabsContent>

            <TabsContent value="visualize" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Network className="h-4 w-4" />
                    PyGraphistry Visualization
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label>Graph (nodes / edges JSON)</Label>
                    <Textarea
                      value={graphPayload}
                      onChange={(event) => setGraphPayload(event.target.value)}
                      className="min-h-[160px] font-mono text-xs"
                    />
                  </div>
                  <Button onClick={runVisualize} disabled={loading}>
                    Visualize
                  </Button>
                </CardContent>
              </Card>

              {vizResult && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="border-[#30363d]">
                      {vizResult.status}
                    </Badge>
                    {vizResult.visualization_url && (
                      <span className="text-xs text-muted-foreground break-all">
                        {vizResult.visualization_url}
                      </span>
                    )}
                  </div>
                  {vizResult.message && (
                    <div className="text-sm text-muted-foreground">{vizResult.message}</div>
                  )}
                  {vizResult.visualization_url && (
                    <iframe
                      title="PyGraphistry Visualization"
                      src={vizResult.visualization_url}
                      className="h-[480px] w-full rounded border border-[#30363d] bg-white"
                    />
                  )}
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};
