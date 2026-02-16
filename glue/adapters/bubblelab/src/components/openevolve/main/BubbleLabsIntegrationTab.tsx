import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { openevolveApi } from "../../../lib/openevolveApi";
import type {
  BubbleLabsStatusResponse,
  BubbleLabsControlCatalogResponse,
} from "../../../lib/types";

const parseJson = (value: string) => {
  if (!value.trim()) return undefined;
  return JSON.parse(value);
};

const controlPayloadTemplate = (component: string, action: string): Record<string, unknown> => {
  if (component !== "openevolve_workflows") {
    return {};
  }

  switch (action) {
    case "create_definition":
      return {
        name: "OpenEvolve Workflow",
        description: "Managed from BubbleLabs control tab",
        workflow_type: "evolution",
        parameters: {},
      };
    case "get_definition":
      return { definition_id: "" };
    case "create_instance":
      return { definition_id: "", instance_name: "bubble-instance", inputs: {}, parameters: {} };
    case "get_instance_status":
      return { instance_id: "" };
    case "start_instance":
    case "pause_instance":
    case "resume_instance":
    case "stop_instance":
    case "cancel_instance":
    case "restart_instance":
    case "delete_instance":
      return { instance_id: "" };
    case "sync_parameters":
      return { instance_id: "", parameters: {} };
    default:
      return {};
  }
};

export const BubbleLabsIntegrationTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [status, setStatus] = useState<BubbleLabsStatusResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const [aceSkillbookName, setAceSkillbookName] = useState("");
  const [aceSkills, setAceSkills] = useState("[]");
  const [acePatterns, setAcePatterns] = useState("[]");

  const [z3Variables, setZ3Variables] = useState("[]");
  const [z3Constraints, setZ3Constraints] = useState("");
  const [z3Theorem, setZ3Theorem] = useState("");

  const [romaProblem, setRomaProblem] = useState("");
  const [romaDepth, setRomaDepth] = useState("3");
  const [romaConfig, setRomaConfig] = useState("{}");

  const [knowledgeArtifact, setKnowledgeArtifact] = useState("{}");
  const [knowledgeQuery, setKnowledgeQuery] = useState("");

  const [analyticsWorkflowId, setAnalyticsWorkflowId] = useState("");
  const [analyticsMetrics, setAnalyticsMetrics] = useState("{}");
  const [analyticsDashboard, setAnalyticsDashboard] = useState<Record<string, unknown> | null>(null);

  const [leanTheorem, setLeanTheorem] = useState("");

  const [actionResult, setActionResult] = useState<Record<string, unknown> | null>(null);
  const [controlCatalog, setControlCatalog] = useState<BubbleLabsControlCatalogResponse | null>(null);
  const [controlComponent, setControlComponent] = useState("");
  const [controlAction, setControlAction] = useState("");
  const [controlPayload, setControlPayload] = useState("{}");

  const controlComponents = useMemo(
    () => Object.keys(controlCatalog?.components ?? {}).sort(),
    [controlCatalog],
  );

  const controlActions = useMemo(() => {
    if (!controlComponent || !controlCatalog?.components[controlComponent]) {
      return [];
    }
    return [...controlCatalog.components[controlComponent]].sort();
  }, [controlCatalog, controlComponent]);

  const refreshStatus = async () => {
    setErrorMessage(null);
    try {
      const response = await openevolveApi.getBubblelabsStatus(apiConfig);
      setStatus(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load BubbleLabs status.");
    }
  };

  const refreshControlCatalog = async () => {
    setErrorMessage(null);
    try {
      const response = await openevolveApi.bubblelabsControlCatalog(apiConfig);
      setControlCatalog(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load BubbleLabs control catalog.");
    }
  };

  useEffect(() => {
    refreshStatus();
    refreshControlCatalog();
  }, [apiConfig.apiKey]);

  useEffect(() => {
    if (!controlComponents.length) {
      setControlComponent("");
      return;
    }
    if (!controlComponents.includes(controlComponent)) {
      setControlComponent(controlComponents[0]);
    }
  }, [controlComponent, controlComponents]);

  useEffect(() => {
    if (!controlActions.length) {
      setControlAction("");
      setControlPayload("{}");
      return;
    }
    if (!controlActions.includes(controlAction)) {
      const nextAction = controlActions[0];
      setControlAction(nextAction);
      setControlPayload(JSON.stringify(controlPayloadTemplate(controlComponent, nextAction), null, 2));
    }
  }, [controlAction, controlActions, controlComponent]);

  const runAction = async (fn: () => Promise<Record<string, unknown>>) => {
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const response = await fn();
      setActionResult(response);
      setStatusMessage("Action completed.");
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Action failed.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>BubbleLabs Integration</CardTitle>
          <CardDescription>Manage extended integrations (ACE, Z3, ROMA, LeanAide).</CardDescription>
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
                    // ignore
                  }
                }}
              />
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={refreshStatus}>
                Refresh Status
              </Button>
              <Button
                variant="outline"
                onClick={() =>
                  runAction(async () => {
                    const response = await openevolveApi.initializeBubblelabs(apiConfig);
                    await refreshStatus();
                    return response as Record<string, unknown>;
                  })
                }
              >
                Initialize
              </Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid gap-3 md:grid-cols-3 text-sm">
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Components</div>
              <div className="text-lg font-semibold">{status?.total_components ?? 0}</div>
            </div>
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Available</div>
              <div className="text-lg font-semibold">{status?.available_components ?? 0}</div>
            </div>
            <div className="rounded border p-3">
              <div className="text-xs text-muted-foreground">Health</div>
              <div className="text-lg font-semibold">
                {status
                  ? `${Math.round(
                      ((status.available_components ?? 0) / (status.total_components || 1)) * 100,
                    )}%`
                  : "n/a"}
              </div>
            </div>
          </div>

          {status?.components ? (
            <div className="grid gap-3 md:grid-cols-2">
              {Object.entries(status.components).map(([key, component]) => (
                <Card key={key}>
                  <CardHeader>
                    <CardTitle className="text-sm">{String(component.component ?? key)}</CardTitle>
                    <CardDescription>{String(component.status ?? "unknown")}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2 text-xs">
                    <div>Version: {String(component.version ?? "n/a")}</div>
                    {Array.isArray(component.capabilities) && component.capabilities.length ? (
                      <div className="flex flex-wrap gap-2">
                        {component.capabilities.map((cap: string) => (
                          <Badge key={cap} variant="secondary">
                            {cap}
                          </Badge>
                        ))}
                      </div>
                    ) : null}
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Tabs defaultValue="workflows" className="w-full">
        <TabsList className="grid w-full grid-cols-8">
          <TabsTrigger value="workflows">Workflows</TabsTrigger>
          <TabsTrigger value="control">Control</TabsTrigger>
          <TabsTrigger value="ace">ACE</TabsTrigger>
          <TabsTrigger value="z3">Z3</TabsTrigger>
          <TabsTrigger value="roma">ROMA</TabsTrigger>
          <TabsTrigger value="knowledge">Knowledge</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="lean">LeanAide</TabsTrigger>
        </TabsList>

        <TabsContent value="workflows" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Workflow Templates</CardTitle>
              <CardDescription>
                Pre-built workflows that combine multiple BubbleLabs capabilities
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded border p-3">
                  <h4 className="font-medium">Research Assistant</h4>
                  <p className="text-xs text-muted-foreground mb-2">
                    Search knowledge base, analyze results, and generate insights
                  </p>
                  <Badge variant="secondary">RAGBits + Datapizza</Badge>
                </div>
                <div className="rounded border p-3">
                  <h4 className="font-medium">Data Analysis Pipeline</h4>
                  <p className="text-xs text-muted-foreground mb-2">
                    Process raw data through ETL pipeline and generate analytics
                  </p>
                  <Badge variant="secondary">Datapizza + RAGBits</Badge>
                </div>
                <div className="rounded border p-3">
                  <h4 className="font-medium">Proof Verification</h4>
                  <p className="text-xs text-muted-foreground mb-2">
                    Verify mathematical theorems using Z3 and LeanAide
                  </p>
                  <Badge variant="secondary">Z3 + LeanAide</Badge>
                </div>
                <div className="rounded border p-3">
                  <h4 className="font-medium">Knowledge Extraction</h4>
                  <p className="text-xs text-muted-foreground mb-2">
                    Extract structured knowledge from unstructured documents
                  </p>
                  <Badge variant="secondary">Knowledge + RAGBits</Badge>
                </div>
                <div className="rounded border p-3">
                  <h4 className="font-medium">Problem Solving</h4>
                  <p className="text-xs text-muted-foreground mb-2">
                    Analyze complex problems and generate solutions
                  </p>
                  <Badge variant="secondary">ROMA + Z3</Badge>
                </div>
              </div>
              <Button
                onClick={() => {
                  // Navigate to workflow execution tab
                  const tabElement = document.querySelector('[value="workflow-execution"]') as HTMLElement;
                  tabElement?.click();
                }}
              >
                Open Workflow Executor
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="control" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Unified Control Plane</CardTitle>
              <CardDescription>
                Dynamically execute discovered BubbleLabs and OpenEvolve integration actions.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Button variant="outline" onClick={refreshControlCatalog}>
                  Refresh Catalog
                </Button>
                <Button
                  variant="outline"
                  onClick={() =>
                    runAction(async () => {
                      const response = await openevolveApi.bubblelabsControlDiscover(
                        { force: true },
                        apiConfig,
                      );
                      await refreshControlCatalog();
                      return response as Record<string, unknown>;
                    })
                  }
                >
                  Discover Integrations
                </Button>
                <Badge variant="secondary">
                  Components: {controlComponents.length}
                </Badge>
              </div>

              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Component</Label>
                  <select
                    value={controlComponent}
                    onChange={(event) => setControlComponent(event.target.value)}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    {controlComponents.length === 0 ? (
                      <option value="">No components available</option>
                    ) : null}
                    {controlComponents.map((component) => (
                      <option key={component} value={component}>
                        {component}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-2">
                  <Label>Action</Label>
                  <select
                    value={controlAction}
                    onChange={(event) => {
                      const nextAction = event.target.value;
                      setControlAction(nextAction);
                      setControlPayload(
                        JSON.stringify(controlPayloadTemplate(controlComponent, nextAction), null, 2),
                      );
                    }}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    {controlActions.length === 0 ? (
                      <option value="">No actions available</option>
                    ) : null}
                    {controlActions.map((action) => (
                      <option key={action} value={action}>
                        {action}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="space-y-2">
                <Label>Payload (JSON)</Label>
                <Textarea
                  value={controlPayload}
                  onChange={(event) => setControlPayload(event.target.value)}
                  rows={7}
                />
              </div>

              <Button
                onClick={() =>
                  runAction(async () => {
                    const parsed = parseJson(controlPayload) ?? {};
                    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
                      throw new Error("Payload must be a JSON object");
                    }
                    return await openevolveApi.bubblelabsControlExecute(
                      { component: controlComponent, action: controlAction, payload: parsed },
                      apiConfig,
                    );
                  })
                }
                disabled={!controlComponent || !controlAction}
              >
                Execute Control Action
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="ace" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Create Skillbook</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                placeholder="Skillbook name"
                value={aceSkillbookName}
                onChange={(event) => setAceSkillbookName(event.target.value)}
              />
              <Textarea
                value={aceSkills}
                onChange={(event) => setAceSkills(event.target.value)}
                rows={4}
              />
              <Button
                onClick={() =>
                  runAction(async () =>
                    openevolveApi.bubblelabsAceSkillbook(
                      { name: aceSkillbookName, skills: parseJson(aceSkills) ?? [] },
                      apiConfig,
                    ),
                  )
                }
              >
                Create Skillbook
              </Button>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Extract Patterns</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                value={acePatterns}
                onChange={(event) => setAcePatterns(event.target.value)}
                rows={4}
              />
              <Button
                onClick={() =>
                  runAction(async () =>
                    openevolveApi.bubblelabsAcePatterns(
                      { workflow_results: parseJson(acePatterns) ?? [] },
                      apiConfig,
                    ),
                  )
                }
              >
                Extract Patterns
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="z3" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Solve Constraints</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                value={z3Variables}
                onChange={(event) => setZ3Variables(event.target.value)}
                rows={3}
                placeholder='[{"name": "x", "type": "Int"}]'
              />
              <Textarea
                value={z3Constraints}
                onChange={(event) => setZ3Constraints(event.target.value)}
                rows={3}
                placeholder="(> x 0)\n(< x 10)"
              />
              <Button
                onClick={() =>
                  runAction(async () =>
                    openevolveApi.bubblelabsZ3Solve(
                      {
                        variables: parseJson(z3Variables) ?? [],
                        constraints: z3Constraints.split("\n").filter(Boolean),
                      },
                      apiConfig,
                    ),
                  )
                }
              >
                Solve
              </Button>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Prove Theorem</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                value={z3Theorem}
                onChange={(event) => setZ3Theorem(event.target.value)}
                placeholder="forall x. x > 0"
              />
              <Button onClick={() => runAction(() => openevolveApi.bubblelabsZ3Prove({ theorem: z3Theorem }, apiConfig))}>
                Prove
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="roma" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Analyze Problem</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                value={romaProblem}
                onChange={(event) => setRomaProblem(event.target.value)}
                rows={3}
              />
              <Input
                value={romaDepth}
                onChange={(event) => setRomaDepth(event.target.value)}
                placeholder="Max depth"
              />
              <Button
                onClick={() =>
                  runAction(() =>
                    openevolveApi.bubblelabsRomaAnalyze(
                      { problem: romaProblem, max_depth: Number(romaDepth) },
                      apiConfig,
                    ),
                  )
                }
              >
                Analyze
              </Button>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Create Config</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                value={romaConfig}
                onChange={(event) => setRomaConfig(event.target.value)}
                rows={4}
              />
              <Button
                onClick={() =>
                  runAction(() =>
                    openevolveApi.bubblelabsRomaConfig(
                      { config: parseJson(romaConfig) ?? {} },
                      apiConfig,
                    ),
                  )
                }
              >
                Create Config
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="knowledge" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Store Artifact</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                value={knowledgeArtifact}
                onChange={(event) => setKnowledgeArtifact(event.target.value)}
                rows={4}
              />
              <Button
                onClick={() =>
                  runAction(() =>
                    openevolveApi.bubblelabsKnowledgeStore(
                      { artifact: parseJson(knowledgeArtifact) ?? {} },
                      apiConfig,
                    ),
                  )
                }
              >
                Store Artifact
              </Button>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Query Patterns</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                value={knowledgeQuery}
                onChange={(event) => setKnowledgeQuery(event.target.value)}
              />
              <Button
                onClick={() =>
                  runAction(() =>
                    openevolveApi.bubblelabsKnowledgeQuery({ query: knowledgeQuery }, apiConfig),
                  )
                }
              >
                Query
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Track Workflow Metrics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                value={analyticsWorkflowId}
                onChange={(event) => setAnalyticsWorkflowId(event.target.value)}
                placeholder="Workflow ID"
              />
              <Textarea
                value={analyticsMetrics}
                onChange={(event) => setAnalyticsMetrics(event.target.value)}
                rows={4}
              />
              <Button
                onClick={() =>
                  runAction(() =>
                    openevolveApi.bubblelabsAnalyticsTrack(
                      { workflow_id: analyticsWorkflowId, metrics: parseJson(analyticsMetrics) ?? {} },
                      apiConfig,
                    ),
                  )
                }
              >
                Track Metrics
              </Button>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Analytics Dashboard</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button
                onClick={() =>
                  runAction(async () => {
                    const response = await openevolveApi.bubblelabsAnalyticsDashboard(apiConfig);
                    setAnalyticsDashboard(response);
                    return response as Record<string, unknown>;
                  })
                }
              >
                Load Dashboard
              </Button>
              {analyticsDashboard ? (
                <pre className="rounded border p-2 text-xs whitespace-pre-wrap">
                  {JSON.stringify(analyticsDashboard, null, 2)}
                </pre>
              ) : null}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="lean" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">LeanAide Prover</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Input
                value={leanTheorem}
                onChange={(event) => setLeanTheorem(event.target.value)}
                placeholder="Theorem to prove"
              />
              <Button
                onClick={() =>
                  runAction(() =>
                    openevolveApi.bubblelabsLeanAideProve({ theorem: leanTheorem }, apiConfig),
                  )
                }
              >
                Prove
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {actionResult ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Action Result</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="rounded border p-2 text-xs whitespace-pre-wrap">
              {JSON.stringify(actionResult, null, 2)}
            </pre>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
};
