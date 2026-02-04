import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { openevolveApi } from "@/lib/openevolveApi";
import type { MakerDelegation, MakerExecutionResult, MakerToolDefinition } from "@/lib/types";

const STATUS_OPTIONS = ["draft", "testing", "validated", "deployed", "deprecated"];
const MODE_OPTIONS = ["sequential", "recursive", "hybrid"];

const readApiKey = () => {
  try {
    return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
  } catch {
    return "";
  }
};

const parseJsonOrError = (value: string, label: string) => {
  if (!value.trim()) return { parsed: undefined, error: undefined };
  try {
    return { parsed: JSON.parse(value), error: undefined };
  } catch (error: any) {
    return { parsed: undefined, error: `${label} JSON is invalid: ${error?.message ?? "error"}` };
  }
};

export const MakerStudioTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(readApiKey);
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [tools, setTools] = useState<MakerToolDefinition[]>([]);
  const [delegations, setDelegations] = useState<MakerDelegation[]>([]);
  const [toolFilter, setToolFilter] = useState({ status: "", mode: "", search: "" });
  const [delegationFilter, setDelegationFilter] = useState({ status: "", type: "" });
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [toolForm, setToolForm] = useState({
    name: "",
    description: "",
    task: "",
    maker_mode: "recursive",
    k_ahead: 3,
    max_depth: 5,
    contextJson: "",
    promptTemplate: "",
    systemPrompt: "",
    expectedSchemaJson: "",
    metadataJson: "",
  });

  const [toolTestInputs, setToolTestInputs] = useState<Record<string, string>>({});
  const [toolTestResults, setToolTestResults] = useState<Record<string, MakerExecutionResult | null>>({});

  const [executorState, setExecutorState] = useState({
    toolId: "",
    inputJson: "{\n  \"task\": \"Describe desired output\",\n  \"context\": {}\n}",
    delegateToCrewAI: false,
  });
  const [executorResult, setExecutorResult] = useState<MakerExecutionResult | null>(null);

  const refreshTools = useCallback(async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const result = await openevolveApi.listMakerTools(
        {
          status: toolFilter.status || undefined,
          maker_mode: toolFilter.mode || undefined,
          search: toolFilter.search || undefined,
        },
        apiConfig,
      );
      setTools(result.tools ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load tools.");
    } finally {
      setLoading(false);
    }
  }, [apiConfig, toolFilter]);

  const refreshDelegations = useCallback(async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const result = await openevolveApi.listMakerDelegations(
        {
          status: delegationFilter.status || undefined,
          delegation_type: delegationFilter.type || undefined,
        },
        apiConfig,
      );
      setDelegations(result.delegations ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load delegations.");
    } finally {
      setLoading(false);
    }
  }, [apiConfig, delegationFilter]);

  useEffect(() => {
    refreshTools();
    refreshDelegations();
  }, [refreshTools, refreshDelegations]);

  const handleCreateTool = async () => {
    setStatusMessage(null);
    setErrorMessage(null);
    if (!toolForm.name.trim() || !toolForm.description.trim() || !toolForm.task.trim()) {
      setErrorMessage("Name, description, and task are required.");
      return;
    }
    const context = parseJsonOrError(toolForm.contextJson, "Context");
    if (context.error) {
      setErrorMessage(context.error);
      return;
    }
    const expectedSchema = parseJsonOrError(toolForm.expectedSchemaJson, "Expected schema");
    if (expectedSchema.error) {
      setErrorMessage(expectedSchema.error);
      return;
    }
    const metadata = parseJsonOrError(toolForm.metadataJson, "Metadata");
    if (metadata.error) {
      setErrorMessage(metadata.error);
      return;
    }

    setLoading(true);
    try {
      const response = await openevolveApi.createMakerTool(
        {
          name: toolForm.name,
          description: toolForm.description,
          task: toolForm.task,
          maker_mode: toolForm.maker_mode,
          k_ahead: toolForm.k_ahead,
          max_depth: toolForm.max_depth,
          context: context.parsed,
          prompt_template: toolForm.promptTemplate || undefined,
          system_prompt: toolForm.systemPrompt || undefined,
          expected_schema: expectedSchema.parsed,
          metadata: metadata.parsed,
        },
        apiConfig,
      );
      setStatusMessage(`Tool ${response.tool.tool_id} created.`);
      setToolForm({
        name: "",
        description: "",
        task: "",
        maker_mode: "recursive",
        k_ahead: 3,
        max_depth: 5,
        contextJson: "",
        promptTemplate: "",
        systemPrompt: "",
        expectedSchemaJson: "",
        metadataJson: "",
      });
      await refreshTools();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to create tool.");
    } finally {
      setLoading(false);
    }
  };

  const handleTestTool = async (toolId: string) => {
    setErrorMessage(null);
    const inputRaw = toolTestInputs[toolId] ?? "{\n  \"task\": \"Run a validation test\"\n}";
    const parsed = parseJsonOrError(inputRaw, "Tool input");
    if (parsed.error) {
      setErrorMessage(parsed.error);
      return;
    }
    setLoading(true);
    try {
      const result = await openevolveApi.testMakerTool(
        toolId,
        { input_data: parsed.parsed ?? {} },
        apiConfig,
      );
      setToolTestResults((prev) => ({ ...prev, [toolId]: result.result }));
      await refreshTools();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Tool test failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleValidateTool = async (toolId: string) => {
    setErrorMessage(null);
    setLoading(true);
    try {
      await openevolveApi.validateMakerTool(toolId, apiConfig);
      setStatusMessage(`Tool ${toolId} validated.`);
      await refreshTools();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Tool validation failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleExecuteTool = async () => {
    setErrorMessage(null);
    setExecutorResult(null);
    const parsed = parseJsonOrError(executorState.inputJson, "Execution input");
    if (parsed.error) {
      setErrorMessage(parsed.error);
      return;
    }
    if (!executorState.toolId) {
      setErrorMessage("Select a tool to execute.");
      return;
    }
    setLoading(true);
    try {
      const result = await openevolveApi.executeMakerTool(
        executorState.toolId,
        { input_data: parsed.parsed ?? {}, delegate_to_crewai: executorState.delegateToCrewAI },
        apiConfig,
      );
      setExecutorResult(result.result);
      await refreshTools();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Tool execution failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleSyncDelegations = async () => {
    setErrorMessage(null);
    setLoading(true);
    try {
      const result = await openevolveApi.syncMakerDelegations(apiConfig);
      setStatusMessage(`Synced ${result.synced} delegations.`);
      await refreshDelegations();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to sync delegations.");
    } finally {
      setLoading(false);
    }
  };

  const toolStats = useMemo(() => {
    const statusCounts: Record<string, number> = {};
    tools.forEach((tool) => {
      statusCounts[tool.status] = (statusCounts[tool.status] || 0) + 1;
    });
    return {
      total: tools.length,
      statusCounts,
      validated: tools.filter((tool) => tool.status === "validated").length,
      totalUsage: tools.reduce((sum, tool) => sum + (tool.usage_count || 0), 0),
      activeDelegations: delegations.filter((del) =>
        ["pending", "assigned", "in_progress", "in_review"].includes(del.status),
      ).length,
    };
  }, [tools, delegations]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Maker Studio</CardTitle>
          <CardDescription>Build and execute Maker-powered tools with CrewAI delegation.</CardDescription>
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
            <div className="flex gap-2">
              <Button variant="outline" onClick={refreshTools} disabled={loading}>
                Refresh Tools
              </Button>
              <Button variant="outline" onClick={refreshDelegations} disabled={loading}>
                Refresh Delegations
              </Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-emerald-600">{statusMessage}</div> : null}

          <Tabs defaultValue="creator" className="w-full">
            <TabsList className="flex flex-wrap gap-2">
              <TabsTrigger value="creator">Tool Creator</TabsTrigger>
              <TabsTrigger value="repository">Repository</TabsTrigger>
              <TabsTrigger value="executor">Executor</TabsTrigger>
              <TabsTrigger value="delegations">CrewAI Tracker</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
            </TabsList>

            <TabsContent value="creator" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Create New Tool</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Name</Label>
                      <Input
                        value={toolForm.name}
                        onChange={(event) => setToolForm((prev) => ({ ...prev, name: event.target.value }))}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Maker Mode</Label>
                      <Select
                        value={toolForm.maker_mode}
                        onValueChange={(value) => setToolForm((prev) => ({ ...prev, maker_mode: value }))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {MODE_OPTIONS.map((mode) => (
                            <SelectItem key={mode} value={mode}>
                              {mode}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Description</Label>
                    <Textarea
                      value={toolForm.description}
                      onChange={(event) => setToolForm((prev) => ({ ...prev, description: event.target.value }))}
                      className="min-h-[120px]"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Task Definition</Label>
                    <Textarea
                      value={toolForm.task}
                      onChange={(event) => setToolForm((prev) => ({ ...prev, task: event.target.value }))}
                      className="min-h-[120px]"
                    />
                  </div>

                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="space-y-2">
                      <Label>k-ahead</Label>
                      <Input
                        type="number"
                        value={toolForm.k_ahead}
                        onChange={(event) =>
                          setToolForm((prev) => ({ ...prev, k_ahead: Number(event.target.value) || 0 }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Max Depth</Label>
                      <Input
                        type="number"
                        value={toolForm.max_depth}
                        onChange={(event) =>
                          setToolForm((prev) => ({ ...prev, max_depth: Number(event.target.value) || 0 }))
                        }
                      />
                    </div>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Context (JSON)</Label>
                      <Textarea
                        value={toolForm.contextJson}
                        onChange={(event) => setToolForm((prev) => ({ ...prev, contextJson: event.target.value }))}
                        className="min-h-[120px]"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Expected Schema (JSON)</Label>
                      <Textarea
                        value={toolForm.expectedSchemaJson}
                        onChange={(event) =>
                          setToolForm((prev) => ({ ...prev, expectedSchemaJson: event.target.value }))
                        }
                        className="min-h-[120px]"
                      />
                    </div>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Prompt Template</Label>
                      <Textarea
                        value={toolForm.promptTemplate}
                        onChange={(event) => setToolForm((prev) => ({ ...prev, promptTemplate: event.target.value }))}
                        className="min-h-[120px]"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>System Prompt</Label>
                      <Textarea
                        value={toolForm.systemPrompt}
                        onChange={(event) => setToolForm((prev) => ({ ...prev, systemPrompt: event.target.value }))}
                        className="min-h-[120px]"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Metadata (JSON)</Label>
                    <Textarea
                      value={toolForm.metadataJson}
                      onChange={(event) => setToolForm((prev) => ({ ...prev, metadataJson: event.target.value }))}
                      className="min-h-[120px]"
                    />
                  </div>

                  <Button onClick={handleCreateTool} disabled={loading}>
                    Create Tool
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="repository" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Tool Repository</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="space-y-2">
                      <Label>Status Filter</Label>
                      <Select value={toolFilter.status} onValueChange={(value) => setToolFilter((prev) => ({ ...prev, status: value }))}>
                        <SelectTrigger>
                          <SelectValue placeholder="All" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="">All</SelectItem>
                          {STATUS_OPTIONS.map((status) => (
                            <SelectItem key={status} value={status}>
                              {status}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Mode Filter</Label>
                      <Select value={toolFilter.mode} onValueChange={(value) => setToolFilter((prev) => ({ ...prev, mode: value }))}>
                        <SelectTrigger>
                          <SelectValue placeholder="All" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="">All</SelectItem>
                          {MODE_OPTIONS.map((mode) => (
                            <SelectItem key={mode} value={mode}>
                              {mode}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Search</Label>
                      <Input
                        value={toolFilter.search}
                        onChange={(event) => setToolFilter((prev) => ({ ...prev, search: event.target.value }))}
                        placeholder="Search tools"
                      />
                    </div>
                  </div>

                  <Button variant="outline" onClick={refreshTools} disabled={loading}>
                    Apply Filters
                  </Button>

                  <div className="space-y-4">
                    {tools.length === 0 ? (
                      <div className="text-sm text-muted-foreground">No tools found.</div>
                    ) : (
                      tools.map((tool) => (
                        <Card key={tool.tool_id}>
                          <CardHeader>
                            <CardTitle className="text-base flex items-center justify-between">
                              {tool.name}
                              <Badge variant="secondary">{tool.status}</Badge>
                            </CardTitle>
                            <CardDescription>{tool.description}</CardDescription>
                          </CardHeader>
                          <CardContent className="space-y-3">
                            <div className="grid gap-2 md:grid-cols-3 text-sm">
                              <div>Mode: {tool.maker_mode}</div>
                              <div>Version: {tool.version}</div>
                              <div>Usage: {tool.usage_count ?? 0}</div>
                            </div>
                            <div className="space-y-2">
                              <Label>Test Input (JSON)</Label>
                              <Textarea
                                value={toolTestInputs[tool.tool_id] ?? "{\n  \"task\": \"Run validation test\"\n}"}
                                onChange={(event) =>
                                  setToolTestInputs((prev) => ({ ...prev, [tool.tool_id]: event.target.value }))
                                }
                              />
                            </div>
                            <div className="flex flex-wrap gap-2">
                              <Button variant="outline" onClick={() => handleTestTool(tool.tool_id)} disabled={loading}>
                                Test
                              </Button>
                              <Button variant="outline" onClick={() => handleValidateTool(tool.tool_id)} disabled={loading}>
                                Validate
                              </Button>
                            </div>
                            {toolTestResults[tool.tool_id] ? (
                              <div className="rounded-md border p-3 text-sm">
                                <div className="font-medium">Test Result</div>
                                <pre className="mt-2 text-xs whitespace-pre-wrap">
                                  {JSON.stringify(toolTestResults[tool.tool_id], null, 2)}
                                </pre>
                              </div>
                            ) : null}
                          </CardContent>
                        </Card>
                      ))
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="executor" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Tool Executor</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Select Tool</Label>
                    <Select
                      value={executorState.toolId}
                      onValueChange={(value) => setExecutorState((prev) => ({ ...prev, toolId: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select validated tool" />
                      </SelectTrigger>
                      <SelectContent>
                        {tools.map((tool) => (
                          <SelectItem key={tool.tool_id} value={tool.tool_id}>
                            {tool.name} ({tool.status})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Input Data (JSON)</Label>
                    <Textarea
                      value={executorState.inputJson}
                      onChange={(event) =>
                        setExecutorState((prev) => ({ ...prev, inputJson: event.target.value }))
                      }
                      className="min-h-[160px]"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch
                      checked={executorState.delegateToCrewAI}
                      onCheckedChange={(checked) =>
                        setExecutorState((prev) => ({ ...prev, delegateToCrewAI: checked }))
                      }
                    />
                    <Label>Delegate to CrewAI</Label>
                  </div>
                  <Button onClick={handleExecuteTool} disabled={loading}>
                    Execute
                  </Button>
                  {executorResult ? (
                    <div className="rounded-md border p-3 text-sm">
                      <div className="font-medium">Execution Result</div>
                      <pre className="mt-2 text-xs whitespace-pre-wrap">
                        {JSON.stringify(executorResult, null, 2)}
                      </pre>
                    </div>
                  ) : null}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="delegations" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">CrewAI Workflow Tracker</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Status Filter</Label>
                      <Select value={delegationFilter.status} onValueChange={(value) => setDelegationFilter((prev) => ({ ...prev, status: value }))}>
                        <SelectTrigger>
                          <SelectValue placeholder="All" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="">All</SelectItem>
                          {["pending", "assigned", "in_progress", "in_review", "complete", "failed"].map((status) => (
                            <SelectItem key={status} value={status}>
                              {status}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Type Filter</Label>
                      <Select value={delegationFilter.type} onValueChange={(value) => setDelegationFilter((prev) => ({ ...prev, type: value }))}>
                        <SelectTrigger>
                          <SelectValue placeholder="All" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="">All</SelectItem>
                          {["maker_run", "mdap_task", "custom_tool"].map((type) => (
                            <SelectItem key={type} value={type}>
                              {type}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" onClick={refreshDelegations} disabled={loading}>
                      Refresh
                    </Button>
                    <Button variant="outline" onClick={handleSyncDelegations} disabled={loading}>
                      Sync From CrewAI
                    </Button>
                  </div>
                  <div className="space-y-3">
                    {delegations.length === 0 ? (
                      <div className="text-sm text-muted-foreground">No delegations found.</div>
                    ) : (
                      delegations.map((delegation) => (
                        <Card key={delegation.delegation_id}>
                          <CardHeader>
                            <CardTitle className="text-base flex items-center justify-between">
                              {delegation.title}
                              <Badge variant="secondary">{delegation.status}</Badge>
                            </CardTitle>
                            <CardDescription>{delegation.description}</CardDescription>
                          </CardHeader>
                          <CardContent className="text-sm space-y-2">
                            <div>Ticket: {delegation.task_id}</div>
                            <div>Type: {delegation.delegation_type}</div>
                            <div>Created: {delegation.created_at}</div>
                            {delegation.result ? (
                              <div className="rounded-md border p-2 text-xs whitespace-pre-wrap">
                                {JSON.stringify(delegation.result, null, 2)}
                              </div>
                            ) : null}
                          </CardContent>
                        </Card>
                      ))
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="analytics" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Workflow Analytics</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-4">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Total Tools</CardTitle>
                      </CardHeader>
                      <CardContent className="text-2xl font-semibold">{toolStats.total}</CardContent>
                    </Card>
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Validated Tools</CardTitle>
                      </CardHeader>
                      <CardContent className="text-2xl font-semibold">{toolStats.validated}</CardContent>
                    </Card>
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Total Executions</CardTitle>
                      </CardHeader>
                      <CardContent className="text-2xl font-semibold">{toolStats.totalUsage}</CardContent>
                    </Card>
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Active Delegations</CardTitle>
                      </CardHeader>
                      <CardContent className="text-2xl font-semibold">{toolStats.activeDelegations}</CardContent>
                    </Card>
                  </div>

                  <Separator />

                  <div className="space-y-2 text-sm">
                    {Object.entries(toolStats.statusCounts).map(([status, count]) => (
                      <div key={status} className="flex items-center justify-between">
                        <span>{status}</span>
                        <Badge variant="outline">{count}</Badge>
                      </div>
                    ))}
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
