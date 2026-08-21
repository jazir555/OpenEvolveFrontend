import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { openevolveApi } from "@/lib/openevolveApi";
import type { WorkflowSummary, WorkflowTelemetry } from "@/lib/types";

const formatNumber = (value?: number | null, decimals = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toFixed(decimals);
};

const renderHeatmap = (grid: number[][]) => {
  const maxValue = Math.max(...grid.flat());
  return (
    <div className="overflow-auto">
      <div className="grid" style={{ gridTemplateColumns: `repeat(${grid[0]?.length || 0}, minmax(24px, 1fr))` }}>
        {grid.flatMap((row, rowIndex) =>
          row.map((value, colIndex) => {
            const intensity = maxValue ? Math.round((value / maxValue) * 255) : 0;
            const background = `rgb(${255 - intensity}, ${255 - intensity}, 255)`;
            return (
              <div
                key={`${rowIndex}-${colIndex}`}
                className="h-8 w-8 border text-[10px] flex items-center justify-center"
                style={{ background }}
              >
                {value.toFixed(2)}
              </div>
            );
          }),
        )}
      </div>
    </div>
  );
};

export const OpenEvolveVisualizationTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>("");
  const [telemetry, setTelemetry] = useState<WorkflowTelemetry | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [configState, setConfigState] = useState({
    maxIterations: 100,
    populationSize: 50,
    numIslands: 5,
    archiveSize: 100,
    eliteRatio: 0.1,
    explorationRatio: 0.2,
    exploitationRatio: 0.7,
    featureDims: ["complexity", "diversity"],
    cascadeEvaluation: false,
    enableArtifacts: true,
    llmFeedback: false,
    tracing: false,
  });

  const refresh = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const workflowsRes = await openevolveApi.listWorkflows(apiConfig);
      setWorkflows(workflowsRes.workflows || []);
      if (!selectedWorkflowId && workflowsRes.workflows?.length) {
        setSelectedWorkflowId(workflowsRes.workflows[0].workflow_id);
      }
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load workflows.");
    } finally {
      setLoading(false);
    }
  };

  const refreshTelemetry = async (workflowId: string) => {
    if (!workflowId) return;
    try {
      const data = await openevolveApi.getWorkflowTelemetry(workflowId, apiConfig);
      setTelemetry(data);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load workflow telemetry.");
    }
  };

  useEffect(() => {
    refresh();
  }, [apiConfig.apiKey]);

  useEffect(() => {
    refreshTelemetry(selectedWorkflowId);
  }, [selectedWorkflowId, apiConfig.apiKey]);

  const metrics = telemetry?.openevolve_metrics as Record<string, any> | undefined;
  const mapElitesGrid = metrics?.map_elites_grid as number[][] | undefined;
  const featureDimensions = metrics?.feature_dimensions as string[] | undefined;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>OpenEvolve Visualization</CardTitle>
          <CardDescription>Evolution analytics, diagnostics, and configuration insights.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key</label>
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
            <Button variant="outline" onClick={refresh} disabled={loading}>
              Refresh Workflows
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="space-y-2">
            <label className="text-sm font-medium">Workflow</label>
            <Select value={selectedWorkflowId} onValueChange={setSelectedWorkflowId}>
              <SelectTrigger>
                <SelectValue placeholder="Select workflow" />
              </SelectTrigger>
              <SelectContent>
                {workflows.map((workflow) => (
                  <SelectItem key={workflow.workflow_id} value={workflow.workflow_id}>
                    {workflow.workflow_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Tabs defaultValue="evolution">
            <TabsList className="flex flex-wrap gap-2">
              <TabsTrigger value="evolution">Evolution Dashboard</TabsTrigger>
              <TabsTrigger value="diagnostics">Advanced Diagnostics</TabsTrigger>
              <TabsTrigger value="configuration">Configuration</TabsTrigger>
              <TabsTrigger value="performance">Performance Metrics</TabsTrigger>
            </TabsList>

            <TabsContent value="evolution" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Evolution Summary</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div>Iterations: {metrics?.iterations_completed ?? "n/a"}</div>
                  <div>Best Fitness: {formatNumber(metrics?.best_fitness, 3)}</div>
                  <div>Population Size: {metrics?.population_size ?? "n/a"}</div>
                  <div>Archive Size: {metrics?.archive_size ?? "n/a"}</div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">MAP-Elites Grid</CardTitle>
                  <CardDescription>
                    {featureDimensions?.length
                      ? `Dimensions: ${featureDimensions.join(", ")}`
                      : "No feature dimensions recorded."}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {mapElitesGrid ? (
                    renderHeatmap(mapElitesGrid)
                  ) : (
                    <div className="text-muted-foreground">No MAP-Elites grid data available.</div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="diagnostics" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Diagnostics Overview</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {metrics?.diagnostics ? (
                    Object.entries(metrics.diagnostics as Record<string, unknown>).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span>{key}</span>
                        <Badge variant="secondary">{String(value)}</Badge>
                      </div>
                    ))
                  ) : (
                    <div className="text-muted-foreground">No diagnostics data available.</div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="configuration" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Local Configuration</CardTitle>
                  <CardDescription>Configure parameters for upcoming evolution runs.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Max Iterations</label>
                      <Input
                        type="number"
                        value={configState.maxIterations}
                        onChange={(event) =>
                          setConfigState((prev) => ({
                            ...prev,
                            maxIterations: Number(event.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Population Size</label>
                      <Input
                        type="number"
                        value={configState.populationSize}
                        onChange={(event) =>
                          setConfigState((prev) => ({
                            ...prev,
                            populationSize: Number(event.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Islands</label>
                      <Input
                        type="number"
                        value={configState.numIslands}
                        onChange={(event) =>
                          setConfigState((prev) => ({
                            ...prev,
                            numIslands: Number(event.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Archive Size</label>
                      <Input
                        type="number"
                        value={configState.archiveSize}
                        onChange={(event) =>
                          setConfigState((prev) => ({
                            ...prev,
                            archiveSize: Number(event.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Elite Ratio</label>
                      <Input
                        type="number"
                        value={configState.eliteRatio}
                        step="0.01"
                        onChange={(event) =>
                          setConfigState((prev) => ({
                            ...prev,
                            eliteRatio: Number(event.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Exploration Ratio</label>
                      <Input
                        type="number"
                        value={configState.explorationRatio}
                        step="0.01"
                        onChange={(event) =>
                          setConfigState((prev) => ({
                            ...prev,
                            explorationRatio: Number(event.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Exploitation Ratio</label>
                      <Input
                        type="number"
                        value={configState.exploitationRatio}
                        step="0.01"
                        onChange={(event) =>
                          setConfigState((prev) => ({
                            ...prev,
                            exploitationRatio: Number(event.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                  </div>
                  <div className="rounded border p-2 text-xs text-muted-foreground">
                    Feature Dimensions: {configState.featureDims.join(", ")}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="performance" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Performance Metrics</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {telemetry?.performance_metrics ? (
                    Object.entries(telemetry.performance_metrics).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span>{key}</span>
                        <Badge variant="secondary">{String(value)}</Badge>
                      </div>
                    ))
                  ) : (
                    <div className="text-muted-foreground">No performance metrics available.</div>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Resource Usage</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {telemetry?.resource_usage ? (
                    Object.entries(telemetry.resource_usage).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span>{key}</span>
                        <Badge variant="secondary">{String(value)}</Badge>
                      </div>
                    ))
                  ) : (
                    <div className="text-muted-foreground">No resource usage data.</div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};
