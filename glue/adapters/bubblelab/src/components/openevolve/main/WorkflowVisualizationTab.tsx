import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { openevolveApi } from "../../../lib/openevolveApi";
import type { WorkflowSummary, WorkflowDetail, WorkflowTelemetry, WorkflowPlanResponse } from "../../../lib/types";

const formatNumber = (value?: number | null, decimals = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toFixed(decimals);
};

const getStagesForType = (workflowType?: string) => {
  if (!workflowType) {
    return ["Input", "Processing", "Output"];
  }
  const lower = workflowType.toLowerCase();
  if (lower.includes("evolution")) {
    return ["Input", "Analysis", "Evolution", "Evaluation", "Output"];
  }
  if (lower.includes("adversarial")) {
    return ["Input", "Red Team", "Blue Team", "Evaluator", "Output"];
  }
  if (lower.includes("sovereign") || lower.includes("decomposition")) {
    return ["Input", "Analysis", "Decomposition", "Solving", "Assembly", "Verification", "Output"];
  }
  return ["Input", "Processing", "Output"];
};

const extractHistorySeries = (metrics?: Record<string, unknown>) => {
  if (!metrics) return null;
  const history =
    (metrics as any).history ||
    (metrics as any).evolution_history ||
    (metrics as any).fitness_history ||
    (metrics as any).generations;
  if (!history) return null;
  if (Array.isArray(history)) {
    if (history.length === 0) return null;
    if (typeof history[0] === "number") {
      return {
        generations: history.map((_, index) => index + 1),
        best: history,
        avg: [],
        diversity: (metrics as any).diversity_history || [],
      };
    }
    return {
      generations: history.map((entry: any, index: number) => entry.generation ?? index + 1),
      best: history.map((entry: any) => entry.best_fitness ?? entry.best_score ?? null).filter((v) => v != null),
      avg: history.map((entry: any) => entry.avg_fitness ?? entry.average_fitness ?? null).filter((v) => v != null),
      diversity: history
        .map((entry: any) => entry.diversity ?? entry.diversity_score ?? null)
        .filter((v) => v != null),
    };
  }
  return null;
};

export const WorkflowVisualizationTab: React.FC = () => {
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
  const [workflowDetail, setWorkflowDetail] = useState<WorkflowDetail | null>(null);
  const [workflowTelemetry, setWorkflowTelemetry] = useState<WorkflowTelemetry | null>(null);
  const [workflowPlan, setWorkflowPlan] = useState<WorkflowPlanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const refreshList = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.listWorkflows(apiConfig);
      setWorkflows(response.workflows || []);
      if (!selectedWorkflowId && response.workflows?.length) {
        setSelectedWorkflowId(response.workflows[0].workflow_id);
      }
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load workflows.");
    } finally {
      setLoading(false);
    }
  };

  const refreshWorkflow = async (workflowId: string) => {
    if (!workflowId) return;
    setErrorMessage(null);
    try {
      const [detail, telemetry, plan] = await Promise.all([
        openevolveApi.getWorkflow(workflowId, apiConfig),
        openevolveApi.getWorkflowTelemetry(workflowId, apiConfig),
        openevolveApi.getWorkflowPlan(workflowId, apiConfig).catch(() => null),
      ]);
      setWorkflowDetail(detail);
      setWorkflowTelemetry(telemetry);
      setWorkflowPlan(plan);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load workflow data.");
    }
  };

  useEffect(() => {
    refreshList();
  }, [apiConfig.apiKey]);

  useEffect(() => {
    refreshWorkflow(selectedWorkflowId);
  }, [selectedWorkflowId, apiConfig.apiKey]);

  const stages = getStagesForType(workflowTelemetry?.workflow_type);
  const history = extractHistorySeries(workflowTelemetry?.openevolve_metrics || undefined);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Workflow Visualization</CardTitle>
          <CardDescription>Live workflow metrics and execution flow overview.</CardDescription>
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
            <Button variant="outline" onClick={refreshList} disabled={loading}>
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

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Workflow Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {workflowDetail ? (
                  <>
                    <div className="flex items-center justify-between">
                      <div className="font-semibold">{workflowDetail.workflow_id}</div>
                      <Badge variant="secondary">{workflowDetail.status}</Badge>
                    </div>
                    <div>Stage: {workflowDetail.current_stage}</div>
                    <div>Progress: {formatNumber(workflowDetail.progress * 100, 1)}%</div>
                    <div>
                      Sub-problems: {workflowDetail.solved_sub_problems}/
                      {workflowDetail.total_sub_problems}
                    </div>
                  </>
                ) : (
                  <div className="text-muted-foreground">Select a workflow to view status.</div>
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Execution Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div>
                  Execution Time: {formatNumber(workflowTelemetry?.execution_time_seconds, 1)}s
                </div>
                <div>Tokens Used: {workflowTelemetry?.resource_usage?.tokens_used ?? "n/a"}</div>
                <div>
                  Memory Usage: {workflowTelemetry?.resource_usage?.memory_usage_mb ?? "n/a"} MB
                </div>
                <div>CPU Usage: {workflowTelemetry?.resource_usage?.cpu_usage ?? "n/a"}</div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Workflow Flow</CardTitle>
              <CardDescription>{workflowTelemetry?.workflow_type ?? "Generic"} pipeline</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-wrap items-center gap-2 text-sm">
              {stages.map((stage, index) => {
                const isCurrent = workflowTelemetry?.current_stage === stage;
                return (
                  <div key={stage} className="flex items-center gap-2">
                    <Badge variant={isCurrent ? "default" : "secondary"}>{stage}</Badge>
                    {index < stages.length - 1 && <span className="text-muted-foreground">→</span>}
                  </div>
                );
              })}
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Evolution Progress</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {history ? (
                  <div className="space-y-2">
                    <div>
                      Latest Best Fitness: {formatNumber(history.best[history.best.length - 1], 3)}
                    </div>
                    <div className="space-y-1">
                      {history.best.slice(-10).map((value, index) => (
                        <div key={`fit-${index}`} className="flex items-center gap-2">
                          <div className="h-2 w-24 rounded bg-muted">
                            <div
                              className="h-2 rounded bg-emerald-500"
                              style={{ width: `${Math.min(100, (value || 0) * 100)}%` }}
                            />
                          </div>
                          <span className="text-xs">{formatNumber(value, 3)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-muted-foreground">No evolution history available.</div>
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Diversity Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {history?.diversity?.length ? (
                  <div className="space-y-1">
                    <div>
                      Latest Diversity:{" "}
                      {formatNumber(history.diversity[history.diversity.length - 1], 3)}
                    </div>
                    {history.diversity.slice(-10).map((value, index) => (
                      <div key={`div-${index}`} className="flex items-center gap-2">
                        <div className="h-2 w-24 rounded bg-muted">
                          <div
                            className="h-2 rounded bg-blue-500"
                            style={{ width: `${Math.min(100, (value || 0) * 100)}%` }}
                          />
                        </div>
                        <span className="text-xs">{formatNumber(value, 3)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-muted-foreground">No diversity history available.</div>
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Decomposition Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              {workflowPlan ? (
                <>
                  <div>
                    Sub-problems: {workflowPlan.plan.sub_problems.length} · Parallel:{" "}
                    {workflowPlan.plan.parallel_processing_enabled ? "enabled" : "disabled"}
                  </div>
                  <div>
                    Planner Team: {workflowPlan.plan.planner_team_name ?? "n/a"} · Assembler Team:{" "}
                    {workflowPlan.plan.assembler_team_name ?? "n/a"}
                  </div>
                </>
              ) : (
                <div className="text-muted-foreground">No decomposition plan loaded.</div>
              )}
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
};
