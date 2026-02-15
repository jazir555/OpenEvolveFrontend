import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { openevolveApi } from "../../../lib/openevolveApi";
import type {
  WorkflowSummary,
  WorkflowResourceUsageResponse,
  WorkflowResourceOptimizationResponse,
  ResourceUsageSummary,
} from "../../../lib/types";

const readApiKey = () => {
  try {
    return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
  } catch {
    return "";
  }
};

export const ResourceManagerTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(readApiKey);
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>("");
  const [usage, setUsage] = useState<ResourceUsageSummary | null>(null);
  const [optimization, setOptimization] = useState<Record<string, unknown> | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadWorkflows = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.listWorkflows(apiConfig);
      setWorkflows(response.workflows ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load workflows.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadWorkflows();
  }, [apiConfig.apiKey]);

  const loadUsage = async () => {
    if (!selectedWorkflowId) return;
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const response: WorkflowResourceUsageResponse = await openevolveApi.getWorkflowResourceUsage(
        selectedWorkflowId,
        apiConfig,
      );
      setUsage(response.resource_usage ?? null);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load resource usage.");
    }
  };

  const loadOptimization = async () => {
    if (!selectedWorkflowId) return;
    setErrorMessage(null);
    try {
      const response: WorkflowResourceOptimizationResponse = await openevolveApi.optimizeWorkflowResources(
        selectedWorkflowId,
        apiConfig,
      );
      setOptimization(response.suggestions ?? null);
      setStatusMessage("Optimization suggestions generated.");
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to optimize resources.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Resource Manager</CardTitle>
          <CardDescription>Inspect workflow resource usage and optimization hints.</CardDescription>
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
                    // ignore
                  }
                }}
              />
            </div>
            <Button variant="outline" onClick={loadWorkflows} disabled={loading}>
              Refresh
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-[260px_1fr]">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Workflows</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <select
                  className="w-full rounded border border-input bg-background px-3 py-2 text-sm"
                  value={selectedWorkflowId}
                  onChange={(event) => setSelectedWorkflowId(event.target.value)}
                >
                  <option value="">Select workflow</option>
                  {workflows.map((workflow) => (
                    <option key={workflow.workflow_id} value={workflow.workflow_id}>
                      {workflow.workflow_id}
                    </option>
                  ))}
                </select>
                <Button className="w-full" variant="secondary" onClick={loadUsage}>
                  Load Usage
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Usage Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {!usage && <div className="text-sm text-muted-foreground">No usage data loaded.</div>}
                {usage && (
                  <div className="grid gap-3 md:grid-cols-2 text-sm">
                    <div className="rounded border p-3">
                      <div className="text-xs text-muted-foreground">API Calls</div>
                      <div className="text-lg font-semibold">{usage.api_calls ?? 0}</div>
                    </div>
                    <div className="rounded border p-3">
                      <div className="text-xs text-muted-foreground">Tokens Used</div>
                      <div className="text-lg font-semibold">{usage.tokens_used ?? 0}</div>
                    </div>
                    <div className="rounded border p-3">
                      <div className="text-xs text-muted-foreground">Estimated Cost</div>
                      <div className="text-lg font-semibold">${usage.estimated_cost ?? 0}</div>
                    </div>
                    <div className="rounded border p-3">
                      <div className="text-xs text-muted-foreground">Execution Time</div>
                      <div className="text-lg font-semibold">{usage.execution_time_seconds ?? 0}s</div>
                    </div>
                  </div>
                )}

                {usage?.limits ? (
                  <div className="space-y-2">
                    <div className="text-sm font-semibold">Limits</div>
                    <pre className="rounded border p-2 text-xs whitespace-pre-wrap">
                      {JSON.stringify(usage.limits, null, 2)}
                    </pre>
                  </div>
                ) : null}

                {usage?.component_breakdown ? (
                  <div className="space-y-2">
                    <div className="text-sm font-semibold">Component Breakdown</div>
                    {Object.entries(usage.component_breakdown).map(([component, metrics]) => (
                      <div key={component} className="rounded border p-2 text-xs">
                        <div className="font-semibold">{component}</div>
                        <pre className="whitespace-pre-wrap">{JSON.stringify(metrics, null, 2)}</pre>
                      </div>
                    ))}
                  </div>
                ) : null}

                <Separator />
                <Button variant="outline" onClick={loadOptimization}>
                  Generate Optimization Suggestions
                </Button>
                {optimization && (
                  <pre className="rounded border p-2 text-xs whitespace-pre-wrap">
                    {JSON.stringify(optimization, null, 2)}
                  </pre>
                )}
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
