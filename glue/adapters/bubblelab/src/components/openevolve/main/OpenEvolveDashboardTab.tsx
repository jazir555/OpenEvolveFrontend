import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { openevolveApi } from "../../../lib/openevolveApi";
import type { StatisticsSummary, WorkflowSummary } from "../../../lib/types";

const readStorage = <T,>(key: string, fallback: T): T => {
  try {
    const raw = globalThis.localStorage?.getItem(key);
    if (!raw) {
      return fallback;
    }
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

export const OpenEvolveDashboardTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [statistics, setStatistics] = useState<StatisticsSummary | null>(null);
  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [mdapHealth, setMdapHealth] = useState<Record<string, unknown> | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const [stats, workflowList, healthStatus] = await Promise.all([
        openevolveApi.getStatistics(apiConfig),
        openevolveApi.listWorkflows(apiConfig),
        openevolveApi.getHealth(apiConfig),
      ]);
      setStatistics(stats);
      setWorkflows(workflowList.workflows || []);
      setHealth(healthStatus);
      try {
        const mdap = await openevolveApi.getAdaptiveMdapHealth(apiConfig);
        setMdapHealth(mdap);
      } catch {
        setMdapHealth(null);
      }
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load dashboard data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, [apiConfig.apiKey]);

  const recentState = readStorage<Record<string, any> | null>("openevolve-state", null);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>OpenEvolve Dashboard</CardTitle>
          <CardDescription>System status and workflow overview.</CardDescription>
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
              Refresh Dashboard
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">API Health</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div>
                  Status:{" "}
                  <Badge variant={health?.status === "ok" ? "default" : "secondary"}>
                    {String(health?.status ?? "unknown")}
                  </Badge>
                </div>
                <div>Version: {String(health?.version ?? "n/a")}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Adaptive MDAP</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div>
                  Status:{" "}
                  <Badge variant={mdapHealth?.status === "healthy" ? "default" : "secondary"}>
                    {String(mdapHealth?.status ?? "unavailable")}
                  </Badge>
                </div>
                <div>Details: {mdapHealth ? "Available" : "Not configured"}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Workflows</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>Total: {statistics?.total_workflows ?? 0}</div>
                <div>Running: {statistics?.running ?? 0}</div>
                <div>Completed: {statistics?.completed ?? 0}</div>
                <div>Failed: {statistics?.failed ?? 0}</div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Active Workflows</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {workflows.length === 0 && (
                  <div className="text-muted-foreground">No workflows found.</div>
                )}
                {workflows.map((workflow) => (
                  <div key={workflow.workflow_id} className="rounded border p-2 space-y-1">
                    <div className="flex items-center justify-between">
                      <div className="font-semibold">{workflow.workflow_id}</div>
                      <Badge variant="secondary">{workflow.status}</Badge>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Stage: {workflow.current_stage}
                    </div>
                    <div className="h-2 w-full rounded bg-muted">
                      <div
                        className="h-2 rounded bg-blue-500"
                        style={{ width: `${Math.round(workflow.progress * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Recent Evolution Snapshot</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div>
                  Last Evolution Status:{" "}
                  {recentState?.evolutionStatusMessage || "No recent runs."}
                </div>
                <div>
                  Best Score: {String(recentState?.evolutionBestScore ?? "n/a")}
                </div>
                <div className="rounded border p-2 bg-muted">
                  {recentState?.evolutionCurrentBest
                    ? recentState.evolutionCurrentBest.slice(0, 240)
                    : "No evolved content stored."}
                </div>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
