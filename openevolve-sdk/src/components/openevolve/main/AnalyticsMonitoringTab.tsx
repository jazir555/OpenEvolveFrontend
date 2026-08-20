import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { openevolveApi } from "@/lib/openevolveApi";
import type {
  StatisticsSummary,
  MonitoringDashboardMetrics,
  MonitoringAlert,
  MonitoringService,
  AnalyticsWorkflowMetric,
  WorkflowSummary,
} from "@/lib/types";

const average = (values: Array<number | null | undefined>) => {
  const filtered = values.filter((value): value is number => typeof value === "number");
  if (!filtered.length) return null;
  return filtered.reduce((acc, value) => acc + value, 0) / filtered.length;
};

const formatNumber = (value?: number | null, decimals = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toFixed(decimals);
};

export const AnalyticsMonitoringTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [statistics, setStatistics] = useState<StatisticsSummary | null>(null);
  const [workflowMetrics, setWorkflowMetrics] = useState<AnalyticsWorkflowMetric[]>([]);
  const [monitoringDashboard, setMonitoringDashboard] = useState<MonitoringDashboardMetrics | null>(
    null,
  );
  const [monitoringAlerts, setMonitoringAlerts] = useState<MonitoringAlert[]>([]);
  const [monitoringServices, setMonitoringServices] = useState<MonitoringService[]>([]);
  const [workflowSummaries, setWorkflowSummaries] = useState<WorkflowSummary[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(15);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const refresh = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const [
        stats,
        metrics,
        monitoring,
        alerts,
        services,
        workflows,
      ] = await Promise.all([
        openevolveApi.getStatistics(apiConfig),
        openevolveApi.getWorkflowMetrics(apiConfig),
        openevolveApi.getMonitoringDashboard(apiConfig),
        openevolveApi.getMonitoringAlerts(apiConfig),
        openevolveApi.getMonitoringServices(apiConfig),
        openevolveApi.listWorkflows(apiConfig),
      ]);
      setStatistics(stats);
      setWorkflowMetrics(metrics.metrics || []);
      setMonitoringDashboard(monitoring);
      setMonitoringAlerts(alerts.alerts || []);
      setMonitoringServices(services.services || []);
      setWorkflowSummaries(workflows.workflows || []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load analytics monitoring data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, [apiConfig.apiKey]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = globalThis.setInterval(() => {
      refresh();
    }, Math.max(5, refreshInterval) * 1000);
    return () => globalThis.clearInterval(interval);
  }, [autoRefresh, refreshInterval, apiConfig.apiKey]);

  const avgBestFitness = average(workflowMetrics.map((metric) => metric.best_fitness));
  const avgDiversity = average(workflowMetrics.map((metric) => metric.diversity));
  const avgExecution = average(workflowMetrics.map((metric) => metric.execution_time));
  const totalTokens = workflowMetrics.reduce(
    (acc, metric) => acc + (metric.tokens_used ?? 0),
    0,
  );

  const statusCounts = workflowSummaries.reduce<Record<string, number>>((acc, workflow) => {
    const status = workflow.status ?? "unknown";
    acc[status] = (acc[status] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Analytics & Monitoring</CardTitle>
          <CardDescription>Performance analytics with live monitoring signals.</CardDescription>
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
              Refresh
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-4">
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
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Fitness / Diversity</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>Avg Best Fitness: {formatNumber(avgBestFitness, 3)}</div>
                <div>Avg Diversity: {formatNumber(avgDiversity, 3)}</div>
                <div>Avg Execution: {formatNumber(avgExecution, 1)}s</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Tokens</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>Total Tokens: {totalTokens.toLocaleString()}</div>
                <div>Samples: {workflowMetrics.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Monitoring</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>
                  Health:{" "}
                  <Badge
                    variant={
                      monitoringDashboard?.health?.status === "healthy" ? "default" : "secondary"
                    }
                  >
                    {monitoringDashboard?.health?.status ?? "unknown"}
                  </Badge>
                </div>
                <div>Alerts: {monitoringAlerts.length}</div>
                <div>Services: {monitoringServices.length}</div>
              </CardContent>
            </Card>
          </div>

          <Tabs defaultValue="performance" className="w-full">
            <TabsList className="flex flex-wrap gap-2">
              <TabsTrigger value="performance">Performance</TabsTrigger>
              <TabsTrigger value="system">System Monitoring</TabsTrigger>
              <TabsTrigger value="workflows">Workflow Analytics</TabsTrigger>
              <TabsTrigger value="reporting">Advanced Reporting</TabsTrigger>
              <TabsTrigger value="controls">Controls</TabsTrigger>
            </TabsList>

            <TabsContent value="performance" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Workflow Performance Snapshot</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {workflowMetrics.length === 0 ? (
                      <div className="text-muted-foreground">No workflow metrics yet.</div>
                    ) : (
                      workflowMetrics.slice(0, 8).map((metric) => (
                        <div key={metric.workflow_id} className="rounded border p-2">
                          <div className="flex items-center justify-between">
                            <div className="font-semibold">{metric.workflow_id}</div>
                            <Badge variant="secondary">{metric.status ?? "unknown"}</Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Fitness: {formatNumber(metric.best_fitness, 3)} · Diversity:{" "}
                            {formatNumber(metric.diversity, 3)}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Tokens: {metric.tokens_used ?? 0} · Execution:{" "}
                            {formatNumber(metric.execution_time, 1)}s
                          </div>
                          <div className="mt-2 h-2 w-full rounded bg-muted">
                            <div
                              className="h-2 rounded bg-blue-500"
                              style={{ width: `${Math.round((metric.progress ?? 0) * 100)}%` }}
                            />
                          </div>
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Resource Utilization</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {workflowMetrics.length === 0 ? (
                      <div className="text-muted-foreground">No resource metrics available.</div>
                    ) : (
                      workflowMetrics.slice(0, 8).map((metric) => (
                        <div key={`resource-${metric.workflow_id}`} className="rounded border p-2">
                          <div className="font-semibold">{metric.workflow_id}</div>
                          <div className="text-xs text-muted-foreground">
                            Memory: {formatNumber(metric.memory_usage, 1)} MB · CPU:{" "}
                            {formatNumber(metric.cpu_usage, 2)}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Population: {metric.population_size ?? "n/a"} · Generation:{" "}
                            {metric.generation ?? "n/a"}
                          </div>
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="system" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">System Metrics</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    <div>CPU: {formatNumber(monitoringDashboard?.system?.system?.cpu_percent, 1)}%</div>
                    <div>
                      Memory: {formatNumber(monitoringDashboard?.system?.system?.memory_percent, 1)}%
                    </div>
                    <div>
                      Uptime:{" "}
                      {monitoringDashboard?.health?.uptime_seconds
                        ? `${monitoringDashboard.health.uptime_seconds.toFixed(0)}s`
                        : "n/a"}
                    </div>
                    <div>
                      Health:{" "}
                      <Badge
                        variant={
                          monitoringDashboard?.health?.status === "healthy" ? "default" : "secondary"
                        }
                      >
                        {monitoringDashboard?.health?.status ?? "unknown"}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Service Health Checks</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {monitoringServices.length === 0 ? (
                      <div className="text-muted-foreground">No service checks found.</div>
                    ) : (
                      monitoringServices.map((service) => (
                        <div key={service.name} className="rounded border p-2">
                          <div className="flex items-center justify-between">
                            <div className="font-semibold">{service.name}</div>
                            <Badge variant={service.healthy ? "default" : "secondary"}>
                              {service.status ?? "unknown"}
                            </Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Exec: {formatNumber(service.execution_time, 3)}s ·{" "}
                            {service.timestamp ?? "n/a"}
                          </div>
                          {service.error ? (
                            <div className="text-xs text-red-500">{service.error}</div>
                          ) : null}
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
              </div>
              <Card className="mt-4">
                <CardHeader>
                  <CardTitle className="text-sm">Active Alerts</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {monitoringAlerts.length === 0 ? (
                    <div className="text-muted-foreground">No active alerts.</div>
                  ) : (
                    monitoringAlerts.map((alert, index) => (
                      <div key={`alert-${index}`} className="rounded border p-2">
                        <div className="font-semibold">{alert.name ?? "alert"}</div>
                        <div className="text-xs text-muted-foreground">
                          {alert.description ?? alert.metric_name ?? "threshold triggered"}
                        </div>
                      </div>
                    ))
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="workflows" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Status Distribution</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {Object.keys(statusCounts).length === 0 ? (
                      <div className="text-muted-foreground">No workflow data.</div>
                    ) : (
                      Object.entries(statusCounts).map(([status, count]) => (
                        <div key={status} className="flex items-center justify-between">
                          <span>{status}</span>
                          <Badge variant="secondary">{count}</Badge>
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Current Progress</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {workflowSummaries.length === 0 ? (
                      <div className="text-muted-foreground">No workflows available.</div>
                    ) : (
                      workflowSummaries.slice(0, 10).map((workflow) => (
                        <div key={workflow.workflow_id} className="rounded border p-2">
                          <div className="flex items-center justify-between">
                            <div className="font-semibold">{workflow.workflow_id}</div>
                            <Badge variant="secondary">{workflow.status}</Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Stage: {workflow.current_stage ?? "n/a"}
                          </div>
                          <div className="mt-2 h-2 w-full rounded bg-muted">
                            <div
                              className="h-2 rounded bg-emerald-500"
                              style={{ width: `${Math.round((workflow.progress ?? 0) * 100)}%` }}
                            />
                          </div>
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="reporting" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Generated Summary</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <div>
                    Success Rate:{" "}
                    {statistics
                      ? `${((statistics.completed / Math.max(statistics.total_workflows, 1)) * 100).toFixed(1)}%`
                      : "n/a"}
                  </div>
                  <div>Average Execution: {formatNumber(avgExecution, 1)}s</div>
                  <div>Average Best Fitness: {formatNumber(avgBestFitness, 3)}</div>
                  <div>Total Tokens Used: {totalTokens.toLocaleString()}</div>
                  <Button
                    variant="outline"
                    onClick={() => {
                      const payload = {
                        generated_at: new Date().toISOString(),
                        statistics,
                        averages: { avgBestFitness, avgDiversity, avgExecution },
                        totalTokens,
                      };
                      globalThis.navigator?.clipboard?.writeText(JSON.stringify(payload, null, 2));
                    }}
                  >
                    Copy Report JSON
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="controls" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Dashboard Controls</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <div className="flex items-center justify-between">
                    <span>Auto refresh</span>
                    <Button
                      variant={autoRefresh ? "default" : "outline"}
                      onClick={() => setAutoRefresh((prev) => !prev)}
                    >
                      {autoRefresh ? "Enabled" : "Disabled"}
                    </Button>
                  </div>
                  <div className="space-y-1">
                    <label className="text-sm font-medium">Refresh Interval (seconds)</label>
                    <Input
                      type="number"
                      value={refreshInterval}
                      onChange={(event) => setRefreshInterval(Number(event.target.value) || 10)}
                    />
                  </div>
                  <Button variant="outline" onClick={refresh}>
                    Refresh Now
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};
