import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { openevolveApi } from "@/lib/openevolveApi";
import type {
  AdaptiveMdapDashboard,
  AdaptiveMdapProfiles,
  AuditLogEntry,
  MonitoringDashboardMetrics,
  MonitoringAlert,
  MonitoringMetric,
} from "@/lib/types";

const DEFAULT_MODELS = [
  "gpt-4o-mini",
  "gpt-4o",
  "gpt-4",
  "claude-3-5-sonnet",
  "claude-3-5-haiku",
  "gemini-1-5-pro",
  "gemini-1-5-flash",
];

const parseJson = (value: string) => {
  if (!value.trim()) {
    return undefined;
  }
  return JSON.parse(value);
};

export const MonitoringTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [health, setHealth] = useState<Record<string, any> | null>(null);
  const [dashboard, setDashboard] = useState<AdaptiveMdapDashboard | null>(null);
  const [profiles, setProfiles] = useState<AdaptiveMdapProfiles | null>(null);
  const [profileConfig, setProfileConfig] = useState<Record<string, unknown> | null>(null);
  const [selectedProfile, setSelectedProfile] = useState<string>("");
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([]);
  const [systemDashboard, setSystemDashboard] = useState<MonitoringDashboardMetrics | null>(null);
  const [monitoringAlerts, setMonitoringAlerts] = useState<MonitoringAlert[]>([]);
  const [metricName, setMetricName] = useState("system_cpu_percent");
  const [metricSamples, setMetricSamples] = useState<MonitoringMetric[]>([]);
  const [monitoringWsStatus, setMonitoringWsStatus] = useState("disconnected");
  const [monitoringUpdates, setMonitoringUpdates] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [costInputs, setCostInputs] = useState({
    numProblems: "100",
    model: DEFAULT_MODELS[0],
    distribution: "",
  });
  const [costResult, setCostResult] = useState<Record<string, unknown> | null>(null);

  const [complexityInputs, setComplexityInputs] = useState({
    description: "",
    domain: "",
    depth: "1",
    dependencies: "",
    constraints: "",
    successCriteria: "",
  });
  const [complexityResult, setComplexityResult] = useState<Record<string, unknown> | null>(null);

  const [allocationInputs, setAllocationInputs] = useState({
    score: "0.5",
    context: "",
  });
  const [allocationResult, setAllocationResult] = useState<Record<string, unknown> | null>(null);

  const refresh = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const [healthRes, dashboardRes, profilesRes, logsRes, monitoringDashboardRes, alertsRes] =
        await Promise.all([
        openevolveApi.getAdaptiveMdapHealth(apiConfig),
        openevolveApi.getAdaptiveMdapDashboard(apiConfig),
        openevolveApi.getAdaptiveMdapProfiles(apiConfig),
        openevolveApi.listAuditLogs(100, apiConfig),
        openevolveApi.getMonitoringDashboard(apiConfig),
        openevolveApi.getMonitoringAlerts(apiConfig),
      ]);
      setHealth(healthRes);
      setDashboard(dashboardRes);
      setProfiles(profilesRes);
      setSelectedProfile(profilesRes.default);
      setAuditLogs(logsRes.logs || []);
      setSystemDashboard(monitoringDashboardRes);
      setMonitoringAlerts(alertsRes.alerts || []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load monitoring data.");
    } finally {
      setLoading(false);
    }
  };

  const loadMetricSamples = async () => {
    setErrorMessage(null);
    try {
      const result = await openevolveApi.getMonitoringMetrics({ name: metricName }, apiConfig);
      setMetricSamples(result.metrics || []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load monitoring metrics.");
    }
  };

  const connectMonitoringWebsocket = () => {
    try {
      const baseUrl = (globalThis as any)?.OPENEVOLVE_API_BASE as string | undefined;
      const apiBase =
        baseUrl ||
        (() => {
          try {
            return globalThis.localStorage?.getItem("openevolve_api_base") || "";
          } catch {
            return "";
          }
        })();
      if (!apiBase) {
        setErrorMessage("Set openevolve_api_base to use monitoring websocket.");
        return;
      }
      const wsUrl = apiBase.replace(/^http/, "ws");
      const socket = new WebSocket(`${wsUrl}/ws/monitoring`);
      socket.onopen = () => setMonitoringWsStatus("connected");
      socket.onclose = () => setMonitoringWsStatus("disconnected");
      socket.onerror = () => setMonitoringWsStatus("error");
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          setMonitoringUpdates((prev) => [payload, ...prev].slice(0, 20));
        } catch {
          // ignore malformed messages
        }
      };
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to connect websocket.");
    }
  };

  const loadProfileConfig = async (profileName: string) => {
    try {
      const config = await openevolveApi.getAdaptiveMdapProfileConfig(profileName, apiConfig);
      setProfileConfig(config);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load profile config.");
    }
  };

  const calculateCost = async () => {
    setErrorMessage(null);
    try {
      const payload: Record<string, unknown> = {
        num_problems: Number(costInputs.numProblems),
        model: costInputs.model,
      };
      const distribution = parseJson(costInputs.distribution || "");
      if (distribution) {
        payload.workload_distribution = distribution;
      }
      const result = await openevolveApi.calculateAdaptiveMdapCost(payload, apiConfig);
      setCostResult(result);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to calculate cost.");
    }
  };

  const classifyComplexity = async () => {
    setErrorMessage(null);
    try {
      const payload = {
        description: complexityInputs.description,
        domain: complexityInputs.domain || undefined,
        depth: Number(complexityInputs.depth || 1),
        dependencies: complexityInputs.dependencies
          ? complexityInputs.dependencies.split(",").map((item) => item.trim()).filter(Boolean)
          : undefined,
        constraints: complexityInputs.constraints
          ? complexityInputs.constraints.split(",").map((item) => item.trim()).filter(Boolean)
          : undefined,
        success_criteria: complexityInputs.successCriteria
          ? complexityInputs.successCriteria.split(",").map((item) => item.trim()).filter(Boolean)
          : undefined,
      };
      const result = await openevolveApi.classifyAdaptiveMdapComplexity(payload, apiConfig);
      setComplexityResult(result);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to classify complexity.");
    }
  };

  const allocateResources = async () => {
    setErrorMessage(null);
    try {
      const payload: Record<string, unknown> = {
        complexity_score: Number(allocationInputs.score),
      };
      const context = parseJson(allocationInputs.context || "");
      if (context) {
        payload.context = context;
      }
      const result = await openevolveApi.allocateAdaptiveMdapResources(payload, apiConfig);
      setAllocationResult(result);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to allocate resources.");
    }
  };

  useEffect(() => {
    refresh();
  }, [apiConfig.apiKey]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Adaptive MDAP Monitoring</CardTitle>
          <CardDescription>Health checks, allocation metrics, and cost controls.</CardDescription>
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
            <Button variant="outline" onClick={refresh} disabled={loading}>
              Refresh Metrics
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Health</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div>
                  Status:{" "}
                  <Badge variant={health?.status === "healthy" ? "default" : "secondary"}>
                    {String(health?.status ?? "unknown")}
                  </Badge>
                </div>
                <div>Version: {String(health?.version ?? "n/a")}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                <div>Total Classifications: {dashboard?.summary?.total_classifications ?? 0}</div>
                <div>Successful: {dashboard?.summary?.successful_classifications ?? 0}</div>
                <div>Failed: {dashboard?.summary?.failed_classifications ?? 0}</div>
                <div>Total Allocations: {dashboard?.summary?.total_allocations ?? 0}</div>
                <div>Total Executions: {dashboard?.summary?.total_executions ?? 0}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Allocations</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-sm">
                {dashboard?.allocations ? (
                  Object.entries(dashboard.allocations).map(([key, value]) => (
                    <div key={key}>
                      {key}: {value}
                    </div>
                  ))
                ) : (
                  <div className="text-muted-foreground">No allocation data.</div>
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">System Monitoring</CardTitle>
              <CardDescription>System health, resource usage, and alerts.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded border p-2">
                  <div className="font-semibold">Health Status</div>
                  <div>Status: {systemDashboard?.health?.status ?? "unknown"}</div>
                  <div>Uptime: {systemDashboard?.health?.uptime_seconds?.toFixed(0) ?? "0"}s</div>
                </div>
                <div className="rounded border p-2">
                  <div className="font-semibold">CPU</div>
                  <div>
                    {systemDashboard?.system?.system?.cpu_percent !== undefined
                      ? `${systemDashboard.system.system.cpu_percent.toFixed(1)}%`
                      : "n/a"}
                  </div>
                </div>
                <div className="rounded border p-2">
                  <div className="font-semibold">Memory</div>
                  <div>
                    {systemDashboard?.system?.system?.memory_percent !== undefined
                      ? `${systemDashboard.system.system.memory_percent.toFixed(1)}%`
                      : "n/a"}
                  </div>
                </div>
              </div>

              <div className="rounded border p-2">
                <div className="font-semibold">Workflow Success Rates</div>
                {systemDashboard?.workflow ? (
                  Object.entries(systemDashboard.workflow).map(([key, value]) => (
                    <div key={key} className="text-xs text-muted-foreground">
                      {key}: {(value * 100).toFixed(1)}%
                    </div>
                  ))
                ) : (
                  <div className="text-muted-foreground">No workflow metrics.</div>
                )}
              </div>

              <div className="rounded border p-2">
                <div className="font-semibold">Active Alerts</div>
                {monitoringAlerts.length === 0 ? (
                  <div className="text-muted-foreground">No active alerts.</div>
                ) : (
                  monitoringAlerts.map((alert, index) => (
                    <div key={`alert-${index}`} className="text-xs text-muted-foreground">
                      {alert.name ?? "alert"} · {alert.description ?? "threshold breached"}
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Monitoring Metrics Stream</CardTitle>
              <CardDescription>Fetch metrics or connect to the websocket feed.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex flex-col gap-3 md:flex-row md:items-center">
                <div className="flex-1">
                  <Label>Metric Name</Label>
                  <Input value={metricName} onChange={(event) => setMetricName(event.target.value)} />
                </div>
                <Button variant="outline" onClick={loadMetricSamples}>
                  Load Metrics
                </Button>
                <Button variant="outline" onClick={connectMonitoringWebsocket}>
                  Connect WebSocket
                </Button>
              </div>
              <div className="text-xs text-muted-foreground">
                WebSocket Status: {monitoringWsStatus}
              </div>
              <div className="grid gap-2 md:grid-cols-2">
                <div className="rounded border p-2">
                  <div className="font-semibold">Recent Metric Samples</div>
                  {metricSamples.length === 0 ? (
                    <div className="text-muted-foreground">No samples loaded.</div>
                  ) : (
                    metricSamples.slice(0, 6).map((metric, index) => (
                      <div key={`metric-${index}`} className="text-xs text-muted-foreground">
                        {metric.timestamp ?? "time"} · {metric.value ?? "n/a"}
                      </div>
                    ))
                  )}
                </div>
                <div className="rounded border p-2">
                  <div className="font-semibold">Live Updates</div>
                  {monitoringUpdates.length === 0 ? (
                    <div className="text-muted-foreground">No live updates yet.</div>
                  ) : (
                    monitoringUpdates.slice(0, 6).map((update, index) => (
                      <div key={`update-${index}`} className="text-xs text-muted-foreground">
                        {JSON.stringify(update)}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Execution Strategies</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {dashboard?.execution ? (
                  Object.entries(dashboard.execution).map(([strategy, data]) => (
                    <div key={strategy} className="rounded border p-2">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">{strategy}</div>
                        <Badge variant="secondary">
                          {Math.round((data.success_rate || 0) * 100)}% success
                        </Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Total: {data.total_executions ?? 0} · Success: {data.success_count ?? 0}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Avg Latency: {data.latency_ms?.mean ?? "n/a"} ms
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-muted-foreground">No execution data.</div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Cost Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                {dashboard?.costs ? (
                  Object.entries(dashboard.costs).map(([strategy, data]) => (
                    <div key={strategy} className="rounded border p-2">
                      <div className="font-semibold">{strategy}</div>
                      <div className="text-xs text-muted-foreground">
                        Mean: {data.mean_cost ?? 0} · Max: {data.max_cost ?? 0} · P95: {data.p95_cost ?? 0}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-muted-foreground">No cost data.</div>
                )}
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Profiles</CardTitle>
          <CardDescription>Adaptive MDAP tuning profiles.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Profile</Label>
              <Select
                value={selectedProfile}
                onValueChange={(value) => {
                  setSelectedProfile(value);
                  loadProfileConfig(value);
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select profile" />
                </SelectTrigger>
                <SelectContent>
                  {profiles &&
                    Object.entries(profiles.profiles).map(([profileName, description]) => (
                      <SelectItem key={profileName} value={profileName}>
                        {profileName} - {description}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Profile Config</Label>
              <Textarea
                value={profileConfig ? JSON.stringify(profileConfig, null, 2) : ""}
                readOnly
                className="min-h-[120px]"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Cost Calculator</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="space-y-2">
              <Label>Number of Problems</Label>
              <Input
                value={costInputs.numProblems}
                onChange={(event) =>
                  setCostInputs((prev) => ({ ...prev, numProblems: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label>Model</Label>
              <Select
                value={costInputs.model}
                onValueChange={(value) => setCostInputs((prev) => ({ ...prev, model: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {DEFAULT_MODELS.map((model) => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Workload Distribution (JSON)</Label>
              <Textarea
                value={costInputs.distribution}
                onChange={(event) =>
                  setCostInputs((prev) => ({ ...prev, distribution: event.target.value }))
                }
                placeholder='{"easy":0.3,"medium":0.4,"hard":0.3}'
              />
            </div>
            <Button onClick={calculateCost}>Calculate Cost</Button>
            <Textarea
              value={costResult ? JSON.stringify(costResult, null, 2) : ""}
              readOnly
              className="min-h-[120px]"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Complexity Classifier</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="space-y-2">
              <Label>Description</Label>
              <Textarea
                value={complexityInputs.description}
                onChange={(event) =>
                  setComplexityInputs((prev) => ({ ...prev, description: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label>Domain</Label>
              <Input
                value={complexityInputs.domain}
                onChange={(event) =>
                  setComplexityInputs((prev) => ({ ...prev, domain: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label>Depth</Label>
              <Input
                value={complexityInputs.depth}
                onChange={(event) =>
                  setComplexityInputs((prev) => ({ ...prev, depth: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label>Dependencies (comma separated)</Label>
              <Input
                value={complexityInputs.dependencies}
                onChange={(event) =>
                  setComplexityInputs((prev) => ({ ...prev, dependencies: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label>Constraints (comma separated)</Label>
              <Input
                value={complexityInputs.constraints}
                onChange={(event) =>
                  setComplexityInputs((prev) => ({ ...prev, constraints: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label>Success Criteria (comma separated)</Label>
              <Input
                value={complexityInputs.successCriteria}
                onChange={(event) =>
                  setComplexityInputs((prev) => ({
                    ...prev,
                    successCriteria: event.target.value,
                  }))
                }
              />
            </div>
            <Button onClick={classifyComplexity} disabled={!complexityInputs.description}>
              Classify
            </Button>
            <Textarea
              value={complexityResult ? JSON.stringify(complexityResult, null, 2) : ""}
              readOnly
              className="min-h-[120px]"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Resource Allocator</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="space-y-2">
              <Label>Complexity Score</Label>
              <Input
                value={allocationInputs.score}
                onChange={(event) =>
                  setAllocationInputs((prev) => ({ ...prev, score: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label>Context (JSON)</Label>
              <Textarea
                value={allocationInputs.context}
                onChange={(event) =>
                  setAllocationInputs((prev) => ({ ...prev, context: event.target.value }))
                }
              />
            </div>
            <Button onClick={allocateResources}>Allocate</Button>
            <Textarea
              value={allocationResult ? JSON.stringify(allocationResult, null, 2) : ""}
              readOnly
              className="min-h-[120px]"
            />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Audit Logs</CardTitle>
          <CardDescription>Recent system events and admin operations.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          {auditLogs.length === 0 && (
            <div className="text-muted-foreground">No audit logs.</div>
          )}
          {auditLogs.map((log, index) => (
            <div key={`${log.resource_id ?? "log"}-${index}`} className="rounded border p-2">
              <div className="flex items-center justify-between">
                <div className="font-semibold">{log.operation ?? "Event"}</div>
                <Badge variant={log.success ? "default" : "destructive"}>
                  {log.success ? "success" : "failure"}
                </Badge>
              </div>
              <div className="text-xs text-muted-foreground">
                {log.timestamp ?? "unknown"} · {log.user ?? "system"} · {log.resource ?? "resource"}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
};
