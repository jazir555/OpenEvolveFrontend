"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.AnalyticsMonitoringTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const badge_1 = require("@/components/ui/badge");
const tabs_1 = require("@/components/ui/tabs");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const average = (values) => {
    const filtered = values.filter((value) => typeof value === "number");
    if (!filtered.length)
        return null;
    return filtered.reduce((acc, value) => acc + value, 0) / filtered.length;
};
const formatNumber = (value, decimals = 2) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "n/a";
    }
    return value.toFixed(decimals);
};
const AnalyticsMonitoringTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [statistics, setStatistics] = (0, react_1.useState)(null);
    const [workflowMetrics, setWorkflowMetrics] = (0, react_1.useState)([]);
    const [monitoringDashboard, setMonitoringDashboard] = (0, react_1.useState)(null);
    const [monitoringAlerts, setMonitoringAlerts] = (0, react_1.useState)([]);
    const [monitoringServices, setMonitoringServices] = (0, react_1.useState)([]);
    const [workflowSummaries, setWorkflowSummaries] = (0, react_1.useState)([]);
    const [autoRefresh, setAutoRefresh] = (0, react_1.useState)(true);
    const [refreshInterval, setRefreshInterval] = (0, react_1.useState)(15);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const refresh = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const [stats, metrics, monitoring, alerts, services, workflows,] = await Promise.all([
                openevolveApi_1.openevolveApi.getStatistics(apiConfig),
                openevolveApi_1.openevolveApi.getWorkflowMetrics(apiConfig),
                openevolveApi_1.openevolveApi.getMonitoringDashboard(apiConfig),
                openevolveApi_1.openevolveApi.getMonitoringAlerts(apiConfig),
                openevolveApi_1.openevolveApi.getMonitoringServices(apiConfig),
                openevolveApi_1.openevolveApi.listWorkflows(apiConfig),
            ]);
            setStatistics(stats);
            setWorkflowMetrics(metrics.metrics || []);
            setMonitoringDashboard(monitoring);
            setMonitoringAlerts(alerts.alerts || []);
            setMonitoringServices(services.services || []);
            setWorkflowSummaries(workflows.workflows || []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load analytics monitoring data.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        refresh();
    }, [apiConfig.apiKey]);
    (0, react_1.useEffect)(() => {
        if (!autoRefresh)
            return;
        const interval = globalThis.setInterval(() => {
            refresh();
        }, Math.max(5, refreshInterval) * 1000);
        return () => globalThis.clearInterval(interval);
    }, [autoRefresh, refreshInterval, apiConfig.apiKey]);
    const avgBestFitness = average(workflowMetrics.map((metric) => metric.best_fitness));
    const avgDiversity = average(workflowMetrics.map((metric) => metric.diversity));
    const avgExecution = average(workflowMetrics.map((metric) => metric.execution_time));
    const totalTokens = workflowMetrics.reduce((acc, metric) => acc + (metric.tokens_used ?? 0), 0);
    const statusCounts = workflowSummaries.reduce((acc, workflow) => {
        const status = workflow.status ?? "unknown";
        acc[status] = (acc[status] ?? 0) + 1;
        return acc;
    }, {});
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Analytics & Monitoring</card_1.CardTitle>
          <card_1.CardDescription>Performance analytics with live monitoring signals.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key</label>
              <input_1.Input value={apiKey} type="password" onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            try {
                globalThis.localStorage?.setItem("openevolve_api_key", value);
            }
            catch {
                // ignore storage errors
            }
        }}/>
            </div>
            <button_1.Button variant="outline" onClick={refresh} disabled={loading}>
              Refresh
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-4">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Workflows</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                <div>Total: {statistics?.total_workflows ?? 0}</div>
                <div>Running: {statistics?.running ?? 0}</div>
                <div>Completed: {statistics?.completed ?? 0}</div>
                <div>Failed: {statistics?.failed ?? 0}</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Fitness / Diversity</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                <div>Avg Best Fitness: {formatNumber(avgBestFitness, 3)}</div>
                <div>Avg Diversity: {formatNumber(avgDiversity, 3)}</div>
                <div>Avg Execution: {formatNumber(avgExecution, 1)}s</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Tokens</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                <div>Total Tokens: {totalTokens.toLocaleString()}</div>
                <div>Samples: {workflowMetrics.length}</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Monitoring</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                <div>
                  Health:{" "}
                  <badge_1.Badge variant={monitoringDashboard?.health?.status === "healthy" ? "default" : "secondary"}>
                    {monitoringDashboard?.health?.status ?? "unknown"}
                  </badge_1.Badge>
                </div>
                <div>Alerts: {monitoringAlerts.length}</div>
                <div>Services: {monitoringServices.length}</div>
              </card_1.CardContent>
            </card_1.Card>
          </div>

          <tabs_1.Tabs defaultValue="performance" className="w-full">
            <tabs_1.TabsList className="flex flex-wrap gap-2">
              <tabs_1.TabsTrigger value="performance">Performance</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="system">System Monitoring</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="workflows">Workflow Analytics</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="reporting">Advanced Reporting</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="controls">Controls</tabs_1.TabsTrigger>
            </tabs_1.TabsList>

            <tabs_1.TabsContent value="performance" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Workflow Performance Snapshot</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2 text-sm">
                    {workflowMetrics.length === 0 ? (<div className="text-muted-foreground">No workflow metrics yet.</div>) : (workflowMetrics.slice(0, 8).map((metric) => (<div key={metric.workflow_id} className="rounded border p-2">
                          <div className="flex items-center justify-between">
                            <div className="font-semibold">{metric.workflow_id}</div>
                            <badge_1.Badge variant="secondary">{metric.status ?? "unknown"}</badge_1.Badge>
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
                            <div className="h-2 rounded bg-blue-500" style={{ width: `${Math.round((metric.progress ?? 0) * 100)}%` }}/>
                          </div>
                        </div>)))}
                  </card_1.CardContent>
                </card_1.Card>
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Resource Utilization</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2 text-sm">
                    {workflowMetrics.length === 0 ? (<div className="text-muted-foreground">No resource metrics available.</div>) : (workflowMetrics.slice(0, 8).map((metric) => (<div key={`resource-${metric.workflow_id}`} className="rounded border p-2">
                          <div className="font-semibold">{metric.workflow_id}</div>
                          <div className="text-xs text-muted-foreground">
                            Memory: {formatNumber(metric.memory_usage, 1)} MB · CPU:{" "}
                            {formatNumber(metric.cpu_usage, 2)}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Population: {metric.population_size ?? "n/a"} · Generation:{" "}
                            {metric.generation ?? "n/a"}
                          </div>
                        </div>)))}
                  </card_1.CardContent>
                </card_1.Card>
              </div>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="system" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">System Metrics</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2 text-sm">
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
                      <badge_1.Badge variant={monitoringDashboard?.health?.status === "healthy" ? "default" : "secondary"}>
                        {monitoringDashboard?.health?.status ?? "unknown"}
                      </badge_1.Badge>
                    </div>
                  </card_1.CardContent>
                </card_1.Card>
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Service Health Checks</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2 text-sm">
                    {monitoringServices.length === 0 ? (<div className="text-muted-foreground">No service checks found.</div>) : (monitoringServices.map((service) => (<div key={service.name} className="rounded border p-2">
                          <div className="flex items-center justify-between">
                            <div className="font-semibold">{service.name}</div>
                            <badge_1.Badge variant={service.healthy ? "default" : "secondary"}>
                              {service.status ?? "unknown"}
                            </badge_1.Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Exec: {formatNumber(service.execution_time, 3)}s ·{" "}
                            {service.timestamp ?? "n/a"}
                          </div>
                          {service.error ? (<div className="text-xs text-red-500">{service.error}</div>) : null}
                        </div>)))}
                  </card_1.CardContent>
                </card_1.Card>
              </div>
              <card_1.Card className="mt-4">
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Active Alerts</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  {monitoringAlerts.length === 0 ? (<div className="text-muted-foreground">No active alerts.</div>) : (monitoringAlerts.map((alert, index) => (<div key={`alert-${index}`} className="rounded border p-2">
                        <div className="font-semibold">{alert.name ?? "alert"}</div>
                        <div className="text-xs text-muted-foreground">
                          {alert.description ?? alert.metric_name ?? "threshold triggered"}
                        </div>
                      </div>)))}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="workflows" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Status Distribution</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2 text-sm">
                    {Object.keys(statusCounts).length === 0 ? (<div className="text-muted-foreground">No workflow data.</div>) : (Object.entries(statusCounts).map(([status, count]) => (<div key={status} className="flex items-center justify-between">
                          <span>{status}</span>
                          <badge_1.Badge variant="secondary">{count}</badge_1.Badge>
                        </div>)))}
                  </card_1.CardContent>
                </card_1.Card>
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Current Progress</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2 text-sm">
                    {workflowSummaries.length === 0 ? (<div className="text-muted-foreground">No workflows available.</div>) : (workflowSummaries.slice(0, 10).map((workflow) => (<div key={workflow.workflow_id} className="rounded border p-2">
                          <div className="flex items-center justify-between">
                            <div className="font-semibold">{workflow.workflow_id}</div>
                            <badge_1.Badge variant="secondary">{workflow.status}</badge_1.Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Stage: {workflow.current_stage ?? "n/a"}
                          </div>
                          <div className="mt-2 h-2 w-full rounded bg-muted">
                            <div className="h-2 rounded bg-emerald-500" style={{ width: `${Math.round((workflow.progress ?? 0) * 100)}%` }}/>
                          </div>
                        </div>)))}
                  </card_1.CardContent>
                </card_1.Card>
              </div>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="reporting" className="mt-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Generated Summary</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3 text-sm">
                  <div>
                    Success Rate:{" "}
                    {statistics
            ? `${((statistics.completed / Math.max(statistics.total_workflows, 1)) * 100).toFixed(1)}%`
            : "n/a"}
                  </div>
                  <div>Average Execution: {formatNumber(avgExecution, 1)}s</div>
                  <div>Average Best Fitness: {formatNumber(avgBestFitness, 3)}</div>
                  <div>Total Tokens Used: {totalTokens.toLocaleString()}</div>
                  <button_1.Button variant="outline" onClick={() => {
            const payload = {
                generated_at: new Date().toISOString(),
                statistics,
                averages: { avgBestFitness, avgDiversity, avgExecution },
                totalTokens,
            };
            globalThis.navigator?.clipboard?.writeText(JSON.stringify(payload, null, 2));
        }}>
                    Copy Report JSON
                  </button_1.Button>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="controls" className="mt-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Dashboard Controls</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3 text-sm">
                  <div className="flex items-center justify-between">
                    <span>Auto refresh</span>
                    <button_1.Button variant={autoRefresh ? "default" : "outline"} onClick={() => setAutoRefresh((prev) => !prev)}>
                      {autoRefresh ? "Enabled" : "Disabled"}
                    </button_1.Button>
                  </div>
                  <div className="space-y-1">
                    <label className="text-sm font-medium">Refresh Interval (seconds)</label>
                    <input_1.Input type="number" value={refreshInterval} onChange={(event) => setRefreshInterval(Number(event.target.value) || 10)}/>
                  </div>
                  <button_1.Button variant="outline" onClick={refresh}>
                    Refresh Now
                  </button_1.Button>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>
          </tabs_1.Tabs>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.AnalyticsMonitoringTab = AnalyticsMonitoringTab;
//# sourceMappingURL=AnalyticsMonitoringTab.js.map