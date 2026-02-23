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
exports.MonitoringTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4",
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
    "gemini-1-5-pro",
    "gemini-1-5-flash",
];
const parseJson = (value) => {
    if (!value.trim()) {
        return undefined;
    }
    return JSON.parse(value);
};
const MonitoringTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [health, setHealth] = (0, react_1.useState)(null);
    const [dashboard, setDashboard] = (0, react_1.useState)(null);
    const [profiles, setProfiles] = (0, react_1.useState)(null);
    const [profileConfig, setProfileConfig] = (0, react_1.useState)(null);
    const [selectedProfile, setSelectedProfile] = (0, react_1.useState)("");
    const [auditLogs, setAuditLogs] = (0, react_1.useState)([]);
    const [systemDashboard, setSystemDashboard] = (0, react_1.useState)(null);
    const [monitoringAlerts, setMonitoringAlerts] = (0, react_1.useState)([]);
    const [metricName, setMetricName] = (0, react_1.useState)("system_cpu_percent");
    const [metricSamples, setMetricSamples] = (0, react_1.useState)([]);
    const [monitoringWsStatus, setMonitoringWsStatus] = (0, react_1.useState)("disconnected");
    const [monitoringUpdates, setMonitoringUpdates] = (0, react_1.useState)([]);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [costInputs, setCostInputs] = (0, react_1.useState)({
        numProblems: "100",
        model: DEFAULT_MODELS[0],
        distribution: "",
    });
    const [costResult, setCostResult] = (0, react_1.useState)(null);
    const [complexityInputs, setComplexityInputs] = (0, react_1.useState)({
        description: "",
        domain: "",
        depth: "1",
        dependencies: "",
        constraints: "",
        successCriteria: "",
    });
    const [complexityResult, setComplexityResult] = (0, react_1.useState)(null);
    const [allocationInputs, setAllocationInputs] = (0, react_1.useState)({
        score: "0.5",
        context: "",
    });
    const [allocationResult, setAllocationResult] = (0, react_1.useState)(null);
    const refresh = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const [healthRes, dashboardRes, profilesRes, logsRes, monitoringDashboardRes, alertsRes] = await Promise.all([
                openevolveApi_1.openevolveApi.getAdaptiveMdapHealth(apiConfig),
                openevolveApi_1.openevolveApi.getAdaptiveMdapDashboard(apiConfig),
                openevolveApi_1.openevolveApi.getAdaptiveMdapProfiles(apiConfig),
                openevolveApi_1.openevolveApi.listAuditLogs(100, apiConfig),
                openevolveApi_1.openevolveApi.getMonitoringDashboard(apiConfig),
                openevolveApi_1.openevolveApi.getMonitoringAlerts(apiConfig),
            ]);
            setHealth(healthRes);
            setDashboard(dashboardRes);
            setProfiles(profilesRes);
            setSelectedProfile(profilesRes.default);
            setAuditLogs(logsRes.logs || []);
            setSystemDashboard(monitoringDashboardRes);
            setMonitoringAlerts(alertsRes.alerts || []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load monitoring data.");
        }
        finally {
            setLoading(false);
        }
    };
    const loadMetricSamples = async () => {
        setErrorMessage(null);
        try {
            const result = await openevolveApi_1.openevolveApi.getMonitoringMetrics({ name: metricName }, apiConfig);
            setMetricSamples(result.metrics || []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load monitoring metrics.");
        }
    };
    const connectMonitoringWebsocket = () => {
        try {
            const baseUrl = globalThis?.OPENEVOLVE_API_BASE;
            const apiBase = baseUrl ||
                (() => {
                    try {
                        return globalThis.localStorage?.getItem("openevolve_api_base") || "";
                    }
                    catch {
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
                }
                catch {
                    // ignore malformed messages
                }
            };
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to connect websocket.");
        }
    };
    const loadProfileConfig = async (profileName) => {
        try {
            const config = await openevolveApi_1.openevolveApi.getAdaptiveMdapProfileConfig(profileName, apiConfig);
            setProfileConfig(config);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load profile config.");
        }
    };
    const calculateCost = async () => {
        setErrorMessage(null);
        try {
            const payload = {
                num_problems: Number(costInputs.numProblems),
                model: costInputs.model,
            };
            const distribution = parseJson(costInputs.distribution || "");
            if (distribution) {
                payload.workload_distribution = distribution;
            }
            const result = await openevolveApi_1.openevolveApi.calculateAdaptiveMdapCost(payload, apiConfig);
            setCostResult(result);
        }
        catch (error) {
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
            const result = await openevolveApi_1.openevolveApi.classifyAdaptiveMdapComplexity(payload, apiConfig);
            setComplexityResult(result);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to classify complexity.");
        }
    };
    const allocateResources = async () => {
        setErrorMessage(null);
        try {
            const payload = {
                complexity_score: Number(allocationInputs.score),
            };
            const context = parseJson(allocationInputs.context || "");
            if (context) {
                payload.context = context;
            }
            const result = await openevolveApi_1.openevolveApi.allocateAdaptiveMdapResources(payload, apiConfig);
            setAllocationResult(result);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to allocate resources.");
        }
    };
    (0, react_1.useEffect)(() => {
        refresh();
    }, [apiConfig.apiKey]);
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Adaptive MDAP Monitoring</card_1.CardTitle>
          <card_1.CardDescription>Health checks, allocation metrics, and cost controls.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label_1.Label>API Key</label_1.Label>
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
              Refresh Metrics
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-3">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Health</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                <div>
                  Status:{" "}
                  <badge_1.Badge variant={health?.status === "healthy" ? "default" : "secondary"}>
                    {String(health?.status ?? "unknown")}
                  </badge_1.Badge>
                </div>
                <div>Version: {String(health?.version ?? "n/a")}</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Summary</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                <div>Total Classifications: {dashboard?.summary?.total_classifications ?? 0}</div>
                <div>Successful: {dashboard?.summary?.successful_classifications ?? 0}</div>
                <div>Failed: {dashboard?.summary?.failed_classifications ?? 0}</div>
                <div>Total Allocations: {dashboard?.summary?.total_allocations ?? 0}</div>
                <div>Total Executions: {dashboard?.summary?.total_executions ?? 0}</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Allocations</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                {dashboard?.allocations ? (Object.entries(dashboard.allocations).map(([key, value]) => (<div key={key}>
                      {key}: {value}
                    </div>))) : (<div className="text-muted-foreground">No allocation data.</div>)}
              </card_1.CardContent>
            </card_1.Card>
          </div>

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">System Monitoring</card_1.CardTitle>
              <card_1.CardDescription>System health, resource usage, and alerts.</card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3 text-sm">
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
                {systemDashboard?.workflow ? (Object.entries(systemDashboard.workflow).map(([key, value]) => (<div key={key} className="text-xs text-muted-foreground">
                      {key}: {(value * 100).toFixed(1)}%
                    </div>))) : (<div className="text-muted-foreground">No workflow metrics.</div>)}
              </div>

              <div className="rounded border p-2">
                <div className="font-semibold">Active Alerts</div>
                {monitoringAlerts.length === 0 ? (<div className="text-muted-foreground">No active alerts.</div>) : (monitoringAlerts.map((alert, index) => (<div key={`alert-${index}`} className="text-xs text-muted-foreground">
                      {alert.name ?? "alert"} · {alert.description ?? "threshold breached"}
                    </div>)))}
              </div>
            </card_1.CardContent>
          </card_1.Card>

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Monitoring Metrics Stream</card_1.CardTitle>
              <card_1.CardDescription>Fetch metrics or connect to the websocket feed.</card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3 text-sm">
              <div className="flex flex-col gap-3 md:flex-row md:items-center">
                <div className="flex-1">
                  <label_1.Label>Metric Name</label_1.Label>
                  <input_1.Input value={metricName} onChange={(event) => setMetricName(event.target.value)}/>
                </div>
                <button_1.Button variant="outline" onClick={loadMetricSamples}>
                  Load Metrics
                </button_1.Button>
                <button_1.Button variant="outline" onClick={connectMonitoringWebsocket}>
                  Connect WebSocket
                </button_1.Button>
              </div>
              <div className="text-xs text-muted-foreground">
                WebSocket Status: {monitoringWsStatus}
              </div>
              <div className="grid gap-2 md:grid-cols-2">
                <div className="rounded border p-2">
                  <div className="font-semibold">Recent Metric Samples</div>
                  {metricSamples.length === 0 ? (<div className="text-muted-foreground">No samples loaded.</div>) : (metricSamples.slice(0, 6).map((metric, index) => (<div key={`metric-${index}`} className="text-xs text-muted-foreground">
                        {metric.timestamp ?? "time"} · {metric.value ?? "n/a"}
                      </div>)))}
                </div>
                <div className="rounded border p-2">
                  <div className="font-semibold">Live Updates</div>
                  {monitoringUpdates.length === 0 ? (<div className="text-muted-foreground">No live updates yet.</div>) : (monitoringUpdates.slice(0, 6).map((update, index) => (<div key={`update-${index}`} className="text-xs text-muted-foreground">
                        {JSON.stringify(update)}
                      </div>)))}
                </div>
              </div>
            </card_1.CardContent>
          </card_1.Card>

          <div className="grid gap-4 md:grid-cols-2">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Execution Strategies</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {dashboard?.execution ? (Object.entries(dashboard.execution).map(([strategy, data]) => (<div key={strategy} className="rounded border p-2">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">{strategy}</div>
                        <badge_1.Badge variant="secondary">
                          {Math.round((data.success_rate || 0) * 100)}% success
                        </badge_1.Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Total: {data.total_executions ?? 0} · Success: {data.success_count ?? 0}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Avg Latency: {data.latency_ms?.mean ?? "n/a"} ms
                      </div>
                    </div>))) : (<div className="text-muted-foreground">No execution data.</div>)}
              </card_1.CardContent>
            </card_1.Card>

            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Cost Metrics</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {dashboard?.costs ? (Object.entries(dashboard.costs).map(([strategy, data]) => (<div key={strategy} className="rounded border p-2">
                      <div className="font-semibold">{strategy}</div>
                      <div className="text-xs text-muted-foreground">
                        Mean: {data.mean_cost ?? 0} · Max: {data.max_cost ?? 0} · P95: {data.p95_cost ?? 0}
                      </div>
                    </div>))) : (<div className="text-muted-foreground">No cost data.</div>)}
              </card_1.CardContent>
            </card_1.Card>
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Profiles</card_1.CardTitle>
          <card_1.CardDescription>Adaptive MDAP tuning profiles.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>Profile</label_1.Label>
              <select_1.Select value={selectedProfile} onValueChange={(value) => {
            setSelectedProfile(value);
            loadProfileConfig(value);
        }}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue placeholder="Select profile"/>
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  {profiles &&
            Object.entries(profiles.profiles).map(([profileName, description]) => (<select_1.SelectItem key={profileName} value={profileName}>
                        {profileName} - {description}
                      </select_1.SelectItem>))}
                </select_1.SelectContent>
              </select_1.Select>
            </div>
            <div className="space-y-2">
              <label_1.Label>Profile Config</label_1.Label>
              <textarea_1.Textarea value={profileConfig ? JSON.stringify(profileConfig, null, 2) : ""} readOnly className="min-h-[120px]"/>
            </div>
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <div className="grid gap-4 lg:grid-cols-3">
        <card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="text-sm">Cost Calculator</card_1.CardTitle>
          </card_1.CardHeader>
          <card_1.CardContent className="space-y-3 text-sm">
            <div className="space-y-2">
              <label_1.Label>Number of Problems</label_1.Label>
              <input_1.Input value={costInputs.numProblems} onChange={(event) => setCostInputs((prev) => ({ ...prev, numProblems: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Model</label_1.Label>
              <select_1.Select value={costInputs.model} onValueChange={(value) => setCostInputs((prev) => ({ ...prev, model: value }))}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue />
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  {DEFAULT_MODELS.map((model) => (<select_1.SelectItem key={model} value={model}>
                      {model}
                    </select_1.SelectItem>))}
                </select_1.SelectContent>
              </select_1.Select>
            </div>
            <div className="space-y-2">
              <label_1.Label>Workload Distribution (JSON)</label_1.Label>
              <textarea_1.Textarea value={costInputs.distribution} onChange={(event) => setCostInputs((prev) => ({ ...prev, distribution: event.target.value }))} placeholder='{"easy":0.3,"medium":0.4,"hard":0.3}'/>
            </div>
            <button_1.Button onClick={calculateCost}>Calculate Cost</button_1.Button>
            <textarea_1.Textarea value={costResult ? JSON.stringify(costResult, null, 2) : ""} readOnly className="min-h-[120px]"/>
          </card_1.CardContent>
        </card_1.Card>

        <card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="text-sm">Complexity Classifier</card_1.CardTitle>
          </card_1.CardHeader>
          <card_1.CardContent className="space-y-3 text-sm">
            <div className="space-y-2">
              <label_1.Label>Description</label_1.Label>
              <textarea_1.Textarea value={complexityInputs.description} onChange={(event) => setComplexityInputs((prev) => ({ ...prev, description: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Domain</label_1.Label>
              <input_1.Input value={complexityInputs.domain} onChange={(event) => setComplexityInputs((prev) => ({ ...prev, domain: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Depth</label_1.Label>
              <input_1.Input value={complexityInputs.depth} onChange={(event) => setComplexityInputs((prev) => ({ ...prev, depth: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Dependencies (comma separated)</label_1.Label>
              <input_1.Input value={complexityInputs.dependencies} onChange={(event) => setComplexityInputs((prev) => ({ ...prev, dependencies: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Constraints (comma separated)</label_1.Label>
              <input_1.Input value={complexityInputs.constraints} onChange={(event) => setComplexityInputs((prev) => ({ ...prev, constraints: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Success Criteria (comma separated)</label_1.Label>
              <input_1.Input value={complexityInputs.successCriteria} onChange={(event) => setComplexityInputs((prev) => ({
            ...prev,
            successCriteria: event.target.value,
        }))}/>
            </div>
            <button_1.Button onClick={classifyComplexity} disabled={!complexityInputs.description}>
              Classify
            </button_1.Button>
            <textarea_1.Textarea value={complexityResult ? JSON.stringify(complexityResult, null, 2) : ""} readOnly className="min-h-[120px]"/>
          </card_1.CardContent>
        </card_1.Card>

        <card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="text-sm">Resource Allocator</card_1.CardTitle>
          </card_1.CardHeader>
          <card_1.CardContent className="space-y-3 text-sm">
            <div className="space-y-2">
              <label_1.Label>Complexity Score</label_1.Label>
              <input_1.Input value={allocationInputs.score} onChange={(event) => setAllocationInputs((prev) => ({ ...prev, score: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Context (JSON)</label_1.Label>
              <textarea_1.Textarea value={allocationInputs.context} onChange={(event) => setAllocationInputs((prev) => ({ ...prev, context: event.target.value }))}/>
            </div>
            <button_1.Button onClick={allocateResources}>Allocate</button_1.Button>
            <textarea_1.Textarea value={allocationResult ? JSON.stringify(allocationResult, null, 2) : ""} readOnly className="min-h-[120px]"/>
          </card_1.CardContent>
        </card_1.Card>
      </div>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Audit Logs</card_1.CardTitle>
          <card_1.CardDescription>Recent system events and admin operations.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-2 text-sm">
          {auditLogs.length === 0 && (<div className="text-muted-foreground">No audit logs.</div>)}
          {auditLogs.map((log, index) => (<div key={`${log.resource_id ?? "log"}-${index}`} className="rounded border p-2">
              <div className="flex items-center justify-between">
                <div className="font-semibold">{log.operation ?? "Event"}</div>
                <badge_1.Badge variant={log.success ? "default" : "destructive"}>
                  {log.success ? "success" : "failure"}
                </badge_1.Badge>
              </div>
              <div className="text-xs text-muted-foreground">
                {log.timestamp ?? "unknown"} · {log.user ?? "system"} · {log.resource ?? "resource"}
              </div>
            </div>))}
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.MonitoringTab = MonitoringTab;
//# sourceMappingURL=MonitoringTab.js.map