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
exports.SystemMonitoringTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const badge_1 = require("@/components/ui/badge");
const textarea_1 = require("@/components/ui/textarea");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];
const formatValue = (value, suffix = "") => {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "n/a";
    }
    return `${value.toFixed(1)}${suffix}`;
};
const SystemMonitoringTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [dashboard, setDashboard] = (0, react_1.useState)(null);
    const [alerts, setAlerts] = (0, react_1.useState)([]);
    const [services, setServices] = (0, react_1.useState)([]);
    const [logs, setLogs] = (0, react_1.useState)([]);
    const [selectedService, setSelectedService] = (0, react_1.useState)("all");
    const [logLevels, setLogLevels] = (0, react_1.useState)(["INFO", "WARNING", "ERROR"]);
    const [logSource, setLogSource] = (0, react_1.useState)("");
    const [logLimit, setLogLimit] = (0, react_1.useState)(200);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const refresh = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const [dashboardRes, alertsRes, servicesRes, logsRes] = await Promise.all([
                openevolveApi_1.openevolveApi.getMonitoringDashboard(apiConfig),
                openevolveApi_1.openevolveApi.getMonitoringAlerts(apiConfig),
                openevolveApi_1.openevolveApi.getMonitoringServices(apiConfig),
                openevolveApi_1.openevolveApi.getMonitoringLogs(logLimit, logSource || undefined, apiConfig),
            ]);
            setDashboard(dashboardRes);
            setAlerts(alertsRes.alerts || []);
            setServices(servicesRes.services || []);
            setLogs(logsRes.entries || []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load monitoring data.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        refresh();
    }, [apiConfig.apiKey]);
    const filteredServices = selectedService === "all"
        ? services
        : services.filter((service) => service.name === selectedService);
    const filteredLogs = logs.filter((entry) => logLevels.length
        ? logLevels.some((level) => entry.line.includes(level))
        : true);
    const servicesUp = services.filter((service) => service.healthy).length;
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>System Monitoring</card_1.CardTitle>
          <card_1.CardDescription>Service health checks, logs, and alerting signals.</card_1.CardDescription>
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
              Refresh Dashboard
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-4">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Services Up</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                <div>
                  {servicesUp} / {services.length}
                </div>
                <div className="text-xs text-muted-foreground">healthy checks</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">CPU Usage</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                <div>{formatValue(dashboard?.system?.system?.cpu_percent, "%")}</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Memory Usage</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                <div>{formatValue(dashboard?.system?.system?.memory_percent, "%")}</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Active Alerts</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-1 text-sm">
                <div>{alerts.length}</div>
              </card_1.CardContent>
            </card_1.Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Service Status</card_1.CardTitle>
                <card_1.CardDescription>Health check output per subsystem.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-3 text-sm">
                <div className="flex flex-col gap-2 md:flex-row md:items-center">
                  <div className="text-sm font-medium">Filter Service</div>
                  <select_1.Select value={selectedService} onValueChange={setSelectedService}>
                    <select_1.SelectTrigger className="w-[220px]">
                      <select_1.SelectValue placeholder="All services"/>
                    </select_1.SelectTrigger>
                    <select_1.SelectContent>
                      <select_1.SelectItem value="all">All</select_1.SelectItem>
                      {services.map((service) => (<select_1.SelectItem key={service.name} value={service.name}>
                          {service.name}
                        </select_1.SelectItem>))}
                    </select_1.SelectContent>
                  </select_1.Select>
                </div>
                {filteredServices.length === 0 ? (<div className="text-muted-foreground">No service checks available.</div>) : (filteredServices.map((service) => (<div key={service.name} className="rounded border p-2">
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">{service.name}</div>
                        <badge_1.Badge variant={service.healthy ? "default" : "secondary"}>
                          {service.status ?? "unknown"}
                        </badge_1.Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Exec: {formatValue(service.execution_time, "s")} ·{" "}
                        {service.timestamp ?? "n/a"}
                      </div>
                      {service.error ? (<div className="text-xs text-red-500">{service.error}</div>) : null}
                    </div>)))}
              </card_1.CardContent>
            </card_1.Card>

            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Alerts</card_1.CardTitle>
                <card_1.CardDescription>Triggered alert rules from monitoring.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {alerts.length === 0 ? (<div className="text-muted-foreground">No active alerts.</div>) : (alerts.map((alert, index) => (<div key={`alert-${index}`} className="rounded border p-2">
                      <div className="font-semibold">{alert.name ?? "alert"}</div>
                      <div className="text-xs text-muted-foreground">
                        {alert.description ?? alert.metric_name ?? "threshold triggered"}
                      </div>
                    </div>)))}
              </card_1.CardContent>
            </card_1.Card>
          </div>

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Logs</card_1.CardTitle>
              <card_1.CardDescription>Tail logs from known sources.</card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3 text-sm">
              <div className="grid gap-3 md:grid-cols-3">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Log Source</label>
                  <input_1.Input value={logSource} onChange={(event) => setLogSource(event.target.value)} placeholder="backend_stdout.log"/>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Log Limit</label>
                  <input_1.Input type="number" value={logLimit} onChange={(event) => setLogLimit(Number(event.target.value) || 200)}/>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Filter Levels</label>
                  <div className="flex flex-wrap gap-2">
                    {LOG_LEVELS.map((level) => (<button_1.Button key={level} size="sm" variant={logLevels.includes(level) ? "default" : "outline"} onClick={() => {
                setLogLevels((prev) => prev.includes(level)
                    ? prev.filter((value) => value !== level)
                    : [...prev, level]);
            }}>
                        {level}
                      </button_1.Button>))}
                  </div>
                </div>
              </div>
              <button_1.Button variant="outline" onClick={refresh}>
                Reload Logs
              </button_1.Button>
              <textarea_1.Textarea className="min-h-[240px] text-xs" readOnly value={filteredLogs.map((entry) => `[${entry.source}] ${entry.line}`).join("\n")}/>
            </card_1.CardContent>
          </card_1.Card>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.SystemMonitoringTab = SystemMonitoringTab;
//# sourceMappingURL=SystemMonitoringTab.js.map