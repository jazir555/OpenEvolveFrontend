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
exports.OpenEvolveDashboardTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const badge_1 = require("@/components/ui/badge");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const readStorage = (key, fallback) => {
    try {
        const raw = globalThis.localStorage?.getItem(key);
        if (!raw) {
            return fallback;
        }
        return JSON.parse(raw);
    }
    catch {
        return fallback;
    }
};
const OpenEvolveDashboardTab = () => {
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
    const [workflows, setWorkflows] = (0, react_1.useState)([]);
    const [health, setHealth] = (0, react_1.useState)(null);
    const [mdapHealth, setMdapHealth] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const refresh = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const [stats, workflowList, healthStatus] = await Promise.all([
                openevolveApi_1.openevolveApi.getStatistics(apiConfig),
                openevolveApi_1.openevolveApi.listWorkflows(apiConfig),
                openevolveApi_1.openevolveApi.getHealth(apiConfig),
            ]);
            setStatistics(stats);
            setWorkflows(workflowList.workflows || []);
            setHealth(healthStatus);
            try {
                const mdap = await openevolveApi_1.openevolveApi.getAdaptiveMdapHealth(apiConfig);
                setMdapHealth(mdap);
            }
            catch {
                setMdapHealth(null);
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load dashboard data.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        refresh();
    }, [apiConfig.apiKey]);
    const recentState = readStorage("openevolve-state", null);
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>OpenEvolve Dashboard</card_1.CardTitle>
          <card_1.CardDescription>System status and workflow overview.</card_1.CardDescription>
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

          <div className="grid gap-4 md:grid-cols-3">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">API Health</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                <div>
                  Status:{" "}
                  <badge_1.Badge variant={health?.status === "ok" ? "default" : "secondary"}>
                    {String(health?.status ?? "unknown")}
                  </badge_1.Badge>
                </div>
                <div>Version: {String(health?.version ?? "n/a")}</div>
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Adaptive MDAP</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                <div>
                  Status:{" "}
                  <badge_1.Badge variant={mdapHealth?.status === "healthy" ? "default" : "secondary"}>
                    {String(mdapHealth?.status ?? "unavailable")}
                  </badge_1.Badge>
                </div>
                <div>Details: {mdapHealth ? "Available" : "Not configured"}</div>
              </card_1.CardContent>
            </card_1.Card>
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
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Active Workflows</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {workflows.length === 0 && (<div className="text-muted-foreground">No workflows found.</div>)}
                {workflows.map((workflow) => (<div key={workflow.workflow_id} className="rounded border p-2 space-y-1">
                    <div className="flex items-center justify-between">
                      <div className="font-semibold">{workflow.workflow_id}</div>
                      <badge_1.Badge variant="secondary">{workflow.status}</badge_1.Badge>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Stage: {workflow.current_stage}
                    </div>
                    <div className="h-2 w-full rounded bg-muted">
                      <div className="h-2 rounded bg-blue-500" style={{ width: `${Math.round(workflow.progress * 100)}%` }}/>
                    </div>
                  </div>))}
              </card_1.CardContent>
            </card_1.Card>

            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Recent Evolution Snapshot</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
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
              </card_1.CardContent>
            </card_1.Card>
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.OpenEvolveDashboardTab = OpenEvolveDashboardTab;
//# sourceMappingURL=OpenEvolveDashboardTab.js.map