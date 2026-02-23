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
exports.ActivityFeedTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const badge_1 = require("@/components/ui/badge");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const ActivityFeedTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [logs, setLogs] = (0, react_1.useState)([]);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const loadLogs = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const result = await openevolveApi_1.openevolveApi.listAuditLogs(200, apiConfig);
            setLogs(result.logs ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load audit logs.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        loadLogs();
    }, [apiConfig.apiKey]);
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Activity Feed</card_1.CardTitle>
          <card_1.CardDescription>Audit log events across workflows, teams, and gauntlets.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key (admin required)</label>
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
            <button_1.Button variant="outline" onClick={loadLogs} disabled={loading}>
              Refresh Logs
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="space-y-3">
            {logs.length === 0 && (<div className="text-sm text-muted-foreground">No audit events available.</div>)}
            {logs.map((log, index) => (<div key={`${log.resource_id ?? "log"}-${index}`} className="rounded border p-3">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <div className="text-sm font-semibold">{log.operation ?? "Event"}</div>
                    <div className="text-xs text-muted-foreground">
                      {log.timestamp ?? "Unknown time"} · {log.user ?? "system"}
                    </div>
                  </div>
                  <badge_1.Badge variant={log.success ? "default" : "destructive"}>
                    {log.success ? "success" : "failure"}
                  </badge_1.Badge>
                </div>
                <div className="mt-2 text-sm text-muted-foreground">
                  {log.resource ?? "resource"} · {log.resource_id ?? "n/a"}
                </div>
                {log.details ? (<pre className="mt-2 overflow-auto rounded bg-muted p-2 text-xs">
                    {JSON.stringify(log.details, null, 2)}
                  </pre>) : null}
              </div>))}
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.ActivityFeedTab = ActivityFeedTab;
//# sourceMappingURL=ActivityFeedTab.js.map