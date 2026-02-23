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
exports.ResourceManagerTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const separator_1 = require("@/components/ui/separator");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const readApiKey = () => {
    try {
        return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    }
    catch {
        return "";
    }
};
const ResourceManagerTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(readApiKey);
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [workflows, setWorkflows] = (0, react_1.useState)([]);
    const [selectedWorkflowId, setSelectedWorkflowId] = (0, react_1.useState)("");
    const [usage, setUsage] = (0, react_1.useState)(null);
    const [optimization, setOptimization] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const loadWorkflows = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.listWorkflows(apiConfig);
            setWorkflows(response.workflows ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load workflows.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        loadWorkflows();
    }, [apiConfig.apiKey]);
    const loadUsage = async () => {
        if (!selectedWorkflowId)
            return;
        setErrorMessage(null);
        setStatusMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.getWorkflowResourceUsage(selectedWorkflowId, apiConfig);
            setUsage(response.resource_usage ?? null);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load resource usage.");
        }
    };
    const loadOptimization = async () => {
        if (!selectedWorkflowId)
            return;
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.optimizeWorkflowResources(selectedWorkflowId, apiConfig);
            setOptimization(response.suggestions ?? null);
            setStatusMessage("Optimization suggestions generated.");
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to optimize resources.");
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Resource Manager</card_1.CardTitle>
          <card_1.CardDescription>Inspect workflow resource usage and optimization hints.</card_1.CardDescription>
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
                // ignore
            }
        }}/>
            </div>
            <button_1.Button variant="outline" onClick={loadWorkflows} disabled={loading}>
              Refresh
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-[260px_1fr]">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Workflows</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2">
                <select className="w-full rounded border border-input bg-background px-3 py-2 text-sm" value={selectedWorkflowId} onChange={(event) => setSelectedWorkflowId(event.target.value)}>
                  <option value="">Select workflow</option>
                  {workflows.map((workflow) => (<option key={workflow.workflow_id} value={workflow.workflow_id}>
                      {workflow.workflow_id}
                    </option>))}
                </select>
                <button_1.Button className="w-full" variant="secondary" onClick={loadUsage}>
                  Load Usage
                </button_1.Button>
              </card_1.CardContent>
            </card_1.Card>

            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Usage Summary</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-3">
                {!usage && <div className="text-sm text-muted-foreground">No usage data loaded.</div>}
                {usage && (<div className="grid gap-3 md:grid-cols-2 text-sm">
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
                  </div>)}

                {usage?.limits ? (<div className="space-y-2">
                    <div className="text-sm font-semibold">Limits</div>
                    <pre className="rounded border p-2 text-xs whitespace-pre-wrap">
                      {JSON.stringify(usage.limits, null, 2)}
                    </pre>
                  </div>) : null}

                {usage?.component_breakdown ? (<div className="space-y-2">
                    <div className="text-sm font-semibold">Component Breakdown</div>
                    {Object.entries(usage.component_breakdown).map(([component, metrics]) => (<div key={component} className="rounded border p-2 text-xs">
                        <div className="font-semibold">{component}</div>
                        <pre className="whitespace-pre-wrap">{JSON.stringify(metrics, null, 2)}</pre>
                      </div>))}
                  </div>) : null}

                <separator_1.Separator />
                <button_1.Button variant="outline" onClick={loadOptimization}>
                  Generate Optimization Suggestions
                </button_1.Button>
                {optimization && (<pre className="rounded border p-2 text-xs whitespace-pre-wrap">
                    {JSON.stringify(optimization, null, 2)}
                  </pre>)}
              </card_1.CardContent>
            </card_1.Card>
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.ResourceManagerTab = ResourceManagerTab;
//# sourceMappingURL=ResourceManagerTab.js.map