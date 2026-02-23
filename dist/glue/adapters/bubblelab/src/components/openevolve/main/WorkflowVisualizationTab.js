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
exports.WorkflowVisualizationTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const badge_1 = require("@/components/ui/badge");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const formatNumber = (value, decimals = 2) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "n/a";
    }
    return value.toFixed(decimals);
};
const getStagesForType = (workflowType) => {
    if (!workflowType) {
        return ["Input", "Processing", "Output"];
    }
    const lower = workflowType.toLowerCase();
    if (lower.includes("evolution")) {
        return ["Input", "Analysis", "Evolution", "Evaluation", "Output"];
    }
    if (lower.includes("adversarial")) {
        return ["Input", "Red Team", "Blue Team", "Evaluator", "Output"];
    }
    if (lower.includes("sovereign") || lower.includes("decomposition")) {
        return ["Input", "Analysis", "Decomposition", "Solving", "Assembly", "Verification", "Output"];
    }
    return ["Input", "Processing", "Output"];
};
const extractHistorySeries = (metrics) => {
    if (!metrics)
        return null;
    const history = metrics.history ||
        metrics.evolution_history ||
        metrics.fitness_history ||
        metrics.generations;
    if (!history)
        return null;
    if (Array.isArray(history)) {
        if (history.length === 0)
            return null;
        if (typeof history[0] === "number") {
            return {
                generations: history.map((_, index) => index + 1),
                best: history,
                avg: [],
                diversity: metrics.diversity_history || [],
            };
        }
        return {
            generations: history.map((entry, index) => entry.generation ?? index + 1),
            best: history.map((entry) => entry.best_fitness ?? entry.best_score ?? null).filter((v) => v != null),
            avg: history.map((entry) => entry.avg_fitness ?? entry.average_fitness ?? null).filter((v) => v != null),
            diversity: history
                .map((entry) => entry.diversity ?? entry.diversity_score ?? null)
                .filter((v) => v != null),
        };
    }
    return null;
};
const WorkflowVisualizationTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [workflows, setWorkflows] = (0, react_1.useState)([]);
    const [selectedWorkflowId, setSelectedWorkflowId] = (0, react_1.useState)("");
    const [workflowDetail, setWorkflowDetail] = (0, react_1.useState)(null);
    const [workflowTelemetry, setWorkflowTelemetry] = (0, react_1.useState)(null);
    const [workflowPlan, setWorkflowPlan] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const refreshList = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.listWorkflows(apiConfig);
            setWorkflows(response.workflows || []);
            if (!selectedWorkflowId && response.workflows?.length) {
                setSelectedWorkflowId(response.workflows[0].workflow_id);
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load workflows.");
        }
        finally {
            setLoading(false);
        }
    };
    const refreshWorkflow = async (workflowId) => {
        if (!workflowId)
            return;
        setErrorMessage(null);
        try {
            const [detail, telemetry, plan] = await Promise.all([
                openevolveApi_1.openevolveApi.getWorkflow(workflowId, apiConfig),
                openevolveApi_1.openevolveApi.getWorkflowTelemetry(workflowId, apiConfig),
                openevolveApi_1.openevolveApi.getWorkflowPlan(workflowId, apiConfig).catch(() => null),
            ]);
            setWorkflowDetail(detail);
            setWorkflowTelemetry(telemetry);
            setWorkflowPlan(plan);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load workflow data.");
        }
    };
    (0, react_1.useEffect)(() => {
        refreshList();
    }, [apiConfig.apiKey]);
    (0, react_1.useEffect)(() => {
        refreshWorkflow(selectedWorkflowId);
    }, [selectedWorkflowId, apiConfig.apiKey]);
    const stages = getStagesForType(workflowTelemetry?.workflow_type);
    const history = extractHistorySeries(workflowTelemetry?.openevolve_metrics || undefined);
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Workflow Visualization</card_1.CardTitle>
          <card_1.CardDescription>Live workflow metrics and execution flow overview.</card_1.CardDescription>
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
            <button_1.Button variant="outline" onClick={refreshList} disabled={loading}>
              Refresh Workflows
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="space-y-2">
            <label className="text-sm font-medium">Workflow</label>
            <select_1.Select value={selectedWorkflowId} onValueChange={setSelectedWorkflowId}>
              <select_1.SelectTrigger>
                <select_1.SelectValue placeholder="Select workflow"/>
              </select_1.SelectTrigger>
              <select_1.SelectContent>
                {workflows.map((workflow) => (<select_1.SelectItem key={workflow.workflow_id} value={workflow.workflow_id}>
                    {workflow.workflow_id}
                  </select_1.SelectItem>))}
              </select_1.SelectContent>
            </select_1.Select>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Workflow Status</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {workflowDetail ? (<>
                    <div className="flex items-center justify-between">
                      <div className="font-semibold">{workflowDetail.workflow_id}</div>
                      <badge_1.Badge variant="secondary">{workflowDetail.status}</badge_1.Badge>
                    </div>
                    <div>Stage: {workflowDetail.current_stage}</div>
                    <div>Progress: {formatNumber(workflowDetail.progress * 100, 1)}%</div>
                    <div>
                      Sub-problems: {workflowDetail.solved_sub_problems}/
                      {workflowDetail.total_sub_problems}
                    </div>
                  </>) : (<div className="text-muted-foreground">Select a workflow to view status.</div>)}
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Execution Metrics</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                <div>
                  Execution Time: {formatNumber(workflowTelemetry?.execution_time_seconds, 1)}s
                </div>
                <div>Tokens Used: {workflowTelemetry?.resource_usage?.tokens_used ?? "n/a"}</div>
                <div>
                  Memory Usage: {workflowTelemetry?.resource_usage?.memory_usage_mb ?? "n/a"} MB
                </div>
                <div>CPU Usage: {workflowTelemetry?.resource_usage?.cpu_usage ?? "n/a"}</div>
              </card_1.CardContent>
            </card_1.Card>
          </div>

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Workflow Flow</card_1.CardTitle>
              <card_1.CardDescription>{workflowTelemetry?.workflow_type ?? "Generic"} pipeline</card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="flex flex-wrap items-center gap-2 text-sm">
              {stages.map((stage, index) => {
            const isCurrent = workflowTelemetry?.current_stage === stage;
            return (<div key={stage} className="flex items-center gap-2">
                    <badge_1.Badge variant={isCurrent ? "default" : "secondary"}>{stage}</badge_1.Badge>
                    {index < stages.length - 1 && <span className="text-muted-foreground">→</span>}
                  </div>);
        })}
            </card_1.CardContent>
          </card_1.Card>

          <div className="grid gap-4 md:grid-cols-2">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Evolution Progress</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {history ? (<div className="space-y-2">
                    <div>
                      Latest Best Fitness: {formatNumber(history.best[history.best.length - 1], 3)}
                    </div>
                    <div className="space-y-1">
                      {history.best.slice(-10).map((value, index) => (<div key={`fit-${index}`} className="flex items-center gap-2">
                          <div className="h-2 w-24 rounded bg-muted">
                            <div className="h-2 rounded bg-emerald-500" style={{ width: `${Math.min(100, (value || 0) * 100)}%` }}/>
                          </div>
                          <span className="text-xs">{formatNumber(value, 3)}</span>
                        </div>))}
                    </div>
                  </div>) : (<div className="text-muted-foreground">No evolution history available.</div>)}
              </card_1.CardContent>
            </card_1.Card>
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Diversity Metrics</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                {history?.diversity?.length ? (<div className="space-y-1">
                    <div>
                      Latest Diversity:{" "}
                      {formatNumber(history.diversity[history.diversity.length - 1], 3)}
                    </div>
                    {history.diversity.slice(-10).map((value, index) => (<div key={`div-${index}`} className="flex items-center gap-2">
                        <div className="h-2 w-24 rounded bg-muted">
                          <div className="h-2 rounded bg-blue-500" style={{ width: `${Math.min(100, (value || 0) * 100)}%` }}/>
                        </div>
                        <span className="text-xs">{formatNumber(value, 3)}</span>
                      </div>))}
                  </div>) : (<div className="text-muted-foreground">No diversity history available.</div>)}
              </card_1.CardContent>
            </card_1.Card>
          </div>

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Decomposition Summary</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-2 text-sm">
              {workflowPlan ? (<>
                  <div>
                    Sub-problems: {workflowPlan.plan.sub_problems.length} · Parallel:{" "}
                    {workflowPlan.plan.parallel_processing_enabled ? "enabled" : "disabled"}
                  </div>
                  <div>
                    Planner Team: {workflowPlan.plan.planner_team_name ?? "n/a"} · Assembler Team:{" "}
                    {workflowPlan.plan.assembler_team_name ?? "n/a"}
                  </div>
                </>) : (<div className="text-muted-foreground">No decomposition plan loaded.</div>)}
            </card_1.CardContent>
          </card_1.Card>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.WorkflowVisualizationTab = WorkflowVisualizationTab;
//# sourceMappingURL=WorkflowVisualizationTab.js.map