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
exports.OpenEvolveVisualizationTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const badge_1 = require("@/components/ui/badge");
const tabs_1 = require("@/components/ui/tabs");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const formatNumber = (value, decimals = 2) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "n/a";
    }
    return value.toFixed(decimals);
};
const renderHeatmap = (grid) => {
    const maxValue = Math.max(...grid.flat());
    return (<div className="overflow-auto">
      <div className="grid" style={{ gridTemplateColumns: `repeat(${grid[0]?.length || 0}, minmax(24px, 1fr))` }}>
        {grid.flatMap((row, rowIndex) => row.map((value, colIndex) => {
            const intensity = maxValue ? Math.round((value / maxValue) * 255) : 0;
            const background = `rgb(${255 - intensity}, ${255 - intensity}, 255)`;
            return (<div key={`${rowIndex}-${colIndex}`} className="h-8 w-8 border text-[10px] flex items-center justify-center" style={{ background }}>
                {value.toFixed(2)}
              </div>);
        }))}
      </div>
    </div>);
};
const OpenEvolveVisualizationTab = () => {
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
    const [telemetry, setTelemetry] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [configState, setConfigState] = (0, react_1.useState)({
        maxIterations: 100,
        populationSize: 50,
        numIslands: 5,
        archiveSize: 100,
        eliteRatio: 0.1,
        explorationRatio: 0.2,
        exploitationRatio: 0.7,
        featureDims: ["complexity", "diversity"],
        cascadeEvaluation: false,
        enableArtifacts: true,
        llmFeedback: false,
        tracing: false,
    });
    const refresh = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const workflowsRes = await openevolveApi_1.openevolveApi.listWorkflows(apiConfig);
            setWorkflows(workflowsRes.workflows || []);
            if (!selectedWorkflowId && workflowsRes.workflows?.length) {
                setSelectedWorkflowId(workflowsRes.workflows[0].workflow_id);
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load workflows.");
        }
        finally {
            setLoading(false);
        }
    };
    const refreshTelemetry = async (workflowId) => {
        if (!workflowId)
            return;
        try {
            const data = await openevolveApi_1.openevolveApi.getWorkflowTelemetry(workflowId, apiConfig);
            setTelemetry(data);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load workflow telemetry.");
        }
    };
    (0, react_1.useEffect)(() => {
        refresh();
    }, [apiConfig.apiKey]);
    (0, react_1.useEffect)(() => {
        refreshTelemetry(selectedWorkflowId);
    }, [selectedWorkflowId, apiConfig.apiKey]);
    const metrics = telemetry?.openevolve_metrics;
    const mapElitesGrid = metrics?.map_elites_grid;
    const featureDimensions = metrics?.feature_dimensions;
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>OpenEvolve Visualization</card_1.CardTitle>
          <card_1.CardDescription>Evolution analytics, diagnostics, and configuration insights.</card_1.CardDescription>
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

          <tabs_1.Tabs defaultValue="evolution">
            <tabs_1.TabsList className="flex flex-wrap gap-2">
              <tabs_1.TabsTrigger value="evolution">Evolution Dashboard</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="diagnostics">Advanced Diagnostics</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="configuration">Configuration</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="performance">Performance Metrics</tabs_1.TabsTrigger>
            </tabs_1.TabsList>

            <tabs_1.TabsContent value="evolution" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Evolution Summary</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  <div>Iterations: {metrics?.iterations_completed ?? "n/a"}</div>
                  <div>Best Fitness: {formatNumber(metrics?.best_fitness, 3)}</div>
                  <div>Population Size: {metrics?.population_size ?? "n/a"}</div>
                  <div>Archive Size: {metrics?.archive_size ?? "n/a"}</div>
                </card_1.CardContent>
              </card_1.Card>

              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">MAP-Elites Grid</card_1.CardTitle>
                  <card_1.CardDescription>
                    {featureDimensions?.length
            ? `Dimensions: ${featureDimensions.join(", ")}`
            : "No feature dimensions recorded."}
                  </card_1.CardDescription>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  {mapElitesGrid ? (renderHeatmap(mapElitesGrid)) : (<div className="text-muted-foreground">No MAP-Elites grid data available.</div>)}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="diagnostics" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Diagnostics Overview</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  {metrics?.diagnostics ? (Object.entries(metrics.diagnostics).map(([key, value]) => (<div key={key} className="flex items-center justify-between">
                        <span>{key}</span>
                        <badge_1.Badge variant="secondary">{String(value)}</badge_1.Badge>
                      </div>))) : (<div className="text-muted-foreground">No diagnostics data available.</div>)}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="configuration" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Local Configuration</card_1.CardTitle>
                  <card_1.CardDescription>Configure parameters for upcoming evolution runs.</card_1.CardDescription>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3 text-sm">
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Max Iterations</label>
                      <input_1.Input type="number" value={configState.maxIterations} onChange={(event) => setConfigState((prev) => ({
            ...prev,
            maxIterations: Number(event.target.value) || 0,
        }))}/>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Population Size</label>
                      <input_1.Input type="number" value={configState.populationSize} onChange={(event) => setConfigState((prev) => ({
            ...prev,
            populationSize: Number(event.target.value) || 0,
        }))}/>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Islands</label>
                      <input_1.Input type="number" value={configState.numIslands} onChange={(event) => setConfigState((prev) => ({
            ...prev,
            numIslands: Number(event.target.value) || 0,
        }))}/>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Archive Size</label>
                      <input_1.Input type="number" value={configState.archiveSize} onChange={(event) => setConfigState((prev) => ({
            ...prev,
            archiveSize: Number(event.target.value) || 0,
        }))}/>
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Elite Ratio</label>
                      <input_1.Input type="number" value={configState.eliteRatio} step="0.01" onChange={(event) => setConfigState((prev) => ({
            ...prev,
            eliteRatio: Number(event.target.value) || 0,
        }))}/>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Exploration Ratio</label>
                      <input_1.Input type="number" value={configState.explorationRatio} step="0.01" onChange={(event) => setConfigState((prev) => ({
            ...prev,
            explorationRatio: Number(event.target.value) || 0,
        }))}/>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Exploitation Ratio</label>
                      <input_1.Input type="number" value={configState.exploitationRatio} step="0.01" onChange={(event) => setConfigState((prev) => ({
            ...prev,
            exploitationRatio: Number(event.target.value) || 0,
        }))}/>
                    </div>
                  </div>
                  <div className="rounded border p-2 text-xs text-muted-foreground">
                    Feature Dimensions: {configState.featureDims.join(", ")}
                  </div>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="performance" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Performance Metrics</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  {telemetry?.performance_metrics ? (Object.entries(telemetry.performance_metrics).map(([key, value]) => (<div key={key} className="flex items-center justify-between">
                        <span>{key}</span>
                        <badge_1.Badge variant="secondary">{String(value)}</badge_1.Badge>
                      </div>))) : (<div className="text-muted-foreground">No performance metrics available.</div>)}
                </card_1.CardContent>
              </card_1.Card>
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Resource Usage</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  {telemetry?.resource_usage ? (Object.entries(telemetry.resource_usage).map(([key, value]) => (<div key={key} className="flex items-center justify-between">
                        <span>{key}</span>
                        <badge_1.Badge variant="secondary">{String(value)}</badge_1.Badge>
                      </div>))) : (<div className="text-muted-foreground">No resource usage data.</div>)}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>
          </tabs_1.Tabs>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.OpenEvolveVisualizationTab = OpenEvolveVisualizationTab;
//# sourceMappingURL=OpenEvolveVisualizationTab.js.map