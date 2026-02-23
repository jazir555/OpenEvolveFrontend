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
exports.SgdMonitoringTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const badge_1 = require("@/components/ui/badge");
const tabs_1 = require("@/components/ui/tabs");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const formatPercent = (value) => {
    if (value === null || value === undefined || Number.isNaN(value))
        return "n/a";
    return `${(value * 100).toFixed(1)}%`;
};
const SgdMonitoringTab = () => {
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
    const [crewaiWorkflows, setCrewaiWorkflows] = (0, react_1.useState)([]);
    const [selectedWorkflowId, setSelectedWorkflowId] = (0, react_1.useState)("");
    const [selectedCrewaiWorkflowId, setSelectedCrewaiWorkflowId] = (0, react_1.useState)("");
    const [workflowDetail, setWorkflowDetail] = (0, react_1.useState)(null);
    const [workflowTelemetry, setWorkflowTelemetry] = (0, react_1.useState)(null);
    const [workflowPlan, setWorkflowPlan] = (0, react_1.useState)(null);
    const [crewaiTickets, setCrewaiTickets] = (0, react_1.useState)([]);
    const [ticketBreakdown, setTicketBreakdown] = (0, react_1.useState)({});
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const refreshLists = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const [workflowRes, crewaiRes] = await Promise.all([
                openevolveApi_1.openevolveApi.listWorkflows(apiConfig),
                openevolveApi_1.openevolveApi.listCrewaiWorkflows(apiConfig),
            ]);
            setWorkflows(workflowRes.workflows || []);
            setCrewaiWorkflows(crewaiRes.workflows || []);
            if (!selectedWorkflowId && workflowRes.workflows?.length) {
                setSelectedWorkflowId(workflowRes.workflows[0].workflow_id);
            }
            if (!selectedCrewaiWorkflowId && crewaiRes.workflows?.length) {
                setSelectedCrewaiWorkflowId(crewaiRes.workflows[0].workflow_id);
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load SGD monitoring data.");
        }
        finally {
            setLoading(false);
        }
    };
    const refreshWorkflowData = async (workflowId) => {
        if (!workflowId)
            return;
        setErrorMessage(null);
        try {
            const [detailRes, telemetryRes, planRes] = await Promise.all([
                openevolveApi_1.openevolveApi.getWorkflow(workflowId, apiConfig),
                openevolveApi_1.openevolveApi.getWorkflowTelemetry(workflowId, apiConfig),
                openevolveApi_1.openevolveApi.getWorkflowPlan(workflowId, apiConfig).catch(() => null),
            ]);
            setWorkflowDetail(detailRes);
            setWorkflowTelemetry(telemetryRes);
            setWorkflowPlan(planRes);
            if (telemetryRes?.crewai_workflow_id) {
                setSelectedCrewaiWorkflowId(telemetryRes.crewai_workflow_id);
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load workflow telemetry.");
        }
    };
    const refreshCrewaiTickets = async (workflowId) => {
        if (!workflowId) {
            setCrewaiTickets([]);
            setTicketBreakdown({});
            return;
        }
        try {
            const result = await openevolveApi_1.openevolveApi.getCrewaiWorkflowTickets(workflowId, apiConfig);
            setCrewaiTickets(result.tickets || []);
            setTicketBreakdown(result.status_breakdown || {});
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load CrewAI ticket data.");
        }
    };
    (0, react_1.useEffect)(() => {
        refreshLists();
    }, [apiConfig.apiKey]);
    (0, react_1.useEffect)(() => {
        refreshWorkflowData(selectedWorkflowId);
    }, [selectedWorkflowId, apiConfig.apiKey]);
    (0, react_1.useEffect)(() => {
        refreshCrewaiTickets(selectedCrewaiWorkflowId);
    }, [selectedCrewaiWorkflowId, apiConfig.apiKey]);
    const gauntletSummary = workflowTelemetry?.gauntlet_summary;
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>SGD Monitoring</card_1.CardTitle>
          <card_1.CardDescription>Sovereign-Grade Decomposition + CrewAI integration status.</card_1.CardDescription>
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
            <button_1.Button variant="outline" onClick={refreshLists} disabled={loading}>
              Refresh Lists
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">OpenEvolve Workflow</label>
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
            <div className="space-y-2">
              <label className="text-sm font-medium">CrewAI Workflow</label>
              <select_1.Select value={selectedCrewaiWorkflowId} onValueChange={setSelectedCrewaiWorkflowId}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue placeholder="Select CrewAI workflow"/>
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  {crewaiWorkflows.map((workflow) => (<select_1.SelectItem key={workflow.workflow_id} value={workflow.workflow_id}>
                      {workflow.workflow_id}
                    </select_1.SelectItem>))}
                </select_1.SelectContent>
              </select_1.Select>
            </div>
          </div>

          <tabs_1.Tabs defaultValue="progress">
            <tabs_1.TabsList className="flex flex-wrap gap-2">
              <tabs_1.TabsTrigger value="progress">Workflow Progress</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="tickets">Ticket Status</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="gauntlet">Gauntlet Performance</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="analysis">Detailed Analysis</tabs_1.TabsTrigger>
            </tabs_1.TabsList>

            <tabs_1.TabsContent value="progress" className="mt-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Current Workflow</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  {workflowDetail ? (<>
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">{workflowDetail.workflow_id}</div>
                        <badge_1.Badge variant="secondary">{workflowDetail.status}</badge_1.Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Stage: {workflowDetail.current_stage} · Progress:{" "}
                        {formatPercent(workflowDetail.progress)}
                      </div>
                      <div className="mt-2 h-2 w-full rounded bg-muted">
                        <div className="h-2 rounded bg-emerald-500" style={{ width: `${Math.round((workflowDetail.progress ?? 0) * 100)}%` }}/>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Sub-problems solved: {workflowDetail.solved_sub_problems}/
                        {workflowDetail.total_sub_problems}
                      </div>
                      {workflowTelemetry?.crewai_workflow_id ? (<div className="text-xs text-muted-foreground">
                          CrewAI Workflow: {workflowTelemetry.crewai_workflow_id}
                        </div>) : null}
                      {workflowPlan?.plan?.problem_statement ? (<div className="text-xs text-muted-foreground">
                          Problem: {workflowPlan.plan.problem_statement.slice(0, 120)}
                        </div>) : null}
                    </>) : (<div className="text-muted-foreground">Select a workflow to view progress.</div>)}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="tickets" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Ticket Breakdown</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2 text-sm">
                    {Object.keys(ticketBreakdown).length === 0 ? (<div className="text-muted-foreground">No ticket data available.</div>) : (Object.entries(ticketBreakdown).map(([status, count]) => (<div key={status} className="flex items-center justify-between">
                          <span>{status}</span>
                          <badge_1.Badge variant="secondary">{count}</badge_1.Badge>
                        </div>)))}
                  </card_1.CardContent>
                </card_1.Card>
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Tickets</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-2 text-sm">
                    {crewaiTickets.length === 0 ? (<div className="text-muted-foreground">No tickets found.</div>) : (crewaiTickets.slice(0, 12).map((ticket) => (<div key={ticket.id} className="rounded border p-2">
                          <div className="flex items-center justify-between">
                            <div className="font-semibold">{ticket.title ?? ticket.id}</div>
                            <badge_1.Badge variant="secondary">{ticket.status ?? "pending"}</badge_1.Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Assigned: {ticket.assigned_agent_id ?? "unassigned"}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Dependencies: {ticket.dependencies?.join(", ") || "none"}
                          </div>
                        </div>)))}
                  </card_1.CardContent>
                </card_1.Card>
              </div>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="gauntlet" className="mt-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Gauntlet Summary</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3 text-sm">
                  {gauntletSummary ? (<>
                      <div className="grid gap-4 md:grid-cols-3">
                        <div className="rounded border p-2">
                          <div className="font-semibold">Critique Reports</div>
                          <div>Total: {gauntletSummary.critique_total}</div>
                          <div>Approved: {gauntletSummary.critique_approved}</div>
                          <div>Avg Score: {gauntletSummary.critique_avg_score.toFixed(3)}</div>
                        </div>
                        <div className="rounded border p-2">
                          <div className="font-semibold">Verification Reports</div>
                          <div>Total: {gauntletSummary.verification_total}</div>
                          <div>Approved: {gauntletSummary.verification_approved}</div>
                          <div>Avg Score: {gauntletSummary.verification_avg_score.toFixed(3)}</div>
                        </div>
                        <div className="rounded border p-2">
                          <div className="font-semibold">Approval Rate</div>
                          <div>
                            {formatPercent(gauntletSummary.verification_total
                ? gauntletSummary.verification_approved /
                    gauntletSummary.verification_total
                : null)}
                          </div>
                        </div>
                      </div>
                    </>) : (<div className="text-muted-foreground">No gauntlet summary available.</div>)}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="analysis" className="mt-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Stage Timeline</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  {workflowTelemetry ? (<div className="space-y-2">
                      {[
                "INITIALIZING",
                "Content Analysis",
                "AI-Assisted Decomposition",
                "Manual Review & Override",
                "Delegate to CrewAI",
                "Monitoring",
                "Sub-Problem Solving Loop",
                "Configurable Reassembly",
                "Final Verification & Self-Healing Loop",
            ].map((stage) => {
                const isCurrent = workflowTelemetry.current_stage === stage;
                return (<div key={stage} className="flex items-center justify-between">
                            <span>{stage}</span>
                            <badge_1.Badge variant={isCurrent ? "default" : "secondary"}>
                              {isCurrent ? "current" : "pending"}
                            </badge_1.Badge>
                          </div>);
            })}
                    </div>) : (<div className="text-muted-foreground">Select a workflow to view details.</div>)}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>
          </tabs_1.Tabs>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.SgdMonitoringTab = SgdMonitoringTab;
//# sourceMappingURL=SgdMonitoringTab.js.map