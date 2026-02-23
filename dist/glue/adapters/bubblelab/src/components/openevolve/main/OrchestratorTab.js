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
exports.OrchestratorTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const select_1 = require("@/components/ui/select");
const badge_1 = require("@/components/ui/badge");
const separator_1 = require("@/components/ui/separator");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const createDefaultWorkflowRequest = () => ({
    problem_statement: "",
    content_analyzer_team: "",
    planner_team: "",
    solver_team: "",
    patcher_team: "",
    assembler_team: "",
    sub_problem_red_gauntlet: "",
    sub_problem_gold_gauntlet: "",
    final_red_gauntlet: "",
    final_gold_gauntlet: "",
    solver_generation_gauntlet: "",
    max_refinement_loops: 3,
    mdap_enabled: false,
    mdap_config: {},
    maker_enabled: false,
    maker_config: {},
});
const OrchestratorTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [teams, setTeams] = (0, react_1.useState)([]);
    const [gauntlets, setGauntlets] = (0, react_1.useState)([]);
    const [workflows, setWorkflows] = (0, react_1.useState)([]);
    const [selectedWorkflowId, setSelectedWorkflowId] = (0, react_1.useState)(null);
    const [workflowDetail, setWorkflowDetail] = (0, react_1.useState)(null);
    const [workflowResults, setWorkflowResults] = (0, react_1.useState)(null);
    const [form, setForm] = (0, react_1.useState)(createDefaultWorkflowRequest());
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const loadTeams = async () => {
        const result = await openevolveApi_1.openevolveApi.listTeams(apiConfig);
        setTeams(result.teams ?? []);
    };
    const loadGauntlets = async () => {
        const result = await openevolveApi_1.openevolveApi.listGauntlets(apiConfig);
        setGauntlets(result.gauntlets ?? []);
    };
    const loadWorkflows = async () => {
        const result = await openevolveApi_1.openevolveApi.listWorkflows(apiConfig);
        setWorkflows(result.workflows ?? []);
    };
    const loadWorkflowDetail = async (workflowId) => {
        const detail = await openevolveApi_1.openevolveApi.getWorkflow(workflowId, apiConfig);
        setWorkflowDetail(detail);
    };
    const loadWorkflowResults = async (workflowId) => {
        const results = await openevolveApi_1.openevolveApi.getWorkflowResults(workflowId, apiConfig);
        setWorkflowResults(results);
    };
    const refreshAll = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            await Promise.all([loadTeams(), loadGauntlets(), loadWorkflows()]);
            if (selectedWorkflowId) {
                await loadWorkflowDetail(selectedWorkflowId);
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to refresh data.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        refreshAll();
    }, [apiConfig.apiKey]);
    const handleSelectWorkflow = async (workflowId) => {
        setSelectedWorkflowId(workflowId);
        setWorkflowResults(null);
        setErrorMessage(null);
        try {
            await loadWorkflowDetail(workflowId);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load workflow details.");
        }
    };
    const handleCreateWorkflow = async () => {
        setErrorMessage(null);
        setStatusMessage(null);
        if (!form.problem_statement.trim()) {
            setErrorMessage("Problem statement is required.");
            return;
        }
        if (!form.content_analyzer_team ||
            !form.planner_team ||
            !form.solver_team ||
            !form.patcher_team ||
            !form.assembler_team) {
            setErrorMessage("All team selections are required.");
            return;
        }
        if (!form.sub_problem_red_gauntlet ||
            !form.sub_problem_gold_gauntlet ||
            !form.final_red_gauntlet ||
            !form.final_gold_gauntlet ||
            !form.solver_generation_gauntlet) {
            setErrorMessage("All gauntlet selections are required.");
            return;
        }
        try {
            const result = await openevolveApi_1.openevolveApi.createWorkflow(form, apiConfig);
            setStatusMessage(`Workflow ${result.workflow_id} created.`);
            setForm(createDefaultWorkflowRequest());
            await loadWorkflows();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to create workflow.");
        }
    };
    const updateForm = (field, value) => {
        setForm((prev) => ({ ...prev, [field]: value }));
    };
    const loadTemplateFromCache = () => {
        try {
            const raw = globalThis.localStorage?.getItem("openevolve_active_workflow_template");
            if (!raw) {
                setStatusMessage("No cached workflow template found.");
                return;
            }
            const parsed = JSON.parse(raw);
            if (parsed?.config) {
                setForm((prev) => ({
                    ...prev,
                    ...parsed.config,
                }));
                setStatusMessage(`Loaded template ${parsed.name ?? "template"} into form.`);
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load cached template.");
        }
    };
    const blueTeams = teams.filter((team) => team.role === "Blue");
    const redTeams = teams.filter((team) => team.role === "Red");
    const goldTeams = teams.filter((team) => team.role === "Gold");
    const gauntletOptions = gauntlets;
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Workflow Orchestrator</card_1.CardTitle>
          <card_1.CardDescription>Create, monitor, and manage sovereign workflows.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key (X-API-Key)</label>
              <input_1.Input value={apiKey} type="password" placeholder="Paste API key for workflow endpoints" onChange={(event) => {
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
            <div className="flex gap-2">
              <button_1.Button variant="outline" onClick={refreshAll} disabled={loading}>
                Refresh
              </button_1.Button>
              <button_1.Button variant="outline" onClick={loadTemplateFromCache}>
                Load Template
              </button_1.Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid grid-cols-1 gap-6 xl:grid-cols-[420px_1fr]">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-base">Create Workflow</card_1.CardTitle>
                <card_1.CardDescription>Configure the full sovereign gauntlet pipeline.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Problem Statement</label>
                  <textarea_1.Textarea value={form.problem_statement} onChange={(event) => updateForm("problem_statement", event.target.value)} rows={5} placeholder="Describe the problem to solve"/>
                </div>

                <separator_1.Separator />
                <div className="space-y-3">
                  <h4 className="text-sm font-semibold">Teams</h4>
                  <div className="grid gap-3">
                    <select_1.Select value={form.content_analyzer_team} onValueChange={(value) => updateForm("content_analyzer_team", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue placeholder="Content Analyzer Team"/>
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {blueTeams.map((team) => (<select_1.SelectItem key={team.name} value={team.name}>
                            {team.name}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                    <select_1.Select value={form.planner_team} onValueChange={(value) => updateForm("planner_team", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue placeholder="Planner Team"/>
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {blueTeams.map((team) => (<select_1.SelectItem key={team.name} value={team.name}>
                            {team.name}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                    <select_1.Select value={form.solver_team} onValueChange={(value) => updateForm("solver_team", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue placeholder="Solver Team"/>
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {blueTeams.map((team) => (<select_1.SelectItem key={team.name} value={team.name}>
                            {team.name}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                    <select_1.Select value={form.patcher_team} onValueChange={(value) => updateForm("patcher_team", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue placeholder="Patcher Team"/>
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {blueTeams.map((team) => (<select_1.SelectItem key={team.name} value={team.name}>
                            {team.name}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                    <select_1.Select value={form.assembler_team} onValueChange={(value) => updateForm("assembler_team", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue placeholder="Assembler Team"/>
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {blueTeams.map((team) => (<select_1.SelectItem key={team.name} value={team.name}>
                            {team.name}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                </div>

                <separator_1.Separator />
                <div className="space-y-3">
                  <h4 className="text-sm font-semibold">Gauntlets</h4>
                  <div className="grid gap-3">
                    <select_1.Select value={form.sub_problem_red_gauntlet} onValueChange={(value) => updateForm("sub_problem_red_gauntlet", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue placeholder="Sub-problem Red Gauntlet"/>
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {gauntletOptions.map((gauntlet) => (<select_1.SelectItem key={gauntlet.name} value={gauntlet.name}>
                            {gauntlet.name}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                    <select_1.Select value={form.sub_problem_gold_gauntlet} onValueChange={(value) => updateForm("sub_problem_gold_gauntlet", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue placeholder="Sub-problem Gold Gauntlet"/>
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {gauntletOptions.map((gauntlet) => (<select_1.SelectItem key={gauntlet.name} value={gauntlet.name}>
                            {gauntlet.name}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                    <select_1.Select value={form.final_red_gauntlet} onValueChange={(value) => updateForm("final_red_gauntlet", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue placeholder="Final Red Gauntlet"/>
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {gauntletOptions.map((gauntlet) => (<select_1.SelectItem key={gauntlet.name} value={gauntlet.name}>
                            {gauntlet.name}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                    <select_1.Select value={form.final_gold_gauntlet} onValueChange={(value) => updateForm("final_gold_gauntlet", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue placeholder="Final Gold Gauntlet"/>
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {gauntletOptions.map((gauntlet) => (<select_1.SelectItem key={gauntlet.name} value={gauntlet.name}>
                            {gauntlet.name}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                    <select_1.Select value={form.solver_generation_gauntlet} onValueChange={(value) => updateForm("solver_generation_gauntlet", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue placeholder="Solver Generation Gauntlet"/>
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {gauntletOptions.map((gauntlet) => (<select_1.SelectItem key={gauntlet.name} value={gauntlet.name}>
                            {gauntlet.name}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                </div>

                <separator_1.Separator />
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Max Refinement Loops</label>
                    <input_1.Input type="number" value={form.max_refinement_loops ?? 3} onChange={(event) => updateForm("max_refinement_loops", Number(event.target.value) || 1)}/>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">MDAP Enabled</label>
                    <select_1.Select value={form.mdap_enabled ? "enabled" : "disabled"} onValueChange={(value) => updateForm("mdap_enabled", value === "enabled")}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue />
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        <select_1.SelectItem value="enabled">Enabled</select_1.SelectItem>
                        <select_1.SelectItem value="disabled">Disabled</select_1.SelectItem>
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">MAKER Enabled</label>
                    <select_1.Select value={form.maker_enabled ? "enabled" : "disabled"} onValueChange={(value) => updateForm("maker_enabled", value === "enabled")}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue />
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        <select_1.SelectItem value="enabled">Enabled</select_1.SelectItem>
                        <select_1.SelectItem value="disabled">Disabled</select_1.SelectItem>
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">MDAP Config (JSON)</label>
                  <textarea_1.Textarea value={JSON.stringify(form.mdap_config ?? {}, null, 2)} onChange={(event) => {
            try {
                updateForm("mdap_config", JSON.parse(event.target.value));
            }
            catch {
                // ignore invalid JSON
            }
        }} rows={6}/>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">MAKER Config (JSON)</label>
                  <textarea_1.Textarea value={JSON.stringify(form.maker_config ?? {}, null, 2)} onChange={(event) => {
            try {
                updateForm("maker_config", JSON.parse(event.target.value));
            }
            catch {
                // ignore invalid JSON
            }
        }} rows={6}/>
                </div>

                <div className="flex justify-end">
                  <button_1.Button onClick={handleCreateWorkflow}>Create Workflow</button_1.Button>
                </div>
              </card_1.CardContent>
            </card_1.Card>

            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-base">Active Workflows</card_1.CardTitle>
                <card_1.CardDescription>Monitor progress and control execution.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-4">
                {workflows.length === 0 && (<div className="text-sm text-muted-foreground">No workflows found.</div>)}
                {workflows.map((workflow) => (<div key={workflow.workflow_id} className={`rounded border p-3 ${selectedWorkflowId === workflow.workflow_id ? "border-primary" : "border-border"}`}>
                    <div className="flex items-center justify-between">
                      <div className="space-y-1">
                        <div className="text-sm font-semibold">{workflow.workflow_id}</div>
                        <div className="text-xs text-muted-foreground">
                          {workflow.current_stage} · {Math.round(workflow.progress * 100)}%
                        </div>
                      </div>
                      <badge_1.Badge variant="outline">{workflow.status}</badge_1.Badge>
                    </div>
                    <div className="mt-3 flex gap-2">
                      <button_1.Button size="sm" variant="secondary" onClick={() => handleSelectWorkflow(workflow.workflow_id)}>
                        View
                      </button_1.Button>
                    </div>
                  </div>))}

                {workflowDetail ? (<div className="space-y-3 rounded border p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-sm font-semibold">Workflow Detail</div>
                        <div className="text-xs text-muted-foreground">
                          {workflowDetail.workflow_id}
                        </div>
                      </div>
                      <badge_1.Badge variant="outline">{workflowDetail.status}</badge_1.Badge>
                    </div>
                    <div className="text-sm">Stage: {workflowDetail.current_stage}</div>
                    <div className="text-sm">
                      Progress: {Math.round(workflowDetail.progress * 100)}%
                    </div>
                    <div className="text-sm">
                      Sub-problems: {workflowDetail.solved_sub_problems} /{" "}
                      {workflowDetail.total_sub_problems}
                    </div>
                    <div className="text-sm">Refinement loops: {workflowDetail.refinement_loop_count}</div>
                    <div className="flex flex-wrap gap-2 pt-2">
                      <button_1.Button size="sm" variant="outline" onClick={async () => {
                if (!workflowDetail)
                    return;
                try {
                    await openevolveApi_1.openevolveApi.pauseWorkflow(workflowDetail.workflow_id, apiConfig);
                    await loadWorkflowDetail(workflowDetail.workflow_id);
                    await loadWorkflows();
                }
                catch (error) {
                    setErrorMessage(error?.message ?? "Failed to pause workflow.");
                }
            }}>
                        Pause
                      </button_1.Button>
                      <button_1.Button size="sm" variant="outline" onClick={async () => {
                if (!workflowDetail)
                    return;
                try {
                    await openevolveApi_1.openevolveApi.resumeWorkflow(workflowDetail.workflow_id, apiConfig);
                    await loadWorkflowDetail(workflowDetail.workflow_id);
                    await loadWorkflows();
                }
                catch (error) {
                    setErrorMessage(error?.message ?? "Failed to resume workflow.");
                }
            }}>
                        Resume
                      </button_1.Button>
                      <button_1.Button size="sm" variant="secondary" onClick={async () => {
                if (!workflowDetail)
                    return;
                try {
                    await loadWorkflowResults(workflowDetail.workflow_id);
                }
                catch (error) {
                    setErrorMessage(error?.message ?? "Failed to load workflow results.");
                }
            }}>
                        Fetch Results
                      </button_1.Button>
                    </div>

                    {workflowResults ? (<div className="space-y-2 rounded border border-dashed p-3 text-sm">
                        <div className="font-semibold">Final Solution</div>
                        <div className="text-xs text-muted-foreground">
                          {workflowResults.final_solution?.generated_by ?? "N/A"}
                        </div>
                        <textarea_1.Textarea value={workflowResults.final_solution?.content ?? ""} readOnly rows={6}/>
                        <div className="font-semibold">Sub-problem Solutions</div>
                        {Object.entries(workflowResults.sub_problem_solutions ?? {}).map(([subId, solution]) => (<div key={subId} className="space-y-1">
                              <div className="text-xs font-medium">{subId}</div>
                              <textarea_1.Textarea value={solution.content} readOnly rows={4}/>
                            </div>))}
                      </div>) : null}
                  </div>) : null}
              </card_1.CardContent>
            </card_1.Card>
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.OrchestratorTab = OrchestratorTab;
//# sourceMappingURL=OrchestratorTab.js.map