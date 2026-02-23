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
exports.DecompositionReviewTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const separator_1 = require("@/components/ui/separator");
const checkbox_1 = require("@/components/ui/checkbox");
const switch_1 = require("@/components/ui/switch");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const CONTENT_TYPES = [
    "text_general",
    "code_python",
    "code_javascript",
    "document_legal",
    "document_medical",
    "document_technical",
    "prompt",
    "protocol",
];
const parseList = (value) => value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
const joinList = (values) => (values && values.length ? values.join(", ") : "");
const readApiKey = () => {
    try {
        return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    }
    catch {
        return "";
    }
};
const toJson = (value) => JSON.stringify(value ?? {}, null, 2);
const DecompositionReviewTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(readApiKey);
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [workflows, setWorkflows] = (0, react_1.useState)([]);
    const [teams, setTeams] = (0, react_1.useState)([]);
    const [gauntlets, setGauntlets] = (0, react_1.useState)([]);
    const [selectedWorkflowId, setSelectedWorkflowId] = (0, react_1.useState)("");
    const [planResponse, setPlanResponse] = (0, react_1.useState)(null);
    const [planDraft, setPlanDraft] = (0, react_1.useState)(null);
    const [selectedIds, setSelectedIds] = (0, react_1.useState)(new Set());
    const [jsonDrafts, setJsonDrafts] = (0, react_1.useState)({});
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const refreshBaseData = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const [workflowRes, teamRes, gauntletRes] = await Promise.all([
                openevolveApi_1.openevolveApi.listWorkflows(apiConfig),
                openevolveApi_1.openevolveApi.listTeams(apiConfig),
                openevolveApi_1.openevolveApi.listGauntlets(apiConfig),
            ]);
            setWorkflows(workflowRes.workflows ?? []);
            setTeams(teamRes.teams ?? []);
            setGauntlets(gauntletRes.gauntlets ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load base data.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        refreshBaseData();
    }, [apiConfig.apiKey]);
    const loadPlan = async (workflowId) => {
        setErrorMessage(null);
        setStatusMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.getWorkflowPlan(workflowId, apiConfig);
            setPlanResponse(response);
            setPlanDraft(response.plan);
            setSelectedIds(new Set());
            setJsonDrafts({
                auto_approval_criteria: toJson(response.plan.auto_approval_criteria ?? {}),
                mdap_config: toJson(response.plan.mdap_config ?? {}),
                maker_config: toJson(response.plan.maker_config ?? {}),
                resource_limits: toJson(response.plan.resource_limits ?? {}),
                learning_config: toJson(response.plan.learning_config ?? {}),
                metadata: toJson(response.plan.metadata ?? {}),
            });
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load decomposition plan.");
        }
    };
    const updatePlan = (updates) => {
        setPlanDraft((prev) => (prev ? { ...prev, ...updates } : prev));
    };
    const updateSubProblem = (id, updates) => {
        setPlanDraft((prev) => {
            if (!prev)
                return prev;
            return {
                ...prev,
                sub_problems: prev.sub_problems.map((sp) => (sp.id === id ? { ...sp, ...updates } : sp)),
            };
        });
    };
    const toggleSelection = (id) => {
        setSelectedIds((prev) => {
            const next = new Set(prev);
            if (next.has(id)) {
                next.delete(id);
            }
            else {
                next.add(id);
            }
            return next;
        });
    };
    const applyBatch = (updates) => {
        if (!planDraft)
            return;
        if (!selectedIds.size) {
            setErrorMessage("Select at least one sub-problem for batch operations.");
            return;
        }
        setErrorMessage(null);
        setPlanDraft({
            ...planDraft,
            sub_problems: planDraft.sub_problems.map((sp) => selectedIds.has(sp.id) ? { ...sp, ...updates } : sp),
        });
    };
    const handleSavePlan = async () => {
        if (!planDraft || !selectedWorkflowId)
            return;
        setErrorMessage(null);
        setStatusMessage(null);
        try {
            const payload = {
                ...planDraft,
                sub_problems: planDraft.sub_problems,
                auto_approval_criteria: JSON.parse(jsonDrafts.auto_approval_criteria || "{}"),
                mdap_config: JSON.parse(jsonDrafts.mdap_config || "{}"),
                maker_config: JSON.parse(jsonDrafts.maker_config || "{}"),
                resource_limits: JSON.parse(jsonDrafts.resource_limits || "{}"),
                learning_config: JSON.parse(jsonDrafts.learning_config || "{}"),
                metadata: JSON.parse(jsonDrafts.metadata || "{}"),
            };
            await openevolveApi_1.openevolveApi.updateWorkflowPlan(selectedWorkflowId, payload, apiConfig);
            setStatusMessage("Decomposition plan updated.");
            await loadPlan(selectedWorkflowId);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to update decomposition plan.");
        }
    };
    const updateJsonDraft = (field, value) => {
        setJsonDrafts((prev) => ({ ...prev, [field]: value }));
        try {
            const parsed = JSON.parse(value || "{}");
            updatePlan({ [field]: parsed });
        }
        catch {
            // Keep draft text until valid
        }
    };
    const blueTeams = teams.filter((team) => team.role === "Blue");
    const redGauntlets = gauntlets;
    const goldGauntlets = gauntlets;
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Decomposition Review</card_1.CardTitle>
          <card_1.CardDescription>Edit decomposition plans with batch operations.</card_1.CardDescription>
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
            <button_1.Button variant="outline" onClick={refreshBaseData} disabled={loading}>
              Refresh
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-[280px_1fr]">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Workflows</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2">
                {workflows.length === 0 && (<div className="text-sm text-muted-foreground">No workflows available.</div>)}
                <select className="w-full rounded border border-input bg-background px-3 py-2 text-sm" value={selectedWorkflowId} onChange={(event) => setSelectedWorkflowId(event.target.value)}>
                  <option value="">Select workflow</option>
                  {workflows.map((workflow) => (<option key={workflow.workflow_id} value={workflow.workflow_id}>
                      {workflow.workflow_id}
                    </option>))}
                </select>
                <button_1.Button className="w-full" variant="secondary" onClick={() => {
            if (selectedWorkflowId) {
                loadPlan(selectedWorkflowId);
            }
            else {
                setErrorMessage("Select a workflow first.");
            }
        }}>
                  Load Plan
                </button_1.Button>
              </card_1.CardContent>
            </card_1.Card>

            {planDraft ? (<card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Plan Summary</card_1.CardTitle>
                  <card_1.CardDescription>{planDraft.problem_statement}</card_1.CardDescription>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3">
                  <div className="flex flex-wrap gap-2">
                    <badge_1.Badge variant="outline">
                      Sub-problems: {planDraft.sub_problems.length}
                    </badge_1.Badge>
                    {planResponse?.dependency_graph?.execution_order?.length ? (<badge_1.Badge variant="secondary">
                        Execution Order: {planResponse.dependency_graph.execution_order.length}
                      </badge_1.Badge>) : null}
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="space-y-1">
                      <label_1.Label>Max Refinement Loops</label_1.Label>
                      <input_1.Input type="number" value={planDraft.max_refinement_loops ?? 3} onChange={(event) => updatePlan({ max_refinement_loops: Number(event.target.value) })}/>
                    </div>
                    <div className="space-y-1">
                      <label_1.Label>Auto Approval</label_1.Label>
                      <switch_1.Switch checked={Boolean(planDraft.auto_approval_enabled)} onCheckedChange={(value) => updatePlan({ auto_approval_enabled: value })}/>
                    </div>
                    <div className="space-y-1">
                      <label_1.Label>Parallel Processing</label_1.Label>
                      <switch_1.Switch checked={Boolean(planDraft.parallel_processing_enabled)} onCheckedChange={(value) => updatePlan({ parallel_processing_enabled: value })}/>
                    </div>
                  </div>

                  <separator_1.Separator />

                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-1">
                      <label_1.Label>MDAP Enabled</label_1.Label>
                      <switch_1.Switch checked={Boolean(planDraft.mdap_enabled)} onCheckedChange={(value) => updatePlan({ mdap_enabled: value })}/>
                    </div>
                    <div className="space-y-1">
                      <label_1.Label>MAKER Enabled</label_1.Label>
                      <switch_1.Switch checked={Boolean(planDraft.maker_enabled)} onCheckedChange={(value) => updatePlan({ maker_enabled: value })}/>
                    </div>
                  </div>

                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-1">
                      <label_1.Label>Auto Approval Criteria (JSON)</label_1.Label>
                      <textarea_1.Textarea value={jsonDrafts.auto_approval_criteria ?? ""} onChange={(event) => updateJsonDraft("auto_approval_criteria", event.target.value)} rows={6}/>
                    </div>
                    <div className="space-y-1">
                      <label_1.Label>MDAP Config (JSON)</label_1.Label>
                      <textarea_1.Textarea value={jsonDrafts.mdap_config ?? ""} onChange={(event) => updateJsonDraft("mdap_config", event.target.value)} rows={6}/>
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-1">
                      <label_1.Label>MAKER Config (JSON)</label_1.Label>
                      <textarea_1.Textarea value={jsonDrafts.maker_config ?? ""} onChange={(event) => updateJsonDraft("maker_config", event.target.value)} rows={6}/>
                    </div>
                    <div className="space-y-1">
                      <label_1.Label>Resource Limits (JSON)</label_1.Label>
                      <textarea_1.Textarea value={jsonDrafts.resource_limits ?? ""} onChange={(event) => updateJsonDraft("resource_limits", event.target.value)} rows={6}/>
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-1">
                      <label_1.Label>Learning Config (JSON)</label_1.Label>
                      <textarea_1.Textarea value={jsonDrafts.learning_config ?? ""} onChange={(event) => updateJsonDraft("learning_config", event.target.value)} rows={6}/>
                    </div>
                    <div className="space-y-1">
                      <label_1.Label>Metadata (JSON)</label_1.Label>
                      <textarea_1.Textarea value={jsonDrafts.metadata ?? ""} onChange={(event) => updateJsonDraft("metadata", event.target.value)} rows={6}/>
                    </div>
                  </div>

                  <div className="flex justify-end">
                    <button_1.Button onClick={handleSavePlan}>Save Plan</button_1.Button>
                  </div>
                </card_1.CardContent>
              </card_1.Card>) : null}
          </div>
        </card_1.CardContent>
      </card_1.Card>

      {planDraft ? (<card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="text-base">Batch Operations</card_1.CardTitle>
            <card_1.CardDescription>Apply changes to selected sub-problems.</card_1.CardDescription>
          </card_1.CardHeader>
          <card_1.CardContent className="space-y-4">
            <div className="text-sm text-muted-foreground">
              Selected sub-problems: {selectedIds.size}
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <label_1.Label>Assign Solver Team</label_1.Label>
                <select_1.Select onValueChange={(value) => applyBatch({ solver_team_name: value })}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue placeholder="Select team"/>
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    {blueTeams.map((team) => (<select_1.SelectItem key={team.name} value={team.name}>
                        {team.name}
                      </select_1.SelectItem>))}
                  </select_1.SelectContent>
                </select_1.Select>
              </div>
              <div className="space-y-2">
                <label_1.Label>Assign Patcher Team</label_1.Label>
                <select_1.Select onValueChange={(value) => applyBatch({ patcher_team_name: value })}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue placeholder="Select team"/>
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    {blueTeams.map((team) => (<select_1.SelectItem key={team.name} value={team.name}>
                        {team.name}
                      </select_1.SelectItem>))}
                  </select_1.SelectContent>
                </select_1.Select>
              </div>
              <div className="space-y-2">
                <label_1.Label>Assign Red Gauntlet</label_1.Label>
                <select_1.Select onValueChange={(value) => applyBatch({ red_team_gauntlet_name: value })}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue placeholder="Select gauntlet"/>
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    {redGauntlets.map((gauntlet) => (<select_1.SelectItem key={gauntlet.name} value={gauntlet.name}>
                        {gauntlet.name}
                      </select_1.SelectItem>))}
                  </select_1.SelectContent>
                </select_1.Select>
              </div>
              <div className="space-y-2">
                <label_1.Label>Assign Gold Gauntlet</label_1.Label>
                <select_1.Select onValueChange={(value) => applyBatch({ gold_team_gauntlet_name: value })}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue placeholder="Select gauntlet"/>
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    {goldGauntlets.map((gauntlet) => (<select_1.SelectItem key={gauntlet.name} value={gauntlet.name}>
                        {gauntlet.name}
                      </select_1.SelectItem>))}
                  </select_1.SelectContent>
                </select_1.Select>
              </div>
              <div className="space-y-2">
                <label_1.Label>Evolution Mode</label_1.Label>
                <input_1.Input placeholder="standard" onBlur={(event) => {
                if (event.target.value) {
                    applyBatch({ ai_suggested_evolution_mode: event.target.value });
                    event.target.value = "";
                }
            }}/>
              </div>
              <div className="space-y-2">
                <label_1.Label>Complexity Score</label_1.Label>
                <input_1.Input type="number" placeholder="5" onBlur={(event) => {
                if (event.target.value) {
                    applyBatch({ ai_suggested_complexity_score: Number(event.target.value) });
                    event.target.value = "";
                }
            }}/>
              </div>
              <div className="space-y-2">
                <label_1.Label>Content Type</label_1.Label>
                <select_1.Select onValueChange={(value) => applyBatch({ content_type: value })}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue placeholder="Select content type"/>
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    {CONTENT_TYPES.map((type) => (<select_1.SelectItem key={type} value={type}>
                        {type}
                      </select_1.SelectItem>))}
                  </select_1.SelectContent>
                </select_1.Select>
              </div>
            </div>
          </card_1.CardContent>
        </card_1.Card>) : null}

      {planDraft ? (<card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="text-base">Sub-Problems</card_1.CardTitle>
            <card_1.CardDescription>Edit individual sub-problem assignments and metadata.</card_1.CardDescription>
          </card_1.CardHeader>
          <card_1.CardContent className="space-y-4">
            {planDraft.sub_problems.map((sp) => (<card_1.Card key={sp.id}>
                <card_1.CardHeader className="flex flex-row items-center justify-between">
                  <div className="space-y-1">
                    <card_1.CardTitle className="text-sm">{sp.id}</card_1.CardTitle>
                    <card_1.CardDescription>{sp.description}</card_1.CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <checkbox_1.Checkbox checked={selectedIds.has(sp.id)} onCheckedChange={() => toggleSelection(sp.id)}/>
                    <badge_1.Badge variant="secondary">{sp.status ?? "pending"}</badge_1.Badge>
                  </div>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3">
                  <div className="space-y-2">
                    <label_1.Label>Description</label_1.Label>
                    <textarea_1.Textarea value={sp.description} onChange={(event) => updateSubProblem(sp.id, { description: event.target.value })} rows={3}/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Dependencies (comma-separated IDs)</label_1.Label>
                    <input_1.Input value={joinList(sp.dependencies)} onChange={(event) => updateSubProblem(sp.id, { dependencies: parseList(event.target.value) })}/>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <label_1.Label>Suggested Evolution Mode</label_1.Label>
                      <input_1.Input value={sp.ai_suggested_evolution_mode ?? ""} onChange={(event) => updateSubProblem(sp.id, { ai_suggested_evolution_mode: event.target.value })}/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Complexity Score</label_1.Label>
                      <input_1.Input type="number" value={sp.ai_suggested_complexity_score ?? 0} onChange={(event) => updateSubProblem(sp.id, {
                    ai_suggested_complexity_score: Number(event.target.value),
                })}/>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Suggested Evaluation Prompt</label_1.Label>
                    <textarea_1.Textarea value={sp.ai_suggested_evaluation_prompt ?? ""} onChange={(event) => updateSubProblem(sp.id, { ai_suggested_evaluation_prompt: event.target.value })} rows={3}/>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <label_1.Label>Content Type</label_1.Label>
                      <select_1.Select value={sp.content_type ?? "text_general"} onValueChange={(value) => updateSubProblem(sp.id, { content_type: value })}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue placeholder="Select content type"/>
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          {CONTENT_TYPES.map((type) => (<select_1.SelectItem key={type} value={type}>
                              {type}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Solver Team</label_1.Label>
                      <select_1.Select value={sp.solver_team_name ?? ""} onValueChange={(value) => updateSubProblem(sp.id, { solver_team_name: value })}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue placeholder="Select team"/>
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          {blueTeams.map((team) => (<select_1.SelectItem key={team.name} value={team.name}>
                              {team.name}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="space-y-2">
                      <label_1.Label>Red Gauntlet</label_1.Label>
                      <select_1.Select value={sp.red_team_gauntlet_name ?? ""} onValueChange={(value) => updateSubProblem(sp.id, { red_team_gauntlet_name: value })}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue placeholder="Select gauntlet"/>
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          {redGauntlets.map((gauntlet) => (<select_1.SelectItem key={gauntlet.name} value={gauntlet.name}>
                              {gauntlet.name}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Gold Gauntlet</label_1.Label>
                      <select_1.Select value={sp.gold_team_gauntlet_name ?? ""} onValueChange={(value) => updateSubProblem(sp.id, { gold_team_gauntlet_name: value })}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue placeholder="Select gauntlet"/>
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          {goldGauntlets.map((gauntlet) => (<select_1.SelectItem key={gauntlet.name} value={gauntlet.name}>
                              {gauntlet.name}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Patcher Team</label_1.Label>
                      <select_1.Select value={sp.patcher_team_name ?? ""} onValueChange={(value) => updateSubProblem(sp.id, { patcher_team_name: value })}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue placeholder="Select team"/>
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          {blueTeams.map((team) => (<select_1.SelectItem key={team.name} value={team.name}>
                              {team.name}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <label_1.Label>Atomic Mode</label_1.Label>
                      <switch_1.Switch checked={Boolean(sp.atomic_mode)} onCheckedChange={(value) => updateSubProblem(sp.id, { atomic_mode: value })}/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Decomposition Depth</label_1.Label>
                      <input_1.Input type="number" value={sp.decomposition_depth ?? 0} onChange={(event) => updateSubProblem(sp.id, { decomposition_depth: Number(event.target.value) })}/>
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <label_1.Label>Acceptance Criteria</label_1.Label>
                      <textarea_1.Textarea value={(sp.acceptance_criteria ?? []).join("\n")} onChange={(event) => updateSubProblem(sp.id, {
                    acceptance_criteria: event.target.value
                        .split("\n")
                        .map((item) => item.trim())
                        .filter(Boolean),
                })} rows={3}/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Specific Constraints</label_1.Label>
                      <textarea_1.Textarea value={(sp.specific_constraints ?? []).join("\n")} onChange={(event) => updateSubProblem(sp.id, {
                    specific_constraints: event.target.value
                        .split("\n")
                        .map((item) => item.trim())
                        .filter(Boolean),
                })} rows={3}/>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Solution Requirements (JSON)</label_1.Label>
                    <textarea_1.Textarea value={toJson(sp.solution_requirements ?? {})} onChange={(event) => {
                    try {
                        updateSubProblem(sp.id, { solution_requirements: JSON.parse(event.target.value) });
                    }
                    catch {
                        // ignore invalid JSON
                    }
                }} rows={4}/>
                  </div>
                </card_1.CardContent>
              </card_1.Card>))}
          </card_1.CardContent>
        </card_1.Card>) : null}
    </div>);
};
exports.DecompositionReviewTab = DecompositionReviewTab;
//# sourceMappingURL=DecompositionReviewTab.js.map