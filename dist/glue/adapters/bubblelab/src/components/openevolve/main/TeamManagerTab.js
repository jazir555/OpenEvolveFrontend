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
exports.TeamManagerTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const select_1 = require("@/components/ui/select");
const tabs_1 = require("@/components/ui/tabs");
const badge_1 = require("@/components/ui/badge");
const separator_1 = require("@/components/ui/separator");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const types_1 = require("../../../lib/types");
const promptDefaults = {
    content_analysis_system_prompt: "You are a highly skilled content analyzer. Your task is to analyze a problem statement and extract key information, context, and potential challenges. Provide your analysis in a structured JSON format.",
    content_analysis_user_prompt_template: `Analyze the following problem statement and extract:
- domain: (e.g., "Software Development", "Physics", "Legal")
- keywords: List of important terms.
- estimated_complexity: (1-10)
- potential_challenges: List of anticipated difficulties.
- required_expertise: List of expertise areas needed.
- summary: A brief, concise summary of the problem.

Problem Statement:
---
{problem_statement}
---`,
    decomposition_system_prompt: "You are an expert problem decomposer. Your task is to break down a complex problem into smaller, manageable sub-problems. For each sub-problem, suggest an evolution mode, a complexity score (1-10), and a specific evaluation prompt. Provide the output as a JSON array of sub-problem objects.",
    decomposition_user_prompt_template: `Decompose the following problem into a list of sub-problems. For each sub-problem, provide:
- id: A unique identifier (e.g., "sub_1.1")
- description: A clear statement of the sub-problem.
- dependencies: A list of ids of other sub-problems this one depends on.
- ai_suggested_evolution_mode: Suggested evolution mode (e.g., "standard", "adversarial", "quality_diversity").
- ai_suggested_complexity_score: An integer from 1 to 10.
- ai_suggested_evaluation_prompt: A specific prompt for a Gold Team to evaluate this sub-problem's solution.

Problem Statement:
---
{problem_statement}
---`,
    solver_system_prompt: "You are a skilled problem solver. Your task is to generate a high-quality solution for the given sub-problem. Provide your response in a clear, concise format.",
    solver_user_prompt_template: `Solve the following sub-problem:

Sub-Problem Description:
---
{sub_problem_description}
---
`,
    patcher_system_prompt: "You are an expert solution patcher. Your task is to improve and fix a solution based on critique feedback.",
    patcher_user_prompt_template: `Improve the following solution based on critique feedback:

Original Solution:
---
{solution_content}
---

Critique Feedback:
---
{critique_summary}
---`,
    assembler_system_prompt: "You are an expert solution assembler. Your task is to combine sub-problem solutions into a cohesive final output.",
    assembler_user_prompt_template: `Assemble the final solution from the following sub-problem solutions:

Sub-Problem Solutions:
---
{sub_problem_solutions}
---`,
    red_team_system_prompt: "You are a Red Team critic. Identify flaws, risks, and weaknesses in the solution.",
    red_team_user_prompt_template: `Critique the following solution:

Solution:
---
{solution_content}
---`,
    gold_team_system_prompt: "You are a Gold Team verifier. Verify correctness, completeness, and compliance.",
    gold_team_user_prompt_template: `Verify the following solution against requirements:

Solution:
---
{solution_content}
---`,
};
const TeamManagerTab = () => {
    const [teams, setTeams] = (0, react_1.useState)([]);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [editingTeam, setEditingTeam] = (0, react_1.useState)(null);
    const [formTeam, setFormTeam] = (0, react_1.useState)(() => ({
        ...(0, types_1.createDefaultTeam)(),
        ...promptDefaults,
    }));
    const [memberJson, setMemberJson] = (0, react_1.useState)([]);
    const [memberErrors, setMemberErrors] = (0, react_1.useState)([]);
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const syncMembersFromTeam = (team) => {
        const jsonValues = team.members.map((member) => JSON.stringify(member, null, 2));
        setMemberJson(jsonValues);
        setMemberErrors(jsonValues.map(() => ""));
    };
    const resetForm = () => {
        const next = { ...(0, types_1.createDefaultTeam)(), ...promptDefaults };
        setFormTeam(next);
        setEditingTeam(null);
        syncMembersFromTeam(next);
    };
    const loadTeams = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const result = await openevolveApi_1.openevolveApi.listTeams(apiConfig);
            setTeams(result.teams ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load teams.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        syncMembersFromTeam(formTeam);
    }, []);
    (0, react_1.useEffect)(() => {
        loadTeams();
    }, [apiConfig.apiKey]);
    const handleMemberJsonChange = (index, value) => {
        const updatedJson = [...memberJson];
        updatedJson[index] = value;
        setMemberJson(updatedJson);
        try {
            const parsed = JSON.parse(value);
            const updatedMembers = [...formTeam.members];
            updatedMembers[index] = parsed;
            setFormTeam({ ...formTeam, members: updatedMembers });
            const errors = [...memberErrors];
            errors[index] = "";
            setMemberErrors(errors);
        }
        catch (error) {
            const errors = [...memberErrors];
            errors[index] = error?.message ?? "Invalid JSON";
            setMemberErrors(errors);
        }
    };
    const handleAddMember = () => {
        const updatedMembers = [...formTeam.members, (0, types_1.createDefaultModelConfig)()];
        setFormTeam({ ...formTeam, members: updatedMembers });
        setMemberJson([...memberJson, JSON.stringify((0, types_1.createDefaultModelConfig)(), null, 2)]);
        setMemberErrors([...memberErrors, ""]);
    };
    const handleRemoveMember = (index) => {
        const updatedMembers = formTeam.members.filter((_, idx) => idx !== index);
        setFormTeam({ ...formTeam, members: updatedMembers });
        setMemberJson(memberJson.filter((_, idx) => idx !== index));
        setMemberErrors(memberErrors.filter((_, idx) => idx !== index));
    };
    const handleEditTeam = async (team) => {
        setErrorMessage(null);
        try {
            const detailed = await openevolveApi_1.openevolveApi.getTeam(team.name, apiConfig);
            setEditingTeam(team.name);
            setFormTeam(detailed);
            syncMembersFromTeam(detailed);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load team details.");
        }
    };
    const handleDeleteTeam = async (teamName) => {
        if (!confirm(`Delete team "${teamName}"? This cannot be undone.`)) {
            return;
        }
        try {
            await openevolveApi_1.openevolveApi.deleteTeam(teamName, apiConfig);
            setStatusMessage(`Deleted team ${teamName}.`);
            await loadTeams();
            if (editingTeam === teamName) {
                resetForm();
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to delete team.");
        }
    };
    const handleSaveTeam = async () => {
        setErrorMessage(null);
        setStatusMessage(null);
        if (!formTeam.name.trim()) {
            setErrorMessage("Team name is required.");
            return;
        }
        if (!formTeam.members.length || !formTeam.members[0].model_id) {
            setErrorMessage("At least one model with a model_id is required.");
            return;
        }
        if (memberErrors.some((err) => err)) {
            setErrorMessage("Fix invalid member JSON before saving.");
            return;
        }
        try {
            if (editingTeam) {
                await openevolveApi_1.openevolveApi.updateTeam(editingTeam, formTeam, apiConfig);
                setStatusMessage(`Updated team ${formTeam.name}.`);
            }
            else {
                await openevolveApi_1.openevolveApi.createTeam(formTeam, apiConfig);
                setStatusMessage(`Created team ${formTeam.name}.`);
            }
            await loadTeams();
            resetForm();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to save team.");
        }
    };
    const roles = ["Blue", "Red", "Gold"];
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Team Manager</card_1.CardTitle>
          <card_1.CardDescription>Create, edit, and manage AI teams for the workflow.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key (X-API-Key)</label>
              <input_1.Input value={apiKey} type="password" placeholder="Paste API key for /teams endpoints" onChange={(event) => {
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
              <button_1.Button variant="outline" onClick={loadTeams} disabled={loading}>
                Refresh Teams
              </button_1.Button>
              <button_1.Button variant="secondary" onClick={resetForm}>
                New Team
              </button_1.Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
            <card_1.Card className="h-full">
              <card_1.CardHeader>
                <card_1.CardTitle className="text-base">Existing Teams</card_1.CardTitle>
                <card_1.CardDescription>Click a team to edit.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-3">
                {teams.length === 0 && (<div className="text-sm text-muted-foreground">No teams yet.</div>)}
                {teams.map((team) => (<div key={team.name} className={`rounded border p-3 ${editingTeam === team.name ? "border-primary" : "border-border"}`}>
                    <div className="flex items-center justify-between">
                      <div className="space-y-1">
                        <div className="text-sm font-semibold">{team.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {team.role} · {team.member_count} model(s)
                        </div>
                      </div>
                      <badge_1.Badge variant="outline">standard</badge_1.Badge>
                    </div>
                    <div className="mt-3 flex gap-2">
                      <button_1.Button size="sm" variant="secondary" onClick={() => handleEditTeam(team)}>
                        Edit
                      </button_1.Button>
                      <button_1.Button size="sm" variant="destructive" onClick={() => handleDeleteTeam(team.name)}>
                        Delete
                      </button_1.Button>
                    </div>
                  </div>))}
              </card_1.CardContent>
            </card_1.Card>

            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-base">
                  {editingTeam ? `Edit Team: ${editingTeam}` : "Create New Team"}
                </card_1.CardTitle>
                <card_1.CardDescription>Configure team metadata, prompts, and members.</card_1.CardDescription>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-6">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Team Name</label>
                    <input_1.Input value={formTeam.name} onChange={(event) => setFormTeam({ ...formTeam, name: event.target.value })} placeholder="Team name"/>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Role</label>
                    <select_1.Select value={formTeam.role} onValueChange={(value) => setFormTeam({ ...formTeam, role: value })}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue />
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {roles.map((role) => (<select_1.SelectItem key={role} value={role}>
                            {role}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Team Type</label>
                    <select_1.Select value={formTeam.team_type ?? "standard"} onValueChange={(value) => setFormTeam({ ...formTeam, team_type: value })}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue />
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        <select_1.SelectItem value="standard">Standard</select_1.SelectItem>
                        <select_1.SelectItem value="swarm">Swarm</select_1.SelectItem>
                        <select_1.SelectItem value="sovereign">Sovereign</select_1.SelectItem>
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Sub-role (optional)</label>
                    <input_1.Input value={formTeam.sub_role ?? ""} onChange={(event) => setFormTeam({ ...formTeam, sub_role: event.target.value })} placeholder="Planner, Solver, Patcher, Assembler"/>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Description</label>
                  <textarea_1.Textarea value={formTeam.description ?? ""} onChange={(event) => setFormTeam({ ...formTeam, description: event.target.value })} placeholder="Describe this team"/>
                </div>

                <tabs_1.Tabs defaultValue="prompts">
                  <tabs_1.TabsList className="grid w-full grid-cols-2">
                    <tabs_1.TabsTrigger value="prompts">Prompts</tabs_1.TabsTrigger>
                    <tabs_1.TabsTrigger value="members">Members</tabs_1.TabsTrigger>
                  </tabs_1.TabsList>
                  <tabs_1.TabsContent value="prompts" className="space-y-6 pt-4">
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Content Analysis Prompts</h4>
                      <textarea_1.Textarea value={formTeam.content_analysis_system_prompt ?? ""} onChange={(event) => setFormTeam({ ...formTeam, content_analysis_system_prompt: event.target.value })} placeholder="System prompt"/>
                      <textarea_1.Textarea value={formTeam.content_analysis_user_prompt_template ?? ""} onChange={(event) => setFormTeam({ ...formTeam, content_analysis_user_prompt_template: event.target.value })} placeholder="User prompt template" rows={6}/>
                    </div>
                    <separator_1.Separator />
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Decomposition Prompts</h4>
                      <textarea_1.Textarea value={formTeam.decomposition_system_prompt ?? ""} onChange={(event) => setFormTeam({ ...formTeam, decomposition_system_prompt: event.target.value })} placeholder="System prompt"/>
                      <textarea_1.Textarea value={formTeam.decomposition_user_prompt_template ?? ""} onChange={(event) => setFormTeam({ ...formTeam, decomposition_user_prompt_template: event.target.value })} placeholder="User prompt template" rows={6}/>
                    </div>
                    <separator_1.Separator />
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Solver Prompts</h4>
                      <textarea_1.Textarea value={formTeam.solver_system_prompt ?? ""} onChange={(event) => setFormTeam({ ...formTeam, solver_system_prompt: event.target.value })} placeholder="System prompt"/>
                      <textarea_1.Textarea value={formTeam.solver_user_prompt_template ?? ""} onChange={(event) => setFormTeam({ ...formTeam, solver_user_prompt_template: event.target.value })} placeholder="User prompt template" rows={5}/>
                    </div>
                    <separator_1.Separator />
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Patcher Prompts</h4>
                      <textarea_1.Textarea value={formTeam.patcher_system_prompt ?? ""} onChange={(event) => setFormTeam({ ...formTeam, patcher_system_prompt: event.target.value })} placeholder="System prompt"/>
                      <textarea_1.Textarea value={formTeam.patcher_user_prompt_template ?? ""} onChange={(event) => setFormTeam({ ...formTeam, patcher_user_prompt_template: event.target.value })} placeholder="User prompt template" rows={5}/>
                    </div>
                    <separator_1.Separator />
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Assembler Prompts</h4>
                      <textarea_1.Textarea value={formTeam.assembler_system_prompt ?? ""} onChange={(event) => setFormTeam({ ...formTeam, assembler_system_prompt: event.target.value })} placeholder="System prompt"/>
                      <textarea_1.Textarea value={formTeam.assembler_user_prompt_template ?? ""} onChange={(event) => setFormTeam({ ...formTeam, assembler_user_prompt_template: event.target.value })} placeholder="User prompt template" rows={5}/>
                    </div>
                    <separator_1.Separator />
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Red Team Prompts</h4>
                      <textarea_1.Textarea value={formTeam.red_team_system_prompt ?? ""} onChange={(event) => setFormTeam({ ...formTeam, red_team_system_prompt: event.target.value })} placeholder="System prompt"/>
                      <textarea_1.Textarea value={formTeam.red_team_user_prompt_template ?? ""} onChange={(event) => setFormTeam({ ...formTeam, red_team_user_prompt_template: event.target.value })} placeholder="User prompt template" rows={5}/>
                    </div>
                    <separator_1.Separator />
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Gold Team Prompts</h4>
                      <textarea_1.Textarea value={formTeam.gold_team_system_prompt ?? ""} onChange={(event) => setFormTeam({ ...formTeam, gold_team_system_prompt: event.target.value })} placeholder="System prompt"/>
                      <textarea_1.Textarea value={formTeam.gold_team_user_prompt_template ?? ""} onChange={(event) => setFormTeam({ ...formTeam, gold_team_user_prompt_template: event.target.value })} placeholder="User prompt template" rows={5}/>
                    </div>
                  </tabs_1.TabsContent>
                  <tabs_1.TabsContent value="members" className="space-y-4 pt-4">
                    {formTeam.members.map((member, index) => (<card_1.Card key={index}>
                        <card_1.CardHeader className="flex flex-row items-center justify-between">
                          <div>
                            <card_1.CardTitle className="text-sm">Model {index + 1}</card_1.CardTitle>
                            <card_1.CardDescription>{member.model_id || "New model"}</card_1.CardDescription>
                          </div>
                          <button_1.Button variant="destructive" size="sm" onClick={() => handleRemoveMember(index)} disabled={formTeam.members.length === 1}>
                            Remove
                          </button_1.Button>
                        </card_1.CardHeader>
                        <card_1.CardContent className="space-y-3">
                          <div className="grid gap-3 md:grid-cols-2">
                            <div className="space-y-2">
                              <label className="text-sm font-medium">Model ID</label>
                              <input_1.Input value={member.model_id} onChange={(event) => {
                const updated = [...formTeam.members];
                updated[index] = { ...updated[index], model_id: event.target.value };
                setFormTeam({ ...formTeam, members: updated });
                setMemberJson((prev) => {
                    const next = [...prev];
                    next[index] = JSON.stringify(updated[index], null, 2);
                    return next;
                });
            }}/>
                            </div>
                            <div className="space-y-2">
                              <label className="text-sm font-medium">API Key</label>
                              <input_1.Input type="password" value={member.api_key} onChange={(event) => {
                const updated = [...formTeam.members];
                updated[index] = { ...updated[index], api_key: event.target.value };
                setFormTeam({ ...formTeam, members: updated });
                setMemberJson((prev) => {
                    const next = [...prev];
                    next[index] = JSON.stringify(updated[index], null, 2);
                    return next;
                });
            }}/>
                            </div>
                            <div className="space-y-2">
                              <label className="text-sm font-medium">API Base</label>
                              <input_1.Input value={member.api_base ?? ""} onChange={(event) => {
                const updated = [...formTeam.members];
                updated[index] = { ...updated[index], api_base: event.target.value };
                setFormTeam({ ...formTeam, members: updated });
                setMemberJson((prev) => {
                    const next = [...prev];
                    next[index] = JSON.stringify(updated[index], null, 2);
                    return next;
                });
            }}/>
                            </div>
                            <div className="space-y-2">
                              <label className="text-sm font-medium">Max Tokens</label>
                              <input_1.Input type="number" value={member.max_tokens ?? 0} onChange={(event) => {
                const updated = [...formTeam.members];
                updated[index] = {
                    ...updated[index],
                    max_tokens: Number(event.target.value) || 0,
                };
                setFormTeam({ ...formTeam, members: updated });
                setMemberJson((prev) => {
                    const next = [...prev];
                    next[index] = JSON.stringify(updated[index], null, 2);
                    return next;
                });
            }}/>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <label className="text-sm font-medium">Full Model Config (JSON)</label>
                            <textarea_1.Textarea value={memberJson[index] ?? ""} onChange={(event) => handleMemberJsonChange(index, event.target.value)} rows={10}/>
                            {memberErrors[index] ? (<div className="text-xs text-red-500">{memberErrors[index]}</div>) : null}
                          </div>
                        </card_1.CardContent>
                      </card_1.Card>))}

                    <button_1.Button variant="secondary" onClick={handleAddMember}>
                      Add Model
                    </button_1.Button>
                  </tabs_1.TabsContent>
                </tabs_1.Tabs>

                <div className="flex justify-end gap-2">
                  <button_1.Button variant="outline" onClick={resetForm}>
                    Reset
                  </button_1.Button>
                  <button_1.Button onClick={handleSaveTeam}>
                    {editingTeam ? "Update Team" : "Create Team"}
                  </button_1.Button>
                </div>
              </card_1.CardContent>
            </card_1.Card>
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.TeamManagerTab = TeamManagerTab;
//# sourceMappingURL=TeamManagerTab.js.map