import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { openevolveApi } from "../../../lib/openevolveApi";
import {
  Team,
  TeamRole,
  TeamSummary,
  createDefaultTeam,
  createDefaultModelConfig,
} from "../../../lib/types";

const promptDefaults = {
  content_analysis_system_prompt:
    "You are a highly skilled content analyzer. Your task is to analyze a problem statement and extract key information, context, and potential challenges. Provide your analysis in a structured JSON format.",
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
  decomposition_system_prompt:
    "You are an expert problem decomposer. Your task is to break down a complex problem into smaller, manageable sub-problems. For each sub-problem, suggest an evolution mode, a complexity score (1-10), and a specific evaluation prompt. Provide the output as a JSON array of sub-problem objects.",
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
  solver_system_prompt:
    "You are a skilled problem solver. Your task is to generate a high-quality solution for the given sub-problem. Provide your response in a clear, concise format.",
  solver_user_prompt_template: `Solve the following sub-problem:

Sub-Problem Description:
---
{sub_problem_description}
---
`,
  patcher_system_prompt:
    "You are an expert solution patcher. Your task is to improve and fix a solution based on critique feedback.",
  patcher_user_prompt_template: `Improve the following solution based on critique feedback:

Original Solution:
---
{solution_content}
---

Critique Feedback:
---
{critique_summary}
---`,
  assembler_system_prompt:
    "You are an expert solution assembler. Your task is to combine sub-problem solutions into a cohesive final output.",
  assembler_user_prompt_template: `Assemble the final solution from the following sub-problem solutions:

Sub-Problem Solutions:
---
{sub_problem_solutions}
---`,
  red_team_system_prompt:
    "You are a Red Team critic. Identify flaws, risks, and weaknesses in the solution.",
  red_team_user_prompt_template: `Critique the following solution:

Solution:
---
{solution_content}
---`,
  gold_team_system_prompt:
    "You are a Gold Team verifier. Verify correctness, completeness, and compliance.",
  gold_team_user_prompt_template: `Verify the following solution against requirements:

Solution:
---
{solution_content}
---`,
};

export const TeamManagerTab: React.FC = () => {
  const [teams, setTeams] = useState<TeamSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [editingTeam, setEditingTeam] = useState<string | null>(null);
  const [formTeam, setFormTeam] = useState<Team>(() => ({
    ...createDefaultTeam(),
    ...promptDefaults,
  }));
  const [memberJson, setMemberJson] = useState<string[]>([]);
  const [memberErrors, setMemberErrors] = useState<string[]>([]);
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });

  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const syncMembersFromTeam = (team: Team) => {
    const jsonValues = team.members.map((member) => JSON.stringify(member, null, 2));
    setMemberJson(jsonValues);
    setMemberErrors(jsonValues.map(() => ""));
  };

  const resetForm = () => {
    const next = { ...createDefaultTeam(), ...promptDefaults };
    setFormTeam(next);
    setEditingTeam(null);
    syncMembersFromTeam(next);
  };

  const loadTeams = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const result = await openevolveApi.listTeams(apiConfig);
      setTeams(result.teams ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load teams.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    syncMembersFromTeam(formTeam);
  }, []);

  useEffect(() => {
    loadTeams();
  }, [apiConfig.apiKey]);

  const handleMemberJsonChange = (index: number, value: string) => {
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
    } catch (error: any) {
      const errors = [...memberErrors];
      errors[index] = error?.message ?? "Invalid JSON";
      setMemberErrors(errors);
    }
  };

  const handleAddMember = () => {
    const updatedMembers = [...formTeam.members, createDefaultModelConfig()];
    setFormTeam({ ...formTeam, members: updatedMembers });
    setMemberJson([...memberJson, JSON.stringify(createDefaultModelConfig(), null, 2)]);
    setMemberErrors([...memberErrors, ""]);
  };

  const handleRemoveMember = (index: number) => {
    const updatedMembers = formTeam.members.filter((_, idx) => idx !== index);
    setFormTeam({ ...formTeam, members: updatedMembers });
    setMemberJson(memberJson.filter((_, idx) => idx !== index));
    setMemberErrors(memberErrors.filter((_, idx) => idx !== index));
  };

  const handleEditTeam = async (team: TeamSummary) => {
    setErrorMessage(null);
    try {
      const detailed = await openevolveApi.getTeam(team.name, apiConfig);
      setEditingTeam(team.name);
      setFormTeam(detailed);
      syncMembersFromTeam(detailed);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load team details.");
    }
  };

  const handleDeleteTeam = async (teamName: string) => {
    if (!confirm(`Delete team "${teamName}"? This cannot be undone.`)) {
      return;
    }
    try {
      await openevolveApi.deleteTeam(teamName, apiConfig);
      setStatusMessage(`Deleted team ${teamName}.`);
      await loadTeams();
      if (editingTeam === teamName) {
        resetForm();
      }
    } catch (error: any) {
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
        await openevolveApi.updateTeam(editingTeam, formTeam, apiConfig);
        setStatusMessage(`Updated team ${formTeam.name}.`);
      } else {
        await openevolveApi.createTeam(formTeam, apiConfig);
        setStatusMessage(`Created team ${formTeam.name}.`);
      }
      await loadTeams();
      resetForm();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to save team.");
    }
  };

  const roles: TeamRole[] = ["Blue", "Red", "Gold"];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Team Manager</CardTitle>
          <CardDescription>Create, edit, and manage AI teams for the workflow.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key (X-API-Key)</label>
              <Input
                value={apiKey}
                type="password"
                placeholder="Paste API key for /teams endpoints"
                onChange={(event) => {
                  const value = event.target.value;
                  setApiKey(value);
                  try {
                    globalThis.localStorage?.setItem("openevolve_api_key", value);
                  } catch {
                    // ignore storage errors
                  }
                }}
              />
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={loadTeams} disabled={loading}>
                Refresh Teams
              </Button>
              <Button variant="secondary" onClick={resetForm}>
                New Team
              </Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="text-base">Existing Teams</CardTitle>
                <CardDescription>Click a team to edit.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {teams.length === 0 && (
                  <div className="text-sm text-muted-foreground">No teams yet.</div>
                )}
                {teams.map((team) => (
                  <div
                    key={team.name}
                    className={`rounded border p-3 ${editingTeam === team.name ? "border-primary" : "border-border"}`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="space-y-1">
                        <div className="text-sm font-semibold">{team.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {team.role} · {team.member_count} model(s)
                        </div>
                      </div>
                      <Badge variant="outline">standard</Badge>
                    </div>
                    <div className="mt-3 flex gap-2">
                      <Button size="sm" variant="secondary" onClick={() => handleEditTeam(team)}>
                        Edit
                      </Button>
                      <Button size="sm" variant="destructive" onClick={() => handleDeleteTeam(team.name)}>
                        Delete
                      </Button>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  {editingTeam ? `Edit Team: ${editingTeam}` : "Create New Team"}
                </CardTitle>
                <CardDescription>Configure team metadata, prompts, and members.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Team Name</label>
                    <Input
                      value={formTeam.name}
                      onChange={(event) => setFormTeam({ ...formTeam, name: event.target.value })}
                      placeholder="Team name"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Role</label>
                    <Select
                      value={formTeam.role}
                      onValueChange={(value) =>
                        setFormTeam({ ...formTeam, role: value as TeamRole })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {roles.map((role) => (
                          <SelectItem key={role} value={role}>
                            {role}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Team Type</label>
                    <Select
                      value={formTeam.team_type ?? "standard"}
                      onValueChange={(value) => setFormTeam({ ...formTeam, team_type: value as Team["team_type"] })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="standard">Standard</SelectItem>
                        <SelectItem value="swarm">Swarm</SelectItem>
                        <SelectItem value="sovereign">Sovereign</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Sub-role (optional)</label>
                    <Input
                      value={formTeam.sub_role ?? ""}
                      onChange={(event) => setFormTeam({ ...formTeam, sub_role: event.target.value })}
                      placeholder="Planner, Solver, Patcher, Assembler"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">Description</label>
                  <Textarea
                    value={formTeam.description ?? ""}
                    onChange={(event) => setFormTeam({ ...formTeam, description: event.target.value })}
                    placeholder="Describe this team"
                  />
                </div>

                <Tabs defaultValue="prompts">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="prompts">Prompts</TabsTrigger>
                    <TabsTrigger value="members">Members</TabsTrigger>
                  </TabsList>
                  <TabsContent value="prompts" className="space-y-6 pt-4">
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Content Analysis Prompts</h4>
                      <Textarea
                        value={formTeam.content_analysis_system_prompt ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, content_analysis_system_prompt: event.target.value })
                        }
                        placeholder="System prompt"
                      />
                      <Textarea
                        value={formTeam.content_analysis_user_prompt_template ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, content_analysis_user_prompt_template: event.target.value })
                        }
                        placeholder="User prompt template"
                        rows={6}
                      />
                    </div>
                    <Separator />
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Decomposition Prompts</h4>
                      <Textarea
                        value={formTeam.decomposition_system_prompt ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, decomposition_system_prompt: event.target.value })
                        }
                        placeholder="System prompt"
                      />
                      <Textarea
                        value={formTeam.decomposition_user_prompt_template ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, decomposition_user_prompt_template: event.target.value })
                        }
                        placeholder="User prompt template"
                        rows={6}
                      />
                    </div>
                    <Separator />
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Solver Prompts</h4>
                      <Textarea
                        value={formTeam.solver_system_prompt ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, solver_system_prompt: event.target.value })
                        }
                        placeholder="System prompt"
                      />
                      <Textarea
                        value={formTeam.solver_user_prompt_template ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, solver_user_prompt_template: event.target.value })
                        }
                        placeholder="User prompt template"
                        rows={5}
                      />
                    </div>
                    <Separator />
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Patcher Prompts</h4>
                      <Textarea
                        value={formTeam.patcher_system_prompt ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, patcher_system_prompt: event.target.value })
                        }
                        placeholder="System prompt"
                      />
                      <Textarea
                        value={formTeam.patcher_user_prompt_template ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, patcher_user_prompt_template: event.target.value })
                        }
                        placeholder="User prompt template"
                        rows={5}
                      />
                    </div>
                    <Separator />
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Assembler Prompts</h4>
                      <Textarea
                        value={formTeam.assembler_system_prompt ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, assembler_system_prompt: event.target.value })
                        }
                        placeholder="System prompt"
                      />
                      <Textarea
                        value={formTeam.assembler_user_prompt_template ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, assembler_user_prompt_template: event.target.value })
                        }
                        placeholder="User prompt template"
                        rows={5}
                      />
                    </div>
                    <Separator />
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Red Team Prompts</h4>
                      <Textarea
                        value={formTeam.red_team_system_prompt ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, red_team_system_prompt: event.target.value })
                        }
                        placeholder="System prompt"
                      />
                      <Textarea
                        value={formTeam.red_team_user_prompt_template ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, red_team_user_prompt_template: event.target.value })
                        }
                        placeholder="User prompt template"
                        rows={5}
                      />
                    </div>
                    <Separator />
                    <div className="space-y-3">
                      <h4 className="text-sm font-semibold">Gold Team Prompts</h4>
                      <Textarea
                        value={formTeam.gold_team_system_prompt ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, gold_team_system_prompt: event.target.value })
                        }
                        placeholder="System prompt"
                      />
                      <Textarea
                        value={formTeam.gold_team_user_prompt_template ?? ""}
                        onChange={(event) =>
                          setFormTeam({ ...formTeam, gold_team_user_prompt_template: event.target.value })
                        }
                        placeholder="User prompt template"
                        rows={5}
                      />
                    </div>
                  </TabsContent>
                  <TabsContent value="members" className="space-y-4 pt-4">
                    {formTeam.members.map((member, index) => (
                      <Card key={index}>
                        <CardHeader className="flex flex-row items-center justify-between">
                          <div>
                            <CardTitle className="text-sm">Model {index + 1}</CardTitle>
                            <CardDescription>{member.model_id || "New model"}</CardDescription>
                          </div>
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={() => handleRemoveMember(index)}
                            disabled={formTeam.members.length === 1}
                          >
                            Remove
                          </Button>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          <div className="grid gap-3 md:grid-cols-2">
                            <div className="space-y-2">
                              <label className="text-sm font-medium">Model ID</label>
                              <Input
                                value={member.model_id}
                                onChange={(event) => {
                                  const updated = [...formTeam.members];
                                  updated[index] = { ...updated[index], model_id: event.target.value };
                                  setFormTeam({ ...formTeam, members: updated });
                                  setMemberJson((prev) => {
                                    const next = [...prev];
                                    next[index] = JSON.stringify(updated[index], null, 2);
                                    return next;
                                  });
                                }}
                              />
                            </div>
                            <div className="space-y-2">
                              <label className="text-sm font-medium">API Key</label>
                              <Input
                                type="password"
                                value={member.api_key}
                                onChange={(event) => {
                                  const updated = [...formTeam.members];
                                  updated[index] = { ...updated[index], api_key: event.target.value };
                                  setFormTeam({ ...formTeam, members: updated });
                                  setMemberJson((prev) => {
                                    const next = [...prev];
                                    next[index] = JSON.stringify(updated[index], null, 2);
                                    return next;
                                  });
                                }}
                              />
                            </div>
                            <div className="space-y-2">
                              <label className="text-sm font-medium">API Base</label>
                              <Input
                                value={member.api_base ?? ""}
                                onChange={(event) => {
                                  const updated = [...formTeam.members];
                                  updated[index] = { ...updated[index], api_base: event.target.value };
                                  setFormTeam({ ...formTeam, members: updated });
                                  setMemberJson((prev) => {
                                    const next = [...prev];
                                    next[index] = JSON.stringify(updated[index], null, 2);
                                    return next;
                                  });
                                }}
                              />
                            </div>
                            <div className="space-y-2">
                              <label className="text-sm font-medium">Max Tokens</label>
                              <Input
                                type="number"
                                value={member.max_tokens ?? 0}
                                onChange={(event) => {
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
                                }}
                              />
                            </div>
                          </div>

                          <div className="space-y-2">
                            <label className="text-sm font-medium">Full Model Config (JSON)</label>
                            <Textarea
                              value={memberJson[index] ?? ""}
                              onChange={(event) => handleMemberJsonChange(index, event.target.value)}
                              rows={10}
                            />
                            {memberErrors[index] ? (
                              <div className="text-xs text-red-500">{memberErrors[index]}</div>
                            ) : null}
                          </div>
                        </CardContent>
                      </Card>
                    ))}

                    <Button variant="secondary" onClick={handleAddMember}>
                      Add Model
                    </Button>
                  </TabsContent>
                </Tabs>

                <div className="flex justify-end gap-2">
                  <Button variant="outline" onClick={resetForm}>
                    Reset
                  </Button>
                  <Button onClick={handleSaveTeam}>
                    {editingTeam ? "Update Team" : "Create Team"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
