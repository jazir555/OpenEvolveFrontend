import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { openevolveApi } from "@/lib/openevolveApi";
import type {
  WorkflowSummary,
  WorkflowPlanResponse,
  WorkflowDecompositionPlan,
  WorkflowSubProblem,
  TeamSummary,
  GauntletSummary,
} from "@/lib/types";

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

const parseList = (value: string): string[] =>
  value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

const joinList = (values?: string[] | null) => (values && values.length ? values.join(", ") : "");

const readApiKey = () => {
  try {
    return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
  } catch {
    return "";
  }
};

const toJson = (value: unknown) => JSON.stringify(value ?? {}, null, 2);

export const DecompositionReviewTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(readApiKey);
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [teams, setTeams] = useState<TeamSummary[]>([]);
  const [gauntlets, setGauntlets] = useState<GauntletSummary[]>([]);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>("");
  const [planResponse, setPlanResponse] = useState<WorkflowPlanResponse | null>(null);
  const [planDraft, setPlanDraft] = useState<WorkflowDecompositionPlan | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [jsonDrafts, setJsonDrafts] = useState<Record<string, string>>({});
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refreshBaseData = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const [workflowRes, teamRes, gauntletRes] = await Promise.all([
        openevolveApi.listWorkflows(apiConfig),
        openevolveApi.listTeams(apiConfig),
        openevolveApi.listGauntlets(apiConfig),
      ]);
      setWorkflows(workflowRes.workflows ?? []);
      setTeams(teamRes.teams ?? []);
      setGauntlets(gauntletRes.gauntlets ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load base data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshBaseData();
  }, [apiConfig.apiKey]);

  const loadPlan = async (workflowId: string) => {
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const response = await openevolveApi.getWorkflowPlan(workflowId, apiConfig);
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
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load decomposition plan.");
    }
  };

  const updatePlan = (updates: Partial<WorkflowDecompositionPlan>) => {
    setPlanDraft((prev) => (prev ? { ...prev, ...updates } : prev));
  };

  const updateSubProblem = (id: string, updates: Partial<WorkflowSubProblem>) => {
    setPlanDraft((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        sub_problems: prev.sub_problems.map((sp) => (sp.id === id ? { ...sp, ...updates } : sp)),
      };
    });
  };

  const toggleSelection = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const applyBatch = (updates: Partial<WorkflowSubProblem>) => {
    if (!planDraft) return;
    if (!selectedIds.size) {
      setErrorMessage("Select at least one sub-problem for batch operations.");
      return;
    }
    setErrorMessage(null);
    setPlanDraft({
      ...planDraft,
      sub_problems: planDraft.sub_problems.map((sp) =>
        selectedIds.has(sp.id) ? { ...sp, ...updates } : sp,
      ),
    });
  };

  const handleSavePlan = async () => {
    if (!planDraft || !selectedWorkflowId) return;
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
      await openevolveApi.updateWorkflowPlan(selectedWorkflowId, payload, apiConfig);
      setStatusMessage("Decomposition plan updated.");
      await loadPlan(selectedWorkflowId);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to update decomposition plan.");
    }
  };

  const updateJsonDraft = (field: string, value: string) => {
    setJsonDrafts((prev) => ({ ...prev, [field]: value }));
    try {
      const parsed = JSON.parse(value || "{}");
      updatePlan({ [field]: parsed } as Partial<WorkflowDecompositionPlan>);
    } catch {
      // Keep draft text until valid
    }
  };

  const blueTeams = teams.filter((team) => team.role === "Blue");
  const redGauntlets = gauntlets;
  const goldGauntlets = gauntlets;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Decomposition Review</CardTitle>
          <CardDescription>Edit decomposition plans with batch operations.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
                value={apiKey}
                type="password"
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
            <Button variant="outline" onClick={refreshBaseData} disabled={loading}>
              Refresh
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-[280px_1fr]">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Workflows</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {workflows.length === 0 && (
                  <div className="text-sm text-muted-foreground">No workflows available.</div>
                )}
                <select
                  className="w-full rounded border border-input bg-background px-3 py-2 text-sm"
                  value={selectedWorkflowId}
                  onChange={(event) => setSelectedWorkflowId(event.target.value)}
                >
                  <option value="">Select workflow</option>
                  {workflows.map((workflow) => (
                    <option key={workflow.workflow_id} value={workflow.workflow_id}>
                      {workflow.workflow_id}
                    </option>
                  ))}
                </select>
                <Button
                  className="w-full"
                  variant="secondary"
                  onClick={() => {
                    if (selectedWorkflowId) {
                      loadPlan(selectedWorkflowId);
                    } else {
                      setErrorMessage("Select a workflow first.");
                    }
                  }}
                >
                  Load Plan
                </Button>
              </CardContent>
            </Card>

            {planDraft ? (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Plan Summary</CardTitle>
                  <CardDescription>{planDraft.problem_statement}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline">
                      Sub-problems: {planDraft.sub_problems.length}
                    </Badge>
                    {planResponse?.dependency_graph?.execution_order?.length ? (
                      <Badge variant="secondary">
                        Execution Order: {planResponse.dependency_graph.execution_order.length}
                      </Badge>
                    ) : null}
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="space-y-1">
                      <Label>Max Refinement Loops</Label>
                      <Input
                        type="number"
                        value={planDraft.max_refinement_loops ?? 3}
                        onChange={(event) =>
                          updatePlan({ max_refinement_loops: Number(event.target.value) })
                        }
                      />
                    </div>
                    <div className="space-y-1">
                      <Label>Auto Approval</Label>
                      <Switch
                        checked={Boolean(planDraft.auto_approval_enabled)}
                        onCheckedChange={(value) => updatePlan({ auto_approval_enabled: value })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label>Parallel Processing</Label>
                      <Switch
                        checked={Boolean(planDraft.parallel_processing_enabled)}
                        onCheckedChange={(value) =>
                          updatePlan({ parallel_processing_enabled: value })
                        }
                      />
                    </div>
                  </div>

                  <Separator />

                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-1">
                      <Label>MDAP Enabled</Label>
                      <Switch
                        checked={Boolean(planDraft.mdap_enabled)}
                        onCheckedChange={(value) => updatePlan({ mdap_enabled: value })}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label>MAKER Enabled</Label>
                      <Switch
                        checked={Boolean(planDraft.maker_enabled)}
                        onCheckedChange={(value) => updatePlan({ maker_enabled: value })}
                      />
                    </div>
                  </div>

                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-1">
                      <Label>Auto Approval Criteria (JSON)</Label>
                      <Textarea
                        value={jsonDrafts.auto_approval_criteria ?? ""}
                        onChange={(event) =>
                          updateJsonDraft("auto_approval_criteria", event.target.value)
                        }
                        rows={6}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label>MDAP Config (JSON)</Label>
                      <Textarea
                        value={jsonDrafts.mdap_config ?? ""}
                        onChange={(event) => updateJsonDraft("mdap_config", event.target.value)}
                        rows={6}
                      />
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-1">
                      <Label>MAKER Config (JSON)</Label>
                      <Textarea
                        value={jsonDrafts.maker_config ?? ""}
                        onChange={(event) => updateJsonDraft("maker_config", event.target.value)}
                        rows={6}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label>Resource Limits (JSON)</Label>
                      <Textarea
                        value={jsonDrafts.resource_limits ?? ""}
                        onChange={(event) => updateJsonDraft("resource_limits", event.target.value)}
                        rows={6}
                      />
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-1">
                      <Label>Learning Config (JSON)</Label>
                      <Textarea
                        value={jsonDrafts.learning_config ?? ""}
                        onChange={(event) => updateJsonDraft("learning_config", event.target.value)}
                        rows={6}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label>Metadata (JSON)</Label>
                      <Textarea
                        value={jsonDrafts.metadata ?? ""}
                        onChange={(event) => updateJsonDraft("metadata", event.target.value)}
                        rows={6}
                      />
                    </div>
                  </div>

                  <div className="flex justify-end">
                    <Button onClick={handleSavePlan}>Save Plan</Button>
                  </div>
                </CardContent>
              </Card>
            ) : null}
          </div>
        </CardContent>
      </Card>

      {planDraft ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Batch Operations</CardTitle>
            <CardDescription>Apply changes to selected sub-problems.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-sm text-muted-foreground">
              Selected sub-problems: {selectedIds.size}
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label>Assign Solver Team</Label>
                <Select
                  onValueChange={(value) => applyBatch({ solver_team_name: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select team" />
                  </SelectTrigger>
                  <SelectContent>
                    {blueTeams.map((team) => (
                      <SelectItem key={team.name} value={team.name}>
                        {team.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Assign Patcher Team</Label>
                <Select onValueChange={(value) => applyBatch({ patcher_team_name: value })}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select team" />
                  </SelectTrigger>
                  <SelectContent>
                    {blueTeams.map((team) => (
                      <SelectItem key={team.name} value={team.name}>
                        {team.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Assign Red Gauntlet</Label>
                <Select onValueChange={(value) => applyBatch({ red_team_gauntlet_name: value })}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select gauntlet" />
                  </SelectTrigger>
                  <SelectContent>
                    {redGauntlets.map((gauntlet) => (
                      <SelectItem key={gauntlet.name} value={gauntlet.name}>
                        {gauntlet.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Assign Gold Gauntlet</Label>
                <Select onValueChange={(value) => applyBatch({ gold_team_gauntlet_name: value })}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select gauntlet" />
                  </SelectTrigger>
                  <SelectContent>
                    {goldGauntlets.map((gauntlet) => (
                      <SelectItem key={gauntlet.name} value={gauntlet.name}>
                        {gauntlet.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Evolution Mode</Label>
                <Input
                  placeholder="standard"
                  onBlur={(event) => {
                    if (event.target.value) {
                      applyBatch({ ai_suggested_evolution_mode: event.target.value });
                      event.target.value = "";
                    }
                  }}
                />
              </div>
              <div className="space-y-2">
                <Label>Complexity Score</Label>
                <Input
                  type="number"
                  placeholder="5"
                  onBlur={(event) => {
                    if (event.target.value) {
                      applyBatch({ ai_suggested_complexity_score: Number(event.target.value) });
                      event.target.value = "";
                    }
                  }}
                />
              </div>
              <div className="space-y-2">
                <Label>Content Type</Label>
                <Select onValueChange={(value) => applyBatch({ content_type: value })}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select content type" />
                  </SelectTrigger>
                  <SelectContent>
                    {CONTENT_TYPES.map((type) => (
                      <SelectItem key={type} value={type}>
                        {type}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : null}

      {planDraft ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Sub-Problems</CardTitle>
            <CardDescription>Edit individual sub-problem assignments and metadata.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {planDraft.sub_problems.map((sp) => (
              <Card key={sp.id}>
                <CardHeader className="flex flex-row items-center justify-between">
                  <div className="space-y-1">
                    <CardTitle className="text-sm">{sp.id}</CardTitle>
                    <CardDescription>{sp.description}</CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <Checkbox
                      checked={selectedIds.has(sp.id)}
                      onCheckedChange={() => toggleSelection(sp.id)}
                    />
                    <Badge variant="secondary">{sp.status ?? "pending"}</Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label>Description</Label>
                    <Textarea
                      value={sp.description}
                      onChange={(event) => updateSubProblem(sp.id, { description: event.target.value })}
                      rows={3}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Dependencies (comma-separated IDs)</Label>
                    <Input
                      value={joinList(sp.dependencies)}
                      onChange={(event) =>
                        updateSubProblem(sp.id, { dependencies: parseList(event.target.value) })
                      }
                    />
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Suggested Evolution Mode</Label>
                      <Input
                        value={sp.ai_suggested_evolution_mode ?? ""}
                        onChange={(event) =>
                          updateSubProblem(sp.id, { ai_suggested_evolution_mode: event.target.value })
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Complexity Score</Label>
                      <Input
                        type="number"
                        value={sp.ai_suggested_complexity_score ?? 0}
                        onChange={(event) =>
                          updateSubProblem(sp.id, {
                            ai_suggested_complexity_score: Number(event.target.value),
                          })
                        }
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Suggested Evaluation Prompt</Label>
                    <Textarea
                      value={sp.ai_suggested_evaluation_prompt ?? ""}
                      onChange={(event) =>
                        updateSubProblem(sp.id, { ai_suggested_evaluation_prompt: event.target.value })
                      }
                      rows={3}
                    />
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Content Type</Label>
                      <Select
                        value={sp.content_type ?? "text_general"}
                        onValueChange={(value) => updateSubProblem(sp.id, { content_type: value })}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select content type" />
                        </SelectTrigger>
                        <SelectContent>
                          {CONTENT_TYPES.map((type) => (
                            <SelectItem key={type} value={type}>
                              {type}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Solver Team</Label>
                      <Select
                        value={sp.solver_team_name ?? ""}
                        onValueChange={(value) => updateSubProblem(sp.id, { solver_team_name: value })}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select team" />
                        </SelectTrigger>
                        <SelectContent>
                          {blueTeams.map((team) => (
                            <SelectItem key={team.name} value={team.name}>
                              {team.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="space-y-2">
                      <Label>Red Gauntlet</Label>
                      <Select
                        value={sp.red_team_gauntlet_name ?? ""}
                        onValueChange={(value) => updateSubProblem(sp.id, { red_team_gauntlet_name: value })}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select gauntlet" />
                        </SelectTrigger>
                        <SelectContent>
                          {redGauntlets.map((gauntlet) => (
                            <SelectItem key={gauntlet.name} value={gauntlet.name}>
                              {gauntlet.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Gold Gauntlet</Label>
                      <Select
                        value={sp.gold_team_gauntlet_name ?? ""}
                        onValueChange={(value) => updateSubProblem(sp.id, { gold_team_gauntlet_name: value })}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select gauntlet" />
                        </SelectTrigger>
                        <SelectContent>
                          {goldGauntlets.map((gauntlet) => (
                            <SelectItem key={gauntlet.name} value={gauntlet.name}>
                              {gauntlet.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Patcher Team</Label>
                      <Select
                        value={sp.patcher_team_name ?? ""}
                        onValueChange={(value) => updateSubProblem(sp.id, { patcher_team_name: value })}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select team" />
                        </SelectTrigger>
                        <SelectContent>
                          {blueTeams.map((team) => (
                            <SelectItem key={team.name} value={team.name}>
                              {team.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Atomic Mode</Label>
                      <Switch
                        checked={Boolean(sp.atomic_mode)}
                        onCheckedChange={(value) => updateSubProblem(sp.id, { atomic_mode: value })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Decomposition Depth</Label>
                      <Input
                        type="number"
                        value={sp.decomposition_depth ?? 0}
                        onChange={(event) =>
                          updateSubProblem(sp.id, { decomposition_depth: Number(event.target.value) })
                        }
                      />
                    </div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Acceptance Criteria</Label>
                      <Textarea
                        value={(sp.acceptance_criteria ?? []).join("\n")}
                        onChange={(event) =>
                          updateSubProblem(sp.id, {
                            acceptance_criteria: event.target.value
                              .split("\n")
                              .map((item) => item.trim())
                              .filter(Boolean),
                          })
                        }
                        rows={3}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Specific Constraints</Label>
                      <Textarea
                        value={(sp.specific_constraints ?? []).join("\n")}
                        onChange={(event) =>
                          updateSubProblem(sp.id, {
                            specific_constraints: event.target.value
                              .split("\n")
                              .map((item) => item.trim())
                              .filter(Boolean),
                          })
                        }
                        rows={3}
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Solution Requirements (JSON)</Label>
                    <Textarea
                      value={toJson(sp.solution_requirements ?? {})}
                      onChange={(event) => {
                        try {
                          updateSubProblem(sp.id, { solution_requirements: JSON.parse(event.target.value) });
                        } catch {
                          // ignore invalid JSON
                        }
                      }}
                      rows={4}
                    />
                  </div>
                </CardContent>
              </Card>
            ))}
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
};
