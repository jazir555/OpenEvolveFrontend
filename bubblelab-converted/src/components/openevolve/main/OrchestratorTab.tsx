import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { openevolveApi } from "@/lib/openevolveApi";
import type {
  TeamSummary,
  GauntletSummary,
  WorkflowSummary,
  WorkflowDetail,
  WorkflowResults,
  WorkflowCreateRequest,
} from "@/lib/types";

const createDefaultWorkflowRequest = (): WorkflowCreateRequest => ({
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

export const OrchestratorTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [teams, setTeams] = useState<TeamSummary[]>([]);
  const [gauntlets, setGauntlets] = useState<GauntletSummary[]>([]);
  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null);
  const [workflowDetail, setWorkflowDetail] = useState<WorkflowDetail | null>(null);
  const [workflowResults, setWorkflowResults] = useState<WorkflowResults | null>(null);
  const [form, setForm] = useState<WorkflowCreateRequest>(createDefaultWorkflowRequest());
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const loadTeams = async () => {
    const result = await openevolveApi.listTeams(apiConfig);
    setTeams(result.teams ?? []);
  };

  const loadGauntlets = async () => {
    const result = await openevolveApi.listGauntlets(apiConfig);
    setGauntlets(result.gauntlets ?? []);
  };

  const loadWorkflows = async () => {
    const result = await openevolveApi.listWorkflows(apiConfig);
    setWorkflows(result.workflows ?? []);
  };

  const loadWorkflowDetail = async (workflowId: string) => {
    const detail = await openevolveApi.getWorkflow(workflowId, apiConfig);
    setWorkflowDetail(detail);
  };

  const loadWorkflowResults = async (workflowId: string) => {
    const results = await openevolveApi.getWorkflowResults(workflowId, apiConfig);
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
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to refresh data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshAll();
  }, [apiConfig.apiKey]);

  const handleSelectWorkflow = async (workflowId: string) => {
    setSelectedWorkflowId(workflowId);
    setWorkflowResults(null);
    setErrorMessage(null);
    try {
      await loadWorkflowDetail(workflowId);
    } catch (error: any) {
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
    if (
      !form.content_analyzer_team ||
      !form.planner_team ||
      !form.solver_team ||
      !form.patcher_team ||
      !form.assembler_team
    ) {
      setErrorMessage("All team selections are required.");
      return;
    }
    if (
      !form.sub_problem_red_gauntlet ||
      !form.sub_problem_gold_gauntlet ||
      !form.final_red_gauntlet ||
      !form.final_gold_gauntlet ||
      !form.solver_generation_gauntlet
    ) {
      setErrorMessage("All gauntlet selections are required.");
      return;
    }

    try {
      const result = await openevolveApi.createWorkflow(form, apiConfig);
      setStatusMessage(`Workflow ${result.workflow_id} created.`);
      setForm(createDefaultWorkflowRequest());
      await loadWorkflows();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to create workflow.");
    }
  };

  const updateForm = (field: keyof WorkflowCreateRequest, value: any) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const blueTeams = teams.filter((team) => team.role === "Blue");
  const redTeams = teams.filter((team) => team.role === "Red");
  const goldTeams = teams.filter((team) => team.role === "Gold");

  const gauntletOptions = gauntlets;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Workflow Orchestrator</CardTitle>
          <CardDescription>Create, monitor, and manage sovereign workflows.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key (X-API-Key)</label>
              <Input
                value={apiKey}
                type="password"
                placeholder="Paste API key for workflow endpoints"
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
              <Button variant="outline" onClick={refreshAll} disabled={loading}>
                Refresh
              </Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid grid-cols-1 gap-6 xl:grid-cols-[420px_1fr]">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Create Workflow</CardTitle>
                <CardDescription>Configure the full sovereign gauntlet pipeline.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Problem Statement</label>
                  <Textarea
                    value={form.problem_statement}
                    onChange={(event) => updateForm("problem_statement", event.target.value)}
                    rows={5}
                    placeholder="Describe the problem to solve"
                  />
                </div>

                <Separator />
                <div className="space-y-3">
                  <h4 className="text-sm font-semibold">Teams</h4>
                  <div className="grid gap-3">
                    <Select
                      value={form.content_analyzer_team}
                      onValueChange={(value) => updateForm("content_analyzer_team", value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Content Analyzer Team" />
                      </SelectTrigger>
                      <SelectContent>
                        {blueTeams.map((team) => (
                          <SelectItem key={team.name} value={team.name}>
                            {team.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Select
                      value={form.planner_team}
                      onValueChange={(value) => updateForm("planner_team", value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Planner Team" />
                      </SelectTrigger>
                      <SelectContent>
                        {blueTeams.map((team) => (
                          <SelectItem key={team.name} value={team.name}>
                            {team.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Select
                      value={form.solver_team}
                      onValueChange={(value) => updateForm("solver_team", value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Solver Team" />
                      </SelectTrigger>
                      <SelectContent>
                        {blueTeams.map((team) => (
                          <SelectItem key={team.name} value={team.name}>
                            {team.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Select
                      value={form.patcher_team}
                      onValueChange={(value) => updateForm("patcher_team", value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Patcher Team" />
                      </SelectTrigger>
                      <SelectContent>
                        {blueTeams.map((team) => (
                          <SelectItem key={team.name} value={team.name}>
                            {team.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Select
                      value={form.assembler_team}
                      onValueChange={(value) => updateForm("assembler_team", value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Assembler Team" />
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

                <Separator />
                <div className="space-y-3">
                  <h4 className="text-sm font-semibold">Gauntlets</h4>
                  <div className="grid gap-3">
                    <Select
                      value={form.sub_problem_red_gauntlet}
                      onValueChange={(value) => updateForm("sub_problem_red_gauntlet", value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Sub-problem Red Gauntlet" />
                      </SelectTrigger>
                      <SelectContent>
                        {gauntletOptions.map((gauntlet) => (
                          <SelectItem key={gauntlet.name} value={gauntlet.name}>
                            {gauntlet.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Select
                      value={form.sub_problem_gold_gauntlet}
                      onValueChange={(value) => updateForm("sub_problem_gold_gauntlet", value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Sub-problem Gold Gauntlet" />
                      </SelectTrigger>
                      <SelectContent>
                        {gauntletOptions.map((gauntlet) => (
                          <SelectItem key={gauntlet.name} value={gauntlet.name}>
                            {gauntlet.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Select
                      value={form.final_red_gauntlet}
                      onValueChange={(value) => updateForm("final_red_gauntlet", value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Final Red Gauntlet" />
                      </SelectTrigger>
                      <SelectContent>
                        {gauntletOptions.map((gauntlet) => (
                          <SelectItem key={gauntlet.name} value={gauntlet.name}>
                            {gauntlet.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Select
                      value={form.final_gold_gauntlet}
                      onValueChange={(value) => updateForm("final_gold_gauntlet", value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Final Gold Gauntlet" />
                      </SelectTrigger>
                      <SelectContent>
                        {gauntletOptions.map((gauntlet) => (
                          <SelectItem key={gauntlet.name} value={gauntlet.name}>
                            {gauntlet.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Select
                      value={form.solver_generation_gauntlet}
                      onValueChange={(value) => updateForm("solver_generation_gauntlet", value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Solver Generation Gauntlet" />
                      </SelectTrigger>
                      <SelectContent>
                        {gauntletOptions.map((gauntlet) => (
                          <SelectItem key={gauntlet.name} value={gauntlet.name}>
                            {gauntlet.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <Separator />
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Max Refinement Loops</label>
                    <Input
                      type="number"
                      value={form.max_refinement_loops ?? 3}
                      onChange={(event) =>
                        updateForm("max_refinement_loops", Number(event.target.value) || 1)
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">MDAP Enabled</label>
                    <Select
                      value={form.mdap_enabled ? "enabled" : "disabled"}
                      onValueChange={(value) => updateForm("mdap_enabled", value === "enabled")}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="enabled">Enabled</SelectItem>
                        <SelectItem value="disabled">Disabled</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">MAKER Enabled</label>
                    <Select
                      value={form.maker_enabled ? "enabled" : "disabled"}
                      onValueChange={(value) => updateForm("maker_enabled", value === "enabled")}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="enabled">Enabled</SelectItem>
                        <SelectItem value="disabled">Disabled</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">MDAP Config (JSON)</label>
                  <Textarea
                    value={JSON.stringify(form.mdap_config ?? {}, null, 2)}
                    onChange={(event) => {
                      try {
                        updateForm("mdap_config", JSON.parse(event.target.value));
                      } catch {
                        // ignore invalid JSON
                      }
                    }}
                    rows={6}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">MAKER Config (JSON)</label>
                  <Textarea
                    value={JSON.stringify(form.maker_config ?? {}, null, 2)}
                    onChange={(event) => {
                      try {
                        updateForm("maker_config", JSON.parse(event.target.value));
                      } catch {
                        // ignore invalid JSON
                      }
                    }}
                    rows={6}
                  />
                </div>

                <div className="flex justify-end">
                  <Button onClick={handleCreateWorkflow}>Create Workflow</Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Active Workflows</CardTitle>
                <CardDescription>Monitor progress and control execution.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {workflows.length === 0 && (
                  <div className="text-sm text-muted-foreground">No workflows found.</div>
                )}
                {workflows.map((workflow) => (
                  <div
                    key={workflow.workflow_id}
                    className={`rounded border p-3 ${
                      selectedWorkflowId === workflow.workflow_id ? "border-primary" : "border-border"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="space-y-1">
                        <div className="text-sm font-semibold">{workflow.workflow_id}</div>
                        <div className="text-xs text-muted-foreground">
                          {workflow.current_stage} · {Math.round(workflow.progress * 100)}%
                        </div>
                      </div>
                      <Badge variant="outline">{workflow.status}</Badge>
                    </div>
                    <div className="mt-3 flex gap-2">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => handleSelectWorkflow(workflow.workflow_id)}
                      >
                        View
                      </Button>
                    </div>
                  </div>
                ))}

                {workflowDetail ? (
                  <div className="space-y-3 rounded border p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-sm font-semibold">Workflow Detail</div>
                        <div className="text-xs text-muted-foreground">
                          {workflowDetail.workflow_id}
                        </div>
                      </div>
                      <Badge variant="outline">{workflowDetail.status}</Badge>
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
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={async () => {
                          if (!workflowDetail) return;
                          try {
                            await openevolveApi.pauseWorkflow(workflowDetail.workflow_id, apiConfig);
                            await loadWorkflowDetail(workflowDetail.workflow_id);
                            await loadWorkflows();
                          } catch (error: any) {
                            setErrorMessage(error?.message ?? "Failed to pause workflow.");
                          }
                        }}
                      >
                        Pause
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={async () => {
                          if (!workflowDetail) return;
                          try {
                            await openevolveApi.resumeWorkflow(workflowDetail.workflow_id, apiConfig);
                            await loadWorkflowDetail(workflowDetail.workflow_id);
                            await loadWorkflows();
                          } catch (error: any) {
                            setErrorMessage(error?.message ?? "Failed to resume workflow.");
                          }
                        }}
                      >
                        Resume
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={async () => {
                          if (!workflowDetail) return;
                          try {
                            await loadWorkflowResults(workflowDetail.workflow_id);
                          } catch (error: any) {
                            setErrorMessage(error?.message ?? "Failed to load workflow results.");
                          }
                        }}
                      >
                        Fetch Results
                      </Button>
                    </div>

                    {workflowResults ? (
                      <div className="space-y-2 rounded border border-dashed p-3 text-sm">
                        <div className="font-semibold">Final Solution</div>
                        <div className="text-xs text-muted-foreground">
                          {workflowResults.final_solution?.generated_by ?? "N/A"}
                        </div>
                        <Textarea
                          value={workflowResults.final_solution?.content ?? ""}
                          readOnly
                          rows={6}
                        />
                        <div className="font-semibold">Sub-problem Solutions</div>
                        {Object.entries(workflowResults.sub_problem_solutions ?? {}).map(
                          ([subId, solution]) => (
                            <div key={subId} className="space-y-1">
                              <div className="text-xs font-medium">{subId}</div>
                              <Textarea value={solution.content} readOnly rows={4} />
                            </div>
                          ),
                        )}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
