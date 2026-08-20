import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { openevolveApi } from "@/lib/openevolveApi";
import type {
  WorkflowSummary,
  WorkflowDetail,
  WorkflowTelemetry,
  CrewAIWorkflowSummary,
  CrewAIWorkflowTicket,
  WorkflowPlanResponse,
} from "@/lib/types";

const formatPercent = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(1)}%`;
};

export const SgdMonitoringTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [crewaiWorkflows, setCrewaiWorkflows] = useState<CrewAIWorkflowSummary[]>([]);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>("");
  const [selectedCrewaiWorkflowId, setSelectedCrewaiWorkflowId] = useState<string>("");
  const [workflowDetail, setWorkflowDetail] = useState<WorkflowDetail | null>(null);
  const [workflowTelemetry, setWorkflowTelemetry] = useState<WorkflowTelemetry | null>(null);
  const [workflowPlan, setWorkflowPlan] = useState<WorkflowPlanResponse | null>(null);
  const [crewaiTickets, setCrewaiTickets] = useState<CrewAIWorkflowTicket[]>([]);
  const [ticketBreakdown, setTicketBreakdown] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const refreshLists = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const [workflowRes, crewaiRes] = await Promise.all([
        openevolveApi.listWorkflows(apiConfig),
        openevolveApi.listCrewaiWorkflows(apiConfig),
      ]);
      setWorkflows(workflowRes.workflows || []);
      setCrewaiWorkflows(crewaiRes.workflows || []);
      if (!selectedWorkflowId && workflowRes.workflows?.length) {
        setSelectedWorkflowId(workflowRes.workflows[0].workflow_id);
      }
      if (!selectedCrewaiWorkflowId && crewaiRes.workflows?.length) {
        setSelectedCrewaiWorkflowId(crewaiRes.workflows[0].workflow_id);
      }
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load SGD monitoring data.");
    } finally {
      setLoading(false);
    }
  };

  const refreshWorkflowData = async (workflowId: string) => {
    if (!workflowId) return;
    setErrorMessage(null);
    try {
      const [detailRes, telemetryRes, planRes] = await Promise.all([
        openevolveApi.getWorkflow(workflowId, apiConfig),
        openevolveApi.getWorkflowTelemetry(workflowId, apiConfig),
        openevolveApi.getWorkflowPlan(workflowId, apiConfig).catch(() => null),
      ]);
      setWorkflowDetail(detailRes);
      setWorkflowTelemetry(telemetryRes);
      setWorkflowPlan(planRes);
      if (telemetryRes?.crewai_workflow_id) {
        setSelectedCrewaiWorkflowId(telemetryRes.crewai_workflow_id);
      }
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load workflow telemetry.");
    }
  };

  const refreshCrewaiTickets = async (workflowId: string) => {
    if (!workflowId) {
      setCrewaiTickets([]);
      setTicketBreakdown({});
      return;
    }
    try {
      const result = await openevolveApi.getCrewaiWorkflowTickets(workflowId, apiConfig);
      setCrewaiTickets(result.tickets || []);
      setTicketBreakdown(result.status_breakdown || {});
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load CrewAI ticket data.");
    }
  };

  useEffect(() => {
    refreshLists();
  }, [apiConfig.apiKey]);

  useEffect(() => {
    refreshWorkflowData(selectedWorkflowId);
  }, [selectedWorkflowId, apiConfig.apiKey]);

  useEffect(() => {
    refreshCrewaiTickets(selectedCrewaiWorkflowId);
  }, [selectedCrewaiWorkflowId, apiConfig.apiKey]);

  const gauntletSummary = workflowTelemetry?.gauntlet_summary;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>SGD Monitoring</CardTitle>
          <CardDescription>Sovereign-Grade Decomposition + CrewAI integration status.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label className="text-sm font-medium">API Key</label>
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
            <Button variant="outline" onClick={refreshLists} disabled={loading}>
              Refresh Lists
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">OpenEvolve Workflow</label>
              <Select value={selectedWorkflowId} onValueChange={setSelectedWorkflowId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select workflow" />
                </SelectTrigger>
                <SelectContent>
                  {workflows.map((workflow) => (
                    <SelectItem key={workflow.workflow_id} value={workflow.workflow_id}>
                      {workflow.workflow_id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">CrewAI Workflow</label>
              <Select value={selectedCrewaiWorkflowId} onValueChange={setSelectedCrewaiWorkflowId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select CrewAI workflow" />
                </SelectTrigger>
                <SelectContent>
                  {crewaiWorkflows.map((workflow) => (
                    <SelectItem key={workflow.workflow_id} value={workflow.workflow_id}>
                      {workflow.workflow_id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <Tabs defaultValue="progress">
            <TabsList className="flex flex-wrap gap-2">
              <TabsTrigger value="progress">Workflow Progress</TabsTrigger>
              <TabsTrigger value="tickets">Ticket Status</TabsTrigger>
              <TabsTrigger value="gauntlet">Gauntlet Performance</TabsTrigger>
              <TabsTrigger value="analysis">Detailed Analysis</TabsTrigger>
            </TabsList>

            <TabsContent value="progress" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Current Workflow</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {workflowDetail ? (
                    <>
                      <div className="flex items-center justify-between">
                        <div className="font-semibold">{workflowDetail.workflow_id}</div>
                        <Badge variant="secondary">{workflowDetail.status}</Badge>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Stage: {workflowDetail.current_stage} · Progress:{" "}
                        {formatPercent(workflowDetail.progress)}
                      </div>
                      <div className="mt-2 h-2 w-full rounded bg-muted">
                        <div
                          className="h-2 rounded bg-emerald-500"
                          style={{ width: `${Math.round((workflowDetail.progress ?? 0) * 100)}%` }}
                        />
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Sub-problems solved: {workflowDetail.solved_sub_problems}/
                        {workflowDetail.total_sub_problems}
                      </div>
                      {workflowTelemetry?.crewai_workflow_id ? (
                        <div className="text-xs text-muted-foreground">
                          CrewAI Workflow: {workflowTelemetry.crewai_workflow_id}
                        </div>
                      ) : null}
                      {workflowPlan?.plan?.problem_statement ? (
                        <div className="text-xs text-muted-foreground">
                          Problem: {workflowPlan.plan.problem_statement.slice(0, 120)}
                        </div>
                      ) : null}
                    </>
                  ) : (
                    <div className="text-muted-foreground">Select a workflow to view progress.</div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="tickets" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Ticket Breakdown</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {Object.keys(ticketBreakdown).length === 0 ? (
                      <div className="text-muted-foreground">No ticket data available.</div>
                    ) : (
                      Object.entries(ticketBreakdown).map(([status, count]) => (
                        <div key={status} className="flex items-center justify-between">
                          <span>{status}</span>
                          <Badge variant="secondary">{count}</Badge>
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Tickets</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {crewaiTickets.length === 0 ? (
                      <div className="text-muted-foreground">No tickets found.</div>
                    ) : (
                      crewaiTickets.slice(0, 12).map((ticket) => (
                        <div key={ticket.id} className="rounded border p-2">
                          <div className="flex items-center justify-between">
                            <div className="font-semibold">{ticket.title ?? ticket.id}</div>
                            <Badge variant="secondary">{ticket.status ?? "pending"}</Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Assigned: {ticket.assigned_agent_id ?? "unassigned"}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Dependencies: {ticket.dependencies?.join(", ") || "none"}
                          </div>
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="gauntlet" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Gauntlet Summary</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  {gauntletSummary ? (
                    <>
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
                            {formatPercent(
                              gauntletSummary.verification_total
                                ? gauntletSummary.verification_approved /
                                    gauntletSummary.verification_total
                                : null,
                            )}
                          </div>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="text-muted-foreground">No gauntlet summary available.</div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="analysis" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Stage Timeline</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {workflowTelemetry ? (
                    <div className="space-y-2">
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
                        return (
                          <div key={stage} className="flex items-center justify-between">
                            <span>{stage}</span>
                            <Badge variant={isCurrent ? "default" : "secondary"}>
                              {isCurrent ? "current" : "pending"}
                            </Badge>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="text-muted-foreground">Select a workflow to view details.</div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};
