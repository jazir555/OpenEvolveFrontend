import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Bot, RefreshCw, Ticket } from "lucide-react";
import { openevolveApi } from "@/lib/openevolveApi";
import type { CrewAIWorkflowSummary, CrewAIWorkflowTicket } from "@/lib/types";

const readApiKey = () => {
  try {
    return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
  } catch {
    return "";
  }
};

export const CrewaiTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(readApiKey);
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [workflows, setWorkflows] = useState<CrewAIWorkflowSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [workflow, setWorkflow] = useState<Record<string, unknown> | null>(null);
  const [ticketsResponse, setTicketsResponse] = useState<{
    tickets: CrewAIWorkflowTicket[];
    total: number;
    status_breakdown?: Record<string, number>;
  } | null>(null);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadWorkflows = async () => {
    setErrorMessage(null);
    try {
      const response = await openevolveApi.listCrewaiWorkflows(apiConfig);
      const list = response.workflows ?? [];
      setWorkflows(list);
      if (list.length > 0 && !selectedId) {
        setSelectedId(list[0].workflow_id);
      }
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load CrewAI workflows.");
    }
  };

  const loadSelected = async (id: string) => {
    if (!id) return;
    setLoading(true);
    setErrorMessage(null);
    try {
      const [wf, tickets] = await Promise.all([
        openevolveApi.getCrewaiWorkflow(id, apiConfig),
        openevolveApi.getCrewaiWorkflowTickets(id, apiConfig),
      ]);
      setWorkflow(wf);
      setTicketsResponse(tickets);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load CrewAI workflow details.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadWorkflows();
  }, [apiConfig.apiKey]);

  useEffect(() => {
    if (selectedId) {
      loadSelected(selectedId);
    }
  }, [selectedId, apiConfig.apiKey]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-5 w-5" />
            CrewAI Workflow Tickets & Telemetry
          </CardTitle>
          <CardDescription>
            Select a CrewAI workflow to inspect its derived tickets and full telemetry state.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2 md:w-1/2">
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
            <Button variant="outline" onClick={loadWorkflows} disabled={loading}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh Workflows
            </Button>
          </div>

          <div className="space-y-2">
            <Label>CrewAI Workflow</Label>
            <select
              value={selectedId}
              onChange={(event) => setSelectedId(event.target.value)}
              className="flex h-10 w-full rounded-md border border-[#30363d] bg-[#0d1117] px-3 py-2 text-sm text-gray-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              {workflows.length === 0 && <option value="">No workflows found</option>}
              {workflows.map((wf) => (
                <option key={wf.workflow_id} value={wf.workflow_id}>
                  {wf.workflow_id}
                  {wf.status ? ` (${wf.status})` : ""}
                </option>
              ))}
            </select>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {loading ? <div className="text-sm text-muted-foreground">Loading…</div> : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Ticket className="h-4 w-4" />
            Tickets
          </CardTitle>
          <CardDescription>
            Ticket-like entries derived from the selected CrewAI workflow state.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {ticketsResponse?.status_breakdown ? (
            <div className="flex flex-wrap gap-2">
              {Object.entries(ticketsResponse.status_breakdown).map(([status, count]) => (
                <Badge key={status} variant="outline" className="border-[#30363d]">
                  {status}: {count}
                </Badge>
              ))}
            </div>
          ) : null}

          {ticketsResponse && ticketsResponse.tickets.length > 0 ? (
            <div className="overflow-x-auto rounded border border-[#30363d]">
              <table className="w-full text-left text-sm text-gray-300">
                <thead className="bg-[#0d1117] text-xs uppercase text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2">ID</th>
                    <th className="px-3 py-2">Title</th>
                    <th className="px-3 py-2">Status</th>
                    <th className="px-3 py-2">Assigned Agent</th>
                    <th className="px-3 py-2">Priority</th>
                    <th className="px-3 py-2">Dependencies</th>
                  </tr>
                </thead>
                <tbody>
                  {ticketsResponse.tickets.map((ticket) => (
                    <tr key={ticket.id} className="border-t border-[#30363d]">
                      <td className="px-3 py-2 font-mono">{ticket.id}</td>
                      <td className="px-3 py-2">{ticket.title ?? "—"}</td>
                      <td className="px-3 py-2">{ticket.status ?? "—"}</td>
                      <td className="px-3 py-2">{ticket.assigned_agent_id ?? "—"}</td>
                      <td className="px-3 py-2">{ticket.priority ?? "—"}</td>
                      <td className="px-3 py-2">
                        {ticket.dependencies && ticket.dependencies.length > 0
                          ? ticket.dependencies.join(", ")
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">No tickets returned.</div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Bot className="h-4 w-4" />
            Telemetry
          </CardTitle>
          <CardDescription>
            Full workflow state returned by the backend (GET /crewai/workflows/{"{id}"}).
          </CardDescription>
        </CardHeader>
        <CardContent>
          {workflow ? (
            <pre className="max-h-[480px] overflow-auto rounded border border-[#30363d] bg-[#0d1117] p-3 text-sm text-gray-300">
              {JSON.stringify(workflow, null, 2)}
            </pre>
          ) : (
            <div className="text-sm text-muted-foreground">No telemetry available.</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
