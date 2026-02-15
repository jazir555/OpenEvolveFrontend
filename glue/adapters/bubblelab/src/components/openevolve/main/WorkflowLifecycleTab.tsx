import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { openevolveApi } from "../../../lib/openevolveApi";
import type {
  WorkflowDefinitionSummary,
  WorkflowInstanceSummary,
  WorkflowInstanceDetail,
} from "../../../lib/types";

const readStorage = (key: string, fallback = "") => {
  try {
    return globalThis.localStorage?.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
};

const writeStorage = (key: string, value: string) => {
  try {
    globalThis.localStorage?.setItem(key, value);
  } catch {
    // ignore
  }
};

const DEFAULT_PARAMETERS = {
  max_iterations: 100,
  population_size: 50,
  temperature: 0.7,
  top_p: 1.0,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
  max_tokens: 4096,
  num_islands: 5,
  migration_rate: 0.1,
  archive_size: 100,
  enable_qd_evolution: false,
  enable_multi_objective: false,
  enable_adversarial: false,
  memory_limit_mb: 2048,
  cpu_limit: 1.0,
};

const formatDuration = (start?: number | null, end?: number | null) => {
  if (!start) return "N/A";
  const endTime = end ?? Date.now() / 1000;
  const duration = Math.max(0, endTime - start);
  const minutes = Math.floor(duration / 60);
  const seconds = Math.floor(duration % 60);
  return `${minutes}m ${seconds}s`;
};

export const WorkflowLifecycleTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(() => readStorage("openevolve_api_key"));
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [definitions, setDefinitions] = useState<WorkflowDefinitionSummary[]>([]);
  const [instances, setInstances] = useState<WorkflowInstanceSummary[]>([]);
  const [selectedInstanceId, setSelectedInstanceId] = useState<string>("");
  const [instanceDetail, setInstanceDetail] = useState<WorkflowInstanceDetail | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [definitionForm, setDefinitionForm] = useState({
    name: "Sovereign Workflow",
    description: "OpenEvolve workflow",
    workflow_type: "sovereign",
    problem_statement: "",
    parameters_json: JSON.stringify(DEFAULT_PARAMETERS, null, 2),
  });

  const [instanceForm, setInstanceForm] = useState({
    definition_id: "",
    instance_name: "Instance",
    inputs_json: '{"content": "Enter your content here", "problem_statement": "Enter problem"}',
    start_after_create: true,
  });

  const refresh = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const [defs, inst] = await Promise.all([
        openevolveApi.listWorkflowDefinitions(apiConfig),
        openevolveApi.listWorkflowInstances(apiConfig),
      ]);
      setDefinitions(defs.definitions ?? []);
      setInstances(inst.instances ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load workflows.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, [apiConfig.apiKey]);

  const loadInstanceDetail = async (instanceId: string) => {
    setErrorMessage(null);
    try {
      const detail = await openevolveApi.getWorkflowInstance(instanceId, apiConfig);
      setInstanceDetail(detail);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load instance details.");
    }
  };

  const handleSelectInstance = (instanceId: string) => {
    setSelectedInstanceId(instanceId);
    if (instanceId) {
      loadInstanceDetail(instanceId);
    } else {
      setInstanceDetail(null);
    }
  };

  const runAction = async (action: "start" | "pause" | "resume" | "stop" | "cancel" | "restart") => {
    if (!selectedInstanceId) return;
    setErrorMessage(null);
    try {
      const callMap = {
        start: openevolveApi.startWorkflowInstance,
        pause: openevolveApi.pauseWorkflowInstance,
        resume: openevolveApi.resumeWorkflowInstance,
        stop: openevolveApi.stopWorkflowInstance,
        cancel: openevolveApi.cancelWorkflowInstance,
        restart: openevolveApi.restartWorkflowInstance,
      };
      const response = await callMap[action](selectedInstanceId, apiConfig);
      if (response && typeof response === "object" && "error" in response) {
        setErrorMessage(String((response as { error?: string }).error ?? "Action failed."));
        return;
      }
      setStatusMessage((response as { message?: string })?.message ?? `Action ${action} executed.`);
      await refresh();
      await loadInstanceDetail(selectedInstanceId);
    } catch (error: any) {
      setErrorMessage(error?.message ?? `Failed to ${action} workflow.`);
    }
  };

  const handleCreateDefinition = async () => {
    setErrorMessage(null);
    setStatusMessage(null);
    if (!definitionForm.problem_statement.trim()) {
      setErrorMessage("Problem statement is required.");
      return;
    }
    try {
      const parsed = JSON.parse(definitionForm.parameters_json || "{}");
      if (!parsed.problem_statement) {
        parsed.problem_statement = definitionForm.problem_statement;
      }
      const response = await openevolveApi.createWorkflowDefinition(
        {
          name: definitionForm.name,
          description: definitionForm.description,
          workflow_type: definitionForm.workflow_type,
          parameters: parsed,
        },
        apiConfig,
      );
      setStatusMessage(`Workflow definition created: ${response.definition_id}`);
      await refresh();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to create workflow definition.");
    }
  };

  const handleCreateInstance = async () => {
    setErrorMessage(null);
    setStatusMessage(null);
    if (!instanceForm.definition_id) {
      setErrorMessage("Select a workflow definition.");
      return;
    }
    try {
      const inputs = JSON.parse(instanceForm.inputs_json || "{}");
      const response = await openevolveApi.createWorkflowInstance(
        {
          definition_id: instanceForm.definition_id,
          instance_name: instanceForm.instance_name,
          inputs,
        },
        apiConfig,
      );
      const instanceId = response.instance_id;
      if (instanceForm.start_after_create) {
        const startResponse = await openevolveApi.startWorkflowInstance(instanceId, apiConfig);
        if (startResponse && typeof startResponse === "object" && "error" in startResponse) {
          setErrorMessage(String((startResponse as { error?: string }).error ?? "Failed to start instance."));
        }
      }
      setStatusMessage(`Instance created: ${instanceId}`);
      await refresh();
      setSelectedInstanceId(instanceId);
      await loadInstanceDetail(instanceId);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to create instance.");
    }
  };

  const currentStatus = instanceDetail?.status?.status ?? "";
  const canStart = ["created", "pending", "stopped", "cancelled", "failed"].includes(currentStatus);
  const canPause = currentStatus === "running";
  const canResume = currentStatus === "paused";
  const canStop = ["running", "pending"].includes(currentStatus);
  const canCancel = ["running", "pending", "paused"].includes(currentStatus);
  const canRestart = ["completed", "failed", "cancelled", "stopped"].includes(currentStatus);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Workflow Lifecycle</CardTitle>
          <CardDescription>Start, pause, resume, and manage BubbleLabs workflow instances.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
                value={apiKey}
                type="password"
                onChange={(event) => {
                  const value = event.target.value;
                  setApiKey(value);
                  writeStorage("openevolve_api_key", value);
                }}
              />
            </div>
            <div className="flex items-end gap-2">
              <Button variant="outline" onClick={refresh} disabled={loading}>
                Refresh
              </Button>
            </div>
          </div>
          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}
        </CardContent>
      </Card>

      <Tabs defaultValue="controls" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="controls">Controls</TabsTrigger>
          <TabsTrigger value="create">Create Definition</TabsTrigger>
          <TabsTrigger value="instances">Create Instance</TabsTrigger>
        </TabsList>

        <TabsContent value="controls" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Workflow Instances</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <Label>Select Instance</Label>
                <Select value={selectedInstanceId} onValueChange={handleSelectInstance}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select instance" />
                  </SelectTrigger>
                  <SelectContent>
                    {instances.map((instance) => (
                      <SelectItem key={instance.instance_id} value={instance.instance_id}>
                        {instance.instance_id.slice(0, 8)} - {instance.status}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {instanceDetail ? (
                <div className="space-y-4">
                  <div className="grid gap-3 md:grid-cols-4">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-xs">Status</CardTitle>
                      </CardHeader>
                      <CardContent className="text-sm">
                        <Badge>{instanceDetail.status.status}</Badge>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-xs">Stage</CardTitle>
                      </CardHeader>
                      <CardContent className="text-sm">{instanceDetail.status.current_stage ?? "N/A"}</CardContent>
                    </Card>
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-xs">Progress</CardTitle>
                      </CardHeader>
                      <CardContent className="text-sm">
                        {instanceDetail.status.progress ? `${(instanceDetail.status.progress * 100).toFixed(1)}%` : "0%"}
                      </CardContent>
                    </Card>
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-xs">Duration</CardTitle>
                      </CardHeader>
                      <CardContent className="text-sm">
                        {formatDuration(instanceDetail.status.start_time, instanceDetail.status.end_time)}
                      </CardContent>
                    </Card>
                  </div>

                  <div className="grid gap-2 md:grid-cols-6">
                    <Button size="sm" onClick={() => runAction("start")} disabled={!canStart}>
                      Start
                    </Button>
                    <Button size="sm" onClick={() => runAction("pause")} disabled={!canPause}>
                      Pause
                    </Button>
                    <Button size="sm" onClick={() => runAction("resume")} disabled={!canResume}>
                      Resume
                    </Button>
                    <Button size="sm" onClick={() => runAction("stop")} disabled={!canStop}>
                      Stop
                    </Button>
                    <Button size="sm" onClick={() => runAction("cancel")} disabled={!canCancel}>
                      Cancel
                    </Button>
                    <Button size="sm" onClick={() => runAction("restart")} disabled={!canRestart}>
                      Restart
                    </Button>
                  </div>

                  <Tabs defaultValue="status" className="w-full">
                    <TabsList className="grid w-full grid-cols-4">
                      <TabsTrigger value="status">Status</TabsTrigger>
                      <TabsTrigger value="parameters">Parameters</TabsTrigger>
                      <TabsTrigger value="timeline">Timeline</TabsTrigger>
                      <TabsTrigger value="errors">Errors</TabsTrigger>
                    </TabsList>
                    <TabsContent value="status" className="mt-3">
                      <pre className="text-xs bg-muted p-3 rounded">
                        {JSON.stringify(instanceDetail.status, null, 2)}
                      </pre>
                    </TabsContent>
                    <TabsContent value="parameters" className="mt-3">
                      <pre className="text-xs bg-muted p-3 rounded">
                        {JSON.stringify(instanceDetail.parameters, null, 2)}
                      </pre>
                    </TabsContent>
                    <TabsContent value="timeline" className="mt-3">
                      <div className="text-sm space-y-1">
                        <div>Created: {instanceDetail.status.start_time ? new Date(instanceDetail.status.start_time * 1000).toLocaleString() : "N/A"}</div>
                        <div>Completed: {instanceDetail.status.end_time ? new Date(instanceDetail.status.end_time * 1000).toLocaleString() : "In progress"}</div>
                      </div>
                    </TabsContent>
                    <TabsContent value="errors" className="mt-3">
                      {instanceDetail.status.error_message ? (
                        <div className="text-sm text-red-500">{instanceDetail.status.error_message}</div>
                      ) : (
                        <div className="text-sm text-muted-foreground">No errors reported.</div>
                      )}
                    </TabsContent>
                  </Tabs>
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">Select an instance to view details.</div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="create" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Create Workflow Definition</CardTitle>
              <CardDescription>Define a new BubbleLabs workflow template.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Name</Label>
                  <Input
                    value={definitionForm.name}
                    onChange={(event) => setDefinitionForm((prev) => ({ ...prev, name: event.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Workflow Type</Label>
                  <Select
                    value={definitionForm.workflow_type}
                    onValueChange={(value) =>
                      setDefinitionForm((prev) => ({ ...prev, workflow_type: value }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="evolution">Evolution</SelectItem>
                      <SelectItem value="adversarial">Adversarial</SelectItem>
                      <SelectItem value="sovereign">Sovereign</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="space-y-2">
                <Label>Description</Label>
                <Input
                  value={definitionForm.description}
                  onChange={(event) =>
                    setDefinitionForm((prev) => ({ ...prev, description: event.target.value }))
                  }
                />
              </div>
              <div className="space-y-2">
                <Label>Problem Statement</Label>
                <Textarea
                  value={definitionForm.problem_statement}
                  onChange={(event) =>
                    setDefinitionForm((prev) => ({ ...prev, problem_statement: event.target.value }))
                  }
                  className="min-h-[120px]"
                />
              </div>
              <div className="space-y-2">
                <Label>Parameters (JSON)</Label>
                <Textarea
                  value={definitionForm.parameters_json}
                  onChange={(event) =>
                    setDefinitionForm((prev) => ({ ...prev, parameters_json: event.target.value }))
                  }
                  className="min-h-[160px]"
                />
              </div>
              <Button onClick={handleCreateDefinition}>Create Definition</Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="instances" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Create Workflow Instance</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <Label>Workflow Definition</Label>
                <Select
                  value={instanceForm.definition_id}
                  onValueChange={(value) => setInstanceForm((prev) => ({ ...prev, definition_id: value }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select definition" />
                  </SelectTrigger>
                  <SelectContent>
                    {definitions.map((definition) => (
                      <SelectItem key={definition.id} value={definition.id}>
                        {definition.name} ({definition.workflow_type})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Instance Name</Label>
                <Input
                  value={instanceForm.instance_name}
                  onChange={(event) =>
                    setInstanceForm((prev) => ({ ...prev, instance_name: event.target.value }))
                  }
                />
              </div>
              <div className="space-y-2">
                <Label>Inputs (JSON)</Label>
                <Textarea
                  value={instanceForm.inputs_json}
                  onChange={(event) =>
                    setInstanceForm((prev) => ({ ...prev, inputs_json: event.target.value }))
                  }
                  className="min-h-[160px]"
                />
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={instanceForm.start_after_create}
                  onChange={(event) =>
                    setInstanceForm((prev) => ({ ...prev, start_after_create: event.target.checked }))
                  }
                />
                <span className="text-sm">Start after create</span>
              </div>
              <Button onClick={handleCreateInstance}>Create Instance</Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
