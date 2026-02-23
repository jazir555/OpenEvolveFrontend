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
exports.WorkflowLifecycleTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const tabs_1 = require("@/components/ui/tabs");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const readStorage = (key, fallback = "") => {
    try {
        return globalThis.localStorage?.getItem(key) ?? fallback;
    }
    catch {
        return fallback;
    }
};
const writeStorage = (key, value) => {
    try {
        globalThis.localStorage?.setItem(key, value);
    }
    catch {
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
const formatDuration = (start, end) => {
    if (!start)
        return "N/A";
    const endTime = end ?? Date.now() / 1000;
    const duration = Math.max(0, endTime - start);
    const minutes = Math.floor(duration / 60);
    const seconds = Math.floor(duration % 60);
    return `${minutes}m ${seconds}s`;
};
const WorkflowLifecycleTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => readStorage("openevolve_api_key"));
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [definitions, setDefinitions] = (0, react_1.useState)([]);
    const [instances, setInstances] = (0, react_1.useState)([]);
    const [selectedInstanceId, setSelectedInstanceId] = (0, react_1.useState)("");
    const [instanceDetail, setInstanceDetail] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [definitionForm, setDefinitionForm] = (0, react_1.useState)({
        name: "Sovereign Workflow",
        description: "OpenEvolve workflow",
        workflow_type: "sovereign",
        problem_statement: "",
        parameters_json: JSON.stringify(DEFAULT_PARAMETERS, null, 2),
    });
    const [instanceForm, setInstanceForm] = (0, react_1.useState)({
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
                openevolveApi_1.openevolveApi.listWorkflowDefinitions(apiConfig),
                openevolveApi_1.openevolveApi.listWorkflowInstances(apiConfig),
            ]);
            setDefinitions(defs.definitions ?? []);
            setInstances(inst.instances ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load workflows.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        refresh();
    }, [apiConfig.apiKey]);
    const loadInstanceDetail = async (instanceId) => {
        setErrorMessage(null);
        try {
            const detail = await openevolveApi_1.openevolveApi.getWorkflowInstance(instanceId, apiConfig);
            setInstanceDetail(detail);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load instance details.");
        }
    };
    const handleSelectInstance = (instanceId) => {
        setSelectedInstanceId(instanceId);
        if (instanceId) {
            loadInstanceDetail(instanceId);
        }
        else {
            setInstanceDetail(null);
        }
    };
    const runAction = async (action) => {
        if (!selectedInstanceId)
            return;
        setErrorMessage(null);
        try {
            const callMap = {
                start: openevolveApi_1.openevolveApi.startWorkflowInstance,
                pause: openevolveApi_1.openevolveApi.pauseWorkflowInstance,
                resume: openevolveApi_1.openevolveApi.resumeWorkflowInstance,
                stop: openevolveApi_1.openevolveApi.stopWorkflowInstance,
                cancel: openevolveApi_1.openevolveApi.cancelWorkflowInstance,
                restart: openevolveApi_1.openevolveApi.restartWorkflowInstance,
            };
            const response = await callMap[action](selectedInstanceId, apiConfig);
            if (response && typeof response === "object" && "error" in response) {
                setErrorMessage(String(response.error ?? "Action failed."));
                return;
            }
            setStatusMessage(response?.message ?? `Action ${action} executed.`);
            await refresh();
            await loadInstanceDetail(selectedInstanceId);
        }
        catch (error) {
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
            const response = await openevolveApi_1.openevolveApi.createWorkflowDefinition({
                name: definitionForm.name,
                description: definitionForm.description,
                workflow_type: definitionForm.workflow_type,
                parameters: parsed,
            }, apiConfig);
            setStatusMessage(`Workflow definition created: ${response.definition_id}`);
            await refresh();
        }
        catch (error) {
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
            const response = await openevolveApi_1.openevolveApi.createWorkflowInstance({
                definition_id: instanceForm.definition_id,
                instance_name: instanceForm.instance_name,
                inputs,
            }, apiConfig);
            const instanceId = response.instance_id;
            if (instanceForm.start_after_create) {
                const startResponse = await openevolveApi_1.openevolveApi.startWorkflowInstance(instanceId, apiConfig);
                if (startResponse && typeof startResponse === "object" && "error" in startResponse) {
                    setErrorMessage(String(startResponse.error ?? "Failed to start instance."));
                }
            }
            setStatusMessage(`Instance created: ${instanceId}`);
            await refresh();
            setSelectedInstanceId(instanceId);
            await loadInstanceDetail(instanceId);
        }
        catch (error) {
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
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Workflow Lifecycle</card_1.CardTitle>
          <card_1.CardDescription>Start, pause, resume, and manage BubbleLabs workflow instances.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>API Key</label_1.Label>
              <input_1.Input value={apiKey} type="password" onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            writeStorage("openevolve_api_key", value);
        }}/>
            </div>
            <div className="flex items-end gap-2">
              <button_1.Button variant="outline" onClick={refresh} disabled={loading}>
                Refresh
              </button_1.Button>
            </div>
          </div>
          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}
        </card_1.CardContent>
      </card_1.Card>

      <tabs_1.Tabs defaultValue="controls" className="w-full">
        <tabs_1.TabsList className="grid w-full grid-cols-3">
          <tabs_1.TabsTrigger value="controls">Controls</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="create">Create Definition</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="instances">Create Instance</tabs_1.TabsTrigger>
        </tabs_1.TabsList>

        <tabs_1.TabsContent value="controls" className="mt-4 space-y-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Workflow Instances</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <div className="space-y-2">
                <label_1.Label>Select Instance</label_1.Label>
                <select_1.Select value={selectedInstanceId} onValueChange={handleSelectInstance}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue placeholder="Select instance"/>
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    {instances.map((instance) => (<select_1.SelectItem key={instance.instance_id} value={instance.instance_id}>
                        {instance.instance_id.slice(0, 8)} - {instance.status}
                      </select_1.SelectItem>))}
                  </select_1.SelectContent>
                </select_1.Select>
              </div>

              {instanceDetail ? (<div className="space-y-4">
                  <div className="grid gap-3 md:grid-cols-4">
                    <card_1.Card>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-xs">Status</card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-sm">
                        <badge_1.Badge>{instanceDetail.status.status}</badge_1.Badge>
                      </card_1.CardContent>
                    </card_1.Card>
                    <card_1.Card>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-xs">Stage</card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-sm">{instanceDetail.status.current_stage ?? "N/A"}</card_1.CardContent>
                    </card_1.Card>
                    <card_1.Card>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-xs">Progress</card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-sm">
                        {instanceDetail.status.progress ? `${(instanceDetail.status.progress * 100).toFixed(1)}%` : "0%"}
                      </card_1.CardContent>
                    </card_1.Card>
                    <card_1.Card>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-xs">Duration</card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-sm">
                        {formatDuration(instanceDetail.status.start_time, instanceDetail.status.end_time)}
                      </card_1.CardContent>
                    </card_1.Card>
                  </div>

                  <div className="grid gap-2 md:grid-cols-6">
                    <button_1.Button size="sm" onClick={() => runAction("start")} disabled={!canStart}>
                      Start
                    </button_1.Button>
                    <button_1.Button size="sm" onClick={() => runAction("pause")} disabled={!canPause}>
                      Pause
                    </button_1.Button>
                    <button_1.Button size="sm" onClick={() => runAction("resume")} disabled={!canResume}>
                      Resume
                    </button_1.Button>
                    <button_1.Button size="sm" onClick={() => runAction("stop")} disabled={!canStop}>
                      Stop
                    </button_1.Button>
                    <button_1.Button size="sm" onClick={() => runAction("cancel")} disabled={!canCancel}>
                      Cancel
                    </button_1.Button>
                    <button_1.Button size="sm" onClick={() => runAction("restart")} disabled={!canRestart}>
                      Restart
                    </button_1.Button>
                  </div>

                  <tabs_1.Tabs defaultValue="status" className="w-full">
                    <tabs_1.TabsList className="grid w-full grid-cols-4">
                      <tabs_1.TabsTrigger value="status">Status</tabs_1.TabsTrigger>
                      <tabs_1.TabsTrigger value="parameters">Parameters</tabs_1.TabsTrigger>
                      <tabs_1.TabsTrigger value="timeline">Timeline</tabs_1.TabsTrigger>
                      <tabs_1.TabsTrigger value="errors">Errors</tabs_1.TabsTrigger>
                    </tabs_1.TabsList>
                    <tabs_1.TabsContent value="status" className="mt-3">
                      <pre className="text-xs bg-muted p-3 rounded">
                        {JSON.stringify(instanceDetail.status, null, 2)}
                      </pre>
                    </tabs_1.TabsContent>
                    <tabs_1.TabsContent value="parameters" className="mt-3">
                      <pre className="text-xs bg-muted p-3 rounded">
                        {JSON.stringify(instanceDetail.parameters, null, 2)}
                      </pre>
                    </tabs_1.TabsContent>
                    <tabs_1.TabsContent value="timeline" className="mt-3">
                      <div className="text-sm space-y-1">
                        <div>Created: {instanceDetail.status.start_time ? new Date(instanceDetail.status.start_time * 1000).toLocaleString() : "N/A"}</div>
                        <div>Completed: {instanceDetail.status.end_time ? new Date(instanceDetail.status.end_time * 1000).toLocaleString() : "In progress"}</div>
                      </div>
                    </tabs_1.TabsContent>
                    <tabs_1.TabsContent value="errors" className="mt-3">
                      {instanceDetail.status.error_message ? (<div className="text-sm text-red-500">{instanceDetail.status.error_message}</div>) : (<div className="text-sm text-muted-foreground">No errors reported.</div>)}
                    </tabs_1.TabsContent>
                  </tabs_1.Tabs>
                </div>) : (<div className="text-sm text-muted-foreground">Select an instance to view details.</div>)}
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="create" className="mt-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Create Workflow Definition</card_1.CardTitle>
              <card_1.CardDescription>Define a new BubbleLabs workflow template.</card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <label_1.Label>Name</label_1.Label>
                  <input_1.Input value={definitionForm.name} onChange={(event) => setDefinitionForm((prev) => ({ ...prev, name: event.target.value }))}/>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Workflow Type</label_1.Label>
                  <select_1.Select value={definitionForm.workflow_type} onValueChange={(value) => setDefinitionForm((prev) => ({ ...prev, workflow_type: value }))}>
                    <select_1.SelectTrigger>
                      <select_1.SelectValue placeholder="Select type"/>
                    </select_1.SelectTrigger>
                    <select_1.SelectContent>
                      <select_1.SelectItem value="evolution">Evolution</select_1.SelectItem>
                      <select_1.SelectItem value="adversarial">Adversarial</select_1.SelectItem>
                      <select_1.SelectItem value="sovereign">Sovereign</select_1.SelectItem>
                    </select_1.SelectContent>
                  </select_1.Select>
                </div>
              </div>
              <div className="space-y-2">
                <label_1.Label>Description</label_1.Label>
                <input_1.Input value={definitionForm.description} onChange={(event) => setDefinitionForm((prev) => ({ ...prev, description: event.target.value }))}/>
              </div>
              <div className="space-y-2">
                <label_1.Label>Problem Statement</label_1.Label>
                <textarea_1.Textarea value={definitionForm.problem_statement} onChange={(event) => setDefinitionForm((prev) => ({ ...prev, problem_statement: event.target.value }))} className="min-h-[120px]"/>
              </div>
              <div className="space-y-2">
                <label_1.Label>Parameters (JSON)</label_1.Label>
                <textarea_1.Textarea value={definitionForm.parameters_json} onChange={(event) => setDefinitionForm((prev) => ({ ...prev, parameters_json: event.target.value }))} className="min-h-[160px]"/>
              </div>
              <button_1.Button onClick={handleCreateDefinition}>Create Definition</button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="instances" className="mt-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Create Workflow Instance</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <div className="space-y-2">
                <label_1.Label>Workflow Definition</label_1.Label>
                <select_1.Select value={instanceForm.definition_id} onValueChange={(value) => setInstanceForm((prev) => ({ ...prev, definition_id: value }))}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue placeholder="Select definition"/>
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    {definitions.map((definition) => (<select_1.SelectItem key={definition.id} value={definition.id}>
                        {definition.name} ({definition.workflow_type})
                      </select_1.SelectItem>))}
                  </select_1.SelectContent>
                </select_1.Select>
              </div>
              <div className="space-y-2">
                <label_1.Label>Instance Name</label_1.Label>
                <input_1.Input value={instanceForm.instance_name} onChange={(event) => setInstanceForm((prev) => ({ ...prev, instance_name: event.target.value }))}/>
              </div>
              <div className="space-y-2">
                <label_1.Label>Inputs (JSON)</label_1.Label>
                <textarea_1.Textarea value={instanceForm.inputs_json} onChange={(event) => setInstanceForm((prev) => ({ ...prev, inputs_json: event.target.value }))} className="min-h-[160px]"/>
              </div>
              <div className="flex items-center gap-2">
                <input type="checkbox" checked={instanceForm.start_after_create} onChange={(event) => setInstanceForm((prev) => ({ ...prev, start_after_create: event.target.checked }))}/>
                <span className="text-sm">Start after create</span>
              </div>
              <button_1.Button onClick={handleCreateInstance}>Create Instance</button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>
      </tabs_1.Tabs>
    </div>);
};
exports.WorkflowLifecycleTab = WorkflowLifecycleTab;
//# sourceMappingURL=WorkflowLifecycleTab.js.map