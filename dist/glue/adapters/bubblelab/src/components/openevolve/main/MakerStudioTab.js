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
exports.MakerStudioTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const tabs_1 = require("@/components/ui/tabs");
const select_1 = require("@/components/ui/select");
const badge_1 = require("@/components/ui/badge");
const switch_1 = require("@/components/ui/switch");
const separator_1 = require("@/components/ui/separator");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const STATUS_OPTIONS = ["draft", "testing", "validated", "deployed", "deprecated"];
const MODE_OPTIONS = ["sequential", "recursive", "hybrid"];
const readApiKey = () => {
    try {
        return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    }
    catch {
        return "";
    }
};
const parseJsonOrError = (value, label) => {
    if (!value.trim())
        return { parsed: undefined, error: undefined };
    try {
        return { parsed: JSON.parse(value), error: undefined };
    }
    catch (error) {
        return { parsed: undefined, error: `${label} JSON is invalid: ${error?.message ?? "error"}` };
    }
};
const MakerStudioTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(readApiKey);
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [tools, setTools] = (0, react_1.useState)([]);
    const [delegations, setDelegations] = (0, react_1.useState)([]);
    const [toolFilter, setToolFilter] = (0, react_1.useState)({ status: "", mode: "", search: "" });
    const [delegationFilter, setDelegationFilter] = (0, react_1.useState)({ status: "", type: "" });
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [toolForm, setToolForm] = (0, react_1.useState)({
        name: "",
        description: "",
        task: "",
        maker_mode: "recursive",
        k_ahead: 3,
        max_depth: 5,
        contextJson: "",
        promptTemplate: "",
        systemPrompt: "",
        expectedSchemaJson: "",
        metadataJson: "",
    });
    const [toolTestInputs, setToolTestInputs] = (0, react_1.useState)({});
    const [toolTestResults, setToolTestResults] = (0, react_1.useState)({});
    const [executorState, setExecutorState] = (0, react_1.useState)({
        toolId: "",
        inputJson: "{\n  \"task\": \"Describe desired output\",\n  \"context\": {}\n}",
        delegateToCrewAI: false,
    });
    const [executorResult, setExecutorResult] = (0, react_1.useState)(null);
    const refreshTools = (0, react_1.useCallback)(async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const result = await openevolveApi_1.openevolveApi.listMakerTools({
                status: toolFilter.status || undefined,
                maker_mode: toolFilter.mode || undefined,
                search: toolFilter.search || undefined,
            }, apiConfig);
            setTools(result.tools ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load tools.");
        }
        finally {
            setLoading(false);
        }
    }, [apiConfig, toolFilter]);
    const refreshDelegations = (0, react_1.useCallback)(async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const result = await openevolveApi_1.openevolveApi.listMakerDelegations({
                status: delegationFilter.status || undefined,
                delegation_type: delegationFilter.type || undefined,
            }, apiConfig);
            setDelegations(result.delegations ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load delegations.");
        }
        finally {
            setLoading(false);
        }
    }, [apiConfig, delegationFilter]);
    (0, react_1.useEffect)(() => {
        refreshTools();
        refreshDelegations();
    }, [refreshTools, refreshDelegations]);
    const handleCreateTool = async () => {
        setStatusMessage(null);
        setErrorMessage(null);
        if (!toolForm.name.trim() || !toolForm.description.trim() || !toolForm.task.trim()) {
            setErrorMessage("Name, description, and task are required.");
            return;
        }
        const context = parseJsonOrError(toolForm.contextJson, "Context");
        if (context.error) {
            setErrorMessage(context.error);
            return;
        }
        const expectedSchema = parseJsonOrError(toolForm.expectedSchemaJson, "Expected schema");
        if (expectedSchema.error) {
            setErrorMessage(expectedSchema.error);
            return;
        }
        const metadata = parseJsonOrError(toolForm.metadataJson, "Metadata");
        if (metadata.error) {
            setErrorMessage(metadata.error);
            return;
        }
        setLoading(true);
        try {
            const response = await openevolveApi_1.openevolveApi.createMakerTool({
                name: toolForm.name,
                description: toolForm.description,
                task: toolForm.task,
                maker_mode: toolForm.maker_mode,
                k_ahead: toolForm.k_ahead,
                max_depth: toolForm.max_depth,
                context: context.parsed,
                prompt_template: toolForm.promptTemplate || undefined,
                system_prompt: toolForm.systemPrompt || undefined,
                expected_schema: expectedSchema.parsed,
                metadata: metadata.parsed,
            }, apiConfig);
            setStatusMessage(`Tool ${response.tool.tool_id} created.`);
            setToolForm({
                name: "",
                description: "",
                task: "",
                maker_mode: "recursive",
                k_ahead: 3,
                max_depth: 5,
                contextJson: "",
                promptTemplate: "",
                systemPrompt: "",
                expectedSchemaJson: "",
                metadataJson: "",
            });
            await refreshTools();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to create tool.");
        }
        finally {
            setLoading(false);
        }
    };
    const handleTestTool = async (toolId) => {
        setErrorMessage(null);
        const inputRaw = toolTestInputs[toolId] ?? "{\n  \"task\": \"Run a validation test\"\n}";
        const parsed = parseJsonOrError(inputRaw, "Tool input");
        if (parsed.error) {
            setErrorMessage(parsed.error);
            return;
        }
        setLoading(true);
        try {
            const result = await openevolveApi_1.openevolveApi.testMakerTool(toolId, { input_data: parsed.parsed ?? {} }, apiConfig);
            setToolTestResults((prev) => ({ ...prev, [toolId]: result.result }));
            await refreshTools();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Tool test failed.");
        }
        finally {
            setLoading(false);
        }
    };
    const handleValidateTool = async (toolId) => {
        setErrorMessage(null);
        setLoading(true);
        try {
            await openevolveApi_1.openevolveApi.validateMakerTool(toolId, apiConfig);
            setStatusMessage(`Tool ${toolId} validated.`);
            await refreshTools();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Tool validation failed.");
        }
        finally {
            setLoading(false);
        }
    };
    const handleExecuteTool = async () => {
        setErrorMessage(null);
        setExecutorResult(null);
        const parsed = parseJsonOrError(executorState.inputJson, "Execution input");
        if (parsed.error) {
            setErrorMessage(parsed.error);
            return;
        }
        if (!executorState.toolId) {
            setErrorMessage("Select a tool to execute.");
            return;
        }
        setLoading(true);
        try {
            const result = await openevolveApi_1.openevolveApi.executeMakerTool(executorState.toolId, { input_data: parsed.parsed ?? {}, delegate_to_crewai: executorState.delegateToCrewAI }, apiConfig);
            setExecutorResult(result.result);
            await refreshTools();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Tool execution failed.");
        }
        finally {
            setLoading(false);
        }
    };
    const handleSyncDelegations = async () => {
        setErrorMessage(null);
        setLoading(true);
        try {
            const result = await openevolveApi_1.openevolveApi.syncMakerDelegations(apiConfig);
            setStatusMessage(`Synced ${result.synced} delegations.`);
            await refreshDelegations();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to sync delegations.");
        }
        finally {
            setLoading(false);
        }
    };
    const toolStats = (0, react_1.useMemo)(() => {
        const statusCounts = {};
        tools.forEach((tool) => {
            statusCounts[tool.status] = (statusCounts[tool.status] || 0) + 1;
        });
        return {
            total: tools.length,
            statusCounts,
            validated: tools.filter((tool) => tool.status === "validated").length,
            totalUsage: tools.reduce((sum, tool) => sum + (tool.usage_count || 0), 0),
            activeDelegations: delegations.filter((del) => ["pending", "assigned", "in_progress", "in_review"].includes(del.status)).length,
        };
    }, [tools, delegations]);
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Maker Studio</card_1.CardTitle>
          <card_1.CardDescription>Build and execute Maker-powered tools with CrewAI delegation.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label_1.Label>API Key</label_1.Label>
              <input_1.Input type="password" value={apiKey} onChange={(event) => {
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
              <button_1.Button variant="outline" onClick={refreshTools} disabled={loading}>
                Refresh Tools
              </button_1.Button>
              <button_1.Button variant="outline" onClick={refreshDelegations} disabled={loading}>
                Refresh Delegations
              </button_1.Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-emerald-600">{statusMessage}</div> : null}

          <tabs_1.Tabs defaultValue="creator" className="w-full">
            <tabs_1.TabsList className="flex flex-wrap gap-2">
              <tabs_1.TabsTrigger value="creator">Tool Creator</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="repository">Repository</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="executor">Executor</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="delegations">CrewAI Tracker</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="analytics">Analytics</tabs_1.TabsTrigger>
            </tabs_1.TabsList>

            <tabs_1.TabsContent value="creator" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">Create New Tool</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label_1.Label>Name</label_1.Label>
                      <input_1.Input value={toolForm.name} onChange={(event) => setToolForm((prev) => ({ ...prev, name: event.target.value }))}/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Maker Mode</label_1.Label>
                      <select_1.Select value={toolForm.maker_mode} onValueChange={(value) => setToolForm((prev) => ({ ...prev, maker_mode: value }))}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue />
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          {MODE_OPTIONS.map((mode) => (<select_1.SelectItem key={mode} value={mode}>
                              {mode}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label_1.Label>Description</label_1.Label>
                    <textarea_1.Textarea value={toolForm.description} onChange={(event) => setToolForm((prev) => ({ ...prev, description: event.target.value }))} className="min-h-[120px]"/>
                  </div>

                  <div className="space-y-2">
                    <label_1.Label>Task Definition</label_1.Label>
                    <textarea_1.Textarea value={toolForm.task} onChange={(event) => setToolForm((prev) => ({ ...prev, task: event.target.value }))} className="min-h-[120px]"/>
                  </div>

                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="space-y-2">
                      <label_1.Label>k-ahead</label_1.Label>
                      <input_1.Input type="number" value={toolForm.k_ahead} onChange={(event) => setToolForm((prev) => ({ ...prev, k_ahead: Number(event.target.value) || 0 }))}/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Max Depth</label_1.Label>
                      <input_1.Input type="number" value={toolForm.max_depth} onChange={(event) => setToolForm((prev) => ({ ...prev, max_depth: Number(event.target.value) || 0 }))}/>
                    </div>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label_1.Label>Context (JSON)</label_1.Label>
                      <textarea_1.Textarea value={toolForm.contextJson} onChange={(event) => setToolForm((prev) => ({ ...prev, contextJson: event.target.value }))} className="min-h-[120px]"/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Expected Schema (JSON)</label_1.Label>
                      <textarea_1.Textarea value={toolForm.expectedSchemaJson} onChange={(event) => setToolForm((prev) => ({ ...prev, expectedSchemaJson: event.target.value }))} className="min-h-[120px]"/>
                    </div>
                  </div>

                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label_1.Label>Prompt Template</label_1.Label>
                      <textarea_1.Textarea value={toolForm.promptTemplate} onChange={(event) => setToolForm((prev) => ({ ...prev, promptTemplate: event.target.value }))} className="min-h-[120px]"/>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>System Prompt</label_1.Label>
                      <textarea_1.Textarea value={toolForm.systemPrompt} onChange={(event) => setToolForm((prev) => ({ ...prev, systemPrompt: event.target.value }))} className="min-h-[120px]"/>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label_1.Label>Metadata (JSON)</label_1.Label>
                    <textarea_1.Textarea value={toolForm.metadataJson} onChange={(event) => setToolForm((prev) => ({ ...prev, metadataJson: event.target.value }))} className="min-h-[120px]"/>
                  </div>

                  <button_1.Button onClick={handleCreateTool} disabled={loading}>
                    Create Tool
                  </button_1.Button>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="repository" className="mt-4 space-y-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">Tool Repository</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="space-y-2">
                      <label_1.Label>Status Filter</label_1.Label>
                      <select_1.Select value={toolFilter.status} onValueChange={(value) => setToolFilter((prev) => ({ ...prev, status: value }))}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue placeholder="All"/>
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          <select_1.SelectItem value="">All</select_1.SelectItem>
                          {STATUS_OPTIONS.map((status) => (<select_1.SelectItem key={status} value={status}>
                              {status}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Mode Filter</label_1.Label>
                      <select_1.Select value={toolFilter.mode} onValueChange={(value) => setToolFilter((prev) => ({ ...prev, mode: value }))}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue placeholder="All"/>
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          <select_1.SelectItem value="">All</select_1.SelectItem>
                          {MODE_OPTIONS.map((mode) => (<select_1.SelectItem key={mode} value={mode}>
                              {mode}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Search</label_1.Label>
                      <input_1.Input value={toolFilter.search} onChange={(event) => setToolFilter((prev) => ({ ...prev, search: event.target.value }))} placeholder="Search tools"/>
                    </div>
                  </div>

                  <button_1.Button variant="outline" onClick={refreshTools} disabled={loading}>
                    Apply Filters
                  </button_1.Button>

                  <div className="space-y-4">
                    {tools.length === 0 ? (<div className="text-sm text-muted-foreground">No tools found.</div>) : (tools.map((tool) => (<card_1.Card key={tool.tool_id}>
                          <card_1.CardHeader>
                            <card_1.CardTitle className="text-base flex items-center justify-between">
                              {tool.name}
                              <badge_1.Badge variant="secondary">{tool.status}</badge_1.Badge>
                            </card_1.CardTitle>
                            <card_1.CardDescription>{tool.description}</card_1.CardDescription>
                          </card_1.CardHeader>
                          <card_1.CardContent className="space-y-3">
                            <div className="grid gap-2 md:grid-cols-3 text-sm">
                              <div>Mode: {tool.maker_mode}</div>
                              <div>Version: {tool.version}</div>
                              <div>Usage: {tool.usage_count ?? 0}</div>
                            </div>
                            <div className="space-y-2">
                              <label_1.Label>Test Input (JSON)</label_1.Label>
                              <textarea_1.Textarea value={toolTestInputs[tool.tool_id] ?? "{\n  \"task\": \"Run validation test\"\n}"} onChange={(event) => setToolTestInputs((prev) => ({ ...prev, [tool.tool_id]: event.target.value }))}/>
                            </div>
                            <div className="flex flex-wrap gap-2">
                              <button_1.Button variant="outline" onClick={() => handleTestTool(tool.tool_id)} disabled={loading}>
                                Test
                              </button_1.Button>
                              <button_1.Button variant="outline" onClick={() => handleValidateTool(tool.tool_id)} disabled={loading}>
                                Validate
                              </button_1.Button>
                            </div>
                            {toolTestResults[tool.tool_id] ? (<div className="rounded-md border p-3 text-sm">
                                <div className="font-medium">Test Result</div>
                                <pre className="mt-2 text-xs whitespace-pre-wrap">
                                  {JSON.stringify(toolTestResults[tool.tool_id], null, 2)}
                                </pre>
                              </div>) : null}
                          </card_1.CardContent>
                        </card_1.Card>)))}
                  </div>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="executor" className="mt-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">Tool Executor</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label_1.Label>Select Tool</label_1.Label>
                    <select_1.Select value={executorState.toolId} onValueChange={(value) => setExecutorState((prev) => ({ ...prev, toolId: value }))}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue placeholder="Select validated tool"/>
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {tools.map((tool) => (<select_1.SelectItem key={tool.tool_id} value={tool.tool_id}>
                            {tool.name} ({tool.status})
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Input Data (JSON)</label_1.Label>
                    <textarea_1.Textarea value={executorState.inputJson} onChange={(event) => setExecutorState((prev) => ({ ...prev, inputJson: event.target.value }))} className="min-h-[160px]"/>
                  </div>
                  <div className="flex items-center gap-2">
                    <switch_1.Switch checked={executorState.delegateToCrewAI} onCheckedChange={(checked) => setExecutorState((prev) => ({ ...prev, delegateToCrewAI: checked }))}/>
                    <label_1.Label>Delegate to CrewAI</label_1.Label>
                  </div>
                  <button_1.Button onClick={handleExecuteTool} disabled={loading}>
                    Execute
                  </button_1.Button>
                  {executorResult ? (<div className="rounded-md border p-3 text-sm">
                      <div className="font-medium">Execution Result</div>
                      <pre className="mt-2 text-xs whitespace-pre-wrap">
                        {JSON.stringify(executorResult, null, 2)}
                      </pre>
                    </div>) : null}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="delegations" className="mt-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">CrewAI Workflow Tracker</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label_1.Label>Status Filter</label_1.Label>
                      <select_1.Select value={delegationFilter.status} onValueChange={(value) => setDelegationFilter((prev) => ({ ...prev, status: value }))}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue placeholder="All"/>
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          <select_1.SelectItem value="">All</select_1.SelectItem>
                          {["pending", "assigned", "in_progress", "in_review", "complete", "failed"].map((status) => (<select_1.SelectItem key={status} value={status}>
                              {status}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                    <div className="space-y-2">
                      <label_1.Label>Type Filter</label_1.Label>
                      <select_1.Select value={delegationFilter.type} onValueChange={(value) => setDelegationFilter((prev) => ({ ...prev, type: value }))}>
                        <select_1.SelectTrigger>
                          <select_1.SelectValue placeholder="All"/>
                        </select_1.SelectTrigger>
                        <select_1.SelectContent>
                          <select_1.SelectItem value="">All</select_1.SelectItem>
                          {["maker_run", "mdap_task", "custom_tool"].map((type) => (<select_1.SelectItem key={type} value={type}>
                              {type}
                            </select_1.SelectItem>))}
                        </select_1.SelectContent>
                      </select_1.Select>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button_1.Button variant="outline" onClick={refreshDelegations} disabled={loading}>
                      Refresh
                    </button_1.Button>
                    <button_1.Button variant="outline" onClick={handleSyncDelegations} disabled={loading}>
                      Sync From CrewAI
                    </button_1.Button>
                  </div>
                  <div className="space-y-3">
                    {delegations.length === 0 ? (<div className="text-sm text-muted-foreground">No delegations found.</div>) : (delegations.map((delegation) => (<card_1.Card key={delegation.delegation_id}>
                          <card_1.CardHeader>
                            <card_1.CardTitle className="text-base flex items-center justify-between">
                              {delegation.title}
                              <badge_1.Badge variant="secondary">{delegation.status}</badge_1.Badge>
                            </card_1.CardTitle>
                            <card_1.CardDescription>{delegation.description}</card_1.CardDescription>
                          </card_1.CardHeader>
                          <card_1.CardContent className="text-sm space-y-2">
                            <div>Ticket: {delegation.task_id}</div>
                            <div>Type: {delegation.delegation_type}</div>
                            <div>Created: {delegation.created_at}</div>
                            {delegation.result ? (<div className="rounded-md border p-2 text-xs whitespace-pre-wrap">
                                {JSON.stringify(delegation.result, null, 2)}
                              </div>) : null}
                          </card_1.CardContent>
                        </card_1.Card>)))}
                  </div>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="analytics" className="mt-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">Workflow Analytics</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-4">
                  <div className="grid gap-4 md:grid-cols-4">
                    <card_1.Card>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-sm">Total Tools</card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-2xl font-semibold">{toolStats.total}</card_1.CardContent>
                    </card_1.Card>
                    <card_1.Card>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-sm">Validated Tools</card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-2xl font-semibold">{toolStats.validated}</card_1.CardContent>
                    </card_1.Card>
                    <card_1.Card>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-sm">Total Executions</card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-2xl font-semibold">{toolStats.totalUsage}</card_1.CardContent>
                    </card_1.Card>
                    <card_1.Card>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-sm">Active Delegations</card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-2xl font-semibold">{toolStats.activeDelegations}</card_1.CardContent>
                    </card_1.Card>
                  </div>

                  <separator_1.Separator />

                  <div className="space-y-2 text-sm">
                    {Object.entries(toolStats.statusCounts).map(([status, count]) => (<div key={status} className="flex items-center justify-between">
                        <span>{status}</span>
                        <badge_1.Badge variant="outline">{count}</badge_1.Badge>
                      </div>))}
                  </div>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>
          </tabs_1.Tabs>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.MakerStudioTab = MakerStudioTab;
//# sourceMappingURL=MakerStudioTab.js.map