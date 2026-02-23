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
exports.WorkflowTemplatesTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const downloadJson = (filename, payload) => {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
};
const WorkflowTemplatesTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [templates, setTemplates] = (0, react_1.useState)([]);
    const [form, setForm] = (0, react_1.useState)({
        name: "",
        description: "",
        tags: "",
        config: "",
    });
    const [importPayload, setImportPayload] = (0, react_1.useState)("");
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const loadTemplates = async () => {
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.listWorkflowTemplates(apiConfig);
            setTemplates(response.templates ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load templates.");
        }
    };
    (0, react_1.useEffect)(() => {
        loadTemplates();
    }, [apiConfig.apiKey]);
    const handleCreateTemplate = async () => {
        setErrorMessage(null);
        setStatusMessage(null);
        if (!form.name.trim() || !form.config.trim()) {
            setErrorMessage("Template name and config are required.");
            return;
        }
        try {
            const config = JSON.parse(form.config);
            const template = await openevolveApi_1.openevolveApi.createWorkflowTemplate({
                name: form.name.trim(),
                description: form.description.trim(),
                tags: form.tags ? form.tags.split(",").map((tag) => tag.trim()) : [],
                config,
            }, apiConfig);
            setStatusMessage(`Template ${template.name} created.`);
            setForm({ name: "", description: "", tags: "", config: "" });
            await loadTemplates();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to create template.");
        }
    };
    const handleDeleteTemplate = async (templateId) => {
        setErrorMessage(null);
        try {
            await openevolveApi_1.openevolveApi.deleteWorkflowTemplate(templateId, apiConfig);
            await loadTemplates();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to delete template.");
        }
    };
    const handleExport = async () => {
        try {
            const data = await openevolveApi_1.openevolveApi.exportWorkflowTemplates(apiConfig);
            downloadJson("workflow_templates.json", data);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to export templates.");
        }
    };
    const handleImport = async () => {
        if (!importPayload.trim()) {
            setErrorMessage("Paste JSON payload to import.");
            return;
        }
        try {
            const parsed = JSON.parse(importPayload);
            await openevolveApi_1.openevolveApi.importWorkflowTemplates(parsed, apiConfig);
            setImportPayload("");
            setStatusMessage("Templates imported.");
            await loadTemplates();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to import templates.");
        }
    };
    const applyToOrchestrator = (template) => {
        try {
            globalThis.localStorage?.setItem("openevolve_active_workflow_template", JSON.stringify(template));
            setStatusMessage(`Loaded template ${template.name} into orchestrator cache.`);
        }
        catch {
            setErrorMessage("Failed to store template in local cache.");
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Workflow Templates</card_1.CardTitle>
          <card_1.CardDescription>Save and reuse workflow configurations.</card_1.CardDescription>
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
            <button_1.Button variant="outline" onClick={loadTemplates}>
              Refresh
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Create Template</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <label_1.Label>Name</label_1.Label>
                  <input_1.Input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })}/>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Tags (comma separated)</label_1.Label>
                  <input_1.Input value={form.tags} onChange={(event) => setForm({ ...form, tags: event.target.value })}/>
                </div>
              </div>
              <div className="space-y-2">
                <label_1.Label>Description</label_1.Label>
                <input_1.Input value={form.description} onChange={(event) => setForm({ ...form, description: event.target.value })}/>
              </div>
              <div className="space-y-2">
                <label_1.Label>Config (JSON)</label_1.Label>
                <textarea_1.Textarea value={form.config} onChange={(event) => setForm({ ...form, config: event.target.value })} className="min-h-[160px]"/>
              </div>
              <button_1.Button onClick={handleCreateTemplate}>Save Template</button_1.Button>
            </card_1.CardContent>
          </card_1.Card>

          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {templates.map((template) => (<card_1.Card key={template.id}>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">{template.name}</card_1.CardTitle>
                  <card_1.CardDescription>{template.description || "No description"}</card_1.CardDescription>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  <div className="flex flex-wrap gap-2">
                    {template.tags?.map((tag) => (<badge_1.Badge key={tag} variant="secondary">
                        {tag}
                      </badge_1.Badge>))}
                  </div>
                  <textarea_1.Textarea readOnly value={JSON.stringify(template.config, null, 2)} className="min-h-[140px]"/>
                  <div className="flex gap-2">
                    <button_1.Button size="sm" variant="outline" onClick={() => applyToOrchestrator(template)}>
                      Use in Orchestrator
                    </button_1.Button>
                    <button_1.Button size="sm" variant="destructive" onClick={() => handleDeleteTemplate(template.id)}>
                      Delete
                    </button_1.Button>
                  </div>
                </card_1.CardContent>
              </card_1.Card>))}
            {templates.length === 0 && (<div className="text-sm text-muted-foreground">No templates saved yet.</div>)}
          </div>

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Import / Export</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <button_1.Button variant="outline" onClick={handleExport}>
                Export Templates
              </button_1.Button>
              <textarea_1.Textarea value={importPayload} onChange={(event) => setImportPayload(event.target.value)} placeholder="Paste exported templates JSON" className="min-h-[140px]"/>
              <button_1.Button onClick={handleImport}>Import Templates</button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.WorkflowTemplatesTab = WorkflowTemplatesTab;
//# sourceMappingURL=WorkflowTemplatesTab.js.map