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
exports.ExportImportTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const tabs_1 = require("@/components/ui/tabs");
const badge_1 = require("@/components/ui/badge");
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
        // ignore storage errors
    }
};
const readStorageJson = (key, fallback) => {
    try {
        const raw = globalThis.localStorage?.getItem(key);
        if (!raw)
            return fallback;
        return JSON.parse(raw);
    }
    catch {
        return fallback;
    }
};
const writeStorageJson = (key, value) => {
    try {
        globalThis.localStorage?.setItem(key, JSON.stringify(value));
    }
    catch {
        // ignore storage errors
    }
};
const downloadFile = (filename, content, mime = "application/json") => {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
};
const jsonDownload = (filename, payload) => {
    downloadFile(filename, JSON.stringify(payload, null, 2), "application/json");
};
const buildMarkdownExport = (projectName, protocolText) => {
    const safeText = protocolText || "";
    return `# ${projectName || "Untitled Project"}\n\n*Exported on: ${new Date().toISOString()}*\n\n${safeText}`;
};
const buildPdfTemplate = (projectName, protocolText) => {
    return `# Protocol Export\n**Project:** ${projectName || "Untitled Project"}\n**Export Date:** ${new Date().toISOString()}\n\n## Content\n${protocolText || ""}`;
};
const computeShareId = async (payload) => {
    const encoder = new TextEncoder();
    const data = encoder.encode(JSON.stringify(payload));
    const digest = await crypto.subtle.digest("SHA-256", data);
    const hashArray = Array.from(new Uint8Array(digest));
    return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
};
const ExportImportTab = ({ state, updateState }) => {
    const [projectName, setProjectName] = (0, react_1.useState)(() => readStorage("openevolve_project_name", "Untitled Project"));
    const [projectDescription, setProjectDescription] = (0, react_1.useState)(() => readStorage("openevolve_project_description", ""));
    const [exportOptions, setExportOptions] = (0, react_1.useState)({
        includeState: true,
        includeHistory: true,
        includeTemplates: true,
        includeSettings: true,
        includeTasks: true,
        includeNotifications: true,
        includeSensitive: false,
    });
    const [exportPayload, setExportPayload] = (0, react_1.useState)(null);
    const [shareId, setShareId] = (0, react_1.useState)("");
    const [importPayload, setImportPayload] = (0, react_1.useState)(null);
    const [importMessage, setImportMessage] = (0, react_1.useState)(null);
    const [importOptions, setImportOptions] = (0, react_1.useState)({
        applyState: true,
        applyHistory: true,
        applyTemplates: true,
        applySettings: true,
        applyTasks: true,
        applyNotifications: true,
    });
    const storedTemplates = (0, react_1.useMemo)(() => readStorageJson("openevolve_report_templates", {}), []);
    const storedWorkflowTemplate = (0, react_1.useMemo)(() => readStorageJson("openevolve_active_workflow_template", null), []);
    const storedTasks = (0, react_1.useMemo)(() => readStorageJson("openevolve_tasks", []), []);
    const storedNotifications = (0, react_1.useMemo)(() => readStorageJson("openevolve_notifications", []), []);
    const buildExportPayload = () => {
        const settings = {
            openevolve_parameter_overrides: readStorage("openevolve_parameter_overrides", "{}"),
            openevolve_api_base: readStorage("openevolve_api_base", ""),
            openevolve_selected_model: readStorage("openevolve_selected_model", ""),
            openevolve_llm_base_url: readStorage("openevolve_llm_base_url", ""),
            openevolve_llm_model: readStorage("openevolve_llm_model", ""),
        };
        if (exportOptions.includeSensitive) {
            settings.openevolve_api_key = readStorage("openevolve_api_key", "");
            settings.openevolve_llm_api_key = readStorage("openevolve_llm_api_key", "");
            settings.openevolve_github_token = readStorage("openevolve_github_token", "");
        }
        const payload = {
            metadata: {
                project_name: projectName,
                project_description: projectDescription,
                exported_at: new Date().toISOString(),
            },
        };
        if (exportOptions.includeState) {
            payload.state = {
                protocolText: state.protocolText,
                evolutionCurrentBest: state.evolutionCurrentBest,
                evolutionStatusMessage: state.evolutionStatusMessage,
                adversarialStatusMessage: state.adversarialStatusMessage,
                evolutionBestScore: state.evolutionBestScore,
            };
        }
        if (exportOptions.includeHistory) {
            payload.history = {
                evolutionHistory: state.evolutionHistory,
                adversarialResults: state.adversarialResults,
            };
        }
        if (exportOptions.includeTemplates) {
            payload.templates = {
                report_templates: storedTemplates,
                workflow_template: storedWorkflowTemplate,
            };
        }
        if (exportOptions.includeSettings) {
            payload.settings = settings;
        }
        if (exportOptions.includeTasks) {
            payload.tasks = storedTasks;
        }
        if (exportOptions.includeNotifications) {
            payload.notifications = storedNotifications;
        }
        return payload;
    };
    const exportProject = async () => {
        const payload = buildExportPayload();
        setExportPayload(payload);
        const generatedShareId = await computeShareId(payload);
        setShareId(generatedShareId);
        jsonDownload(`openevolve_export_${generatedShareId}.json`, payload);
    };
    const handleImportFile = async (file) => {
        setImportMessage(null);
        try {
            const text = await file.text();
            const parsed = JSON.parse(text);
            setImportPayload(parsed);
            setImportMessage("Import file loaded.");
        }
        catch (error) {
            setImportPayload(null);
            setImportMessage(error?.message ?? "Failed to parse import file.");
        }
    };
    const applyImport = () => {
        if (!importPayload)
            return;
        setImportMessage(null);
        try {
            if (importOptions.applyState && importPayload.state) {
                const payloadState = importPayload.state;
                updateState({
                    protocolText: payloadState.protocolText ?? state.protocolText,
                    evolutionCurrentBest: payloadState.evolutionCurrentBest ?? state.evolutionCurrentBest,
                    evolutionStatusMessage: payloadState.evolutionStatusMessage ?? state.evolutionStatusMessage,
                    adversarialStatusMessage: payloadState.adversarialStatusMessage ?? state.adversarialStatusMessage,
                    evolutionBestScore: payloadState.evolutionBestScore ?? state.evolutionBestScore,
                });
            }
            if (importOptions.applyHistory && importPayload.history) {
                const history = importPayload.history;
                updateState({
                    evolutionHistory: history.evolutionHistory ?? state.evolutionHistory,
                    adversarialResults: history.adversarialResults ?? state.adversarialResults,
                });
            }
            if (importOptions.applyTemplates && importPayload.templates) {
                const templates = importPayload.templates;
                if (templates.report_templates) {
                    writeStorageJson("openevolve_report_templates", templates.report_templates);
                }
                if (templates.workflow_template) {
                    writeStorageJson("openevolve_active_workflow_template", templates.workflow_template);
                }
            }
            if (importOptions.applySettings && importPayload.settings) {
                const settings = importPayload.settings;
                Object.entries(settings).forEach(([key, value]) => {
                    if (typeof value === "string") {
                        writeStorage(key, value);
                    }
                    else {
                        writeStorageJson(key, value);
                    }
                });
            }
            if (importOptions.applyTasks && importPayload.tasks) {
                writeStorageJson("openevolve_tasks", importPayload.tasks);
            }
            if (importOptions.applyNotifications && importPayload.notifications) {
                writeStorageJson("openevolve_notifications", importPayload.notifications);
            }
            if (importPayload.metadata) {
                const metadata = importPayload.metadata;
                if (typeof metadata.project_name === "string") {
                    setProjectName(metadata.project_name);
                    writeStorage("openevolve_project_name", metadata.project_name);
                }
                if (typeof metadata.project_description === "string") {
                    setProjectDescription(metadata.project_description);
                    writeStorage("openevolve_project_description", metadata.project_description);
                }
            }
            setImportMessage("Import applied successfully.");
        }
        catch (error) {
            setImportMessage(error?.message ?? "Failed to apply import.");
        }
    };
    const exportMarkdown = () => {
        const content = buildMarkdownExport(projectName, state.protocolText);
        downloadFile("protocol_export.md", content, "text/markdown");
    };
    const exportText = () => {
        downloadFile("protocol_export.txt", state.protocolText || "", "text/plain");
    };
    const exportPdfTemplate = () => {
        const content = buildPdfTemplate(projectName, state.protocolText);
        downloadFile("protocol_pdf_template.txt", content, "text/plain");
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Export / Import Manager</card_1.CardTitle>
          <card_1.CardDescription>Backup, transfer, and restore OpenEvolve project data.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">Project Name</label>
              <input_1.Input value={projectName} onChange={(event) => {
            const value = event.target.value;
            setProjectName(value);
            writeStorage("openevolve_project_name", value);
        }}/>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Project Description</label>
              <input_1.Input value={projectDescription} onChange={(event) => {
            const value = event.target.value;
            setProjectDescription(value);
            writeStorage("openevolve_project_description", value);
        }}/>
            </div>
          </div>

          <tabs_1.Tabs defaultValue="export" className="w-full">
            <tabs_1.TabsList className="flex flex-wrap gap-2">
              <tabs_1.TabsTrigger value="export">Export</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="import">Import</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="backup">Backup</tabs_1.TabsTrigger>
            </tabs_1.TabsList>

            <tabs_1.TabsContent value="export" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Export Options</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-3 text-sm">
                    {[
            ["includeState", "Core State (protocol, status)", exportOptions.includeState],
            ["includeHistory", "History (evolution + adversarial)", exportOptions.includeHistory],
            ["includeTemplates", "Templates (reports + workflow)", exportOptions.includeTemplates],
            ["includeSettings", "Settings (overrides + base URLs)", exportOptions.includeSettings],
            ["includeTasks", "Tasks", exportOptions.includeTasks],
            ["includeNotifications", "Notifications", exportOptions.includeNotifications],
            ["includeSensitive", "Include sensitive keys (API tokens)", exportOptions.includeSensitive],
        ].map(([key, label, checked]) => (<label key={String(key)} className="flex items-center gap-2">
                        <input type="checkbox" checked={Boolean(checked)} onChange={(event) => setExportOptions((prev) => ({
                ...prev,
                [key]: event.target.checked,
            }))}/>
                        <span>{label}</span>
                      </label>))}
                  </card_1.CardContent>
                </card_1.Card>

                <card_1.Card>
                  <card_1.CardHeader>
                    <card_1.CardTitle className="text-sm">Export Actions</card_1.CardTitle>
                  </card_1.CardHeader>
                  <card_1.CardContent className="space-y-3 text-sm">
                    <button_1.Button onClick={exportProject}>Export JSON</button_1.Button>
                    <div className="flex flex-wrap gap-2">
                      <button_1.Button variant="outline" onClick={exportMarkdown}>
                        Export Markdown
                      </button_1.Button>
                      <button_1.Button variant="outline" onClick={exportText}>
                        Export Text
                      </button_1.Button>
                      <button_1.Button variant="outline" onClick={exportPdfTemplate}>
                        Export PDF Template
                      </button_1.Button>
                    </div>
                    {shareId ? (<div className="rounded border p-2 text-xs">
                        Share ID: <badge_1.Badge variant="secondary">{shareId}</badge_1.Badge>
                      </div>) : null}
                    {exportPayload ? (<textarea_1.Textarea value={JSON.stringify(exportPayload, null, 2)} readOnly className="min-h-[180px]"/>) : null}
                  </card_1.CardContent>
                </card_1.Card>
              </div>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="import" className="mt-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Import Project</card_1.CardTitle>
                  <card_1.CardDescription>Upload an export JSON file to restore data.</card_1.CardDescription>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3 text-sm">
                  <input_1.Input type="file" accept="application/json" onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
                handleImportFile(file);
            }
        }}/>
                  {importMessage ? (<div className="text-xs text-muted-foreground">{importMessage}</div>) : null}
                  {importPayload ? (<div className="space-y-3">
                      <div className="grid gap-2 md:grid-cols-2">
                        {[
                ["applyState", "Apply core state"],
                ["applyHistory", "Apply history"],
                ["applyTemplates", "Apply templates"],
                ["applySettings", "Apply settings"],
                ["applyTasks", "Apply tasks"],
                ["applyNotifications", "Apply notifications"],
            ].map(([key, label]) => (<label key={String(key)} className="flex items-center gap-2">
                            <input type="checkbox" checked={Boolean(importOptions[key])} onChange={(event) => setImportOptions((prev) => ({
                    ...prev,
                    [key]: event.target.checked,
                }))}/>
                            <span>{label}</span>
                          </label>))}
                      </div>
                      <button_1.Button onClick={applyImport}>Apply Import</button_1.Button>
                      <textarea_1.Textarea value={JSON.stringify(importPayload, null, 2)} readOnly className="min-h-[180px]"/>
                    </div>) : null}
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="backup" className="mt-4">
              <card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">Project Backup</card_1.CardTitle>
                  <card_1.CardDescription>Create or restore a full backup snapshot.</card_1.CardDescription>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-3 text-sm">
                  <button_1.Button variant="outline" onClick={() => {
            const payload = buildExportPayload();
            jsonDownload(`openevolve_backup_${Date.now()}.json`, payload);
        }}>
                    Download Full Backup
                  </button_1.Button>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Restore Backup</label>
                    <input_1.Input type="file" accept="application/json" onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
                handleImportFile(file);
            }
        }}/>
                  </div>
                </card_1.CardContent>
              </card_1.Card>
            </tabs_1.TabsContent>
          </tabs_1.Tabs>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.ExportImportTab = ExportImportTab;
//# sourceMappingURL=ExportImportTab.js.map