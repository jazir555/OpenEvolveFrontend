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
exports.ReportTemplatesTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const separator_1 = require("@/components/ui/separator");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const readStorage = (key, fallback) => {
    try {
        const raw = globalThis.localStorage?.getItem(key);
        if (!raw) {
            return fallback;
        }
        return JSON.parse(raw);
    }
    catch {
        return fallback;
    }
};
const writeStorage = (key, value) => {
    try {
        globalThis.localStorage?.setItem(key, JSON.stringify(value));
    }
    catch {
        // ignore storage errors
    }
};
const downloadFile = (filename, content, mime = "text/plain") => {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
};
const buildDefaultMarkdown = (results) => {
    const finalContent = results.final_solution?.content ?? "No final solution.";
    const subSolutions = Object.entries(results.sub_problem_solutions || {})
        .map(([key, value]) => `### ${key}\n\n${value.content}`)
        .join("\n\n");
    return `# OpenEvolve Workflow Report

## Workflow Summary
- Workflow ID: ${results.workflow_id}
- Status: ${results.status}
- Refinement Loops: ${results.refinement_loops}
- Execution Time: ${results.execution_time ?? "n/a"}

## Problem Statement
${results.problem_statement}

## Final Solution
${finalContent}

## Sub-problem Solutions
${subSolutions || "No sub-problem solutions recorded."}
`;
};
const buildDefaultHtml = (results) => {
    const finalContent = results.final_solution?.content ?? "No final solution.";
    const subSolutions = Object.entries(results.sub_problem_solutions || {})
        .map(([key, value]) => `<section><h3>${key}</h3><pre>${value.content}</pre></section>`)
        .join("");
    return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>OpenEvolve Workflow Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }
    h1, h2, h3 { color: #1d4ed8; }
    pre { background: #f3f4f6; padding: 12px; border-radius: 6px; white-space: pre-wrap; }
    .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; }
    .meta div { background: #f8fafc; padding: 8px; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>OpenEvolve Workflow Report</h1>
  <section class="meta">
    <div><strong>Workflow ID:</strong> ${results.workflow_id}</div>
    <div><strong>Status:</strong> ${results.status}</div>
    <div><strong>Refinement Loops:</strong> ${results.refinement_loops}</div>
    <div><strong>Execution Time:</strong> ${results.execution_time ?? "n/a"}</div>
  </section>
  <section>
    <h2>Problem Statement</h2>
    <pre>${results.problem_statement}</pre>
  </section>
  <section>
    <h2>Final Solution</h2>
    <pre>${finalContent}</pre>
  </section>
  <section>
    <h2>Sub-problem Solutions</h2>
    ${subSolutions || "<p>No sub-problem solutions recorded.</p>"}
  </section>
</body>
</html>`;
};
const applyTemplate = (template, results) => {
    const replacements = {
        "{{workflow_id}}": results.workflow_id,
        "{{status}}": results.status,
        "{{problem_statement}}": results.problem_statement,
        "{{final_solution}}": results.final_solution?.content ?? "",
        "{{refinement_loops}}": String(results.refinement_loops),
        "{{execution_time}}": String(results.execution_time ?? ""),
    };
    let output = template;
    Object.entries(replacements).forEach(([key, value]) => {
        output = output.split(key).join(value);
    });
    return output;
};
const ReportTemplatesTab = () => {
    const [templates, setTemplates] = (0, react_1.useState)(() => readStorage("openevolve_report_templates", {}));
    const [templateName, setTemplateName] = (0, react_1.useState)("");
    const [templateContent, setTemplateContent] = (0, react_1.useState)("");
    const [templateError, setTemplateError] = (0, react_1.useState)(null);
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [workflowOptions, setWorkflowOptions] = (0, react_1.useState)([]);
    const [selectedWorkflowId, setSelectedWorkflowId] = (0, react_1.useState)("");
    const [workflowResults, setWorkflowResults] = (0, react_1.useState)(null);
    const [reportFormat, setReportFormat] = (0, react_1.useState)("markdown");
    const [selectedTemplate, setSelectedTemplate] = (0, react_1.useState)("default");
    const [reportOutput, setReportOutput] = (0, react_1.useState)("");
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [metricsFormat, setMetricsFormat] = (0, react_1.useState)("json");
    const [includeStats, setIncludeStats] = (0, react_1.useState)(true);
    const [includeWorkflows, setIncludeWorkflows] = (0, react_1.useState)(true);
    const saveTemplates = (next) => {
        setTemplates(next);
        writeStorage("openevolve_report_templates", next);
    };
    const handleSaveTemplate = () => {
        setTemplateError(null);
        if (!templateName.trim() || !templateContent.trim()) {
            setTemplateError("Template name and content are required.");
            return;
        }
        try {
            JSON.parse(templateContent);
        }
        catch {
            setTemplateError("Template content must be valid JSON.");
            return;
        }
        const next = { ...templates, [templateName.trim()]: templateContent };
        saveTemplates(next);
        setTemplateName("");
        setTemplateContent("");
    };
    const handleDeleteTemplate = (name) => {
        const next = { ...templates };
        delete next[name];
        saveTemplates(next);
    };
    const loadWorkflows = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.listWorkflows(apiConfig);
            setWorkflowOptions(response.workflows.map((wf) => wf.workflow_id));
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load workflows.");
        }
        finally {
            setLoading(false);
        }
    };
    const loadResults = async (workflowId) => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const results = await openevolveApi_1.openevolveApi.getWorkflowResults(workflowId, apiConfig);
            setWorkflowResults(results);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load workflow results.");
        }
        finally {
            setLoading(false);
        }
    };
    const generateReport = () => {
        if (!workflowResults) {
            setErrorMessage("Load workflow results first.");
            return;
        }
        let output = "";
        if (selectedTemplate !== "default" && templates[selectedTemplate]) {
            try {
                const parsed = JSON.parse(templates[selectedTemplate]);
                const templateString = typeof parsed === "string"
                    ? parsed
                    : parsed?.template || parsed?.body || JSON.stringify(parsed, null, 2);
                output = applyTemplate(templateString, workflowResults);
            }
            catch {
                output =
                    reportFormat === "html"
                        ? buildDefaultHtml(workflowResults)
                        : buildDefaultMarkdown(workflowResults);
            }
        }
        else {
            output =
                reportFormat === "html"
                    ? buildDefaultHtml(workflowResults)
                    : buildDefaultMarkdown(workflowResults);
        }
        setReportOutput(output);
    };
    (0, react_1.useEffect)(() => {
        loadWorkflows();
    }, [apiConfig.apiKey]);
    const exportMetrics = async () => {
        setErrorMessage(null);
        try {
            const payload = {};
            if (includeStats) {
                payload.statistics = await openevolveApi_1.openevolveApi.getStatistics(apiConfig);
            }
            if (includeWorkflows) {
                const list = await openevolveApi_1.openevolveApi.listWorkflows(apiConfig);
                payload.workflows = list.workflows ?? [];
            }
            if (metricsFormat === "json") {
                downloadFile("workflow_metrics.json", JSON.stringify(payload, null, 2), "application/json");
            }
            else {
                const workflows = payload.workflows ?? [];
                const csvRows = ["workflow_id,status,current_stage,progress"];
                workflows.forEach((wf) => {
                    csvRows.push(`${wf.workflow_id},${wf.status},${wf.current_stage},${wf.progress}`);
                });
                downloadFile("workflow_metrics.csv", csvRows.join("\n"), "text/csv");
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to export metrics.");
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Report Templates</card_1.CardTitle>
          <card_1.CardDescription>Create and manage custom report templates.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>Existing Templates</label_1.Label>
              {Object.keys(templates).length === 0 && (<div className="text-sm text-muted-foreground">No templates yet.</div>)}
              <div className="space-y-2">
                {Object.entries(templates).map(([name, content]) => (<div key={name} className="rounded border p-2 text-sm space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold">{name}</span>
                      <button_1.Button variant="ghost" onClick={() => handleDeleteTemplate(name)}>
                        Delete
                      </button_1.Button>
                    </div>
                    <pre className="max-h-40 overflow-auto rounded bg-muted p-2 text-xs">
                      {content}
                    </pre>
                  </div>))}
              </div>
            </div>

            <div className="space-y-2">
              <label_1.Label>Create New Template</label_1.Label>
              <input_1.Input value={templateName} onChange={(event) => setTemplateName(event.target.value)} placeholder="Template name"/>
              <textarea_1.Textarea value={templateContent} onChange={(event) => setTemplateContent(event.target.value)} placeholder='{"template": "## {{workflow_id}}"}' className="min-h-[180px]"/>
              {templateError ? <div className="text-sm text-red-500">{templateError}</div> : null}
              <button_1.Button onClick={handleSaveTemplate}>Save Template</button_1.Button>
            </div>
          </div>

          <separator_1.Separator />

          <div className="flex items-center justify-between">
            <button_1.Button variant="outline" onClick={() => downloadFile("report_templates.json", JSON.stringify(templates, null, 2), "application/json")}>
              Download Templates
            </button_1.Button>
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Report Generator</card_1.CardTitle>
          <card_1.CardDescription>Generate workflow reports using saved templates.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-[2fr_1fr]">
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
            <div className="flex items-end">
              <button_1.Button variant="outline" onClick={loadWorkflows} disabled={loading}>
                Refresh Workflows
              </button_1.Button>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <label_1.Label>Workflow</label_1.Label>
              <select_1.Select value={selectedWorkflowId} onValueChange={(value) => {
            setSelectedWorkflowId(value);
            if (value) {
                loadResults(value);
            }
        }}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue placeholder="Select workflow"/>
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  {workflowOptions.map((workflowId) => (<select_1.SelectItem key={workflowId} value={workflowId}>
                      {workflowId}
                    </select_1.SelectItem>))}
                </select_1.SelectContent>
              </select_1.Select>
            </div>
            <div className="space-y-2">
              <label_1.Label>Template</label_1.Label>
              <select_1.Select value={selectedTemplate} onValueChange={setSelectedTemplate}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue />
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  <select_1.SelectItem value="default">Default</select_1.SelectItem>
                  {Object.keys(templates).map((name) => (<select_1.SelectItem key={name} value={name}>
                      {name}
                    </select_1.SelectItem>))}
                </select_1.SelectContent>
              </select_1.Select>
            </div>
            <div className="space-y-2">
              <label_1.Label>Format</label_1.Label>
              <select_1.Select value={reportFormat} onValueChange={(value) => setReportFormat(value)}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue />
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  <select_1.SelectItem value="markdown">Markdown</select_1.SelectItem>
                  <select_1.SelectItem value="html">HTML</select_1.SelectItem>
                </select_1.SelectContent>
              </select_1.Select>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="flex gap-2">
            <button_1.Button onClick={generateReport} disabled={!workflowResults}>
              Generate Report
            </button_1.Button>
            <button_1.Button variant="outline" onClick={() => {
            if (!reportOutput)
                return;
            const extension = reportFormat === "html" ? "html" : "md";
            const mime = reportFormat === "html" ? "text/html" : "text/markdown";
            downloadFile(`workflow_report.${extension}`, reportOutput, mime);
        }} disabled={!reportOutput}>
              Download Report
            </button_1.Button>
          </div>

          <div className="space-y-2">
            <label_1.Label>Report Preview</label_1.Label>
            <textarea_1.Textarea value={reportOutput} readOnly className="min-h-[200px]"/>
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Metrics Export</card_1.CardTitle>
          <card_1.CardDescription>Download workflow metrics for offline analysis.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <label_1.Label>Format</label_1.Label>
              <select_1.Select value={metricsFormat} onValueChange={(value) => setMetricsFormat(value)}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue />
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  <select_1.SelectItem value="json">JSON</select_1.SelectItem>
                  <select_1.SelectItem value="csv">CSV</select_1.SelectItem>
                </select_1.SelectContent>
              </select_1.Select>
            </div>
            <div className="flex items-center gap-2">
              <input type="checkbox" checked={includeStats} onChange={(event) => setIncludeStats(event.target.checked)}/>
              <span className="text-sm">Include Statistics</span>
            </div>
            <div className="flex items-center gap-2">
              <input type="checkbox" checked={includeWorkflows} onChange={(event) => setIncludeWorkflows(event.target.checked)}/>
              <span className="text-sm">Include Workflows</span>
            </div>
          </div>
          <button_1.Button variant="outline" onClick={exportMetrics}>
            Export Metrics
          </button_1.Button>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.ReportTemplatesTab = ReportTemplatesTab;
//# sourceMappingURL=ReportTemplatesTab.js.map