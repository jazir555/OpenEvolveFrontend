import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { openevolveApi } from "../../../lib/openevolveApi";
import type { WorkflowResults } from "../../../lib/types";

type TemplateMap = Record<string, string>;

const readStorage = <T,>(key: string, fallback: T): T => {
  try {
    const raw = globalThis.localStorage?.getItem(key);
    if (!raw) {
      return fallback;
    }
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

const writeStorage = (key: string, value: unknown) => {
  try {
    globalThis.localStorage?.setItem(key, JSON.stringify(value));
  } catch {
    // ignore storage errors
  }
};

const downloadFile = (filename: string, content: string, mime = "text/plain") => {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

const buildDefaultMarkdown = (results: WorkflowResults) => {
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

const buildDefaultHtml = (results: WorkflowResults) => {
  const finalContent = results.final_solution?.content ?? "No final solution.";
  const subSolutions = Object.entries(results.sub_problem_solutions || {})
    .map(
      ([key, value]) =>
        `<section><h3>${key}</h3><pre>${value.content}</pre></section>`,
    )
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

const applyTemplate = (template: string, results: WorkflowResults) => {
  const replacements: Record<string, string> = {
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

export const ReportTemplatesTab: React.FC = () => {
  const [templates, setTemplates] = useState<TemplateMap>(() =>
    readStorage<TemplateMap>("openevolve_report_templates", {}),
  );
  const [templateName, setTemplateName] = useState("");
  const [templateContent, setTemplateContent] = useState("");
  const [templateError, setTemplateError] = useState<string | null>(null);

  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [workflowOptions, setWorkflowOptions] = useState<string[]>([]);
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<string>("");
  const [workflowResults, setWorkflowResults] = useState<WorkflowResults | null>(null);
  const [reportFormat, setReportFormat] = useState<"markdown" | "html">("markdown");
  const [selectedTemplate, setSelectedTemplate] = useState<string>("default");
  const [reportOutput, setReportOutput] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [metricsFormat, setMetricsFormat] = useState<"json" | "csv">("json");
  const [includeStats, setIncludeStats] = useState(true);
  const [includeWorkflows, setIncludeWorkflows] = useState(true);

  const saveTemplates = (next: TemplateMap) => {
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
    } catch {
      setTemplateError("Template content must be valid JSON.");
      return;
    }
    const next = { ...templates, [templateName.trim()]: templateContent };
    saveTemplates(next);
    setTemplateName("");
    setTemplateContent("");
  };

  const handleDeleteTemplate = (name: string) => {
    const next = { ...templates };
    delete next[name];
    saveTemplates(next);
  };

  const loadWorkflows = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.listWorkflows(apiConfig);
      setWorkflowOptions(response.workflows.map((wf) => wf.workflow_id));
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load workflows.");
    } finally {
      setLoading(false);
    }
  };

  const loadResults = async (workflowId: string) => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const results = await openevolveApi.getWorkflowResults(workflowId, apiConfig);
      setWorkflowResults(results);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load workflow results.");
    } finally {
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
        const templateString =
          typeof parsed === "string"
            ? parsed
            : parsed?.template || parsed?.body || JSON.stringify(parsed, null, 2);
        output = applyTemplate(templateString, workflowResults);
      } catch {
        output =
          reportFormat === "html"
            ? buildDefaultHtml(workflowResults)
            : buildDefaultMarkdown(workflowResults);
      }
    } else {
      output =
        reportFormat === "html"
          ? buildDefaultHtml(workflowResults)
          : buildDefaultMarkdown(workflowResults);
    }
    setReportOutput(output);
  };

  useEffect(() => {
    loadWorkflows();
  }, [apiConfig.apiKey]);

  const exportMetrics = async () => {
    setErrorMessage(null);
    try {
      const payload: Record<string, unknown> = {};
      if (includeStats) {
        payload.statistics = await openevolveApi.getStatistics(apiConfig);
      }
      if (includeWorkflows) {
        const list = await openevolveApi.listWorkflows(apiConfig);
        payload.workflows = list.workflows ?? [];
      }
      if (metricsFormat === "json") {
        downloadFile("workflow_metrics.json", JSON.stringify(payload, null, 2), "application/json");
      } else {
        const workflows = (payload.workflows as any[]) ?? [];
        const csvRows = ["workflow_id,status,current_stage,progress"];
        workflows.forEach((wf) => {
          csvRows.push(
            `${wf.workflow_id},${wf.status},${wf.current_stage},${wf.progress}`,
          );
        });
        downloadFile("workflow_metrics.csv", csvRows.join("\n"), "text/csv");
      }
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to export metrics.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Report Templates</CardTitle>
          <CardDescription>Create and manage custom report templates.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Existing Templates</Label>
              {Object.keys(templates).length === 0 && (
                <div className="text-sm text-muted-foreground">No templates yet.</div>
              )}
              <div className="space-y-2">
                {Object.entries(templates).map(([name, content]) => (
                  <div key={name} className="rounded border p-2 text-sm space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold">{name}</span>
                      <Button variant="ghost" onClick={() => handleDeleteTemplate(name)}>
                        Delete
                      </Button>
                    </div>
                    <pre className="max-h-40 overflow-auto rounded bg-muted p-2 text-xs">
                      {content}
                    </pre>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <Label>Create New Template</Label>
              <Input
                value={templateName}
                onChange={(event) => setTemplateName(event.target.value)}
                placeholder="Template name"
              />
              <Textarea
                value={templateContent}
                onChange={(event) => setTemplateContent(event.target.value)}
                placeholder='{"template": "## {{workflow_id}}"}'
                className="min-h-[180px]"
              />
              {templateError ? <div className="text-sm text-red-500">{templateError}</div> : null}
              <Button onClick={handleSaveTemplate}>Save Template</Button>
            </div>
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <Button variant="outline" onClick={() => downloadFile("report_templates.json", JSON.stringify(templates, null, 2), "application/json")}>
              Download Templates
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Report Generator</CardTitle>
          <CardDescription>Generate workflow reports using saved templates.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-[2fr_1fr]">
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
            <div className="flex items-end">
              <Button variant="outline" onClick={loadWorkflows} disabled={loading}>
                Refresh Workflows
              </Button>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Workflow</Label>
              <Select
                value={selectedWorkflowId}
                onValueChange={(value) => {
                  setSelectedWorkflowId(value);
                  if (value) {
                    loadResults(value);
                  }
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select workflow" />
                </SelectTrigger>
                <SelectContent>
                  {workflowOptions.map((workflowId) => (
                    <SelectItem key={workflowId} value={workflowId}>
                      {workflowId}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Template</Label>
              <Select value={selectedTemplate} onValueChange={setSelectedTemplate}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">Default</SelectItem>
                  {Object.keys(templates).map((name) => (
                    <SelectItem key={name} value={name}>
                      {name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Format</Label>
              <Select
                value={reportFormat}
                onValueChange={(value) => setReportFormat(value as "markdown" | "html")}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="markdown">Markdown</SelectItem>
                  <SelectItem value="html">HTML</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <div className="flex gap-2">
            <Button onClick={generateReport} disabled={!workflowResults}>
              Generate Report
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                if (!reportOutput) return;
                const extension = reportFormat === "html" ? "html" : "md";
                const mime = reportFormat === "html" ? "text/html" : "text/markdown";
                downloadFile(`workflow_report.${extension}`, reportOutput, mime);
              }}
              disabled={!reportOutput}
            >
              Download Report
            </Button>
          </div>

          <div className="space-y-2">
            <Label>Report Preview</Label>
            <Textarea value={reportOutput} readOnly className="min-h-[200px]" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Metrics Export</CardTitle>
          <CardDescription>Download workflow metrics for offline analysis.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Format</Label>
              <Select
                value={metricsFormat}
                onValueChange={(value) => setMetricsFormat(value as "json" | "csv")}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="json">JSON</SelectItem>
                  <SelectItem value="csv">CSV</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={includeStats}
                onChange={(event) => setIncludeStats(event.target.checked)}
              />
              <span className="text-sm">Include Statistics</span>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={includeWorkflows}
                onChange={(event) => setIncludeWorkflows(event.target.checked)}
              />
              <span className="text-sm">Include Workflows</span>
            </div>
          </div>
          <Button variant="outline" onClick={exportMetrics}>
            Export Metrics
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};
