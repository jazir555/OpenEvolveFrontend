import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { openevolveApi } from "@/lib/openevolveApi";
import type { WorkflowTemplate } from "@/lib/types";

const downloadJson = (filename: string, payload: unknown) => {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

export const WorkflowTemplatesTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [templates, setTemplates] = useState<WorkflowTemplate[]>([]);
  const [form, setForm] = useState({
    name: "",
    description: "",
    tags: "",
    config: "",
  });
  const [importPayload, setImportPayload] = useState("");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const loadTemplates = async () => {
    setErrorMessage(null);
    try {
      const response = await openevolveApi.listWorkflowTemplates(apiConfig);
      setTemplates(response.templates ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load templates.");
    }
  };

  useEffect(() => {
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
      const template = await openevolveApi.createWorkflowTemplate(
        {
          name: form.name.trim(),
          description: form.description.trim(),
          tags: form.tags ? form.tags.split(",").map((tag) => tag.trim()) : [],
          config,
        },
        apiConfig,
      );
      setStatusMessage(`Template ${template.name} created.`);
      setForm({ name: "", description: "", tags: "", config: "" });
      await loadTemplates();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to create template.");
    }
  };

  const handleDeleteTemplate = async (templateId: string) => {
    setErrorMessage(null);
    try {
      await openevolveApi.deleteWorkflowTemplate(templateId, apiConfig);
      await loadTemplates();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to delete template.");
    }
  };

  const handleExport = async () => {
    try {
      const data = await openevolveApi.exportWorkflowTemplates(apiConfig);
      downloadJson("workflow_templates.json", data);
    } catch (error: any) {
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
      await openevolveApi.importWorkflowTemplates(parsed, apiConfig);
      setImportPayload("");
      setStatusMessage("Templates imported.");
      await loadTemplates();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to import templates.");
    }
  };

  const applyToOrchestrator = (template: WorkflowTemplate) => {
    try {
      globalThis.localStorage?.setItem(
        "openevolve_active_workflow_template",
        JSON.stringify(template),
      );
      setStatusMessage(`Loaded template ${template.name} into orchestrator cache.`);
    } catch {
      setErrorMessage("Failed to store template in local cache.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Workflow Templates</CardTitle>
          <CardDescription>Save and reuse workflow configurations.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
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
            <Button variant="outline" onClick={loadTemplates}>
              Refresh
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Create Template</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Name</Label>
                  <Input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} />
                </div>
                <div className="space-y-2">
                  <Label>Tags (comma separated)</Label>
                  <Input value={form.tags} onChange={(event) => setForm({ ...form, tags: event.target.value })} />
                </div>
              </div>
              <div className="space-y-2">
                <Label>Description</Label>
                <Input value={form.description} onChange={(event) => setForm({ ...form, description: event.target.value })} />
              </div>
              <div className="space-y-2">
                <Label>Config (JSON)</Label>
                <Textarea
                  value={form.config}
                  onChange={(event) => setForm({ ...form, config: event.target.value })}
                  className="min-h-[160px]"
                />
              </div>
              <Button onClick={handleCreateTemplate}>Save Template</Button>
            </CardContent>
          </Card>

          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {templates.map((template) => (
              <Card key={template.id}>
                <CardHeader>
                  <CardTitle className="text-sm">{template.name}</CardTitle>
                  <CardDescription>{template.description || "No description"}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex flex-wrap gap-2">
                    {template.tags?.map((tag) => (
                      <Badge key={tag} variant="secondary">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                  <Textarea
                    readOnly
                    value={JSON.stringify(template.config, null, 2)}
                    className="min-h-[140px]"
                  />
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline" onClick={() => applyToOrchestrator(template)}>
                      Use in Orchestrator
                    </Button>
                    <Button size="sm" variant="destructive" onClick={() => handleDeleteTemplate(template.id)}>
                      Delete
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
            {templates.length === 0 && (
              <div className="text-sm text-muted-foreground">No templates saved yet.</div>
            )}
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Import / Export</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button variant="outline" onClick={handleExport}>
                Export Templates
              </Button>
              <Textarea
                value={importPayload}
                onChange={(event) => setImportPayload(event.target.value)}
                placeholder="Paste exported templates JSON"
                className="min-h-[140px]"
              />
              <Button onClick={handleImport}>Import Templates</Button>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
};
