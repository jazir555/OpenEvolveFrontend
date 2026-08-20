import React, { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";

interface OpenEvolveAppState {
  protocolText: string;
  evolutionRunning: boolean;
  adversarialRunning: boolean;
  evolutionHistory: any[];
  adversarialResults: any;
  evolutionCurrentBest: string;
  evolutionStatusMessage: string;
  adversarialStatusMessage: string;
  evolutionBestScore: number;
}

interface ExportImportTabProps {
  state: OpenEvolveAppState;
  updateState: (updates: Partial<OpenEvolveAppState>) => void;
}

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
    // ignore storage errors
  }
};

const readStorageJson = <T,>(key: string, fallback: T): T => {
  try {
    const raw = globalThis.localStorage?.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

const writeStorageJson = (key: string, value: unknown) => {
  try {
    globalThis.localStorage?.setItem(key, JSON.stringify(value));
  } catch {
    // ignore storage errors
  }
};

const downloadFile = (filename: string, content: string, mime = "application/json") => {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

const jsonDownload = (filename: string, payload: unknown) => {
  downloadFile(filename, JSON.stringify(payload, null, 2), "application/json");
};

const buildMarkdownExport = (projectName: string, protocolText: string) => {
  const safeText = protocolText || "";
  return `# ${projectName || "Untitled Project"}\n\n*Exported on: ${new Date().toISOString()}*\n\n${safeText}`;
};

const buildPdfTemplate = (projectName: string, protocolText: string) => {
  return `# Protocol Export\n**Project:** ${projectName || "Untitled Project"}\n**Export Date:** ${new Date().toISOString()}\n\n## Content\n${protocolText || ""}`;
};

const computeShareId = async (payload: unknown) => {
  const encoder = new TextEncoder();
  const data = encoder.encode(JSON.stringify(payload));
  const digest = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(digest));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
};

export const ExportImportTab: React.FC<ExportImportTabProps> = ({ state, updateState }) => {
  const [projectName, setProjectName] = useState(() => readStorage("openevolve_project_name", "Untitled Project"));
  const [projectDescription, setProjectDescription] = useState(() =>
    readStorage("openevolve_project_description", ""),
  );

  const [exportOptions, setExportOptions] = useState({
    includeState: true,
    includeHistory: true,
    includeTemplates: true,
    includeSettings: true,
    includeTasks: true,
    includeNotifications: true,
    includeSensitive: false,
  });

  const [exportPayload, setExportPayload] = useState<Record<string, unknown> | null>(null);
  const [shareId, setShareId] = useState<string>("");

  const [importPayload, setImportPayload] = useState<Record<string, unknown> | null>(null);
  const [importMessage, setImportMessage] = useState<string | null>(null);
  const [importOptions, setImportOptions] = useState({
    applyState: true,
    applyHistory: true,
    applyTemplates: true,
    applySettings: true,
    applyTasks: true,
    applyNotifications: true,
  });

  const storedTemplates = useMemo(
    () => readStorageJson<Record<string, string>>("openevolve_report_templates", {}),
    [],
  );
  const storedWorkflowTemplate = useMemo(
    () => readStorageJson<Record<string, unknown> | null>("openevolve_active_workflow_template", null),
    [],
  );
  const storedTasks = useMemo(() => readStorageJson("openevolve_tasks", [] as unknown[]), []);
  const storedNotifications = useMemo(
    () => readStorageJson("openevolve_notifications", [] as unknown[]),
    [],
  );

  const buildExportPayload = () => {
    const settings: Record<string, unknown> = {
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

    const payload: Record<string, unknown> = {
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

  const handleImportFile = async (file: File) => {
    setImportMessage(null);
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      setImportPayload(parsed);
      setImportMessage("Import file loaded.");
    } catch (error: any) {
      setImportPayload(null);
      setImportMessage(error?.message ?? "Failed to parse import file.");
    }
  };

  const applyImport = () => {
    if (!importPayload) return;
    setImportMessage(null);

    try {
      if (importOptions.applyState && importPayload.state) {
        const payloadState = importPayload.state as Partial<OpenEvolveAppState>;
        updateState({
          protocolText: payloadState.protocolText ?? state.protocolText,
          evolutionCurrentBest: payloadState.evolutionCurrentBest ?? state.evolutionCurrentBest,
          evolutionStatusMessage: payloadState.evolutionStatusMessage ?? state.evolutionStatusMessage,
          adversarialStatusMessage:
            payloadState.adversarialStatusMessage ?? state.adversarialStatusMessage,
          evolutionBestScore: payloadState.evolutionBestScore ?? state.evolutionBestScore,
        });
      }

      if (importOptions.applyHistory && importPayload.history) {
        const history = importPayload.history as Record<string, unknown>;
        updateState({
          evolutionHistory: (history.evolutionHistory as any[]) ?? state.evolutionHistory,
          adversarialResults: history.adversarialResults ?? state.adversarialResults,
        });
      }

      if (importOptions.applyTemplates && importPayload.templates) {
        const templates = importPayload.templates as Record<string, unknown>;
        if (templates.report_templates) {
          writeStorageJson("openevolve_report_templates", templates.report_templates);
        }
        if (templates.workflow_template) {
          writeStorageJson("openevolve_active_workflow_template", templates.workflow_template);
        }
      }

      if (importOptions.applySettings && importPayload.settings) {
        const settings = importPayload.settings as Record<string, unknown>;
        Object.entries(settings).forEach(([key, value]) => {
          if (typeof value === "string") {
            writeStorage(key, value);
          } else {
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
        const metadata = importPayload.metadata as Record<string, unknown>;
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
    } catch (error: any) {
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

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Export / Import Manager</CardTitle>
          <CardDescription>Backup, transfer, and restore OpenEvolve project data.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">Project Name</label>
              <Input
                value={projectName}
                onChange={(event) => {
                  const value = event.target.value;
                  setProjectName(value);
                  writeStorage("openevolve_project_name", value);
                }}
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Project Description</label>
              <Input
                value={projectDescription}
                onChange={(event) => {
                  const value = event.target.value;
                  setProjectDescription(value);
                  writeStorage("openevolve_project_description", value);
                }}
              />
            </div>
          </div>

          <Tabs defaultValue="export" className="w-full">
            <TabsList className="flex flex-wrap gap-2">
              <TabsTrigger value="export">Export</TabsTrigger>
              <TabsTrigger value="import">Import</TabsTrigger>
              <TabsTrigger value="backup">Backup</TabsTrigger>
            </TabsList>

            <TabsContent value="export" className="mt-4">
              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Export Options</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm">
                    {[
                      ["includeState", "Core State (protocol, status)", exportOptions.includeState],
                      ["includeHistory", "History (evolution + adversarial)", exportOptions.includeHistory],
                      ["includeTemplates", "Templates (reports + workflow)", exportOptions.includeTemplates],
                      ["includeSettings", "Settings (overrides + base URLs)", exportOptions.includeSettings],
                      ["includeTasks", "Tasks", exportOptions.includeTasks],
                      ["includeNotifications", "Notifications", exportOptions.includeNotifications],
                      ["includeSensitive", "Include sensitive keys (API tokens)", exportOptions.includeSensitive],
                    ].map(([key, label, checked]) => (
                      <label key={String(key)} className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={Boolean(checked)}
                          onChange={(event) =>
                            setExportOptions((prev) => ({
                              ...prev,
                              [key as string]: event.target.checked,
                            }))
                          }
                        />
                        <span>{label}</span>
                      </label>
                    ))}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Export Actions</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm">
                    <Button onClick={exportProject}>Export JSON</Button>
                    <div className="flex flex-wrap gap-2">
                      <Button variant="outline" onClick={exportMarkdown}>
                        Export Markdown
                      </Button>
                      <Button variant="outline" onClick={exportText}>
                        Export Text
                      </Button>
                      <Button variant="outline" onClick={exportPdfTemplate}>
                        Export PDF Template
                      </Button>
                    </div>
                    {shareId ? (
                      <div className="rounded border p-2 text-xs">
                        Share ID: <Badge variant="secondary">{shareId}</Badge>
                      </div>
                    ) : null}
                    {exportPayload ? (
                      <Textarea
                        value={JSON.stringify(exportPayload, null, 2)}
                        readOnly
                        className="min-h-[180px]"
                      />
                    ) : null}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="import" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Import Project</CardTitle>
                  <CardDescription>Upload an export JSON file to restore data.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <Input
                    type="file"
                    accept="application/json"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) {
                        handleImportFile(file);
                      }
                    }}
                  />
                  {importMessage ? (
                    <div className="text-xs text-muted-foreground">{importMessage}</div>
                  ) : null}
                  {importPayload ? (
                    <div className="space-y-3">
                      <div className="grid gap-2 md:grid-cols-2">
                        {[
                          ["applyState", "Apply core state"],
                          ["applyHistory", "Apply history"],
                          ["applyTemplates", "Apply templates"],
                          ["applySettings", "Apply settings"],
                          ["applyTasks", "Apply tasks"],
                          ["applyNotifications", "Apply notifications"],
                        ].map(([key, label]) => (
                          <label key={String(key)} className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={Boolean(importOptions[key as keyof typeof importOptions])}
                              onChange={(event) =>
                                setImportOptions((prev) => ({
                                  ...prev,
                                  [key as string]: event.target.checked,
                                }))
                              }
                            />
                            <span>{label}</span>
                          </label>
                        ))}
                      </div>
                      <Button onClick={applyImport}>Apply Import</Button>
                      <Textarea
                        value={JSON.stringify(importPayload, null, 2)}
                        readOnly
                        className="min-h-[180px]"
                      />
                    </div>
                  ) : null}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="backup" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Project Backup</CardTitle>
                  <CardDescription>Create or restore a full backup snapshot.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <Button
                    variant="outline"
                    onClick={() => {
                      const payload = buildExportPayload();
                      jsonDownload(`openevolve_backup_${Date.now()}.json`, payload);
                    }}
                  >
                    Download Full Backup
                  </Button>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Restore Backup</label>
                    <Input
                      type="file"
                      accept="application/json"
                      onChange={(event) => {
                        const file = event.target.files?.[0];
                        if (file) {
                          handleImportFile(file);
                        }
                      }}
                    />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};
