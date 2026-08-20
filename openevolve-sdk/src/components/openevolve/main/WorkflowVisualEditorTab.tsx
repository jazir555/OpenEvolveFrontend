import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  ArrowRight,
  ChevronDown,
  ChevronRight,
  Copy,
  Download,
  Link2,
  Plus,
  Trash2,
  Workflow,
  X,
} from "lucide-react";
import type { PluginInterface } from "@/lib/plugin-registry";
import { getPluginRegistry } from "@/lib/plugin-registry";
import { validateWorkflow, type WorkflowDefinition, type WorkflowStep } from "@/lib/workflow-orchestrator";
import { getAllWorkflowTemplates } from "@/lib/workflow-templates";

const LIFECYCLE_METHODS = new Set([
  "constructor",
  "initialize",
  "updateConfig",
  "resetConfig",
  "healthCheck",
  "getContext",
  "getStatus",
  "destroy",
  "metadata",
  "capabilities",
]);

const PLUGIN_FIELDS = new Set([
  "metadata",
  "capabilities",
  "context",
]);

/**
 * Derive the callable action names exposed by a plugin instance.
 * The WorkflowOrchestrator invokes actions dynamically via `plugin[action](input)`,
 * so any function property that is not part of the PluginInterface lifecycle is treated as an action.
 */
function getPluginActions(plugin: PluginInterface): string[] {
  const names = new Set<string>();
  let current: unknown = plugin;
  while (current && current !== Object.prototype) {
    Object.getOwnPropertyNames(current as object).forEach((name) => names.add(name));
    current = Object.getPrototypeOf(current);
  }
  return Array.from(names)
    .filter(
      (name) =>
        !LIFECYCLE_METHODS.has(name) &&
        !PLUGIN_FIELDS.has(name) &&
        typeof (plugin as unknown as Record<string, unknown>)[name] === "function",
    )
    .sort();
}

interface Point {
  x: number;
  y: number;
}

interface Arrow {
  from: Point;
  to: Point;
}

const emptyWorkflow = (): WorkflowDefinition => ({
  id: `workflow-${Date.now()}`,
  name: "Untitled Workflow",
  description: "",
  steps: [],
  onError: "continue",
  maxRetries: 1,
});

const cloneWorkflow = (source: WorkflowDefinition): WorkflowDefinition =>
  JSON.parse(JSON.stringify(source)) as WorkflowDefinition;

export const WorkflowVisualEditorTab: React.FC = () => {
  const [plugins] = useState<PluginInterface[]>(() => {
    try {
      return getPluginRegistry().getAllPlugins();
    } catch {
      return [];
    }
  });

  const [expandedPlugins, setExpandedPlugins] = useState<Set<string>>(new Set());
  const [workflow, setWorkflow] = useState<WorkflowDefinition>(emptyWorkflow);
  const [selectedStepId, setSelectedStepId] = useState<string | null>(null);
  const [validation, setValidation] = useState<{ valid: boolean; errors: string[] } | null>(null);
  const [exportedJson, setExportedJson] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const stepCounter = useRef(0);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const cardRefs = useRef<Map<string, HTMLDivElement | null>>(new Map());
  const [arrows, setArrows] = useState<Arrow[]>([]);
  const prevArrows = useRef<string>("");

  const pluginActions = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const plugin of plugins) {
      map.set(plugin.metadata.name, getPluginActions(plugin));
    }
    return map;
  }, [plugins]);

  const selectedStep = useMemo(
    () => workflow.steps.find((step) => step.id === selectedStepId) ?? null,
    [workflow.steps, selectedStepId],
  );

  const [inputEditor, setInputEditor] = useState<Array<{ key: string; value: string }>>([]);

  useEffect(() => {
    if (!selectedStep) {
      setInputEditor([]);
      return;
    }
    setInputEditor(
      Object.entries(selectedStep.input ?? {}).map(([key, value]) => ({
        key,
        value: typeof value === "string" ? value : JSON.stringify(value),
      })),
    );
  }, [selectedStepId]); // eslint-disable-line react-hooks/exhaustive-deps

  const showStatus = (message: string) => {
    setStatusMessage(message);
    setErrorMessage(null);
  };

  const showError = (message: string) => {
    setErrorMessage(message);
    setStatusMessage(null);
  };

  const togglePlugin = (pluginName: string) => {
    setExpandedPlugins((prev) => {
      const next = new Set(prev);
      if (next.has(pluginName)) {
        next.delete(pluginName);
      } else {
        next.add(pluginName);
      }
      return next;
    });
  };

  const addStep = (plugin: string, action: string) => {
    const id = `step-${stepCounter.current++}`;
    const newStep: WorkflowStep = {
      id,
      name: `${action} (${plugin})`,
      plugin,
      action,
      input: {},
      dependsOn: [],
    };
    setWorkflow((prev) => ({ ...prev, steps: [...prev.steps, newStep] }));
    setSelectedStepId(id);
    setValidation(null);
    setExportedJson(null);
  };

  const removeStep = (stepId: string) => {
    setWorkflow((prev) => ({
      ...prev,
      steps: prev.steps
        .filter((step) => step.id !== stepId)
        .map((step) => ({
          ...step,
          dependsOn: (step.dependsOn ?? []).filter((dep) => dep !== stepId),
        })),
    }));
    if (selectedStepId === stepId) {
      setSelectedStepId(null);
    }
    setValidation(null);
    setExportedJson(null);
  };

  const updateStep = (stepId: string, updater: (step: WorkflowStep) => WorkflowStep) => {
    setWorkflow((prev) => ({
      ...prev,
      steps: prev.steps.map((step) => (step.id === stepId ? updater(step) : step)),
    }));
    setValidation(null);
    setExportedJson(null);
  };

  const addDependency = (stepId: string, dependencyId: string) => {
    if (stepId === dependencyId) return;
    updateStep(stepId, (step) => {
      const dependsOn = new Set(step.dependsOn ?? []);
      dependsOn.add(dependencyId);
      return { ...step, dependsOn: Array.from(dependsOn) };
    });
  };

  const removeDependency = (stepId: string, dependencyId: string) => {
    updateStep(stepId, (step) => ({
      ...step,
      dependsOn: (step.dependsOn ?? []).filter((dep) => dep !== dependencyId),
    }));
  };

  const writeInputEditor = (entries: Array<{ key: string; value: string }>) => {
    setInputEditor(entries);
    if (!selectedStepId) return;
    const input: Record<string, unknown> = {};
    for (const entry of entries) {
      const key = entry.key.trim();
      if (key) {
        input[key] = entry.value;
      }
    }
    updateStep(selectedStepId, (step) => ({ ...step, input }));
  };

  const loadTemplate = (templateId: string) => {
    const template = getAllWorkflowTemplates().find((t) => t.id === templateId);
    if (!template) return;
    const cloned = cloneWorkflow(template);
    stepCounter.current = cloned.steps.length;
    setWorkflow(cloned);
    setSelectedStepId(null);
    setValidation(null);
    setExportedJson(null);
    showStatus(`Loaded template "${template.name}".`);
  };

  const clearCanvas = () => {
    setWorkflow(emptyWorkflow());
    stepCounter.current = 0;
    setSelectedStepId(null);
    setValidation(null);
    setExportedJson(null);
    showStatus("Canvas cleared.");
  };

  const runValidation = () => {
    const result = validateWorkflow(workflow);
    setValidation(result);
    if (result.valid) {
      showStatus("Workflow is valid.");
    } else {
      setErrorMessage(`${result.errors.length} validation error(s) found.`);
    }
  };

  const buildExport = (): WorkflowDefinition => {
    const steps = workflow.steps.map((step) => ({
      ...step,
      input: step.input ?? {},
      dependsOn: step.dependsOn && step.dependsOn.length > 0 ? step.dependsOn : undefined,
    }));
    return { ...workflow, steps };
  };

  const exportJson = () => {
    const definition = buildExport();
    const json = JSON.stringify(definition, null, 2);
    setExportedJson(json);
    showStatus("Workflow exported as JSON.");
  };

  const copyJson = async () => {
    if (!exportedJson) return;
    try {
      await navigator.clipboard?.writeText(exportedJson);
      showStatus("JSON copied to clipboard.");
    } catch {
      showError("Unable to copy to clipboard.");
    }
  };

  const downloadJson = () => {
    if (!exportedJson) return;
    const blob = new Blob([exportedJson], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${workflow.id || "workflow"}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  useLayoutEffect(() => {
    const container = canvasRef.current;
    if (!container) return;
    const base = container.getBoundingClientRect();
    const next: Arrow[] = [];
    for (const step of workflow.steps) {
      const el = cardRefs.current.get(step.id);
      if (!el) continue;
      const rect = el.getBoundingClientRect();
      const to: Point = {
        x: rect.left - base.left + rect.width / 2,
        y: rect.top - base.top,
      };
      for (const dependencyId of step.dependsOn ?? []) {
        const depEl = cardRefs.current.get(dependencyId);
        if (!depEl) continue;
        const depRect = depEl.getBoundingClientRect();
        const from: Point = {
          x: depRect.left - base.left + depRect.width / 2,
          y: depRect.bottom - base.top,
        };
        next.push({ from, to });
      }
    }
    const serialized = JSON.stringify(next);
    if (serialized !== prevArrows.current) {
      prevArrows.current = serialized;
      setArrows(next);
    }
  }, [workflow.steps]);

  useEffect(() => {
    const handleResize = () => {
      prevArrows.current = "";
      // Force recompute on next layout pass
      setArrows((prev) => [...prev]);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const stepNameById = useMemo(() => {
    const map = new Map<string, string>();
    for (const step of workflow.steps) {
      map.set(step.id, step.name);
    }
    return map;
  }, [workflow.steps]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Workflow className="h-5 w-5" />
            Workflow Visual Editor
          </CardTitle>
          <CardDescription>
            Drag-and-drop builder for multi-step workflows executed by the orchestrator.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Workflow ID</Label>
              <Input
                value={workflow.id}
                onChange={(event) => setWorkflow((prev) => ({ ...prev, id: event.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label>Name</Label>
              <Input
                value={workflow.name}
                onChange={(event) => setWorkflow((prev) => ({ ...prev, name: event.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label>On Error</Label>
              <Select
                value={workflow.onError ?? "continue"}
                onValueChange={(value) =>
                  setWorkflow((prev) => ({
                    ...prev,
                    onError: value as WorkflowDefinition["onError"],
                  }))
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="continue">Continue</SelectItem>
                  <SelectItem value="stop">Stop</SelectItem>
                  <SelectItem value="retry">Retry</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="space-y-2">
            <Label>Description</Label>
            <Input
              value={workflow.description ?? ""}
              onChange={(event) =>
                setWorkflow((prev) => ({ ...prev, description: event.target.value }))
              }
            />
          </div>

          <div className="flex flex-wrap gap-2">
            <Select onValueChange={(value) => loadTemplate(value)}>
              <SelectTrigger className="w-[220px]">
                <SelectValue placeholder="Load template…" />
              </SelectTrigger>
              <SelectContent>
                {getAllWorkflowTemplates().map((template) => (
                  <SelectItem key={template.id} value={template.id}>
                    {template.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="outline" onClick={clearCanvas}>
              Clear
            </Button>
            <Button variant="outline" onClick={runValidation}>
              Validate
            </Button>
            <Button variant="outline" onClick={exportJson}>
              Export JSON
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          {validation && !validation.valid && (
            <div className="space-y-1 rounded border border-red-500/50 bg-red-500/10 p-3 text-sm text-red-300">
              <div className="font-semibold">Validation errors:</div>
              <ul className="list-disc pl-5">
                {validation.errors.map((err, index) => (
                  <li key={index}>{err}</li>
                ))}
              </ul>
            </div>
          )}
          {validation && validation.valid && (
            <div className="rounded border border-green-500/50 bg-green-500/10 p-3 text-sm text-green-300">
              Workflow is valid.
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-[260px_1fr_340px]">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Plugin Palette</CardTitle>
            <CardDescription>Click an action to add a step.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {plugins.length === 0 && (
              <div className="text-sm text-muted-foreground">
                No plugins registered. You can still add steps and edit them manually.
              </div>
            )}
            {plugins.map((plugin) => {
              const name = plugin.metadata.name;
              const expanded = expandedPlugins.has(name);
              const actions = pluginActions.get(name) ?? [];
              return (
                <div key={name} className="rounded border border-[#30363d]">
                  <button
                    type="button"
                    className="flex w-full items-center justify-between px-3 py-2 text-left text-sm font-medium text-gray-200 hover:bg-[#161b22]"
                    onClick={() => togglePlugin(name)}
                  >
                    <span>{name}</span>
                    {expanded ? (
                      <ChevronDown className="h-4 w-4" />
                    ) : (
                      <ChevronRight className="h-4 w-4" />
                    )}
                  </button>
                  {expanded && (
                    <div className="space-y-1 border-t border-[#30363d] p-2">
                      {actions.length === 0 && (
                        <div className="px-1 text-xs text-muted-foreground">No actions available</div>
                      )}
                      {actions.map((action) => (
                        <button
                          key={action}
                          type="button"
                          className="flex w-full items-center gap-2 rounded px-2 py-1 text-left text-xs text-gray-300 hover:bg-[#21262d]"
                          onClick={() => addStep(name, action)}
                        >
                          <Plus className="h-3 w-3" />
                          {action}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Canvas</CardTitle>
            <CardDescription>
              {workflow.steps.length} step(s). Select a card to edit it.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div ref={canvasRef} className="relative min-h-[300px] space-y-4">
              <svg className="pointer-events-none absolute inset-0 h-full w-full">
                <defs>
                  <marker
                    id="wve-arrow"
                    markerWidth="10"
                    markerHeight="10"
                    refX="8"
                    refY="3"
                    orient="auto"
                    markerUnits="strokeWidth"
                  >
                    <path d="M0,0 L8,3 L0,6 Z" fill="#3b82f6" />
                  </marker>
                </defs>
                {arrows.map((arrow, index) => (
                  <line
                    key={index}
                    x1={arrow.from.x}
                    y1={arrow.from.y}
                    x2={arrow.to.x}
                    y2={arrow.to.y}
                    stroke="#3b82f6"
                    strokeWidth={2}
                    markerEnd="url(#wve-arrow)"
                  />
                ))}
              </svg>

              {workflow.steps.map((step) => (
                <div
                  key={step.id}
                  ref={(el) => {
                    cardRefs.current.set(step.id, el);
                  }}
                  onClick={() => setSelectedStepId(step.id)}
                  className={`relative cursor-pointer rounded border bg-[#0d1117] p-3 ${
                    selectedStepId === step.id
                      ? "border-blue-500 ring-1 ring-blue-500"
                      : "border-[#30363d] hover:border-[#484f58]"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold text-gray-100">
                        {step.name || "(unnamed)"}
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-gray-400">
                        <Badge variant="outline" className="border-[#30363d]">
                          {step.plugin || "no-plugin"}
                        </Badge>
                        <ArrowRight className="h-3 w-3" />
                        <span>{step.action || "no-action"}</span>
                      </div>
                      {step.dependsOn && step.dependsOn.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {step.dependsOn.map((dep) => (
                            <span
                              key={dep}
                              className="inline-flex items-center gap-1 rounded bg-[#161b22] px-2 py-0.5 text-xs text-gray-300"
                            >
                              <Link2 className="h-3 w-3 text-blue-400" />
                              {stepNameById.get(dep) ?? dep}
                              <button
                                type="button"
                                className="text-gray-500 hover:text-red-400"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  removeDependency(step.id, dep);
                                }}
                              >
                                <X className="h-3 w-3" />
                              </button>
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    <button
                      type="button"
                      className="shrink-0 text-gray-500 hover:text-red-400"
                      onClick={(event) => {
                        event.stopPropagation();
                        removeStep(step.id);
                      }}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}

              {workflow.steps.length === 0 && (
                <div className="flex min-h-[200px] items-center justify-center rounded border border-dashed border-[#30363d] text-sm text-muted-foreground">
                  Add steps from the plugin palette or load a template.
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Step Editor</CardTitle>
            <CardDescription>
              {selectedStep ? `Editing "${selectedStep.name}"` : "Select a step to edit."}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {!selectedStep && (
              <div className="text-sm text-muted-foreground">No step selected.</div>
            )}

            {selectedStep && (
              <>
                <div className="space-y-2">
                  <Label>Name</Label>
                  <Input
                    value={selectedStep.name}
                    onChange={(event) =>
                      updateStep(selectedStep.id, (step) => ({ ...step, name: event.target.value }))
                    }
                  />
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label>Plugin</Label>
                    <Select
                      value={selectedStep.plugin}
                      onValueChange={(value) =>
                        updateStep(selectedStep.id, (step) => ({ ...step, plugin: value }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select plugin" />
                      </SelectTrigger>
                      <SelectContent>
                        {plugins.map((plugin) => (
                          <SelectItem key={plugin.metadata.name} value={plugin.metadata.name}>
                            {plugin.metadata.name}
                          </SelectItem>
                        ))}
                        {selectedStep.plugin && !plugins.some((p) => p.metadata.name === selectedStep.plugin) && (
                          <SelectItem value={selectedStep.plugin}>{selectedStep.plugin}</SelectItem>
                        )}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Action</Label>
                    {pluginActions.get(selectedStep.plugin) ? (
                      <Select
                        value={selectedStep.action}
                        onValueChange={(value) =>
                          updateStep(selectedStep.id, (step) => ({ ...step, action: value }))
                        }
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select action" />
                        </SelectTrigger>
                        <SelectContent>
                          {(pluginActions.get(selectedStep.plugin) ?? []).map((action) => (
                            <SelectItem key={action} value={action}>
                              {action}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    ) : (
                      <Input
                        value={selectedStep.action}
                        onChange={(event) =>
                          updateStep(selectedStep.id, (step) => ({ ...step, action: event.target.value }))
                        }
                      />
                    )}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Inputs</Label>
                  <div className="space-y-2">
                    {inputEditor.map((entry, index) => (
                      <div key={index} className="flex gap-2">
                        <Input
                          placeholder="key"
                          value={entry.key}
                          onChange={(event) => {
                            const nextEntries = inputEditor.map((e, i) =>
                              i === index ? { ...e, key: event.target.value } : e,
                            );
                            writeInputEditor(nextEntries);
                          }}
                        />
                        <Input
                          placeholder="value"
                          value={entry.value}
                          onChange={(event) => {
                            const nextEntries = inputEditor.map((e, i) =>
                              i === index ? { ...e, value: event.target.value } : e,
                            );
                            writeInputEditor(nextEntries);
                          }}
                        />
                        <Button
                          size="icon"
                          variant="outline"
                          onClick={() =>
                            writeInputEditor(inputEditor.filter((_, i) => i !== index))
                          }
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => writeInputEditor([...inputEditor, { key: "", value: "" }])}
                    >
                      <Plus className="mr-1 h-3 w-3" />
                      Add input
                    </Button>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Add dependency on…</Label>
                  <Select
                    onValueChange={(value) => {
                      addDependency(selectedStep.id, value);
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select a step" />
                    </SelectTrigger>
                    <SelectContent>
                      {workflow.steps
                        .filter((step) => step.id !== selectedStep.id)
                        .map((step) => (
                          <SelectItem key={step.id} value={step.id}>
                            {step.name || step.id}
                          </SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Selected step will run after the chosen step completes.
                  </p>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {exportedJson && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Exported Workflow Definition</CardTitle>
            <CardDescription>Copy or download the JSON below.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={copyJson}>
                <Copy className="mr-1 h-4 w-4" />
                Copy
              </Button>
              <Button variant="outline" size="sm" onClick={downloadJson}>
                <Download className="mr-1 h-4 w-4" />
                Download
              </Button>
            </div>
            <Textarea readOnly value={exportedJson} className="min-h-[260px] font-mono text-xs" />
          </CardContent>
        </Card>
      )}
    </div>
  );
};
