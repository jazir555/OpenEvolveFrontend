import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { openevolveApi } from "@/lib/openevolveApi";

const EVALUATOR_PRESETS = {
  "Quality Assurance": {
    name: "Quality Assurance",
    description: "Focus on content quality, accuracy, and completeness.",
    models: ["openai/gpt-4o", "anthropic/claude-3-sonnet", "google/gemini-1.5-pro"],
    threshold: 90.0,
    consecutive_rounds: 1,
    sample_size: 3,
    system_prompt:
      "You are a quality assurance expert evaluating content for accuracy, completeness, and clarity.",
    weight_factors: {
      accuracy: 0.3,
      completeness: 0.25,
      clarity: 0.2,
      consistency: 0.15,
      professionalism: 0.1,
    },
  },
  "Security Review": {
    name: "Security Review",
    description: "Evaluate content for security vulnerabilities and best practices.",
    models: ["openai/gpt-4o", "anthropic/claude-3-opus", "meta-llama/llama-3-70b-instruct"],
    threshold: 95.0,
    consecutive_rounds: 2,
    sample_size: 3,
    system_prompt:
      "You are a security expert reviewing content for potential security vulnerabilities.",
    weight_factors: {
      vulnerabilities: 0.4,
      compliance: 0.3,
      misuse_potential: 0.15,
      privacy: 0.1,
      auth_issues: 0.05,
    },
  },
  "Legal Compliance": {
    name: "Legal Compliance",
    description: "Ensure content meets legal and regulatory requirements.",
    models: ["openai/gpt-4o", "anthropic/claude-3-sonnet", "google/gemini-1.5-pro"],
    threshold: 98.0,
    consecutive_rounds: 2,
    sample_size: 3,
    system_prompt:
      "You are a legal expert reviewing content for compliance with applicable laws and regulations.",
    weight_factors: {
      regulatory_compliance: 0.4,
      contractual_obligations: 0.25,
      ip_considerations: 0.15,
      liability_management: 0.1,
      industry_requirements: 0.1,
    },
  },
  "Technical Review": {
    name: "Technical Review",
    description: "Assess technical accuracy and implementation feasibility.",
    models: ["openai/gpt-4o", "anthropic/claude-3-sonnet", "codellama/codellama-70b-instruct"],
    threshold: 92.0,
    consecutive_rounds: 1,
    sample_size: 3,
    system_prompt:
      "You are a technical expert reviewing content for technical accuracy and feasibility.",
    weight_factors: {
      technical_accuracy: 0.35,
      implementation_feasibility: 0.25,
      performance: 0.2,
      scalability: 0.1,
      integration: 0.1,
    },
  },
  "User Experience": {
    name: "User Experience",
    description: "Evaluate content from a user experience perspective.",
    models: ["openai/gpt-4o", "anthropic/claude-3-sonnet", "google/gemini-1.5-pro"],
    threshold: 88.0,
    consecutive_rounds: 1,
    sample_size: 3,
    system_prompt:
      "You are a user experience expert evaluating content from the end-user perspective.",
    weight_factors: {
      usability: 0.3,
      clarity: 0.25,
      engagement: 0.2,
      design: 0.15,
      flow: 0.1,
    },
  },
} as const;

type EvaluatorPresetKey = keyof typeof EVALUATOR_PRESETS;

type EvaluatorConfig = {
  name: string;
  description: string;
  models: string[];
  threshold: number;
  consecutive_rounds: number;
  sample_size: number;
  system_prompt: string;
  weight_factors: Record<string, number>;
};

const loadJson = <T,>(key: string, fallback: T): T => {
  try {
    const raw = globalThis.localStorage?.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

const saveJson = (key: string, value: unknown) => {
  try {
    globalThis.localStorage?.setItem(key, JSON.stringify(value));
  } catch {
    // ignore storage errors
  }
};

export const EvaluatorHubTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [evaluatorCode, setEvaluatorCode] = useState("");
  const [evaluators, setEvaluators] = useState<Record<string, string>>({});
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [selectedPreset, setSelectedPreset] = useState<EvaluatorPresetKey>("Quality Assurance");
  const [customConfigs, setCustomConfigs] = useState<Record<string, EvaluatorConfig>>(() =>
    loadJson("openevolve_custom_evaluator_configs", {}),
  );
  const [customConfigName, setCustomConfigName] = useState("");

  const loadEvaluators = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.listEvaluators(apiConfig);
      setEvaluators(response.evaluators ?? {});
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load evaluators.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadEvaluators();
  }, [apiConfig.apiKey]);

  const handleUpload = async () => {
    setStatusMessage(null);
    setErrorMessage(null);
    if (!evaluatorCode.trim()) {
      setErrorMessage("Evaluator code is required.");
      return;
    }
    try {
      const response = await openevolveApi.uploadEvaluator({ code: evaluatorCode }, apiConfig);
      setStatusMessage(`Evaluator uploaded: ${response.evaluator_id}`);
      setEvaluatorCode("");
      await loadEvaluators();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to upload evaluator.");
    }
  };

  const handleDelete = async (evaluatorId: string) => {
    setStatusMessage(null);
    setErrorMessage(null);
    try {
      await openevolveApi.deleteEvaluator(evaluatorId, apiConfig);
      await loadEvaluators();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to delete evaluator.");
    }
  };

  const selectedPresetConfig: EvaluatorConfig = EVALUATOR_PRESETS[selectedPreset];

  const saveCustomConfig = () => {
    setErrorMessage(null);
    if (!customConfigName.trim()) {
      setErrorMessage("Custom config name is required.");
      return;
    }
    const next = {
      ...customConfigs,
      [customConfigName.trim()]: { ...selectedPresetConfig },
    };
    setCustomConfigs(next);
    saveJson("openevolve_custom_evaluator_configs", next);
    setCustomConfigName("");
  };

  const deleteCustomConfig = (name: string) => {
    const next = { ...customConfigs };
    delete next[name];
    setCustomConfigs(next);
    saveJson("openevolve_custom_evaluator_configs", next);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Evaluator Hub</CardTitle>
          <CardDescription>Upload custom evaluators and manage evaluation presets.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
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

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="space-y-3">
            <Label>Evaluator Code</Label>
            <Textarea
              value={evaluatorCode}
              onChange={(event) => setEvaluatorCode(event.target.value)}
              placeholder="Paste evaluator code with an evaluate(program_path) function"
              rows={8}
            />
            <Button onClick={handleUpload}>Upload Evaluator</Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Custom Evaluators</CardTitle>
          <CardDescription>Manage uploaded evaluator functions.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button variant="outline" onClick={loadEvaluators} disabled={loading}>
            Refresh Evaluators
          </Button>
          {Object.keys(evaluators).length === 0 && (
            <div className="text-sm text-muted-foreground">No custom evaluators found.</div>
          )}
          {Object.entries(evaluators).map(([evaluatorId, code]) => (
            <Card key={evaluatorId}>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-sm">{evaluatorId}</CardTitle>
                <Button variant="destructive" size="sm" onClick={() => handleDelete(evaluatorId)}>
                  Delete
                </Button>
              </CardHeader>
              <CardContent>
                <Textarea value={code} readOnly rows={6} className="font-mono text-xs" />
              </CardContent>
            </Card>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Evaluator Presets</CardTitle>
          <CardDescription>Review and save evaluator presets for reuse.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Preset</Label>
              <select
                className="w-full rounded border border-input bg-background px-3 py-2 text-sm"
                value={selectedPreset}
                onChange={(event) => setSelectedPreset(event.target.value as EvaluatorPresetKey)}
              >
                {Object.keys(EVALUATOR_PRESETS).map((preset) => (
                  <option key={preset} value={preset}>
                    {preset}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <Label>Save As Custom Config</Label>
              <div className="flex gap-2">
                <Input
                  value={customConfigName}
                  placeholder="custom-config-name"
                  onChange={(event) => setCustomConfigName(event.target.value)}
                />
                <Button variant="outline" onClick={saveCustomConfig}>
                  Save
                </Button>
              </div>
            </div>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">{selectedPresetConfig.name}</CardTitle>
              <CardDescription>{selectedPresetConfig.description}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex flex-wrap gap-2">
                {selectedPresetConfig.models.map((model) => (
                  <Badge key={model} variant="secondary">
                    {model}
                  </Badge>
                ))}
              </div>
              <div>
                Threshold: {selectedPresetConfig.threshold}% · Sample Size: {selectedPresetConfig.sample_size}
              </div>
              <div>Consecutive Rounds: {selectedPresetConfig.consecutive_rounds}</div>
              <div className="rounded border p-3 text-xs whitespace-pre-wrap">
                {selectedPresetConfig.system_prompt}
              </div>
              <Separator />
              <div className="grid gap-2 md:grid-cols-2">
                {Object.entries(selectedPresetConfig.weight_factors).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between rounded border p-2">
                    <span>{key}</span>
                    <Badge variant="outline">{value}</Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Separator />
          <div className="space-y-3">
            <div className="text-sm font-semibold">Custom Configurations</div>
            {Object.keys(customConfigs).length === 0 && (
              <div className="text-sm text-muted-foreground">No custom configs saved.</div>
            )}
            {Object.entries(customConfigs).map(([name, config]) => (
              <Card key={name}>
                <CardHeader className="flex flex-row items-center justify-between">
                  <CardTitle className="text-sm">{name}</CardTitle>
                  <Button size="sm" variant="destructive" onClick={() => deleteCustomConfig(name)}>
                    Delete
                  </Button>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex flex-wrap gap-2">
                    {config.models.map((model) => (
                      <Badge key={model} variant="secondary">
                        {model}
                      </Badge>
                    ))}
                  </div>
                  <div>Threshold: {config.threshold}%</div>
                  <div className="rounded border p-2 text-xs whitespace-pre-wrap">
                    {config.system_prompt}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
