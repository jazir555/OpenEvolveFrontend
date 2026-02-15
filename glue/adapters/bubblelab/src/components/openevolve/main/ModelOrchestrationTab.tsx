import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { openevolveApi } from "../../../lib/openevolveApi";
import type {
  ModelOrchestrationModel,
  ModelOrchestrationEnsembleRequest,
  ModelOrchestrationListResponse,
} from "../../../lib/types";

export const ModelOrchestrationTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [models, setModels] = useState<ModelOrchestrationModel[]>([]);
  const [metrics, setMetrics] = useState<Record<string, unknown>>({});
  const [strategies, setStrategies] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const [registerForm, setRegisterForm] = useState({
    model_name: "",
    role: "red_team",
    weight: "1.0",
    api_key: "",
    api_base: "https://api.openai.com/v1",
    temperature: "0.7",
    top_p: "1.0",
    max_tokens: "4096",
  });

  const [ensembleForm, setEnsembleForm] = useState({
    role: "red_team",
    selection_strategy: "performance_based",
    num_responses: "1",
    temperature: "0.7",
    max_tokens: "2048",
    input: "",
  });

  const [ensembleResponses, setEnsembleResponses] = useState<Array<Record<string, unknown>>>([]);

  const loadModels = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response: ModelOrchestrationListResponse = await openevolveApi.listOrchestrationModels(
        apiConfig,
      );
      setModels(response.models ?? []);
      setMetrics(response.metrics ?? {});
      setStrategies(response.selection_strategies ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load models.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, [apiConfig.apiKey]);

  const handleRegister = async () => {
    setErrorMessage(null);
    setStatusMessage(null);
    if (!registerForm.model_name.trim()) {
      setErrorMessage("Model name is required.");
      return;
    }
    try {
      await openevolveApi.registerOrchestrationModel(
        {
          model_name: registerForm.model_name.trim(),
          role: registerForm.role,
          weight: Number(registerForm.weight),
          api_key: registerForm.api_key || undefined,
          api_base: registerForm.api_base || undefined,
          temperature: Number(registerForm.temperature),
          top_p: Number(registerForm.top_p),
          max_tokens: Number(registerForm.max_tokens),
        },
        apiConfig,
      );
      setStatusMessage("Model registered.");
      setRegisterForm({
        model_name: "",
        role: registerForm.role,
        weight: "1.0",
        api_key: "",
        api_base: registerForm.api_base,
        temperature: "0.7",
        top_p: "1.0",
        max_tokens: "4096",
      });
      await loadModels();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to register model.");
    }
  };

  const handleEnsemble = async () => {
    setErrorMessage(null);
    setStatusMessage(null);
    if (!ensembleForm.input.trim()) {
      setErrorMessage("Input content is required.");
      return;
    }
    try {
      const payload: ModelOrchestrationEnsembleRequest = {
        role: ensembleForm.role,
        selection_strategy: ensembleForm.selection_strategy,
        num_responses: Number(ensembleForm.num_responses),
        temperature: Number(ensembleForm.temperature),
        max_tokens: Number(ensembleForm.max_tokens),
        messages: [{ role: "user", content: ensembleForm.input }],
      };
      const response = await openevolveApi.executeOrchestrationEnsemble(payload, apiConfig);
      setEnsembleResponses(response.responses ?? []);
      setStatusMessage(`Received ${response.responses?.length ?? 0} responses.`);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Ensemble execution failed.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Model Orchestration</CardTitle>
          <CardDescription>Register models and run ensemble executions.</CardDescription>
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
                    // ignore
                  }
                }}
              />
            </div>
            <Button variant="outline" onClick={loadModels} disabled={loading}>
              Refresh
            </Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Register Model</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <Label>Model Name</Label>
                  <Input
                    value={registerForm.model_name}
                    onChange={(event) =>
                      setRegisterForm({ ...registerForm, model_name: event.target.value })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label>Role</Label>
                  <Select
                    value={registerForm.role}
                    onValueChange={(value) => setRegisterForm({ ...registerForm, role: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select role" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="red_team">Red Team</SelectItem>
                      <SelectItem value="blue_team">Blue Team</SelectItem>
                      <SelectItem value="evaluator">Evaluator</SelectItem>
                      <SelectItem value="generator">Generator</SelectItem>
                      <SelectItem value="analyzer">Analyzer</SelectItem>
                      <SelectItem value="optimizer">Optimizer</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label>Weight</Label>
                    <Input
                      value={registerForm.weight}
                      onChange={(event) =>
                        setRegisterForm({ ...registerForm, weight: event.target.value })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>API Base</Label>
                    <Input
                      value={registerForm.api_base}
                      onChange={(event) =>
                        setRegisterForm({ ...registerForm, api_base: event.target.value })
                      }
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>API Key</Label>
                  <Input
                    type="password"
                    value={registerForm.api_key}
                    onChange={(event) =>
                      setRegisterForm({ ...registerForm, api_key: event.target.value })
                    }
                  />
                </div>
                <div className="grid gap-3 md:grid-cols-3">
                  <div className="space-y-2">
                    <Label>Temp</Label>
                    <Input
                      value={registerForm.temperature}
                      onChange={(event) =>
                        setRegisterForm({ ...registerForm, temperature: event.target.value })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Top P</Label>
                    <Input
                      value={registerForm.top_p}
                      onChange={(event) =>
                        setRegisterForm({ ...registerForm, top_p: event.target.value })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Max Tokens</Label>
                    <Input
                      value={registerForm.max_tokens}
                      onChange={(event) =>
                        setRegisterForm({ ...registerForm, max_tokens: event.target.value })
                      }
                    />
                  </div>
                </div>
                <Button onClick={handleRegister}>Register Model</Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Registered Models</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {models.length === 0 && (
                  <div className="text-sm text-muted-foreground">No models registered.</div>
                )}
                {models.map((model) => (
                  <div key={model.name} className="rounded border p-3 space-y-1">
                    <div className="flex items-center justify-between">
                      <div className="font-semibold">{model.name}</div>
                      <Badge variant="secondary">{model.role}</Badge>
                    </div>
                    <div className="text-xs text-muted-foreground">Weight: {model.weight}</div>
                    <div className="text-xs text-muted-foreground">Base: {model.api_base}</div>
                    {metrics && metrics[model.name] ? (
                      <pre className="text-xs whitespace-pre-wrap rounded border p-2">
                        {JSON.stringify(metrics[model.name], null, 2)}
                      </pre>
                    ) : null}
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          <Separator />

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Ensemble Execution</CardTitle>
              <CardDescription>Run prompts across multiple models.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 md:grid-cols-3">
                <div className="space-y-2">
                  <Label>Role</Label>
                  <Select
                    value={ensembleForm.role}
                    onValueChange={(value) => setEnsembleForm({ ...ensembleForm, role: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select role" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="red_team">Red Team</SelectItem>
                      <SelectItem value="blue_team">Blue Team</SelectItem>
                      <SelectItem value="evaluator">Evaluator</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Selection Strategy</Label>
                  <Select
                    value={ensembleForm.selection_strategy}
                    onValueChange={(value) =>
                      setEnsembleForm({ ...ensembleForm, selection_strategy: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select strategy" />
                    </SelectTrigger>
                    <SelectContent>
                      {strategies.map((strategy) => (
                        <SelectItem key={strategy} value={strategy}>
                          {strategy}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Responses</Label>
                  <Input
                    value={ensembleForm.num_responses}
                    onChange={(event) =>
                      setEnsembleForm({ ...ensembleForm, num_responses: event.target.value })
                    }
                  />
                </div>
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Temperature</Label>
                  <Input
                    value={ensembleForm.temperature}
                    onChange={(event) =>
                      setEnsembleForm({ ...ensembleForm, temperature: event.target.value })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label>Max Tokens</Label>
                  <Input
                    value={ensembleForm.max_tokens}
                    onChange={(event) =>
                      setEnsembleForm({ ...ensembleForm, max_tokens: event.target.value })
                    }
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label>Input Content</Label>
                <Textarea
                  value={ensembleForm.input}
                  onChange={(event) =>
                    setEnsembleForm({ ...ensembleForm, input: event.target.value })
                  }
                  rows={4}
                />
              </div>
              <Button onClick={handleEnsemble}>Execute Ensemble</Button>
              {ensembleResponses.length > 0 && (
                <div className="space-y-3">
                  {ensembleResponses.map((response, index) => (
                    <Card key={`${index}-${response.source_model ?? "model"}`}>
                      <CardHeader>
                        <CardTitle className="text-sm">
                          Response {index + 1} ({String(response.source_model ?? "model")})
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="text-sm whitespace-pre-wrap">
                        {String(response.response ?? "")}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
};
