import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { openevolveApi } from "@/lib/openevolveApi";
import type { Team, TeamSummary } from "@/lib/types";

type OpenRouterModel = {
  id: string;
  name?: string;
  context_length?: number;
  pricing?: Record<string, number>;
  description?: string;
};

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

const MODEL_FALLBACK = [
  "gpt-4o",
  "gpt-4o-mini",
  "gpt-4-turbo",
  "gpt-4",
  "gpt-3.5-turbo",
  "claude-3-opus",
  "claude-3-sonnet",
  "claude-3-haiku",
  "gemini-1.5-pro",
  "gemini-1.5-flash",
  "llama-3-70b",
  "llama-3-8b",
  "mistral-large",
  "mistral-medium",
  "mixtral-8x22b",
  "command-r-plus",
  "command-r",
];

export const ModelDashboardTab: React.FC = () => {
  const [openrouterKey, setOpenrouterKey] = useState(() =>
    readStorage<string>("openevolve_openrouter_key", ""),
  );
  const [models, setModels] = useState<OpenRouterModel[]>([]);
  const [filter, setFilter] = useState("");
  const [modelError, setModelError] = useState<string | null>(null);
  const [loadingModels, setLoadingModels] = useState(false);

  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);
  const [teams, setTeams] = useState<Team[]>([]);
  const [teamError, setTeamError] = useState<string | null>(null);

  const [performanceJson, setPerformanceJson] = useState<string>(() =>
    JSON.stringify(readStorage<Record<string, unknown>>("openevolve_model_performance", {}), null, 2),
  );
  const [performanceError, setPerformanceError] = useState<string | null>(null);

  const loadModels = async () => {
    setLoadingModels(true);
    setModelError(null);
    try {
      const response = await fetch("https://openrouter.ai/api/v1/models", {
        headers: openrouterKey
          ? {
              Authorization: `Bearer ${openrouterKey}`,
            }
          : undefined,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.error?.message || "Failed to load models.");
      }
      const modelList: OpenRouterModel[] = (data?.data ?? data?.models ?? []).map(
        (item: any) => ({
          id: item.id,
          name: item.name,
          context_length: item.context_length,
          pricing: item.pricing,
          description: item.description,
        }),
      );
      setModels(modelList.length ? modelList : MODEL_FALLBACK.map((id) => ({ id })));
    } catch (error: any) {
      setModelError(error?.message ?? "Failed to load models.");
      setModels(MODEL_FALLBACK.map((id) => ({ id })));
    } finally {
      setLoadingModels(false);
    }
  };

  const loadTeams = async () => {
    setTeamError(null);
    try {
      const response = await openevolveApi.listTeams(apiConfig);
      const detailedTeams = await Promise.all(
        response.teams.map((team: TeamSummary) => openevolveApi.getTeam(team.name, apiConfig)),
      );
      setTeams(detailedTeams);
    } catch (error: any) {
      setTeamError(error?.message ?? "Failed to load teams.");
    }
  };

  const applyPerformanceJson = () => {
    setPerformanceError(null);
    try {
      const parsed = JSON.parse(performanceJson || "{}");
      writeStorage("openevolve_model_performance", parsed);
    } catch {
      setPerformanceError("Invalid JSON.");
    }
  };

  useEffect(() => {
    loadTeams();
  }, [apiConfig.apiKey]);

  const filteredModels = models.filter((model) =>
    model.id.toLowerCase().includes(filter.toLowerCase()),
  );

  const modelUsage = useMemo(() => {
    const usage: Record<string, number> = {};
    teams.forEach((team) => {
      team.members.forEach((member) => {
        usage[member.model_id] = (usage[member.model_id] || 0) + 1;
      });
    });
    return usage;
  }, [teams]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Model Catalogue</CardTitle>
          <CardDescription>Browse OpenRouter models and local model usage.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-[2fr_1fr]">
            <div className="space-y-2">
              <Label>OpenRouter API Key (optional)</Label>
              <Input
                value={openrouterKey}
                type="password"
                onChange={(event) => {
                  const value = event.target.value;
                  setOpenrouterKey(value);
                  writeStorage("openevolve_openrouter_key", value);
                }}
              />
            </div>
            <div className="flex items-end">
              <Button onClick={loadModels} disabled={loadingModels}>
                Load Models
              </Button>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Filter Models</Label>
            <Input value={filter} onChange={(event) => setFilter(event.target.value)} />
          </div>

          {modelError ? <div className="text-sm text-red-500">{modelError}</div> : null}

          <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
            {filteredModels.map((model) => (
              <div key={model.id} className="rounded border p-3 text-sm space-y-2">
                <div className="flex items-center justify-between">
                  <div className="font-semibold">{model.id}</div>
                  <Badge variant="secondary">
                    {model.context_length ? `${model.context_length} ctx` : "unknown ctx"}
                  </Badge>
                </div>
                {model.name && <div className="text-xs text-muted-foreground">{model.name}</div>}
                {model.pricing && (
                  <div className="text-xs text-muted-foreground">
                    Pricing: {JSON.stringify(model.pricing)}
                  </div>
                )}
              </div>
            ))}
            {filteredModels.length === 0 && (
              <div className="text-sm text-muted-foreground">No models match the filter.</div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Team Model Usage</CardTitle>
          <CardDescription>Model distribution across OpenEvolve teams.</CardDescription>
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
              <Button variant="outline" onClick={loadTeams}>
                Refresh Teams
              </Button>
            </div>
          </div>

          {teamError ? <div className="text-sm text-red-500">{teamError}</div> : null}

          <div className="grid gap-3 md:grid-cols-2">
            {teams.map((team) => (
              <div key={team.name} className="rounded border p-3 text-sm space-y-2">
                <div className="font-semibold">
                  {team.name} <Badge variant="secondary">{team.role}</Badge>
                </div>
                <div className="text-xs text-muted-foreground">
                  Members: {team.members.length}
                </div>
                <div className="text-xs text-muted-foreground">
                  Models: {team.members.map((member) => member.model_id).join(", ") || "n/a"}
                </div>
              </div>
            ))}
          </div>

          <div className="space-y-2">
            <Label>Model Usage Summary</Label>
            <div className="grid gap-2 md:grid-cols-2">
              {Object.entries(modelUsage).map(([modelId, count]) => (
                <div key={modelId} className="flex items-center justify-between rounded border p-2 text-sm">
                  <span>{modelId}</span>
                  <Badge variant="secondary">{count}</Badge>
                </div>
              ))}
              {Object.keys(modelUsage).length === 0 && (
                <div className="text-sm text-muted-foreground">No model usage recorded.</div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Model Performance Metrics</CardTitle>
          <CardDescription>Store and visualize model evaluation metrics.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            value={performanceJson}
            onChange={(event) => setPerformanceJson(event.target.value)}
            className="min-h-[160px]"
          />
          {performanceError ? <div className="text-sm text-red-500">{performanceError}</div> : null}
          <div className="flex gap-2">
            <Button onClick={applyPerformanceJson}>Save Metrics</Button>
            <Button
              variant="outline"
              onClick={() =>
                setPerformanceJson(
                  JSON.stringify(readStorage("openevolve_model_performance", {}), null, 2),
                )
              }
            >
              Reload
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
