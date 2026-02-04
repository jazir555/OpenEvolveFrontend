import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { openevolveApi } from "@/lib/openevolveApi";
import type { ParameterDefinition, ProviderSummary } from "@/lib/types";

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

const downloadJson = (filename: string, payload: unknown) => {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

export const SettingsTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(() => readStorage("openevolve_api_key"));
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);
  const [baseUrl, setBaseUrl] = useState(() => readStorage("openevolve_api_base", ""));

  const [providers, setProviders] = useState<ProviderSummary[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [providerApiKey, setProviderApiKey] = useState("");
  const [providerModels, setProviderModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");

  const [parameters, setParameters] = useState<ParameterDefinition[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [parameterOverrides, setParameterOverrides] = useState(() =>
    readStorage("openevolve_parameter_overrides", "{}"),
  );
  const [validationResult, setValidationResult] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [analyticsSettings, setAnalyticsSettings] = useState(() => {
    try {
      return JSON.parse(
        readStorage(
          "openevolve_analytics_settings",
          "{\"collect_usage_data\":true,\"collect_performance_data\":true,\"collect_error_data\":true}",
        ),
      ) as Record<string, boolean>;
    } catch {
      return {
        collect_usage_data: true,
        collect_performance_data: true,
        collect_error_data: true,
      };
    }
  });
  const [analyticsSettingsRaw, setAnalyticsSettingsRaw] = useState(() =>
    JSON.stringify(analyticsSettings, null, 2),
  );
  const [reportFormat, setReportFormat] = useState(
    () => readStorage("openevolve_default_report_format", "Markdown"),
  );
  const [retentionDays, setRetentionDays] = useState(
    () => Number(readStorage("openevolve_data_retention_days", "90")),
  );
  const [includePersonalInfo, setIncludePersonalInfo] = useState(
    () => readStorage("openevolve_analytics_include_personal", "false") === "true",
  );

  const loadSettings = async () => {
    setErrorMessage(null);
    try {
      const [providerResponse, schemaResponse, categoryResponse, defaultsResponse] =
        await Promise.all([
          openevolveApi.listProviders(apiConfig),
          openevolveApi.getParameterSchema(apiConfig),
          openevolveApi.getParameterCategories(apiConfig),
          openevolveApi.getParameterDefaults(apiConfig),
        ]);
      setProviders(providerResponse.providers ?? []);
      setParameters(schemaResponse.parameters ?? []);
      setCategories(["All", ...(categoryResponse.categories ?? [])]);
      if (!parameterOverrides || parameterOverrides === "{}") {
        setParameterOverrides(JSON.stringify(defaultsResponse ?? {}, null, 2));
      }
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load settings metadata.");
    }
  };

  useEffect(() => {
    loadSettings();
  }, [apiConfig.apiKey]);

  const fetchProviderModels = async () => {
    if (!selectedProvider) return;
    setErrorMessage(null);
    try {
      const response = await openevolveApi.getProviderModels(
        selectedProvider,
        providerApiKey || undefined,
        apiConfig,
      );
      setProviderModels(response.models ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load provider models.");
    }
  };

  const validateParameters = async () => {
    setErrorMessage(null);
    try {
      const parsed = JSON.parse(parameterOverrides);
      const result = await openevolveApi.validateParameters({ parameters: parsed }, apiConfig);
      if (result.valid) {
        setValidationResult("All parameters are valid.");
      } else {
        setValidationResult(`Errors: ${result.errors.join("; ")}`);
      }
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to validate parameters.");
    }
  };

  const saveOverrides = () => {
    writeStorage("openevolve_parameter_overrides", parameterOverrides);
    setStatusMessage("Parameter overrides saved locally.");
  };

  const saveAnalyticsSettings = () => {
    writeStorage("openevolve_analytics_settings", JSON.stringify(analyticsSettings));
    writeStorage("openevolve_default_report_format", reportFormat);
    writeStorage("openevolve_data_retention_days", String(retentionDays));
    writeStorage("openevolve_analytics_include_personal", String(includePersonalInfo));
    setStatusMessage("Analytics settings saved locally.");
  };

  const filteredParameters = parameters.filter((param) =>
    selectedCategory === "All" ? true : param.category === selectedCategory,
  );

  useEffect(() => {
    setAnalyticsSettingsRaw(JSON.stringify(analyticsSettings, null, 2));
  }, [analyticsSettings]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Settings & Configuration</CardTitle>
          <CardDescription>Provider keys, parameter overrides, and validation.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>OpenEvolve API Key</Label>
              <Input
                value={apiKey}
                type="password"
                onChange={(event) => {
                  const value = event.target.value;
                  setApiKey(value);
                  writeStorage("openevolve_api_key", value);
                }}
              />
            </div>
            <div className="space-y-2">
              <Label>API Base URL</Label>
              <Input
                value={baseUrl}
                onChange={(event) => {
                  const value = event.target.value;
                  setBaseUrl(value);
                  writeStorage("openevolve_api_base", value);
                }}
                placeholder="https://your-api-server"
              />
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Provider Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Provider</Label>
                  <Select
                    value={selectedProvider}
                    onValueChange={(value) => {
                      setSelectedProvider(value);
                      setSelectedModel("");
                      setProviderModels([]);
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select provider" />
                    </SelectTrigger>
                    <SelectContent>
                      {providers.map((provider) => (
                        <SelectItem key={provider.id} value={provider.id}>
                          {provider.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Provider API Key</Label>
                  <Input
                    value={providerApiKey}
                    type="password"
                    onChange={(event) => setProviderApiKey(event.target.value)}
                  />
                </div>
              </div>
              <Button variant="outline" onClick={fetchProviderModels} disabled={!selectedProvider}>
                Fetch Models
              </Button>
              <div className="space-y-2">
                <Label>Models</Label>
                <Select
                  value={selectedModel}
                  onValueChange={(value) => {
                    setSelectedModel(value);
                    writeStorage("openevolve_selected_model", value);
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    {providerModels.map((model) => (
                      <SelectItem key={model} value={model}>
                        {model}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Parameter Overrides</CardTitle>
              <CardDescription>Override defaults with custom values.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                value={parameterOverrides}
                onChange={(event) => setParameterOverrides(event.target.value)}
                className="min-h-[200px]"
              />
              <div className="flex gap-2">
                <Button onClick={saveOverrides}>Save Overrides</Button>
                <Button variant="outline" onClick={validateParameters}>
                  Validate
                </Button>
                <Button
                  variant="outline"
                  onClick={() => downloadJson("parameter_overrides.json", JSON.parse(parameterOverrides))}
                >
                  Export
                </Button>
              </div>
              {validationResult ? (
                <div className="text-sm text-muted-foreground">{validationResult}</div>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Parameter Browser</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                <Label>Category</Label>
                <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {categories.map((category) => (
                      <SelectItem key={category} value={category}>
                        {category}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2 md:grid-cols-2">
                {filteredParameters.slice(0, 20).map((param) => (
                  <div key={param.name} className="rounded border p-2 text-xs">
                    <div className="font-semibold">{param.name}</div>
                    <div className="text-muted-foreground">{param.description}</div>
                    <div>
                      <Badge variant="secondary">{param.type}</Badge>
                    </div>
                    <div>Default: {JSON.stringify(param.default)}</div>
                  </div>
                ))}
                {filteredParameters.length === 0 && (
                  <div className="text-sm text-muted-foreground">No parameters.</div>
                )}
              </div>
              {filteredParameters.length > 20 && (
                <div className="text-xs text-muted-foreground">
                  Showing 20 of {filteredParameters.length} parameters.
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Analytics Settings</CardTitle>
              <CardDescription>Data collection, reporting, and retention preferences.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Data Collection</Label>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={Boolean(analyticsSettings.collect_usage_data)}
                    onChange={(event) => {
                      const value = event.target.checked;
                      setAnalyticsSettings((prev) => ({
                        ...prev,
                        collect_usage_data: value,
                      }));
                    }}
                  />
                  <span className="text-sm">Enable anonymous usage data</span>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={Boolean(analyticsSettings.collect_performance_data)}
                    onChange={(event) => {
                      const value = event.target.checked;
                      setAnalyticsSettings((prev) => ({
                        ...prev,
                        collect_performance_data: value,
                      }));
                    }}
                  />
                  <span className="text-sm">Enable performance metrics</span>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={Boolean(analyticsSettings.collect_error_data)}
                    onChange={(event) => {
                      const value = event.target.checked;
                      setAnalyticsSettings((prev) => ({
                        ...prev,
                        collect_error_data: value,
                      }));
                    }}
                  />
                  <span className="text-sm">Enable error reporting</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label>Default Report Format</Label>
                <Select value={reportFormat} onValueChange={setReportFormat}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Markdown">Markdown</SelectItem>
                    <SelectItem value="JSON">JSON</SelectItem>
                    <SelectItem value="PDF">PDF</SelectItem>
                    <SelectItem value="CSV">CSV</SelectItem>
                    <SelectItem value="Excel">Excel</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Data Retention (days)</Label>
                <Input
                  type="number"
                  value={retentionDays}
                  onChange={(event) => setRetentionDays(Number(event.target.value) || 30)}
                />
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={includePersonalInfo}
                  onChange={(event) => setIncludePersonalInfo(event.target.checked)}
                />
                <span className="text-sm">Include personal information in analytics</span>
              </div>

              <div className="space-y-2">
                <Label>Raw Settings (JSON)</Label>
                <Textarea
                  value={analyticsSettingsRaw}
                  onChange={(event) => {
                    const next = event.target.value;
                    setAnalyticsSettingsRaw(next);
                    try {
                      const parsed = JSON.parse(next);
                      setAnalyticsSettings(parsed);
                    } catch {
                      // keep raw text until valid JSON
                    }
                  }}
                  className="min-h-[120px]"
                />
              </div>

              <Button onClick={saveAnalyticsSettings}>Save Analytics Settings</Button>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  );
};
