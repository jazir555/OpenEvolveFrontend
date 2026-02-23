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
exports.SettingsTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
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
const downloadJson = (filename, payload) => {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
};
const SettingsTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => readStorage("openevolve_api_key"));
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [baseUrl, setBaseUrl] = (0, react_1.useState)(() => readStorage("openevolve_api_base", ""));
    const [providers, setProviders] = (0, react_1.useState)([]);
    const [selectedProvider, setSelectedProvider] = (0, react_1.useState)(() => readStorage("openevolve_provider", ""));
    const [providerKeys, setProviderKeys] = (0, react_1.useState)(() => {
        try {
            return JSON.parse(readStorage("openevolve_provider_keys", "{}"));
        }
        catch {
            return {};
        }
    });
    const [providerApiKey, setProviderApiKey] = (0, react_1.useState)(() => readStorage("openevolve_provider_api_key", ""));
    const [providerBaseUrl, setProviderBaseUrl] = (0, react_1.useState)(() => readStorage("openevolve_provider_base_url", ""));
    const [providerModels, setProviderModels] = (0, react_1.useState)([]);
    const [selectedModel, setSelectedModel] = (0, react_1.useState)(() => readStorage("openevolve_selected_model", ""));
    const [providerStatus, setProviderStatus] = (0, react_1.useState)(null);
    const [parameters, setParameters] = (0, react_1.useState)([]);
    const [categories, setCategories] = (0, react_1.useState)([]);
    const [selectedCategory, setSelectedCategory] = (0, react_1.useState)("All");
    const [parameterOverrides, setParameterOverrides] = (0, react_1.useState)(() => readStorage("openevolve_parameter_overrides", "{}"));
    const [validationResult, setValidationResult] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [analyticsSettings, setAnalyticsSettings] = (0, react_1.useState)(() => {
        try {
            return JSON.parse(readStorage("openevolve_analytics_settings", "{\"collect_usage_data\":true,\"collect_performance_data\":true,\"collect_error_data\":true}"));
        }
        catch {
            return {
                collect_usage_data: true,
                collect_performance_data: true,
                collect_error_data: true,
            };
        }
    });
    const [analyticsSettingsRaw, setAnalyticsSettingsRaw] = (0, react_1.useState)(() => JSON.stringify(analyticsSettings, null, 2));
    const [reportFormat, setReportFormat] = (0, react_1.useState)(() => readStorage("openevolve_default_report_format", "Markdown"));
    const [retentionDays, setRetentionDays] = (0, react_1.useState)(() => Number(readStorage("openevolve_data_retention_days", "90")));
    const [includePersonalInfo, setIncludePersonalInfo] = (0, react_1.useState)(() => readStorage("openevolve_analytics_include_personal", "false") === "true");
    const loadSettings = async () => {
        setErrorMessage(null);
        try {
            const [providerResponse, schemaResponse, categoryResponse, defaultsResponse] = await Promise.all([
                openevolveApi_1.openevolveApi.listProviders(apiConfig),
                openevolveApi_1.openevolveApi.getParameterSchema(apiConfig),
                openevolveApi_1.openevolveApi.getParameterCategories(apiConfig),
                openevolveApi_1.openevolveApi.getParameterDefaults(apiConfig),
            ]);
            setProviders(providerResponse.providers ?? []);
            setParameters(schemaResponse.parameters ?? []);
            setCategories(["All", ...(categoryResponse.categories ?? [])]);
            if (!parameterOverrides || parameterOverrides === "{}") {
                setParameterOverrides(JSON.stringify(defaultsResponse ?? {}, null, 2));
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load settings metadata.");
        }
    };
    (0, react_1.useEffect)(() => {
        loadSettings();
    }, [apiConfig.apiKey]);
    (0, react_1.useEffect)(() => {
        if (!selectedProvider)
            return;
        const provider = providers.find((item) => item.id === selectedProvider);
        if (provider && !providerBaseUrl) {
            setProviderBaseUrl(provider.api_base ?? "");
        }
        const storedKey = providerKeys[selectedProvider];
        if (storedKey && storedKey !== providerApiKey) {
            setProviderApiKey(storedKey);
        }
    }, [providers, selectedProvider]);
    const fetchProviderModels = async () => {
        if (!selectedProvider)
            return;
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.getProviderModels(selectedProvider, providerApiKey || undefined, apiConfig);
            setProviderModels(response.models ?? []);
            setProviderStatus("Connection successful.");
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load provider models.");
            setProviderStatus("Connection failed.");
        }
    };
    const saveProviderSettings = () => {
        if (selectedProvider) {
            const nextKeys = { ...providerKeys, [selectedProvider]: providerApiKey };
            setProviderKeys(nextKeys);
            writeStorage("openevolve_provider_keys", JSON.stringify(nextKeys));
            writeStorage("openevolve_provider", selectedProvider);
        }
        writeStorage("openevolve_provider_api_key", providerApiKey);
        writeStorage("openevolve_provider_base_url", providerBaseUrl);
        writeStorage("openevolve_selected_model", selectedModel);
        setStatusMessage("Provider settings saved locally.");
    };
    const validateParameters = async () => {
        setErrorMessage(null);
        try {
            const parsed = JSON.parse(parameterOverrides);
            const result = await openevolveApi_1.openevolveApi.validateParameters({ parameters: parsed }, apiConfig);
            if (result.valid) {
                setValidationResult("All parameters are valid.");
            }
            else {
                setValidationResult(`Errors: ${result.errors.join("; ")}`);
            }
        }
        catch (error) {
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
    const filteredParameters = parameters.filter((param) => selectedCategory === "All" ? true : param.category === selectedCategory);
    (0, react_1.useEffect)(() => {
        setAnalyticsSettingsRaw(JSON.stringify(analyticsSettings, null, 2));
    }, [analyticsSettings]);
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Settings & Configuration</card_1.CardTitle>
          <card_1.CardDescription>Provider keys, parameter overrides, and validation.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>OpenEvolve API Key</label_1.Label>
              <input_1.Input value={apiKey} type="password" onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            writeStorage("openevolve_api_key", value);
        }}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>API Base URL</label_1.Label>
              <input_1.Input value={baseUrl} onChange={(event) => {
            const value = event.target.value;
            setBaseUrl(value);
            writeStorage("openevolve_api_base", value);
        }} placeholder="https://your-api-server"/>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Provider Configuration</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-3">
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <label_1.Label>Provider</label_1.Label>
                    <select_1.Select value={selectedProvider} onValueChange={(value) => {
            setSelectedProvider(value);
            setSelectedModel("");
            setProviderModels([]);
            setProviderApiKey(providerKeys[value] ?? "");
            const provider = providers.find((item) => item.id === value);
            if (provider?.api_base) {
                setProviderBaseUrl(provider.api_base);
            }
        }}>
                    <select_1.SelectTrigger>
                      <select_1.SelectValue placeholder="Select provider"/>
                    </select_1.SelectTrigger>
                    <select_1.SelectContent>
                      {providers.map((provider) => (<select_1.SelectItem key={provider.id} value={provider.id}>
                          {provider.name}
                        </select_1.SelectItem>))}
                    </select_1.SelectContent>
                  </select_1.Select>
                </div>
                  <div className="space-y-2">
                    <label_1.Label>Provider API Key</label_1.Label>
                    <input_1.Input value={providerApiKey} type="password" onChange={(event) => setProviderApiKey(event.target.value)}/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Provider Base URL</label_1.Label>
                    <input_1.Input value={providerBaseUrl} onChange={(event) => setProviderBaseUrl(event.target.value)} placeholder="https://api.openai.com/v1"/>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button_1.Button variant="outline" onClick={fetchProviderModels} disabled={!selectedProvider}>
                    Test Connection
                  </button_1.Button>
                  <button_1.Button variant="outline" onClick={fetchProviderModels} disabled={!selectedProvider}>
                    Fetch Models
                  </button_1.Button>
                  <button_1.Button onClick={saveProviderSettings} disabled={!selectedProvider}>
                    Save Provider Settings
                  </button_1.Button>
                </div>
                {providerStatus ? (<div className="text-xs text-muted-foreground">{providerStatus}</div>) : null}
                <div className="space-y-2">
                  <label_1.Label>Models</label_1.Label>
                  <select_1.Select value={selectedModel} onValueChange={(value) => {
            setSelectedModel(value);
            writeStorage("openevolve_selected_model", value);
        }}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue placeholder="Select model"/>
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    {providerModels.map((model) => (<select_1.SelectItem key={model} value={model}>
                        {model}
                      </select_1.SelectItem>))}
                  </select_1.SelectContent>
                </select_1.Select>
                </div>
              </card_1.CardContent>
            </card_1.Card>

            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Current Provider Status</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-2 text-sm">
                <div>
                  Provider:{" "}
                  <badge_1.Badge variant={selectedProvider ? "default" : "secondary"}>
                    {selectedProvider || "Not selected"}
                  </badge_1.Badge>
                </div>
                <div>Model: {selectedModel || "Not selected"}</div>
                <div>Base URL: {providerBaseUrl || "Not configured"}</div>
                <div>API Key Set: {providerApiKey ? "[OK] Yes" : "[FAIL] No"}</div>
              </card_1.CardContent>
            </card_1.Card>

            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Parameter Overrides</card_1.CardTitle>
              <card_1.CardDescription>Override defaults with custom values.</card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <textarea_1.Textarea value={parameterOverrides} onChange={(event) => setParameterOverrides(event.target.value)} className="min-h-[200px]"/>
              <div className="flex gap-2">
                <button_1.Button onClick={saveOverrides}>Save Overrides</button_1.Button>
                <button_1.Button variant="outline" onClick={validateParameters}>
                  Validate
                </button_1.Button>
                <button_1.Button variant="outline" onClick={() => downloadJson("parameter_overrides.json", JSON.parse(parameterOverrides))}>
                  Export
                </button_1.Button>
              </div>
              {validationResult ? (<div className="text-sm text-muted-foreground">{validationResult}</div>) : null}
            </card_1.CardContent>
          </card_1.Card>

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Parameter Browser</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <div className="space-y-2">
                <label_1.Label>Category</label_1.Label>
                <select_1.Select value={selectedCategory} onValueChange={setSelectedCategory}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue />
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    {categories.map((category) => (<select_1.SelectItem key={category} value={category}>
                        {category}
                      </select_1.SelectItem>))}
                  </select_1.SelectContent>
                </select_1.Select>
              </div>
              <div className="grid gap-2 md:grid-cols-2">
                {filteredParameters.slice(0, 20).map((param) => (<div key={param.name} className="rounded border p-2 text-xs">
                    <div className="font-semibold">{param.name}</div>
                    <div className="text-muted-foreground">{param.description}</div>
                    <div>
                      <badge_1.Badge variant="secondary">{param.type}</badge_1.Badge>
                    </div>
                    <div>Default: {JSON.stringify(param.default)}</div>
                  </div>))}
                {filteredParameters.length === 0 && (<div className="text-sm text-muted-foreground">No parameters.</div>)}
              </div>
              {filteredParameters.length > 20 && (<div className="text-xs text-muted-foreground">
                  Showing 20 of {filteredParameters.length} parameters.
                </div>)}
            </card_1.CardContent>
          </card_1.Card>

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Analytics Settings</card_1.CardTitle>
              <card_1.CardDescription>Data collection, reporting, and retention preferences.</card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-4">
              <div className="space-y-2">
                <label_1.Label>Data Collection</label_1.Label>
                <div className="flex items-center gap-2">
                  <input type="checkbox" checked={Boolean(analyticsSettings.collect_usage_data)} onChange={(event) => {
            const value = event.target.checked;
            setAnalyticsSettings((prev) => ({
                ...prev,
                collect_usage_data: value,
            }));
        }}/>
                  <span className="text-sm">Enable anonymous usage data</span>
                </div>
                <div className="flex items-center gap-2">
                  <input type="checkbox" checked={Boolean(analyticsSettings.collect_performance_data)} onChange={(event) => {
            const value = event.target.checked;
            setAnalyticsSettings((prev) => ({
                ...prev,
                collect_performance_data: value,
            }));
        }}/>
                  <span className="text-sm">Enable performance metrics</span>
                </div>
                <div className="flex items-center gap-2">
                  <input type="checkbox" checked={Boolean(analyticsSettings.collect_error_data)} onChange={(event) => {
            const value = event.target.checked;
            setAnalyticsSettings((prev) => ({
                ...prev,
                collect_error_data: value,
            }));
        }}/>
                  <span className="text-sm">Enable error reporting</span>
                </div>
              </div>

              <div className="space-y-2">
                <label_1.Label>Default Report Format</label_1.Label>
                <select_1.Select value={reportFormat} onValueChange={setReportFormat}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue />
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    <select_1.SelectItem value="Markdown">Markdown</select_1.SelectItem>
                    <select_1.SelectItem value="JSON">JSON</select_1.SelectItem>
                    <select_1.SelectItem value="PDF">PDF</select_1.SelectItem>
                    <select_1.SelectItem value="CSV">CSV</select_1.SelectItem>
                    <select_1.SelectItem value="Excel">Excel</select_1.SelectItem>
                  </select_1.SelectContent>
                </select_1.Select>
              </div>

              <div className="space-y-2">
                <label_1.Label>Data Retention (days)</label_1.Label>
                <input_1.Input type="number" value={retentionDays} onChange={(event) => setRetentionDays(Number(event.target.value) || 30)}/>
              </div>

              <div className="flex items-center gap-2">
                <input type="checkbox" checked={includePersonalInfo} onChange={(event) => setIncludePersonalInfo(event.target.checked)}/>
                <span className="text-sm">Include personal information in analytics</span>
              </div>

              <div className="space-y-2">
                <label_1.Label>Raw Settings (JSON)</label_1.Label>
                <textarea_1.Textarea value={analyticsSettingsRaw} onChange={(event) => {
            const next = event.target.value;
            setAnalyticsSettingsRaw(next);
            try {
                const parsed = JSON.parse(next);
                setAnalyticsSettings(parsed);
            }
            catch {
                // keep raw text until valid JSON
            }
        }} className="min-h-[120px]"/>
              </div>

              <button_1.Button onClick={saveAnalyticsSettings}>Save Analytics Settings</button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.SettingsTab = SettingsTab;
//# sourceMappingURL=SettingsTab.js.map