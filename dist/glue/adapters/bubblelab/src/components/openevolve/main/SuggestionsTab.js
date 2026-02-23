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
exports.SuggestionsTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const tabs_1 = require("@/components/ui/tabs");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const SuggestionsTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_llm_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const [baseUrl, setBaseUrl] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_llm_base_url") ?? "https://api.openai.com/v1";
        }
        catch {
            return "https://api.openai.com/v1";
        }
    });
    const [model, setModel] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_llm_model") ?? "gpt-4o-mini";
        }
        catch {
            return "gpt-4o-mini";
        }
    });
    const [content, setContent] = (0, react_1.useState)("");
    const [suggestions, setSuggestions] = (0, react_1.useState)([]);
    const [classification, setClassification] = (0, react_1.useState)(null);
    const [vulnerabilities, setVulnerabilities] = (0, react_1.useState)([]);
    const [improvementScore, setImprovementScore] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const requestPayload = (0, react_1.useMemo)(() => ({
        content,
        api_key: apiKey,
        base_url: baseUrl,
        model,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        max_tokens: 1024,
    }), [content, apiKey, baseUrl, model]);
    const runSuggestions = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.getContentSuggestions(requestPayload);
            setSuggestions(response.suggestions ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to generate suggestions.");
        }
        finally {
            setLoading(false);
        }
    };
    const runClassification = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.getContentClassification(requestPayload);
            setClassification(response);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to classify content.");
        }
        finally {
            setLoading(false);
        }
    };
    const runSecurity = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.getSecuritySuggestions(requestPayload);
            setVulnerabilities(response.vulnerabilities ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to scan for security issues.");
        }
        finally {
            setLoading(false);
        }
    };
    const runImprovement = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.getImprovementPotential(requestPayload);
            setImprovementScore(response.score ?? 0);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to calculate improvement potential.");
        }
        finally {
            setLoading(false);
        }
    };
    const persistSettings = (key, value) => {
        try {
            globalThis.localStorage?.setItem(key, value);
        }
        catch {
            // ignore storage errors
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>AI Suggestions</card_1.CardTitle>
          <card_1.CardDescription>Generate improvement guidance, tags, and security checks.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <label_1.Label>API Key</label_1.Label>
              <input_1.Input value={apiKey} type="password" onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            persistSettings("openevolve_llm_api_key", value);
        }}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Base URL</label_1.Label>
              <input_1.Input value={baseUrl} onChange={(event) => {
            const value = event.target.value;
            setBaseUrl(value);
            persistSettings("openevolve_llm_base_url", value);
        }}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Model</label_1.Label>
              <input_1.Input value={model} onChange={(event) => {
            const value = event.target.value;
            setModel(value);
            persistSettings("openevolve_llm_model", value);
        }}/>
            </div>
          </div>

          <div className="space-y-2">
            <label_1.Label>Content</label_1.Label>
            <textarea_1.Textarea value={content} onChange={(event) => setContent(event.target.value)} className="min-h-[160px]" placeholder="Paste content to analyze..."/>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <tabs_1.Tabs defaultValue="suggestions">
            <tabs_1.TabsList className="grid w-full grid-cols-4">
              <tabs_1.TabsTrigger value="suggestions">Suggestions</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="classification">Classification</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="security">Security Check</tabs_1.TabsTrigger>
              <tabs_1.TabsTrigger value="improvement">Improvement Potential</tabs_1.TabsTrigger>
            </tabs_1.TabsList>

            <tabs_1.TabsContent value="suggestions" className="mt-4 space-y-3">
              <button_1.Button onClick={runSuggestions} disabled={loading || !content}>
                Generate Suggestions
              </button_1.Button>
              <div className="space-y-2 text-sm">
                {suggestions.length === 0 && (<div className="text-muted-foreground">No suggestions yet.</div>)}
                {suggestions.map((suggestion, index) => (<div key={index} className="rounded border p-2">
                    {index + 1}. {suggestion}
                  </div>))}
              </div>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="classification" className="mt-4 space-y-3">
              <button_1.Button onClick={runClassification} disabled={loading || !content}>
                Classify Content
              </button_1.Button>
              <textarea_1.Textarea value={classification ? JSON.stringify(classification, null, 2) : ""} readOnly className="min-h-[160px]"/>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="security" className="mt-4 space-y-3">
              <button_1.Button onClick={runSecurity} disabled={loading || !content}>
                Scan for Issues
              </button_1.Button>
              <div className="space-y-2 text-sm">
                {vulnerabilities.length === 0 && (<div className="text-muted-foreground">No vulnerabilities listed.</div>)}
                {vulnerabilities.map((issue, index) => (<div key={index} className="rounded border p-2">
                    {issue}
                  </div>))}
              </div>
            </tabs_1.TabsContent>

            <tabs_1.TabsContent value="improvement" className="mt-4 space-y-3">
              <button_1.Button onClick={runImprovement} disabled={loading || !content}>
                Calculate Improvement Potential
              </button_1.Button>
              <div className="text-sm">
                Score: {improvementScore !== null ? improvementScore.toFixed(2) : "n/a"}
              </div>
            </tabs_1.TabsContent>
          </tabs_1.Tabs>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.SuggestionsTab = SuggestionsTab;
//# sourceMappingURL=SuggestionsTab.js.map