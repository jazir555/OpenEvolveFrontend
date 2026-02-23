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
exports.EvaluatorHubTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const separator_1 = require("@/components/ui/separator");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const EVALUATOR_PRESETS = {
    "Quality Assurance": {
        name: "Quality Assurance",
        description: "Focus on content quality, accuracy, and completeness.",
        models: ["openai/gpt-4o", "anthropic/claude-3-sonnet", "google/gemini-1.5-pro"],
        threshold: 90.0,
        consecutive_rounds: 1,
        sample_size: 3,
        system_prompt: "You are a quality assurance expert evaluating content for accuracy, completeness, and clarity.",
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
        system_prompt: "You are a security expert reviewing content for potential security vulnerabilities.",
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
        system_prompt: "You are a legal expert reviewing content for compliance with applicable laws and regulations.",
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
        system_prompt: "You are a technical expert reviewing content for technical accuracy and feasibility.",
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
        system_prompt: "You are a user experience expert evaluating content from the end-user perspective.",
        weight_factors: {
            usability: 0.3,
            clarity: 0.25,
            engagement: 0.2,
            design: 0.15,
            flow: 0.1,
        },
    },
};
const loadJson = (key, fallback) => {
    try {
        const raw = globalThis.localStorage?.getItem(key);
        if (!raw)
            return fallback;
        return JSON.parse(raw);
    }
    catch {
        return fallback;
    }
};
const saveJson = (key, value) => {
    try {
        globalThis.localStorage?.setItem(key, JSON.stringify(value));
    }
    catch {
        // ignore storage errors
    }
};
const EvaluatorHubTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [evaluatorCode, setEvaluatorCode] = (0, react_1.useState)("");
    const [evaluators, setEvaluators] = (0, react_1.useState)({});
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [selectedPreset, setSelectedPreset] = (0, react_1.useState)("Quality Assurance");
    const [customConfigs, setCustomConfigs] = (0, react_1.useState)(() => loadJson("openevolve_custom_evaluator_configs", {}));
    const [customConfigName, setCustomConfigName] = (0, react_1.useState)("");
    const loadEvaluators = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.listEvaluators(apiConfig);
            setEvaluators(response.evaluators ?? {});
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load evaluators.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
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
            const response = await openevolveApi_1.openevolveApi.uploadEvaluator({ code: evaluatorCode }, apiConfig);
            setStatusMessage(`Evaluator uploaded: ${response.evaluator_id}`);
            setEvaluatorCode("");
            await loadEvaluators();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to upload evaluator.");
        }
    };
    const handleDelete = async (evaluatorId) => {
        setStatusMessage(null);
        setErrorMessage(null);
        try {
            await openevolveApi_1.openevolveApi.deleteEvaluator(evaluatorId, apiConfig);
            await loadEvaluators();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to delete evaluator.");
        }
    };
    const selectedPresetConfig = EVALUATOR_PRESETS[selectedPreset];
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
    const deleteCustomConfig = (name) => {
        const next = { ...customConfigs };
        delete next[name];
        setCustomConfigs(next);
        saveJson("openevolve_custom_evaluator_configs", next);
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Evaluator Hub</card_1.CardTitle>
          <card_1.CardDescription>Upload custom evaluators and manage evaluation presets.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="space-y-2">
            <label_1.Label>API Key</label_1.Label>
            <input_1.Input value={apiKey} type="password" onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            try {
                globalThis.localStorage?.setItem("openevolve_api_key", value);
            }
            catch {
                // ignore storage errors
            }
        }}/>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="space-y-3">
            <label_1.Label>Evaluator Code</label_1.Label>
            <textarea_1.Textarea value={evaluatorCode} onChange={(event) => setEvaluatorCode(event.target.value)} placeholder="Paste evaluator code with an evaluate(program_path) function" rows={8}/>
            <button_1.Button onClick={handleUpload}>Upload Evaluator</button_1.Button>
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle className="text-base">Custom Evaluators</card_1.CardTitle>
          <card_1.CardDescription>Manage uploaded evaluator functions.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <button_1.Button variant="outline" onClick={loadEvaluators} disabled={loading}>
            Refresh Evaluators
          </button_1.Button>
          {Object.keys(evaluators).length === 0 && (<div className="text-sm text-muted-foreground">No custom evaluators found.</div>)}
          {Object.entries(evaluators).map(([evaluatorId, code]) => (<card_1.Card key={evaluatorId}>
              <card_1.CardHeader className="flex flex-row items-center justify-between">
                <card_1.CardTitle className="text-sm">{evaluatorId}</card_1.CardTitle>
                <button_1.Button variant="destructive" size="sm" onClick={() => handleDelete(evaluatorId)}>
                  Delete
                </button_1.Button>
              </card_1.CardHeader>
              <card_1.CardContent>
                <textarea_1.Textarea value={code} readOnly rows={6} className="font-mono text-xs"/>
              </card_1.CardContent>
            </card_1.Card>))}
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle className="text-base">Evaluator Presets</card_1.CardTitle>
          <card_1.CardDescription>Review and save evaluator presets for reuse.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>Preset</label_1.Label>
              <select className="w-full rounded border border-input bg-background px-3 py-2 text-sm" value={selectedPreset} onChange={(event) => setSelectedPreset(event.target.value)}>
                {Object.keys(EVALUATOR_PRESETS).map((preset) => (<option key={preset} value={preset}>
                    {preset}
                  </option>))}
              </select>
            </div>
            <div className="space-y-2">
              <label_1.Label>Save As Custom Config</label_1.Label>
              <div className="flex gap-2">
                <input_1.Input value={customConfigName} placeholder="custom-config-name" onChange={(event) => setCustomConfigName(event.target.value)}/>
                <button_1.Button variant="outline" onClick={saveCustomConfig}>
                  Save
                </button_1.Button>
              </div>
            </div>
          </div>

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">{selectedPresetConfig.name}</card_1.CardTitle>
              <card_1.CardDescription>{selectedPresetConfig.description}</card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3 text-sm">
              <div className="flex flex-wrap gap-2">
                {selectedPresetConfig.models.map((model) => (<badge_1.Badge key={model} variant="secondary">
                    {model}
                  </badge_1.Badge>))}
              </div>
              <div>
                Threshold: {selectedPresetConfig.threshold}% · Sample Size: {selectedPresetConfig.sample_size}
              </div>
              <div>Consecutive Rounds: {selectedPresetConfig.consecutive_rounds}</div>
              <div className="rounded border p-3 text-xs whitespace-pre-wrap">
                {selectedPresetConfig.system_prompt}
              </div>
              <separator_1.Separator />
              <div className="grid gap-2 md:grid-cols-2">
                {Object.entries(selectedPresetConfig.weight_factors).map(([key, value]) => (<div key={key} className="flex items-center justify-between rounded border p-2">
                    <span>{key}</span>
                    <badge_1.Badge variant="outline">{value}</badge_1.Badge>
                  </div>))}
              </div>
            </card_1.CardContent>
          </card_1.Card>

          <separator_1.Separator />
          <div className="space-y-3">
            <div className="text-sm font-semibold">Custom Configurations</div>
            {Object.keys(customConfigs).length === 0 && (<div className="text-sm text-muted-foreground">No custom configs saved.</div>)}
            {Object.entries(customConfigs).map(([name, config]) => (<card_1.Card key={name}>
                <card_1.CardHeader className="flex flex-row items-center justify-between">
                  <card_1.CardTitle className="text-sm">{name}</card_1.CardTitle>
                  <button_1.Button size="sm" variant="destructive" onClick={() => deleteCustomConfig(name)}>
                    Delete
                  </button_1.Button>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  <div className="flex flex-wrap gap-2">
                    {config.models.map((model) => (<badge_1.Badge key={model} variant="secondary">
                        {model}
                      </badge_1.Badge>))}
                  </div>
                  <div>Threshold: {config.threshold}%</div>
                  <div className="rounded border p-2 text-xs whitespace-pre-wrap">
                    {config.system_prompt}
                  </div>
                </card_1.CardContent>
              </card_1.Card>))}
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.EvaluatorHubTab = EvaluatorHubTab;
//# sourceMappingURL=EvaluatorHubTab.js.map