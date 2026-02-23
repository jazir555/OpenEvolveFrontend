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
exports.ModelOrchestrationTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const separator_1 = require("@/components/ui/separator");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const ModelOrchestrationTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [models, setModels] = (0, react_1.useState)([]);
    const [metrics, setMetrics] = (0, react_1.useState)({});
    const [strategies, setStrategies] = (0, react_1.useState)([]);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [registerForm, setRegisterForm] = (0, react_1.useState)({
        model_name: "",
        role: "red_team",
        weight: "1.0",
        api_key: "",
        api_base: "https://api.openai.com/v1",
        temperature: "0.7",
        top_p: "1.0",
        max_tokens: "4096",
    });
    const [ensembleForm, setEnsembleForm] = (0, react_1.useState)({
        role: "red_team",
        selection_strategy: "performance_based",
        num_responses: "1",
        temperature: "0.7",
        max_tokens: "2048",
        input: "",
    });
    const [ensembleResponses, setEnsembleResponses] = (0, react_1.useState)([]);
    const loadModels = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.listOrchestrationModels(apiConfig);
            setModels(response.models ?? []);
            setMetrics(response.metrics ?? {});
            setStrategies(response.selection_strategies ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load models.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
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
            await openevolveApi_1.openevolveApi.registerOrchestrationModel({
                model_name: registerForm.model_name.trim(),
                role: registerForm.role,
                weight: Number(registerForm.weight),
                api_key: registerForm.api_key || undefined,
                api_base: registerForm.api_base || undefined,
                temperature: Number(registerForm.temperature),
                top_p: Number(registerForm.top_p),
                max_tokens: Number(registerForm.max_tokens),
            }, apiConfig);
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
        }
        catch (error) {
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
            const payload = {
                role: ensembleForm.role,
                selection_strategy: ensembleForm.selection_strategy,
                num_responses: Number(ensembleForm.num_responses),
                temperature: Number(ensembleForm.temperature),
                max_tokens: Number(ensembleForm.max_tokens),
                messages: [{ role: "user", content: ensembleForm.input }],
            };
            const response = await openevolveApi_1.openevolveApi.executeOrchestrationEnsemble(payload, apiConfig);
            setEnsembleResponses(response.responses ?? []);
            setStatusMessage(`Received ${response.responses?.length ?? 0} responses.`);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Ensemble execution failed.");
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Model Orchestration</card_1.CardTitle>
          <card_1.CardDescription>Register models and run ensemble executions.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <label_1.Label>API Key</label_1.Label>
              <input_1.Input value={apiKey} type="password" onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            try {
                globalThis.localStorage?.setItem("openevolve_api_key", value);
            }
            catch {
                // ignore
            }
        }}/>
            </div>
            <button_1.Button variant="outline" onClick={loadModels} disabled={loading}>
              Refresh
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid gap-6 md:grid-cols-2">
            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Register Model</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-3">
                <div className="space-y-2">
                  <label_1.Label>Model Name</label_1.Label>
                  <input_1.Input value={registerForm.model_name} onChange={(event) => setRegisterForm({ ...registerForm, model_name: event.target.value })}/>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Role</label_1.Label>
                  <select_1.Select value={registerForm.role} onValueChange={(value) => setRegisterForm({ ...registerForm, role: value })}>
                    <select_1.SelectTrigger>
                      <select_1.SelectValue placeholder="Select role"/>
                    </select_1.SelectTrigger>
                    <select_1.SelectContent>
                      <select_1.SelectItem value="red_team">Red Team</select_1.SelectItem>
                      <select_1.SelectItem value="blue_team">Blue Team</select_1.SelectItem>
                      <select_1.SelectItem value="evaluator">Evaluator</select_1.SelectItem>
                      <select_1.SelectItem value="generator">Generator</select_1.SelectItem>
                      <select_1.SelectItem value="analyzer">Analyzer</select_1.SelectItem>
                      <select_1.SelectItem value="optimizer">Optimizer</select_1.SelectItem>
                    </select_1.SelectContent>
                  </select_1.Select>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <label_1.Label>Weight</label_1.Label>
                    <input_1.Input value={registerForm.weight} onChange={(event) => setRegisterForm({ ...registerForm, weight: event.target.value })}/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>API Base</label_1.Label>
                    <input_1.Input value={registerForm.api_base} onChange={(event) => setRegisterForm({ ...registerForm, api_base: event.target.value })}/>
                  </div>
                </div>
                <div className="space-y-2">
                  <label_1.Label>API Key</label_1.Label>
                  <input_1.Input type="password" value={registerForm.api_key} onChange={(event) => setRegisterForm({ ...registerForm, api_key: event.target.value })}/>
                </div>
                <div className="grid gap-3 md:grid-cols-3">
                  <div className="space-y-2">
                    <label_1.Label>Temp</label_1.Label>
                    <input_1.Input value={registerForm.temperature} onChange={(event) => setRegisterForm({ ...registerForm, temperature: event.target.value })}/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Top P</label_1.Label>
                    <input_1.Input value={registerForm.top_p} onChange={(event) => setRegisterForm({ ...registerForm, top_p: event.target.value })}/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Max Tokens</label_1.Label>
                    <input_1.Input value={registerForm.max_tokens} onChange={(event) => setRegisterForm({ ...registerForm, max_tokens: event.target.value })}/>
                  </div>
                </div>
                <button_1.Button onClick={handleRegister}>Register Model</button_1.Button>
              </card_1.CardContent>
            </card_1.Card>

            <card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">Registered Models</card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-3">
                {models.length === 0 && (<div className="text-sm text-muted-foreground">No models registered.</div>)}
                {models.map((model) => (<div key={model.name} className="rounded border p-3 space-y-1">
                    <div className="flex items-center justify-between">
                      <div className="font-semibold">{model.name}</div>
                      <badge_1.Badge variant="secondary">{model.role}</badge_1.Badge>
                    </div>
                    <div className="text-xs text-muted-foreground">Weight: {model.weight}</div>
                    <div className="text-xs text-muted-foreground">Base: {model.api_base}</div>
                    {metrics && metrics[model.name] ? (<pre className="text-xs whitespace-pre-wrap rounded border p-2">
                        {JSON.stringify(metrics[model.name], null, 2)}
                      </pre>) : null}
                  </div>))}
              </card_1.CardContent>
            </card_1.Card>
          </div>

          <separator_1.Separator />

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Ensemble Execution</card_1.CardTitle>
              <card_1.CardDescription>Run prompts across multiple models.</card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-4">
              <div className="grid gap-3 md:grid-cols-3">
                <div className="space-y-2">
                  <label_1.Label>Role</label_1.Label>
                  <select_1.Select value={ensembleForm.role} onValueChange={(value) => setEnsembleForm({ ...ensembleForm, role: value })}>
                    <select_1.SelectTrigger>
                      <select_1.SelectValue placeholder="Select role"/>
                    </select_1.SelectTrigger>
                    <select_1.SelectContent>
                      <select_1.SelectItem value="red_team">Red Team</select_1.SelectItem>
                      <select_1.SelectItem value="blue_team">Blue Team</select_1.SelectItem>
                      <select_1.SelectItem value="evaluator">Evaluator</select_1.SelectItem>
                    </select_1.SelectContent>
                  </select_1.Select>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Selection Strategy</label_1.Label>
                  <select_1.Select value={ensembleForm.selection_strategy} onValueChange={(value) => setEnsembleForm({ ...ensembleForm, selection_strategy: value })}>
                    <select_1.SelectTrigger>
                      <select_1.SelectValue placeholder="Select strategy"/>
                    </select_1.SelectTrigger>
                    <select_1.SelectContent>
                      {strategies.map((strategy) => (<select_1.SelectItem key={strategy} value={strategy}>
                          {strategy}
                        </select_1.SelectItem>))}
                    </select_1.SelectContent>
                  </select_1.Select>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Responses</label_1.Label>
                  <input_1.Input value={ensembleForm.num_responses} onChange={(event) => setEnsembleForm({ ...ensembleForm, num_responses: event.target.value })}/>
                </div>
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <label_1.Label>Temperature</label_1.Label>
                  <input_1.Input value={ensembleForm.temperature} onChange={(event) => setEnsembleForm({ ...ensembleForm, temperature: event.target.value })}/>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Max Tokens</label_1.Label>
                  <input_1.Input value={ensembleForm.max_tokens} onChange={(event) => setEnsembleForm({ ...ensembleForm, max_tokens: event.target.value })}/>
                </div>
              </div>
              <div className="space-y-2">
                <label_1.Label>Input Content</label_1.Label>
                <textarea_1.Textarea value={ensembleForm.input} onChange={(event) => setEnsembleForm({ ...ensembleForm, input: event.target.value })} rows={4}/>
              </div>
              <button_1.Button onClick={handleEnsemble}>Execute Ensemble</button_1.Button>
              {ensembleResponses.length > 0 && (<div className="space-y-3">
                  {ensembleResponses.map((response, index) => (<card_1.Card key={`${index}-${response.source_model ?? "model"}`}>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-sm">
                          Response {index + 1} ({String(response.source_model ?? "model")})
                        </card_1.CardTitle>
                      </card_1.CardHeader>
                      <card_1.CardContent className="text-sm whitespace-pre-wrap">
                        {String(response.response ?? "")}
                      </card_1.CardContent>
                    </card_1.Card>))}
                </div>)}
            </card_1.CardContent>
          </card_1.Card>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.ModelOrchestrationTab = ModelOrchestrationTab;
//# sourceMappingURL=ModelOrchestrationTab.js.map