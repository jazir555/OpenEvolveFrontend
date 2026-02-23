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
exports.ModelDashboardTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const readStorage = (key, fallback) => {
    try {
        const raw = globalThis.localStorage?.getItem(key);
        if (!raw) {
            return fallback;
        }
        return JSON.parse(raw);
    }
    catch {
        return fallback;
    }
};
const writeStorage = (key, value) => {
    try {
        globalThis.localStorage?.setItem(key, JSON.stringify(value));
    }
    catch {
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
const ModelDashboardTab = () => {
    const [openrouterKey, setOpenrouterKey] = (0, react_1.useState)(() => readStorage("openevolve_openrouter_key", ""));
    const [models, setModels] = (0, react_1.useState)([]);
    const [filter, setFilter] = (0, react_1.useState)("");
    const [modelError, setModelError] = (0, react_1.useState)(null);
    const [loadingModels, setLoadingModels] = (0, react_1.useState)(false);
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [teams, setTeams] = (0, react_1.useState)([]);
    const [teamError, setTeamError] = (0, react_1.useState)(null);
    const [performanceJson, setPerformanceJson] = (0, react_1.useState)(() => JSON.stringify(readStorage("openevolve_model_performance", {}), null, 2));
    const [performanceError, setPerformanceError] = (0, react_1.useState)(null);
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
            const modelList = (data?.data ?? data?.models ?? []).map((item) => ({
                id: item.id,
                name: item.name,
                context_length: item.context_length,
                pricing: item.pricing,
                description: item.description,
            }));
            setModels(modelList.length ? modelList : MODEL_FALLBACK.map((id) => ({ id })));
        }
        catch (error) {
            setModelError(error?.message ?? "Failed to load models.");
            setModels(MODEL_FALLBACK.map((id) => ({ id })));
        }
        finally {
            setLoadingModels(false);
        }
    };
    const loadTeams = async () => {
        setTeamError(null);
        try {
            const response = await openevolveApi_1.openevolveApi.listTeams(apiConfig);
            const detailedTeams = await Promise.all(response.teams.map((team) => openevolveApi_1.openevolveApi.getTeam(team.name, apiConfig)));
            setTeams(detailedTeams);
        }
        catch (error) {
            setTeamError(error?.message ?? "Failed to load teams.");
        }
    };
    const applyPerformanceJson = () => {
        setPerformanceError(null);
        try {
            const parsed = JSON.parse(performanceJson || "{}");
            writeStorage("openevolve_model_performance", parsed);
        }
        catch {
            setPerformanceError("Invalid JSON.");
        }
    };
    (0, react_1.useEffect)(() => {
        loadTeams();
    }, [apiConfig.apiKey]);
    const filteredModels = models.filter((model) => model.id.toLowerCase().includes(filter.toLowerCase()));
    const modelUsage = (0, react_1.useMemo)(() => {
        const usage = {};
        teams.forEach((team) => {
            team.members.forEach((member) => {
                usage[member.model_id] = (usage[member.model_id] || 0) + 1;
            });
        });
        return usage;
    }, [teams]);
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Model Catalogue</card_1.CardTitle>
          <card_1.CardDescription>Browse OpenRouter models and local model usage.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-[2fr_1fr]">
            <div className="space-y-2">
              <label_1.Label>OpenRouter API Key (optional)</label_1.Label>
              <input_1.Input value={openrouterKey} type="password" onChange={(event) => {
            const value = event.target.value;
            setOpenrouterKey(value);
            writeStorage("openevolve_openrouter_key", value);
        }}/>
            </div>
            <div className="flex items-end">
              <button_1.Button onClick={loadModels} disabled={loadingModels}>
                Load Models
              </button_1.Button>
            </div>
          </div>

          <div className="space-y-2">
            <label_1.Label>Filter Models</label_1.Label>
            <input_1.Input value={filter} onChange={(event) => setFilter(event.target.value)}/>
          </div>

          {modelError ? <div className="text-sm text-red-500">{modelError}</div> : null}

          <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
            {filteredModels.map((model) => (<div key={model.id} className="rounded border p-3 text-sm space-y-2">
                <div className="flex items-center justify-between">
                  <div className="font-semibold">{model.id}</div>
                  <badge_1.Badge variant="secondary">
                    {model.context_length ? `${model.context_length} ctx` : "unknown ctx"}
                  </badge_1.Badge>
                </div>
                {model.name && <div className="text-xs text-muted-foreground">{model.name}</div>}
                {model.pricing && (<div className="text-xs text-muted-foreground">
                    Pricing: {JSON.stringify(model.pricing)}
                  </div>)}
              </div>))}
            {filteredModels.length === 0 && (<div className="text-sm text-muted-foreground">No models match the filter.</div>)}
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Team Model Usage</card_1.CardTitle>
          <card_1.CardDescription>Model distribution across OpenEvolve teams.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-[2fr_1fr]">
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
            <div className="flex items-end">
              <button_1.Button variant="outline" onClick={loadTeams}>
                Refresh Teams
              </button_1.Button>
            </div>
          </div>

          {teamError ? <div className="text-sm text-red-500">{teamError}</div> : null}

          <div className="grid gap-3 md:grid-cols-2">
            {teams.map((team) => (<div key={team.name} className="rounded border p-3 text-sm space-y-2">
                <div className="font-semibold">
                  {team.name} <badge_1.Badge variant="secondary">{team.role}</badge_1.Badge>
                </div>
                <div className="text-xs text-muted-foreground">
                  Members: {team.members.length}
                </div>
                <div className="text-xs text-muted-foreground">
                  Models: {team.members.map((member) => member.model_id).join(", ") || "n/a"}
                </div>
              </div>))}
          </div>

          <div className="space-y-2">
            <label_1.Label>Model Usage Summary</label_1.Label>
            <div className="grid gap-2 md:grid-cols-2">
              {Object.entries(modelUsage).map(([modelId, count]) => (<div key={modelId} className="flex items-center justify-between rounded border p-2 text-sm">
                  <span>{modelId}</span>
                  <badge_1.Badge variant="secondary">{count}</badge_1.Badge>
                </div>))}
              {Object.keys(modelUsage).length === 0 && (<div className="text-sm text-muted-foreground">No model usage recorded.</div>)}
            </div>
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Model Performance Metrics</card_1.CardTitle>
          <card_1.CardDescription>Store and visualize model evaluation metrics.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-3">
          <textarea_1.Textarea value={performanceJson} onChange={(event) => setPerformanceJson(event.target.value)} className="min-h-[160px]"/>
          {performanceError ? <div className="text-sm text-red-500">{performanceError}</div> : null}
          <div className="flex gap-2">
            <button_1.Button onClick={applyPerformanceJson}>Save Metrics</button_1.Button>
            <button_1.Button variant="outline" onClick={() => setPerformanceJson(JSON.stringify(readStorage("openevolve_model_performance", {}), null, 2))}>
              Reload
            </button_1.Button>
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.ModelDashboardTab = ModelDashboardTab;
//# sourceMappingURL=ModelDashboardTab.js.map