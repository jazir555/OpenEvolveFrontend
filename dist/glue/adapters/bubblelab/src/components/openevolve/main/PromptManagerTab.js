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
exports.PromptManagerTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const PromptManagerTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [prompts, setPrompts] = (0, react_1.useState)({});
    const [promptName, setPromptName] = (0, react_1.useState)("");
    const [promptContent, setPromptContent] = (0, react_1.useState)("");
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const loadPrompts = async () => {
        setErrorMessage(null);
        try {
            const result = await openevolveApi_1.openevolveApi.listPrompts(apiConfig);
            setPrompts(result.prompts || {});
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load prompts.");
        }
    };
    (0, react_1.useEffect)(() => {
        loadPrompts();
    }, [apiConfig.apiKey]);
    const savePrompt = async () => {
        setErrorMessage(null);
        setStatusMessage(null);
        if (!promptName.trim()) {
            setErrorMessage("Prompt name is required.");
            return;
        }
        try {
            await openevolveApi_1.openevolveApi.savePrompt({ name: promptName.trim(), content: promptContent }, apiConfig);
            setStatusMessage(`Saved prompt '${promptName}'.`);
            await loadPrompts();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to save prompt.");
        }
    };
    const deletePrompt = async (name) => {
        if (!confirm(`Delete prompt '${name}'?`)) {
            return;
        }
        try {
            await openevolveApi_1.openevolveApi.deletePrompt(name, apiConfig);
            await loadPrompts();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to delete prompt.");
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Prompt Manager</card_1.CardTitle>
          <card_1.CardDescription>Store and reuse custom prompts for evolution runs.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="space-y-2">
            <label_1.Label>API Key</label_1.Label>
            <input_1.Input type="password" value={apiKey} onChange={(event) => {
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

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>Prompt Name</label_1.Label>
              <input_1.Input value={promptName} onChange={(event) => setPromptName(event.target.value)}/>
            </div>
            <div className="flex items-end">
              <button_1.Button onClick={savePrompt}>Save Prompt</button_1.Button>
            </div>
          </div>

          <div className="space-y-2">
            <label_1.Label>Prompt Content</label_1.Label>
            <textarea_1.Textarea value={promptContent} onChange={(event) => setPromptContent(event.target.value)} rows={8}/>
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle className="text-sm">Saved Prompts</card_1.CardTitle>
          <card_1.CardDescription>Manage stored prompt templates.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-3 text-sm">
          {Object.keys(prompts).length === 0 ? (<div className="text-muted-foreground">No custom prompts saved.</div>) : (Object.entries(prompts).map(([name, content]) => (<div key={name} className="rounded border p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="font-semibold">{name}</div>
                  <div className="flex items-center gap-2">
                    <badge_1.Badge variant="secondary">{content.length} chars</badge_1.Badge>
                    <button_1.Button variant="outline" size="sm" onClick={() => deletePrompt(name)}>
                      Delete
                    </button_1.Button>
                  </div>
                </div>
                <div className="text-xs text-muted-foreground whitespace-pre-wrap">
                  {content}
                </div>
              </div>)))}
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.PromptManagerTab = PromptManagerTab;
//# sourceMappingURL=PromptManagerTab.js.map