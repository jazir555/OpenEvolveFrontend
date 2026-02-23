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
exports.ContentManagerTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const VALIDATION_TYPES = ["generic", "compliance", "security", "technical"];
const ContentManagerTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [templates, setTemplates] = (0, react_1.useState)([]);
    const [selectedTemplate, setSelectedTemplate] = (0, react_1.useState)("");
    const [templateContent, setTemplateContent] = (0, react_1.useState)("");
    const [newTemplateName, setNewTemplateName] = (0, react_1.useState)("");
    const [newTemplateContent, setNewTemplateContent] = (0, react_1.useState)("");
    const [validationText, setValidationText] = (0, react_1.useState)("");
    const [validationType, setValidationType] = (0, react_1.useState)(VALIDATION_TYPES[0]);
    const [validationResult, setValidationResult] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const loadTemplates = async () => {
        setErrorMessage(null);
        try {
            const result = await openevolveApi_1.openevolveApi.listContentTemplates(apiConfig);
            setTemplates(result.templates || []);
            if (!selectedTemplate && result.templates?.length) {
                setSelectedTemplate(result.templates[0]);
            }
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load templates.");
        }
    };
    const loadTemplate = async (name) => {
        setErrorMessage(null);
        try {
            const result = await openevolveApi_1.openevolveApi.getContentTemplate(name, apiConfig);
            setTemplateContent(result.content || "");
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load template content.");
        }
    };
    (0, react_1.useEffect)(() => {
        loadTemplates();
    }, [apiConfig.apiKey]);
    (0, react_1.useEffect)(() => {
        if (selectedTemplate) {
            loadTemplate(selectedTemplate);
        }
    }, [selectedTemplate]);
    const createTemplate = async () => {
        setErrorMessage(null);
        setStatusMessage(null);
        if (!newTemplateName.trim()) {
            setErrorMessage("Template name is required.");
            return;
        }
        try {
            await openevolveApi_1.openevolveApi.createContentTemplate({ name: newTemplateName.trim(), content: newTemplateContent }, apiConfig);
            setStatusMessage(`Saved template '${newTemplateName}'.`);
            setNewTemplateName("");
            setNewTemplateContent("");
            await loadTemplates();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to save template.");
        }
    };
    const validateProtocol = async () => {
        setErrorMessage(null);
        try {
            const result = await openevolveApi_1.openevolveApi.validateProtocol({ protocol_text: validationText, validation_type: validationType }, apiConfig);
            setValidationResult(result);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to validate protocol.");
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Content Manager</card_1.CardTitle>
          <card_1.CardDescription>Protocol templates and validation tools.</card_1.CardDescription>
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
              <label_1.Label>Templates</label_1.Label>
              <select_1.Select value={selectedTemplate} onValueChange={setSelectedTemplate}>
                <select_1.SelectTrigger>
                  <select_1.SelectValue placeholder="Select template"/>
                </select_1.SelectTrigger>
                <select_1.SelectContent>
                  {templates.map((template) => (<select_1.SelectItem key={template} value={template}>
                      {template}
                    </select_1.SelectItem>))}
                </select_1.SelectContent>
              </select_1.Select>
            </div>
            <div className="space-y-2">
              <label_1.Label>Selected Template Content</label_1.Label>
              <textarea_1.Textarea value={templateContent} readOnly rows={6}/>
            </div>
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle className="text-sm">Create Template</card_1.CardTitle>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-3">
          <div className="space-y-2">
            <label_1.Label>Name</label_1.Label>
            <input_1.Input value={newTemplateName} onChange={(event) => setNewTemplateName(event.target.value)}/>
          </div>
          <div className="space-y-2">
            <label_1.Label>Content</label_1.Label>
            <textarea_1.Textarea value={newTemplateContent} onChange={(event) => setNewTemplateContent(event.target.value)} rows={8}/>
          </div>
          <button_1.Button onClick={createTemplate}>Save Template</button_1.Button>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle className="text-sm">Protocol Validation</card_1.CardTitle>
          <card_1.CardDescription>Validate protocol text against configured rules.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-3">
          <div className="space-y-2">
            <label_1.Label>Validation Type</label_1.Label>
            <select_1.Select value={validationType} onValueChange={setValidationType}>
              <select_1.SelectTrigger>
                <select_1.SelectValue placeholder="Select validation type"/>
              </select_1.SelectTrigger>
              <select_1.SelectContent>
                {VALIDATION_TYPES.map((type) => (<select_1.SelectItem key={type} value={type}>
                    {type}
                  </select_1.SelectItem>))}
              </select_1.SelectContent>
            </select_1.Select>
          </div>
          <div className="space-y-2">
            <label_1.Label>Protocol Text</label_1.Label>
            <textarea_1.Textarea value={validationText} onChange={(event) => setValidationText(event.target.value)} rows={8}/>
          </div>
          <button_1.Button variant="outline" onClick={validateProtocol}>
            Validate
          </button_1.Button>
          {validationResult && (<div className="rounded border p-3 text-sm space-y-1">
              <div>Valid: {validationResult.valid ? "Yes" : "No"}</div>
              <div>Score: {validationResult.score}</div>
              <div>Errors: {validationResult.errors?.join(", ") || "None"}</div>
              <div>Warnings: {validationResult.warnings?.join(", ") || "None"}</div>
              <div>Suggestions: {validationResult.suggestions?.join(", ") || "None"}</div>
            </div>)}
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.ContentManagerTab = ContentManagerTab;
//# sourceMappingURL=ContentManagerTab.js.map