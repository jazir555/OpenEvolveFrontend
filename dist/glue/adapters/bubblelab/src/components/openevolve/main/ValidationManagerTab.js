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
exports.ValidationManagerTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const tabs_1 = require("@/components/ui/tabs");
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
const parseCommaList = (value) => value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
const ValidationManagerTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => readStorage("openevolve_api_key"));
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [rules, setRules] = (0, react_1.useState)({});
    const [ruleNames, setRuleNames] = (0, react_1.useState)([]);
    const [selectedRules, setSelectedRules] = (0, react_1.useState)([]);
    const [form, setForm] = (0, react_1.useState)({
        name: "",
        max_length: "",
        min_length: "",
        required_keywords: "",
        forbidden_patterns: "",
        required_sections: "",
    });
    const [content, setContent] = (0, react_1.useState)("");
    const [validationResult, setValidationResult] = (0, react_1.useState)(null);
    const [complianceFramework, setComplianceFramework] = (0, react_1.useState)("generic");
    const [complianceResult, setComplianceResult] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const loadRules = async () => {
        setLoading(true);
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.listValidationRules(apiConfig);
            setRules(response.rules ?? {});
            setRuleNames(response.rule_names ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load validation rules.");
        }
        finally {
            setLoading(false);
        }
    };
    (0, react_1.useEffect)(() => {
        loadRules();
    }, [apiConfig.apiKey]);
    const toggleRule = (ruleName) => {
        setSelectedRules((prev) => prev.includes(ruleName) ? prev.filter((name) => name !== ruleName) : [...prev, ruleName]);
    };
    const handleCreateRule = async () => {
        setErrorMessage(null);
        setStatusMessage(null);
        if (!form.name.trim()) {
            setErrorMessage("Rule name is required.");
            return;
        }
        const payload = {
            name: form.name.trim(),
            max_length: form.max_length ? Number(form.max_length) : undefined,
            min_length: form.min_length ? Number(form.min_length) : undefined,
            required_keywords: parseCommaList(form.required_keywords),
            forbidden_patterns: parseCommaList(form.forbidden_patterns),
            required_sections: parseCommaList(form.required_sections),
        };
        try {
            await openevolveApi_1.openevolveApi.createValidationRule(payload, apiConfig);
            setStatusMessage(`Rule ${payload.name} created.`);
            setForm({
                name: "",
                max_length: "",
                min_length: "",
                required_keywords: "",
                forbidden_patterns: "",
                required_sections: "",
            });
            await loadRules();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to create rule.");
        }
    };
    const handleDeleteRule = async (ruleName) => {
        setErrorMessage(null);
        try {
            await openevolveApi_1.openevolveApi.deleteValidationRule(ruleName, apiConfig);
            await loadRules();
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to delete rule.");
        }
    };
    const handleRunValidation = async () => {
        setErrorMessage(null);
        setValidationResult(null);
        if (!content.trim()) {
            setErrorMessage("Enter content to validate.");
            return;
        }
        if (selectedRules.length === 0) {
            setErrorMessage("Select at least one rule.");
            return;
        }
        try {
            const result = await openevolveApi_1.openevolveApi.runValidation({ content, rule_names: selectedRules }, apiConfig);
            setValidationResult(result);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Validation failed.");
        }
    };
    const handleComplianceCheck = async () => {
        setErrorMessage(null);
        setComplianceResult(null);
        if (!content.trim()) {
            setErrorMessage("Enter content to check compliance.");
            return;
        }
        try {
            const result = await openevolveApi_1.openevolveApi.runComplianceCheck({ content, framework: complianceFramework }, apiConfig);
            setComplianceResult(result);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Compliance check failed.");
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Validation Manager</card_1.CardTitle>
          <card_1.CardDescription>Manage validation rules and run compliance checks.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>API Key</label_1.Label>
              <input_1.Input value={apiKey} type="password" onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            writeStorage("openevolve_api_key", value);
        }}/>
            </div>
            <div className="flex items-end gap-2">
              <button_1.Button variant="outline" onClick={loadRules} disabled={loading}>
                Refresh Rules
              </button_1.Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}
        </card_1.CardContent>
      </card_1.Card>

      <tabs_1.Tabs defaultValue="manage" className="w-full">
        <tabs_1.TabsList className="grid w-full grid-cols-3">
          <tabs_1.TabsTrigger value="manage">Manage Rules</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="apply">Apply Validation</tabs_1.TabsTrigger>
          <tabs_1.TabsTrigger value="results">Results</tabs_1.TabsTrigger>
        </tabs_1.TabsList>

        <tabs_1.TabsContent value="manage" className="mt-4 space-y-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Current Rules</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              {ruleNames.length === 0 ? (<div className="text-sm text-muted-foreground">No validation rules defined.</div>) : (ruleNames.map((name) => (<div key={name} className="rounded border p-3 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="font-semibold">{name}</div>
                      <button_1.Button size="sm" variant="outline" onClick={() => handleDeleteRule(name)}>
                        Delete
                      </button_1.Button>
                    </div>
                    <pre className="text-xs bg-muted p-2 rounded">
                      {JSON.stringify(rules[name] ?? {}, null, 2)}
                    </pre>
                  </div>)))}
            </card_1.CardContent>
          </card_1.Card>

          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Add Rule</card_1.CardTitle>
              <card_1.CardDescription>Define a new validation rule.</card_1.CardDescription>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <label_1.Label>Rule Name</label_1.Label>
                  <input_1.Input value={form.name} onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}/>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Max Length</label_1.Label>
                  <input_1.Input type="number" value={form.max_length} onChange={(event) => setForm((prev) => ({ ...prev, max_length: event.target.value }))}/>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Min Length</label_1.Label>
                  <input_1.Input type="number" value={form.min_length} onChange={(event) => setForm((prev) => ({ ...prev, min_length: event.target.value }))}/>
                </div>
                <div className="space-y-2">
                  <label_1.Label>Required Sections</label_1.Label>
                  <input_1.Input value={form.required_sections} onChange={(event) => setForm((prev) => ({ ...prev, required_sections: event.target.value }))} placeholder="Comma-separated"/>
                </div>
              </div>
              <div className="space-y-2">
                <label_1.Label>Required Keywords</label_1.Label>
                <input_1.Input value={form.required_keywords} onChange={(event) => setForm((prev) => ({ ...prev, required_keywords: event.target.value }))} placeholder="Comma-separated"/>
              </div>
              <div className="space-y-2">
                <label_1.Label>Forbidden Patterns (regex)</label_1.Label>
                <input_1.Input value={form.forbidden_patterns} onChange={(event) => setForm((prev) => ({ ...prev, forbidden_patterns: event.target.value }))} placeholder="Comma-separated"/>
              </div>
              <button_1.Button onClick={handleCreateRule}>Add Rule</button_1.Button>
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="apply" className="mt-4 space-y-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Validation Target</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              <textarea_1.Textarea value={content} onChange={(event) => setContent(event.target.value)} className="min-h-[180px]" placeholder="Paste the content to validate..."/>
              <div className="space-y-2">
                <label_1.Label>Rules</label_1.Label>
                <div className="flex flex-wrap gap-2">
                  {ruleNames.map((name) => (<button_1.Button key={name} size="sm" variant={selectedRules.includes(name) ? "default" : "outline"} onClick={() => toggleRule(name)}>
                      {name}
                    </button_1.Button>))}
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <button_1.Button onClick={handleRunValidation}>Run Validation</button_1.Button>
                <button_1.Button variant="outline" onClick={handleComplianceCheck}>
                  Run Compliance ({complianceFramework.toUpperCase()})
                </button_1.Button>
                <select className="rounded border px-2 py-1 text-sm" value={complianceFramework} onChange={(event) => setComplianceFramework(event.target.value)}>
                  <option value="generic">Generic</option>
                  <option value="gdpr">GDPR</option>
                  <option value="hipaa">HIPAA</option>
                </select>
              </div>
              {validationResult ? (<div className="rounded border p-3 text-sm space-y-1">
                  <div>
                    Overall: {validationResult.overall_result ? "PASS" : "FAIL"}
                    <badge_1.Badge className="ml-2" variant={validationResult.overall_result ? "default" : "destructive"}>
                      {validationResult.overall_result ? "OK" : "Issues"}
                    </badge_1.Badge>
                  </div>
                  <div>Errors: {validationResult.error_count}</div>
                  <div>Warnings: {validationResult.warning_count}</div>
                  <div>Suggestions: {validationResult.suggestion_count}</div>
                </div>) : null}
              {complianceResult ? (<div className="rounded border p-3 text-sm space-y-1">
                  <div>Compliance result: {complianceResult.valid ? "PASS" : "FAIL"}</div>
                  {complianceResult.errors?.length ? (<div>Errors: {complianceResult.errors.join("; ")}</div>) : null}
                </div>) : null}
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>

        <tabs_1.TabsContent value="results" className="mt-4">
          <card_1.Card>
            <card_1.CardHeader>
              <card_1.CardTitle className="text-sm">Latest Results</card_1.CardTitle>
            </card_1.CardHeader>
            <card_1.CardContent className="space-y-3">
              {validationResult ? (<pre className="text-xs bg-muted p-3 rounded">
                  {JSON.stringify(validationResult, null, 2)}
                </pre>) : (<div className="text-sm text-muted-foreground">Run validation to see results.</div>)}
            </card_1.CardContent>
          </card_1.Card>
        </tabs_1.TabsContent>
      </tabs_1.Tabs>
    </div>);
};
exports.ValidationManagerTab = ValidationManagerTab;
//# sourceMappingURL=ValidationManagerTab.js.map