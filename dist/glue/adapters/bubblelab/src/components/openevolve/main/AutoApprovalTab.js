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
exports.AutoApprovalTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const switch_1 = require("@/components/ui/switch");
const select_1 = require("@/components/ui/select");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const defaultRule = () => ({
    name: "",
    priority: 10,
    action: "approve",
    enabled: true,
    conditions: [
        {
            field: "complexity",
            operator: "<",
            value: 5,
            logical_op: "AND",
        },
    ],
});
const AutoApprovalTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [config, setConfig] = (0, react_1.useState)({ enabled: false, rules: [] });
    const [editingRule, setEditingRule] = (0, react_1.useState)(null);
    const [editingIndex, setEditingIndex] = (0, react_1.useState)(null);
    const [testPlan, setTestPlan] = (0, react_1.useState)({
        complexity: "5",
        confidence: "0.8",
        domain: "software",
        num_sub_problems: "6",
        team_type: "standard",
    });
    const [testResults, setTestResults] = (0, react_1.useState)([]);
    const [auditLog, setAuditLog] = (0, react_1.useState)([]);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const loadConfig = async () => {
        setErrorMessage(null);
        try {
            const data = await openevolveApi_1.openevolveApi.getAutoApprovalConfig(apiConfig);
            setConfig(data);
            const audit = await openevolveApi_1.openevolveApi.getAutoApprovalAudit(apiConfig);
            setAuditLog(audit.logs ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to load auto-approval config.");
        }
    };
    (0, react_1.useEffect)(() => {
        loadConfig();
    }, [apiConfig.apiKey]);
    const saveConfig = async (nextConfig) => {
        setErrorMessage(null);
        try {
            const updated = await openevolveApi_1.openevolveApi.updateAutoApprovalConfig(nextConfig, apiConfig);
            setConfig(updated);
            setStatusMessage("Configuration saved.");
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to save configuration.");
        }
    };
    const startNewRule = () => {
        setEditingRule(defaultRule());
        setEditingIndex(null);
    };
    const handleEditRule = (rule, index) => {
        setEditingRule(JSON.parse(JSON.stringify(rule)));
        setEditingIndex(index);
    };
    const updateRuleField = (field, value) => {
        if (!editingRule)
            return;
        setEditingRule({ ...editingRule, [field]: value });
    };
    const updateCondition = (index, updates) => {
        if (!editingRule)
            return;
        const nextConditions = editingRule.conditions.map((condition, idx) => idx === index ? { ...condition, ...updates } : condition);
        setEditingRule({ ...editingRule, conditions: nextConditions });
    };
    const addCondition = () => {
        if (!editingRule)
            return;
        setEditingRule({
            ...editingRule,
            conditions: [
                ...editingRule.conditions,
                { field: "complexity", operator: ">", value: 5, logical_op: "AND" },
            ],
        });
    };
    const removeCondition = (index) => {
        if (!editingRule)
            return;
        const next = editingRule.conditions.filter((_, idx) => idx !== index);
        setEditingRule({ ...editingRule, conditions: next.length ? next : editingRule.conditions });
    };
    const saveRule = async () => {
        if (!editingRule)
            return;
        if (!editingRule.name.trim()) {
            setErrorMessage("Rule name is required.");
            return;
        }
        const nextRules = [...config.rules];
        if (editingIndex !== null) {
            nextRules[editingIndex] = editingRule;
        }
        else {
            nextRules.push(editingRule);
        }
        await saveConfig({ ...config, rules: nextRules });
        setEditingRule(null);
        setEditingIndex(null);
    };
    const deleteRule = async (index) => {
        const nextRules = config.rules.filter((_, idx) => idx !== index);
        await saveConfig({ ...config, rules: nextRules });
    };
    const runTest = async () => {
        setErrorMessage(null);
        try {
            const response = await openevolveApi_1.openevolveApi.testAutoApproval({
                plan: {
                    complexity: Number(testPlan.complexity),
                    confidence: Number(testPlan.confidence),
                    domain: testPlan.domain,
                    num_sub_problems: Number(testPlan.num_sub_problems),
                    team_type: testPlan.team_type,
                },
            }, apiConfig);
            setTestResults(response.results ?? []);
            const audit = await openevolveApi_1.openevolveApi.getAutoApprovalAudit(apiConfig);
            setAuditLog(audit.logs ?? []);
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to test rules.");
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Auto-Approval Configuration</card_1.CardTitle>
          <card_1.CardDescription>Define rules for automatic plan approvals.</card_1.CardDescription>
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
                // ignore storage errors
            }
        }}/>
            </div>
            <button_1.Button variant="outline" onClick={loadConfig}>
              Refresh
            </button_1.Button>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="flex items-center justify-between rounded border p-3">
            <div>
              <div className="font-semibold">Enable Auto-Approval</div>
              <div className="text-xs text-muted-foreground">
                Automatically approve plans that match configured rules.
              </div>
            </div>
            <switch_1.Switch checked={config.enabled} onCheckedChange={(checked) => saveConfig({ ...config, enabled: checked })}/>
          </div>

          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Rules</h3>
            <button_1.Button variant="outline" size="sm" onClick={startNewRule}>
              Add Rule
            </button_1.Button>
          </div>

          <div className="space-y-3">
            {config.rules.length === 0 && (<div className="text-sm text-muted-foreground">No rules configured.</div>)}
            {config.rules.map((rule, index) => (<card_1.Card key={`${rule.name}-${index}`}>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-sm">{rule.name}</card_1.CardTitle>
                  <card_1.CardDescription>
                    Priority {rule.priority} · Action {rule.action}
                  </card_1.CardDescription>
                </card_1.CardHeader>
                <card_1.CardContent className="space-y-2 text-sm">
                  <div className="flex items-center gap-2">
                    <badge_1.Badge variant={rule.enabled ? "default" : "secondary"}>
                      {rule.enabled ? "Enabled" : "Disabled"}
                    </badge_1.Badge>
                  </div>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    {rule.conditions.map((condition, idx) => (<div key={`${condition.field}-${idx}`}>
                        {condition.field} {condition.operator} {String(condition.value)}{" "}
                        {condition.logical_op}
                      </div>))}
                  </div>
                  <div className="flex gap-2">
                    <button_1.Button size="sm" variant="outline" onClick={() => handleEditRule(rule, index)}>
                      Edit
                    </button_1.Button>
                    <button_1.Button size="sm" variant="destructive" onClick={() => deleteRule(index)}>
                      Delete
                    </button_1.Button>
                  </div>
                </card_1.CardContent>
              </card_1.Card>))}
          </div>

          {editingRule ? (<card_1.Card>
              <card_1.CardHeader>
                <card_1.CardTitle className="text-sm">
                  {editingIndex !== null ? "Edit Rule" : "New Rule"}
                </card_1.CardTitle>
              </card_1.CardHeader>
              <card_1.CardContent className="space-y-3">
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <label_1.Label>Rule Name</label_1.Label>
                    <input_1.Input value={editingRule.name} onChange={(event) => updateRuleField("name", event.target.value)}/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Priority</label_1.Label>
                    <input_1.Input type="number" value={editingRule.priority} onChange={(event) => updateRuleField("priority", Number(event.target.value))}/>
                  </div>
                  <div className="space-y-2">
                    <label_1.Label>Action</label_1.Label>
                    <select_1.Select value={editingRule.action} onValueChange={(value) => updateRuleField("action", value)}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue />
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        <select_1.SelectItem value="approve">Approve</select_1.SelectItem>
                        <select_1.SelectItem value="reject">Reject</select_1.SelectItem>
                        <select_1.SelectItem value="escalate">Escalate</select_1.SelectItem>
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                  <div className="flex items-center gap-2">
                    <switch_1.Switch checked={editingRule.enabled} onCheckedChange={(checked) => updateRuleField("enabled", checked)}/>
                    <span className="text-sm">Enabled</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <label_1.Label>Conditions</label_1.Label>
                  <div className="space-y-2">
                    {editingRule.conditions.map((condition, index) => (<div key={`${condition.field}-${index}`} className="grid gap-2 md:grid-cols-[2fr_1fr_2fr_1fr_auto] items-end">
                        <input_1.Input value={condition.field} onChange={(event) => updateCondition(index, { field: event.target.value })} placeholder="Field"/>
                        <select_1.Select value={condition.operator} onValueChange={(value) => updateCondition(index, { operator: value })}>
                          <select_1.SelectTrigger>
                            <select_1.SelectValue />
                          </select_1.SelectTrigger>
                          <select_1.SelectContent>
                            <select_1.SelectItem value="<">{"<"}</select_1.SelectItem>
                            <select_1.SelectItem value=">">{">"}</select_1.SelectItem>
                            <select_1.SelectItem value="==">==</select_1.SelectItem>
                            <select_1.SelectItem value="!=">!=</select_1.SelectItem>
                            <select_1.SelectItem value="contains">contains</select_1.SelectItem>
                          </select_1.SelectContent>
                        </select_1.Select>
                        <input_1.Input value={String(condition.value)} onChange={(event) => updateCondition(index, { value: event.target.value })} placeholder="Value"/>
                        <select_1.Select value={condition.logical_op || "AND"} onValueChange={(value) => updateCondition(index, { logical_op: value })}>
                          <select_1.SelectTrigger>
                            <select_1.SelectValue />
                          </select_1.SelectTrigger>
                          <select_1.SelectContent>
                            <select_1.SelectItem value="AND">AND</select_1.SelectItem>
                            <select_1.SelectItem value="OR">OR</select_1.SelectItem>
                          </select_1.SelectContent>
                        </select_1.Select>
                        <button_1.Button variant="ghost" onClick={() => removeCondition(index)}>
                          Remove
                        </button_1.Button>
                      </div>))}
                  </div>
                  <button_1.Button variant="outline" size="sm" onClick={addCondition}>
                    Add Condition
                  </button_1.Button>
                </div>

                <div className="flex gap-2">
                  <button_1.Button onClick={saveRule}>Save Rule</button_1.Button>
                  <button_1.Button variant="outline" onClick={() => setEditingRule(null)}>
                    Cancel
                  </button_1.Button>
                </div>
              </card_1.CardContent>
            </card_1.Card>) : null}
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Rule Testing</card_1.CardTitle>
          <card_1.CardDescription>Evaluate rules against a sample plan.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>Complexity</label_1.Label>
              <input_1.Input value={testPlan.complexity} onChange={(event) => setTestPlan((prev) => ({ ...prev, complexity: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Confidence</label_1.Label>
              <input_1.Input value={testPlan.confidence} onChange={(event) => setTestPlan((prev) => ({ ...prev, confidence: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Domain</label_1.Label>
              <input_1.Input value={testPlan.domain} onChange={(event) => setTestPlan((prev) => ({ ...prev, domain: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Sub-problems</label_1.Label>
              <input_1.Input value={testPlan.num_sub_problems} onChange={(event) => setTestPlan((prev) => ({ ...prev, num_sub_problems: event.target.value }))}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Team Type</label_1.Label>
              <input_1.Input value={testPlan.team_type} onChange={(event) => setTestPlan((prev) => ({ ...prev, team_type: event.target.value }))}/>
            </div>
          </div>
          <button_1.Button onClick={runTest}>Run Test</button_1.Button>
          <div className="space-y-2 text-sm">
            {testResults.length === 0 && (<div className="text-muted-foreground">No test results yet.</div>)}
            {testResults.map((result, index) => (<div key={index} className="rounded border p-2">
                <div className="font-semibold">{String(result.rule_name)}</div>
                <div className="text-xs text-muted-foreground">
                  Action: {String(result.action)} · Match: {String(result.matched)}
                </div>
              </div>))}
          </div>
        </card_1.CardContent>
      </card_1.Card>

      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Audit Log</card_1.CardTitle>
          <card_1.CardDescription>Recent auto-approval evaluations.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-2 text-sm">
          {auditLog.length === 0 && <div className="text-muted-foreground">No audit entries.</div>}
          {auditLog.map((entry, index) => (<div key={index} className="rounded border p-2">
              <div className="font-semibold">{String(entry.rule_name)}</div>
              <div className="text-xs text-muted-foreground">
                {String(entry.timestamp)} · {String(entry.action)} · Match:{" "}
                {String(entry.matched)}
              </div>
            </div>))}
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.AutoApprovalTab = AutoApprovalTab;
//# sourceMappingURL=AutoApprovalTab.js.map