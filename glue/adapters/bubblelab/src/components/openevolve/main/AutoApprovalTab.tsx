import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { openevolveApi } from "../../../lib/openevolveApi";
import type { AutoApprovalConfig, AutoApprovalRule, AutoApprovalCondition } from "../../../lib/types";

const defaultRule = (): AutoApprovalRule => ({
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

export const AutoApprovalTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [config, setConfig] = useState<AutoApprovalConfig>({ enabled: false, rules: [] });
  const [editingRule, setEditingRule] = useState<AutoApprovalRule | null>(null);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [testPlan, setTestPlan] = useState({
    complexity: "5",
    confidence: "0.8",
    domain: "software",
    num_sub_problems: "6",
    team_type: "standard",
  });
  const [testResults, setTestResults] = useState<Array<Record<string, unknown>>>([]);
  const [auditLog, setAuditLog] = useState<Array<Record<string, unknown>>>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const loadConfig = async () => {
    setErrorMessage(null);
    try {
      const data = await openevolveApi.getAutoApprovalConfig(apiConfig);
      setConfig(data);
      const audit = await openevolveApi.getAutoApprovalAudit(apiConfig);
      setAuditLog(audit.logs ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load auto-approval config.");
    }
  };

  useEffect(() => {
    loadConfig();
  }, [apiConfig.apiKey]);

  const saveConfig = async (nextConfig: AutoApprovalConfig) => {
    setErrorMessage(null);
    try {
      const updated = await openevolveApi.updateAutoApprovalConfig(nextConfig, apiConfig);
      setConfig(updated);
      setStatusMessage("Configuration saved.");
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to save configuration.");
    }
  };

  const startNewRule = () => {
    setEditingRule(defaultRule());
    setEditingIndex(null);
  };

  const handleEditRule = (rule: AutoApprovalRule, index: number) => {
    setEditingRule(JSON.parse(JSON.stringify(rule)));
    setEditingIndex(index);
  };

  const updateRuleField = (field: keyof AutoApprovalRule, value: any) => {
    if (!editingRule) return;
    setEditingRule({ ...editingRule, [field]: value });
  };

  const updateCondition = (index: number, updates: Partial<AutoApprovalCondition>) => {
    if (!editingRule) return;
    const nextConditions = editingRule.conditions.map((condition, idx) =>
      idx === index ? { ...condition, ...updates } : condition,
    );
    setEditingRule({ ...editingRule, conditions: nextConditions });
  };

  const addCondition = () => {
    if (!editingRule) return;
    setEditingRule({
      ...editingRule,
      conditions: [
        ...editingRule.conditions,
        { field: "complexity", operator: ">", value: 5, logical_op: "AND" },
      ],
    });
  };

  const removeCondition = (index: number) => {
    if (!editingRule) return;
    const next = editingRule.conditions.filter((_, idx) => idx !== index);
    setEditingRule({ ...editingRule, conditions: next.length ? next : editingRule.conditions });
  };

  const saveRule = async () => {
    if (!editingRule) return;
    if (!editingRule.name.trim()) {
      setErrorMessage("Rule name is required.");
      return;
    }
    const nextRules = [...config.rules];
    if (editingIndex !== null) {
      nextRules[editingIndex] = editingRule;
    } else {
      nextRules.push(editingRule);
    }
    await saveConfig({ ...config, rules: nextRules });
    setEditingRule(null);
    setEditingIndex(null);
  };

  const deleteRule = async (index: number) => {
    const nextRules = config.rules.filter((_, idx) => idx !== index);
    await saveConfig({ ...config, rules: nextRules });
  };

  const runTest = async () => {
    setErrorMessage(null);
    try {
      const response = await openevolveApi.testAutoApproval(
        {
          plan: {
            complexity: Number(testPlan.complexity),
            confidence: Number(testPlan.confidence),
            domain: testPlan.domain,
            num_sub_problems: Number(testPlan.num_sub_problems),
            team_type: testPlan.team_type,
          },
        },
        apiConfig,
      );
      setTestResults(response.results ?? []);
      const audit = await openevolveApi.getAutoApprovalAudit(apiConfig);
      setAuditLog(audit.logs ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to test rules.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Auto-Approval Configuration</CardTitle>
          <CardDescription>Define rules for automatic plan approvals.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
                value={apiKey}
                type="password"
                onChange={(event) => {
                  const value = event.target.value;
                  setApiKey(value);
                  try {
                    globalThis.localStorage?.setItem("openevolve_api_key", value);
                  } catch {
                    // ignore storage errors
                  }
                }}
              />
            </div>
            <Button variant="outline" onClick={loadConfig}>
              Refresh
            </Button>
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
            <Switch
              checked={config.enabled}
              onCheckedChange={(checked) => saveConfig({ ...config, enabled: checked })}
            />
          </div>

          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Rules</h3>
            <Button variant="outline" size="sm" onClick={startNewRule}>
              Add Rule
            </Button>
          </div>

          <div className="space-y-3">
            {config.rules.length === 0 && (
              <div className="text-sm text-muted-foreground">No rules configured.</div>
            )}
            {config.rules.map((rule, index) => (
              <Card key={`${rule.name}-${index}`}>
                <CardHeader>
                  <CardTitle className="text-sm">{rule.name}</CardTitle>
                  <CardDescription>
                    Priority {rule.priority} · Action {rule.action}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex items-center gap-2">
                    <Badge variant={rule.enabled ? "default" : "secondary"}>
                      {rule.enabled ? "Enabled" : "Disabled"}
                    </Badge>
                  </div>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    {rule.conditions.map((condition, idx) => (
                      <div key={`${condition.field}-${idx}`}>
                        {condition.field} {condition.operator} {String(condition.value)}{" "}
                        {condition.logical_op}
                      </div>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline" onClick={() => handleEditRule(rule, index)}>
                      Edit
                    </Button>
                    <Button size="sm" variant="destructive" onClick={() => deleteRule(index)}>
                      Delete
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {editingRule ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">
                  {editingIndex !== null ? "Edit Rule" : "New Rule"}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label>Rule Name</Label>
                    <Input
                      value={editingRule.name}
                      onChange={(event) => updateRuleField("name", event.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Priority</Label>
                    <Input
                      type="number"
                      value={editingRule.priority}
                      onChange={(event) => updateRuleField("priority", Number(event.target.value))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Action</Label>
                    <Select
                      value={editingRule.action}
                      onValueChange={(value) => updateRuleField("action", value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="approve">Approve</SelectItem>
                        <SelectItem value="reject">Reject</SelectItem>
                        <SelectItem value="escalate">Escalate</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch
                      checked={editingRule.enabled}
                      onCheckedChange={(checked) => updateRuleField("enabled", checked)}
                    />
                    <span className="text-sm">Enabled</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Conditions</Label>
                  <div className="space-y-2">
                    {editingRule.conditions.map((condition, index) => (
                      <div
                        key={`${condition.field}-${index}`}
                        className="grid gap-2 md:grid-cols-[2fr_1fr_2fr_1fr_auto] items-end"
                      >
                        <Input
                          value={condition.field}
                          onChange={(event) =>
                            updateCondition(index, { field: event.target.value })
                          }
                          placeholder="Field"
                        />
                        <Select
                          value={condition.operator}
                          onValueChange={(value) => updateCondition(index, { operator: value })}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="<">{"<"}</SelectItem>
                            <SelectItem value=">">{">"}</SelectItem>
                            <SelectItem value="==">==</SelectItem>
                            <SelectItem value="!=">!=</SelectItem>
                            <SelectItem value="contains">contains</SelectItem>
                          </SelectContent>
                        </Select>
                        <Input
                          value={String(condition.value)}
                          onChange={(event) => updateCondition(index, { value: event.target.value })}
                          placeholder="Value"
                        />
                        <Select
                          value={condition.logical_op || "AND"}
                          onValueChange={(value) => updateCondition(index, { logical_op: value as "AND" | "OR" })}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="AND">AND</SelectItem>
                            <SelectItem value="OR">OR</SelectItem>
                          </SelectContent>
                        </Select>
                        <Button variant="ghost" onClick={() => removeCondition(index)}>
                          Remove
                        </Button>
                      </div>
                    ))}
                  </div>
                  <Button variant="outline" size="sm" onClick={addCondition}>
                    Add Condition
                  </Button>
                </div>

                <div className="flex gap-2">
                  <Button onClick={saveRule}>Save Rule</Button>
                  <Button variant="outline" onClick={() => setEditingRule(null)}>
                    Cancel
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Rule Testing</CardTitle>
          <CardDescription>Evaluate rules against a sample plan.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Complexity</Label>
              <Input
                value={testPlan.complexity}
                onChange={(event) => setTestPlan((prev) => ({ ...prev, complexity: event.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label>Confidence</Label>
              <Input
                value={testPlan.confidence}
                onChange={(event) => setTestPlan((prev) => ({ ...prev, confidence: event.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label>Domain</Label>
              <Input
                value={testPlan.domain}
                onChange={(event) => setTestPlan((prev) => ({ ...prev, domain: event.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label>Sub-problems</Label>
              <Input
                value={testPlan.num_sub_problems}
                onChange={(event) =>
                  setTestPlan((prev) => ({ ...prev, num_sub_problems: event.target.value }))
                }
              />
            </div>
            <div className="space-y-2">
              <Label>Team Type</Label>
              <Input
                value={testPlan.team_type}
                onChange={(event) => setTestPlan((prev) => ({ ...prev, team_type: event.target.value }))}
              />
            </div>
          </div>
          <Button onClick={runTest}>Run Test</Button>
          <div className="space-y-2 text-sm">
            {testResults.length === 0 && (
              <div className="text-muted-foreground">No test results yet.</div>
            )}
            {testResults.map((result, index) => (
              <div key={index} className="rounded border p-2">
                <div className="font-semibold">{String(result.rule_name)}</div>
                <div className="text-xs text-muted-foreground">
                  Action: {String(result.action)} · Match: {String(result.matched)}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Audit Log</CardTitle>
          <CardDescription>Recent auto-approval evaluations.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          {auditLog.length === 0 && <div className="text-muted-foreground">No audit entries.</div>}
          {auditLog.map((entry, index) => (
            <div key={index} className="rounded border p-2">
              <div className="font-semibold">{String(entry.rule_name)}</div>
              <div className="text-xs text-muted-foreground">
                {String(entry.timestamp)} · {String(entry.action)} · Match:{" "}
                {String(entry.matched)}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
};
