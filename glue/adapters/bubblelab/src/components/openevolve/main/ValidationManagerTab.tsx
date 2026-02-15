import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { openevolveApi } from "../../../lib/openevolveApi";
import type { ValidationRule, ValidationRunResult, ComplianceCheckResult } from "../../../lib/types";

const readStorage = (key: string, fallback = "") => {
  try {
    return globalThis.localStorage?.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
};

const writeStorage = (key: string, value: string) => {
  try {
    globalThis.localStorage?.setItem(key, value);
  } catch {
    // ignore storage errors
  }
};

const parseCommaList = (value: string) =>
  value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

export const ValidationManagerTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(() => readStorage("openevolve_api_key"));
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [rules, setRules] = useState<Record<string, ValidationRule>>({});
  const [ruleNames, setRuleNames] = useState<string[]>([]);
  const [selectedRules, setSelectedRules] = useState<string[]>([]);

  const [form, setForm] = useState({
    name: "",
    max_length: "",
    min_length: "",
    required_keywords: "",
    forbidden_patterns: "",
    required_sections: "",
  });

  const [content, setContent] = useState("");
  const [validationResult, setValidationResult] = useState<ValidationRunResult | null>(null);
  const [complianceFramework, setComplianceFramework] = useState("generic");
  const [complianceResult, setComplianceResult] = useState<ComplianceCheckResult | null>(null);

  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const loadRules = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.listValidationRules(apiConfig);
      setRules(response.rules ?? {});
      setRuleNames(response.rule_names ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load validation rules.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRules();
  }, [apiConfig.apiKey]);

  const toggleRule = (ruleName: string) => {
    setSelectedRules((prev) =>
      prev.includes(ruleName) ? prev.filter((name) => name !== ruleName) : [...prev, ruleName],
    );
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
      await openevolveApi.createValidationRule(payload, apiConfig);
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
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to create rule.");
    }
  };

  const handleDeleteRule = async (ruleName: string) => {
    setErrorMessage(null);
    try {
      await openevolveApi.deleteValidationRule(ruleName, apiConfig);
      await loadRules();
    } catch (error: any) {
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
      const result = await openevolveApi.runValidation(
        { content, rule_names: selectedRules },
        apiConfig,
      );
      setValidationResult(result);
    } catch (error: any) {
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
      const result = await openevolveApi.runComplianceCheck(
        { content, framework: complianceFramework },
        apiConfig,
      );
      setComplianceResult(result);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Compliance check failed.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Validation Manager</CardTitle>
          <CardDescription>Manage validation rules and run compliance checks.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
                value={apiKey}
                type="password"
                onChange={(event) => {
                  const value = event.target.value;
                  setApiKey(value);
                  writeStorage("openevolve_api_key", value);
                }}
              />
            </div>
            <div className="flex items-end gap-2">
              <Button variant="outline" onClick={loadRules} disabled={loading}>
                Refresh Rules
              </Button>
            </div>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}
        </CardContent>
      </Card>

      <Tabs defaultValue="manage" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="manage">Manage Rules</TabsTrigger>
          <TabsTrigger value="apply">Apply Validation</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        <TabsContent value="manage" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Current Rules</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {ruleNames.length === 0 ? (
                <div className="text-sm text-muted-foreground">No validation rules defined.</div>
              ) : (
                ruleNames.map((name) => (
                  <div key={name} className="rounded border p-3 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="font-semibold">{name}</div>
                      <Button size="sm" variant="outline" onClick={() => handleDeleteRule(name)}>
                        Delete
                      </Button>
                    </div>
                    <pre className="text-xs bg-muted p-2 rounded">
                      {JSON.stringify(rules[name] ?? {}, null, 2)}
                    </pre>
                  </div>
                ))
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Add Rule</CardTitle>
              <CardDescription>Define a new validation rule.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>Rule Name</Label>
                  <Input
                    value={form.name}
                    onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Max Length</Label>
                  <Input
                    type="number"
                    value={form.max_length}
                    onChange={(event) => setForm((prev) => ({ ...prev, max_length: event.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Min Length</Label>
                  <Input
                    type="number"
                    value={form.min_length}
                    onChange={(event) => setForm((prev) => ({ ...prev, min_length: event.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Required Sections</Label>
                  <Input
                    value={form.required_sections}
                    onChange={(event) =>
                      setForm((prev) => ({ ...prev, required_sections: event.target.value }))
                    }
                    placeholder="Comma-separated"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label>Required Keywords</Label>
                <Input
                  value={form.required_keywords}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, required_keywords: event.target.value }))
                  }
                  placeholder="Comma-separated"
                />
              </div>
              <div className="space-y-2">
                <Label>Forbidden Patterns (regex)</Label>
                <Input
                  value={form.forbidden_patterns}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, forbidden_patterns: event.target.value }))
                  }
                  placeholder="Comma-separated"
                />
              </div>
              <Button onClick={handleCreateRule}>Add Rule</Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="apply" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Validation Target</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                value={content}
                onChange={(event) => setContent(event.target.value)}
                className="min-h-[180px]"
                placeholder="Paste the content to validate..."
              />
              <div className="space-y-2">
                <Label>Rules</Label>
                <div className="flex flex-wrap gap-2">
                  {ruleNames.map((name) => (
                    <Button
                      key={name}
                      size="sm"
                      variant={selectedRules.includes(name) ? "default" : "outline"}
                      onClick={() => toggleRule(name)}
                    >
                      {name}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button onClick={handleRunValidation}>Run Validation</Button>
                <Button variant="outline" onClick={handleComplianceCheck}>
                  Run Compliance ({complianceFramework.toUpperCase()})
                </Button>
                <select
                  className="rounded border px-2 py-1 text-sm"
                  value={complianceFramework}
                  onChange={(event) => setComplianceFramework(event.target.value)}
                >
                  <option value="generic">Generic</option>
                  <option value="gdpr">GDPR</option>
                  <option value="hipaa">HIPAA</option>
                </select>
              </div>
              {validationResult ? (
                <div className="rounded border p-3 text-sm space-y-1">
                  <div>
                    Overall: {validationResult.overall_result ? "PASS" : "FAIL"}
                    <Badge className="ml-2" variant={validationResult.overall_result ? "default" : "destructive"}>
                      {validationResult.overall_result ? "OK" : "Issues"}
                    </Badge>
                  </div>
                  <div>Errors: {validationResult.error_count}</div>
                  <div>Warnings: {validationResult.warning_count}</div>
                  <div>Suggestions: {validationResult.suggestion_count}</div>
                </div>
              ) : null}
              {complianceResult ? (
                <div className="rounded border p-3 text-sm space-y-1">
                  <div>Compliance result: {complianceResult.valid ? "PASS" : "FAIL"}</div>
                  {complianceResult.errors?.length ? (
                    <div>Errors: {complianceResult.errors.join("; ")}</div>
                  ) : null}
                </div>
              ) : null}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Latest Results</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {validationResult ? (
                <pre className="text-xs bg-muted p-3 rounded">
                  {JSON.stringify(validationResult, null, 2)}
                </pre>
              ) : (
                <div className="text-sm text-muted-foreground">Run validation to see results.</div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
