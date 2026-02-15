import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { openevolveApi } from "../../../lib/openevolveApi";
import type { ContentTemplate, ProtocolValidationResult } from "../../../lib/types";

const VALIDATION_TYPES = ["generic", "compliance", "security", "technical"];

export const ContentManagerTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [templates, setTemplates] = useState<string[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>("");
  const [templateContent, setTemplateContent] = useState<string>("");
  const [newTemplateName, setNewTemplateName] = useState<string>("");
  const [newTemplateContent, setNewTemplateContent] = useState<string>("");
  const [validationText, setValidationText] = useState<string>("");
  const [validationType, setValidationType] = useState<string>(VALIDATION_TYPES[0]);
  const [validationResult, setValidationResult] = useState<ProtocolValidationResult | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const loadTemplates = async () => {
    setErrorMessage(null);
    try {
      const result = await openevolveApi.listContentTemplates(apiConfig);
      setTemplates(result.templates || []);
      if (!selectedTemplate && result.templates?.length) {
        setSelectedTemplate(result.templates[0]);
      }
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load templates.");
    }
  };

  const loadTemplate = async (name: string) => {
    setErrorMessage(null);
    try {
      const result: ContentTemplate = await openevolveApi.getContentTemplate(name, apiConfig);
      setTemplateContent(result.content || "");
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load template content.");
    }
  };

  useEffect(() => {
    loadTemplates();
  }, [apiConfig.apiKey]);

  useEffect(() => {
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
      await openevolveApi.createContentTemplate(
        { name: newTemplateName.trim(), content: newTemplateContent },
        apiConfig,
      );
      setStatusMessage(`Saved template '${newTemplateName}'.`);
      setNewTemplateName("");
      setNewTemplateContent("");
      await loadTemplates();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to save template.");
    }
  };

  const validateProtocol = async () => {
    setErrorMessage(null);
    try {
      const result = await openevolveApi.validateProtocol(
        { protocol_text: validationText, validation_type: validationType },
        apiConfig,
      );
      setValidationResult(result);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to validate protocol.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Content Manager</CardTitle>
          <CardDescription>Protocol templates and validation tools.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>API Key</Label>
            <Input
              type="password"
              value={apiKey}
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
          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Templates</Label>
              <Select value={selectedTemplate} onValueChange={setSelectedTemplate}>
                <SelectTrigger>
                  <SelectValue placeholder="Select template" />
                </SelectTrigger>
                <SelectContent>
                  {templates.map((template) => (
                    <SelectItem key={template} value={template}>
                      {template}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Selected Template Content</Label>
              <Textarea value={templateContent} readOnly rows={6} />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Create Template</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-2">
            <Label>Name</Label>
            <Input value={newTemplateName} onChange={(event) => setNewTemplateName(event.target.value)} />
          </div>
          <div className="space-y-2">
            <Label>Content</Label>
            <Textarea
              value={newTemplateContent}
              onChange={(event) => setNewTemplateContent(event.target.value)}
              rows={8}
            />
          </div>
          <Button onClick={createTemplate}>Save Template</Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Protocol Validation</CardTitle>
          <CardDescription>Validate protocol text against configured rules.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-2">
            <Label>Validation Type</Label>
            <Select value={validationType} onValueChange={setValidationType}>
              <SelectTrigger>
                <SelectValue placeholder="Select validation type" />
              </SelectTrigger>
              <SelectContent>
                {VALIDATION_TYPES.map((type) => (
                  <SelectItem key={type} value={type}>
                    {type}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Protocol Text</Label>
            <Textarea value={validationText} onChange={(event) => setValidationText(event.target.value)} rows={8} />
          </div>
          <Button variant="outline" onClick={validateProtocol}>
            Validate
          </Button>
          {validationResult && (
            <div className="rounded border p-3 text-sm space-y-1">
              <div>Valid: {validationResult.valid ? "Yes" : "No"}</div>
              <div>Score: {validationResult.score}</div>
              <div>Errors: {validationResult.errors?.join(", ") || "None"}</div>
              <div>Warnings: {validationResult.warnings?.join(", ") || "None"}</div>
              <div>Suggestions: {validationResult.suggestions?.join(", ") || "None"}</div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
