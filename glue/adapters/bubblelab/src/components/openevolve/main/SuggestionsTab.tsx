import React, { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { openevolveApi } from "../../../lib/openevolveApi";

export const SuggestionsTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_llm_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const [baseUrl, setBaseUrl] = useState(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_llm_base_url") ?? "https://api.openai.com/v1";
    } catch {
      return "https://api.openai.com/v1";
    }
  });
  const [model, setModel] = useState(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_llm_model") ?? "gpt-4o-mini";
    } catch {
      return "gpt-4o-mini";
    }
  });

  const [content, setContent] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [classification, setClassification] = useState<Record<string, unknown> | null>(null);
  const [vulnerabilities, setVulnerabilities] = useState<string[]>([]);
  const [improvementScore, setImprovementScore] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const requestPayload = useMemo(
    () => ({
      content,
      api_key: apiKey,
      base_url: baseUrl,
      model,
      temperature: 0.7,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0,
      max_tokens: 1024,
    }),
    [content, apiKey, baseUrl, model],
  );

  const runSuggestions = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.getContentSuggestions(requestPayload);
      setSuggestions(response.suggestions ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to generate suggestions.");
    } finally {
      setLoading(false);
    }
  };

  const runClassification = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.getContentClassification(requestPayload);
      setClassification(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to classify content.");
    } finally {
      setLoading(false);
    }
  };

  const runSecurity = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.getSecuritySuggestions(requestPayload);
      setVulnerabilities(response.vulnerabilities ?? []);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to scan for security issues.");
    } finally {
      setLoading(false);
    }
  };

  const runImprovement = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await openevolveApi.getImprovementPotential(requestPayload);
      setImprovementScore(response.score ?? 0);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to calculate improvement potential.");
    } finally {
      setLoading(false);
    }
  };

  const persistSettings = (key: string, value: string) => {
    try {
      globalThis.localStorage?.setItem(key, value);
    } catch {
      // ignore storage errors
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>AI Suggestions</CardTitle>
          <CardDescription>Generate improvement guidance, tags, and security checks.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
                value={apiKey}
                type="password"
                onChange={(event) => {
                  const value = event.target.value;
                  setApiKey(value);
                  persistSettings("openevolve_llm_api_key", value);
                }}
              />
            </div>
            <div className="space-y-2">
              <Label>Base URL</Label>
              <Input
                value={baseUrl}
                onChange={(event) => {
                  const value = event.target.value;
                  setBaseUrl(value);
                  persistSettings("openevolve_llm_base_url", value);
                }}
              />
            </div>
            <div className="space-y-2">
              <Label>Model</Label>
              <Input
                value={model}
                onChange={(event) => {
                  const value = event.target.value;
                  setModel(value);
                  persistSettings("openevolve_llm_model", value);
                }}
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label>Content</Label>
            <Textarea
              value={content}
              onChange={(event) => setContent(event.target.value)}
              className="min-h-[160px]"
              placeholder="Paste content to analyze..."
            />
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

          <Tabs defaultValue="suggestions">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="suggestions">Suggestions</TabsTrigger>
              <TabsTrigger value="classification">Classification</TabsTrigger>
              <TabsTrigger value="security">Security Check</TabsTrigger>
              <TabsTrigger value="improvement">Improvement Potential</TabsTrigger>
            </TabsList>

            <TabsContent value="suggestions" className="mt-4 space-y-3">
              <Button onClick={runSuggestions} disabled={loading || !content}>
                Generate Suggestions
              </Button>
              <div className="space-y-2 text-sm">
                {suggestions.length === 0 && (
                  <div className="text-muted-foreground">No suggestions yet.</div>
                )}
                {suggestions.map((suggestion, index) => (
                  <div key={index} className="rounded border p-2">
                    {index + 1}. {suggestion}
                  </div>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="classification" className="mt-4 space-y-3">
              <Button onClick={runClassification} disabled={loading || !content}>
                Classify Content
              </Button>
              <Textarea
                value={classification ? JSON.stringify(classification, null, 2) : ""}
                readOnly
                className="min-h-[160px]"
              />
            </TabsContent>

            <TabsContent value="security" className="mt-4 space-y-3">
              <Button onClick={runSecurity} disabled={loading || !content}>
                Scan for Issues
              </Button>
              <div className="space-y-2 text-sm">
                {vulnerabilities.length === 0 && (
                  <div className="text-muted-foreground">No vulnerabilities listed.</div>
                )}
                {vulnerabilities.map((issue, index) => (
                  <div key={index} className="rounded border p-2">
                    {issue}
                  </div>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="improvement" className="mt-4 space-y-3">
              <Button onClick={runImprovement} disabled={loading || !content}>
                Calculate Improvement Potential
              </Button>
              <div className="text-sm">
                Score: {improvementScore !== null ? improvementScore.toFixed(2) : "n/a"}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};
