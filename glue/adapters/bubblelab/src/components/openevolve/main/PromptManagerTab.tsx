import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { openevolveApi } from "../../../lib/openevolveApi";
import type { PromptMap } from "../../../lib/types";

export const PromptManagerTab: React.FC = () => {
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [prompts, setPrompts] = useState<PromptMap>({});
  const [promptName, setPromptName] = useState("");
  const [promptContent, setPromptContent] = useState("");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const loadPrompts = async () => {
    setErrorMessage(null);
    try {
      const result = await openevolveApi.listPrompts(apiConfig);
      setPrompts(result.prompts || {});
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to load prompts.");
    }
  };

  useEffect(() => {
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
      await openevolveApi.savePrompt({ name: promptName.trim(), content: promptContent }, apiConfig);
      setStatusMessage(`Saved prompt '${promptName}'.`);
      await loadPrompts();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to save prompt.");
    }
  };

  const deletePrompt = async (name: string) => {
    if (!confirm(`Delete prompt '${name}'?`)) {
      return;
    }
    try {
      await openevolveApi.deletePrompt(name, apiConfig);
      await loadPrompts();
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to delete prompt.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Prompt Manager</CardTitle>
          <CardDescription>Store and reuse custom prompts for evolution runs.</CardDescription>
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
              <Label>Prompt Name</Label>
              <Input value={promptName} onChange={(event) => setPromptName(event.target.value)} />
            </div>
            <div className="flex items-end">
              <Button onClick={savePrompt}>Save Prompt</Button>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Prompt Content</Label>
            <Textarea
              value={promptContent}
              onChange={(event) => setPromptContent(event.target.value)}
              rows={8}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Saved Prompts</CardTitle>
          <CardDescription>Manage stored prompt templates.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          {Object.keys(prompts).length === 0 ? (
            <div className="text-muted-foreground">No custom prompts saved.</div>
          ) : (
            Object.entries(prompts).map(([name, content]) => (
              <div key={name} className="rounded border p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="font-semibold">{name}</div>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">{content.length} chars</Badge>
                    <Button variant="outline" size="sm" onClick={() => deletePrompt(name)}>
                      Delete
                    </Button>
                  </div>
                </div>
                <div className="text-xs text-muted-foreground whitespace-pre-wrap">
                  {content}
                </div>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
};
