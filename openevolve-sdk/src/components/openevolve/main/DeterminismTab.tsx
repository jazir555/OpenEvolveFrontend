import React, { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Cpu } from "lucide-react";
import { openevolveApi } from "@/lib/openevolveApi";
import type {
  DeterminismGenerateRequest,
  DeterminismGenerateResponse,
  DeterminismCheckRequest,
  DeterminismCheckResponse,
} from "@/lib/types";

const readApiKey = () => {
  try {
    return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
  } catch {
    return "";
  }
};

export const DeterminismTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(readApiKey);
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  // Generate
  const [genPrompt, setGenPrompt] = useState("");
  const [genMode, setGenMode] = useState("auto");
  const [genProvider, setGenProvider] = useState("");
  const [genModel, setGenModel] = useState("");
  const [genOutput, setGenOutput] = useState<DeterminismGenerateResponse | null>(null);

  // Check
  const [checkPrompt, setCheckPrompt] = useState("");
  const [checkProvider, setCheckProvider] = useState("");
  const [checkModel, setCheckModel] = useState("");
  const [checkRuns, setCheckRuns] = useState("3");
  const [checkTier, setCheckTier] = useState("2");
  const [checkOutput, setCheckOutput] = useState<DeterminismCheckResponse | null>(null);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const runGenerate = async () => {
    setLoading(true);
    setErrorMessage(null);
    setStatusMessage(null);
    if (!genPrompt.trim()) {
      setErrorMessage("Prompt is required.");
      setLoading(false);
      return;
    }
    try {
      const payload: DeterminismGenerateRequest = {
        prompt: genPrompt,
        mode: genMode,
        cloud_provider: genProvider || undefined,
        cloud_model: genModel || undefined,
      };
      const response = await openevolveApi.generateDeterminism(payload, apiConfig);
      setGenOutput(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Determinism generate failed.");
    } finally {
      setLoading(false);
    }
  };

  const runCheck = async () => {
    setLoading(true);
    setErrorMessage(null);
    setStatusMessage(null);
    if (!checkPrompt.trim()) {
      setErrorMessage("Prompt is required.");
      setLoading(false);
      return;
    }
    try {
      const payload: DeterminismCheckRequest = {
        prompt: checkPrompt,
        tier: parseInt(checkTier, 10) || 2,
        runs: parseInt(checkRuns, 10) || 3,
        provider: checkProvider || undefined,
        model: checkModel || undefined,
      };
      const response = await openevolveApi.checkDeterminism(payload, apiConfig);
      setCheckOutput(response);
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Determinism check failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="h-5 w-5" />
            Deterministic LLM
          </CardTitle>
          <CardDescription>
            Generate deterministic outputs and verify reproducibility across runs.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
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

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <Tabs defaultValue="generate" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="generate">Generate</TabsTrigger>
              <TabsTrigger value="check">Check</TabsTrigger>
            </TabsList>

            <TabsContent value="generate" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Generate Deterministic Output</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label>Prompt</Label>
                    <Textarea
                      value={genPrompt}
                      onChange={(event) => setGenPrompt(event.target.value)}
                      className="min-h-[120px]"
                    />
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <div className="space-y-2">
                      <Label>Mode</Label>
                      <Input
                        value={genMode}
                        onChange={(event) => setGenMode(event.target.value)}
                        placeholder="auto"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Provider</Label>
                      <Input
                        value={genProvider}
                        onChange={(event) => setGenProvider(event.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Model</Label>
                      <Input
                        value={genModel}
                        onChange={(event) => setGenModel(event.target.value)}
                      />
                    </div>
                  </div>
                  <Button onClick={runGenerate} disabled={loading || !genPrompt.trim()}>
                    Generate
                  </Button>
                </CardContent>
              </Card>

              {genOutput && (
                <div className="rounded border border-[#30363d] bg-[#0d1117] p-3 text-sm text-gray-300">
                  <pre className="whitespace-pre-wrap break-words">
                    {JSON.stringify(genOutput, null, 2)}
                  </pre>
                </div>
              )}
            </TabsContent>

            <TabsContent value="check" className="mt-4 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Check Reproducibility</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    <Label>Prompt</Label>
                    <Textarea
                      value={checkPrompt}
                      onChange={(event) => setCheckPrompt(event.target.value)}
                      className="min-h-[120px]"
                    />
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label>Provider</Label>
                      <Input
                        value={checkProvider}
                        onChange={(event) => setCheckProvider(event.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Model</Label>
                      <Input
                        value={checkModel}
                        onChange={(event) => setCheckModel(event.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Runs</Label>
                      <Input
                        value={checkRuns}
                        onChange={(event) => setCheckRuns(event.target.value)}
                        inputMode="numeric"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Tier</Label>
                      <Input
                        value={checkTier}
                        onChange={(event) => setCheckTier(event.target.value)}
                        inputMode="numeric"
                      />
                    </div>
                  </div>
                  <Button onClick={runCheck} disabled={loading || !checkPrompt.trim()}>
                    Check
                  </Button>
                </CardContent>
              </Card>

              {checkOutput && (
                <div className="rounded border border-[#30363d] bg-[#0d1117] p-3 text-sm text-gray-300">
                  <pre className="whitespace-pre-wrap break-words">
                    {JSON.stringify(checkOutput, null, 2)}
                  </pre>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};
