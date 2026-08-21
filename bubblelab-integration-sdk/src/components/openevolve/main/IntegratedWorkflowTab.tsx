import React, { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { openevolveApi } from "@/lib/openevolveApi";
import type { IntegratedWorkflowRequest } from "@/lib/types";

const CONTENT_TYPES = [
  "text_general",
  "code_python",
  "code_javascript",
  "document_legal",
  "document_medical",
  "document_technical",
  "prompt",
  "protocol",
];

const defaultSystemPrompt = "You are a red team reviewer assessing the provided content.";
const defaultEvaluatorPrompt = "Evaluate the content quality on a 0-100 scale.";

const parseList = (value: string) =>
  value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

const downloadHtml = (filename: string, html: string) => {
  const blob = new Blob([html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

const buildReportHtml = (results: Record<string, any>) => {
  const initial = results.initial_content ?? "";
  const finalContent = results.final_content ?? "";
  const adversarial = results.adversarial_results ?? {};
  const evolution = results.evolution_results ?? {};
  const evaluation = results.evaluation_results ?? {};
  const keywordAnalysis = results.keyword_analysis ?? {};
  const integratedScore = results.integrated_score ?? 0.0;
  const totalCost = results.total_cost_usd ?? 0.0;
  const totalTokens = results.total_tokens ?? { prompt: 0, completion: 0 };

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Integrated Workflow Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }
    h1, h2 { color: #1d4ed8; }
    .summary, .section { background: #f8fafc; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; }
    .metric { background: #fff; border-radius: 6px; padding: 8px; }
    pre { background: #fff; padding: 12px; border-radius: 6px; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Integrated Adversarial + Evolution + Evaluation Report</h1>
  <div class="summary">
    <h2>Executive Summary</h2>
    <div class="metrics">
      <div class="metric"><strong>Integrated Score:</strong> ${integratedScore}</div>
      <div class="metric"><strong>Total Cost:</strong> $${totalCost}</div>
      <div class="metric"><strong>Total Tokens:</strong> ${(totalTokens.prompt ?? 0) + (totalTokens.completion ?? 0)}</div>
      <div class="metric"><strong>Content Length:</strong> ${initial.length} -> ${finalContent.length}</div>
    </div>
  </div>
  <div class="section">
    <h2>Adversarial Phase</h2>
    <pre>${JSON.stringify(adversarial, null, 2)}</pre>
  </div>
  <div class="section">
    <h2>Evolution Phase</h2>
    <pre>${JSON.stringify(evolution, null, 2)}</pre>
  </div>
  <div class="section">
    <h2>Evaluation Phase</h2>
    <pre>${JSON.stringify(evaluation, null, 2)}</pre>
  </div>
  <div class="section">
    <h2>Keyword Analysis</h2>
    <pre>${JSON.stringify(keywordAnalysis, null, 2)}</pre>
  </div>
  <div class="section">
    <h2>Final Content</h2>
    <pre>${finalContent}</pre>
  </div>
</body>
</html>`;
};

export const IntegratedWorkflowTab: React.FC = () => {
  const [apiKey, setApiKey] = useState(() => {
    try {
      return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const apiConfig = useMemo(() => ({ apiKey }), [apiKey]);

  const [content, setContent] = useState("");
  const [contentType, setContentType] = useState("text_general");
  const [baseUrl, setBaseUrl] = useState("https://openrouter.ai/api/v1");
  const [redModels, setRedModels] = useState("openai/gpt-4o, anthropic/claude-3-sonnet");
  const [blueModels, setBlueModels] = useState("openai/gpt-4o");
  const [evaluatorModels, setEvaluatorModels] = useState("openai/gpt-4o");
  const [systemPrompt, setSystemPrompt] = useState(defaultSystemPrompt);
  const [evaluatorPrompt, setEvaluatorPrompt] = useState(defaultEvaluatorPrompt);

  const [adversarialIterations, setAdversarialIterations] = useState("3");
  const [evolutionIterations, setEvolutionIterations] = useState("2");
  const [evaluationIterations, setEvaluationIterations] = useState("2");
  const [maxIterations, setMaxIterations] = useState("5");
  const [temperature, setTemperature] = useState("0.7");
  const [topP, setTopP] = useState("0.95");
  const [maxTokens, setMaxTokens] = useState("4096");
  const [confidenceThreshold, setConfidenceThreshold] = useState("0.7");
  const [evaluatorThreshold, setEvaluatorThreshold] = useState("90");
  const [enableAugmentation, setEnableAugmentation] = useState(false);
  const [enableHumanFeedback, setEnableHumanFeedback] = useState(false);

  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const runWorkflow = async () => {
    setErrorMessage(null);
    setStatusMessage(null);
    if (!content.trim()) {
      setErrorMessage("Content is required.");
      return;
    }
    setLoading(true);
    try {
      const payload: IntegratedWorkflowRequest = {
        current_content: content,
        content_type: contentType,
        api_key: apiKey,
        base_url: baseUrl,
        red_team_models: parseList(redModels),
        blue_team_models: parseList(blueModels),
        evaluator_models: parseList(evaluatorModels),
        max_iterations: Number(maxIterations),
        adversarial_iterations: Number(adversarialIterations),
        evolution_iterations: Number(evolutionIterations),
        evaluation_iterations: Number(evaluationIterations),
        system_prompt: systemPrompt,
        evaluator_system_prompt: evaluatorPrompt,
        temperature: Number(temperature),
        top_p: Number(topP),
        max_tokens: Number(maxTokens),
        confidence_threshold: Number(confidenceThreshold),
        evaluator_threshold: Number(evaluatorThreshold),
        enable_data_augmentation: enableAugmentation,
        enable_human_feedback: enableHumanFeedback,
      };
      const response = await openevolveApi.runIntegratedWorkflow(payload, apiConfig);
      setResults(response as Record<string, unknown>);
      setStatusMessage("Integrated workflow completed.");
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Integrated workflow failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Integrated Workflow</CardTitle>
          <CardDescription>Run adversarial testing + evolution + evaluation in one pass.</CardDescription>
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
                  // ignore
                }
              }}
            />
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="space-y-2">
            <Label>Content</Label>
            <Textarea value={content} onChange={(event) => setContent(event.target.value)} rows={6} />
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Content Type</Label>
              <select
                className="w-full rounded border border-input bg-background px-3 py-2 text-sm"
                value={contentType}
                onChange={(event) => setContentType(event.target.value)}
              >
                {CONTENT_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <Label>Base URL</Label>
              <Input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} />
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Red Team Models</Label>
              <Input value={redModels} onChange={(event) => setRedModels(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Blue Team Models</Label>
              <Input value={blueModels} onChange={(event) => setBlueModels(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Evaluator Models</Label>
              <Input
                value={evaluatorModels}
                onChange={(event) => setEvaluatorModels(event.target.value)}
              />
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <Label>System Prompt</Label>
              <Textarea value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} rows={3} />
            </div>
            <div className="space-y-2">
              <Label>Evaluator Prompt</Label>
              <Textarea value={evaluatorPrompt} onChange={(event) => setEvaluatorPrompt(event.target.value)} rows={3} />
            </div>
          </div>

          <Separator />

          <div className="grid gap-3 md:grid-cols-4">
            <div className="space-y-2">
              <Label>Max Iterations</Label>
              <Input value={maxIterations} onChange={(event) => setMaxIterations(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Adversarial Iterations</Label>
              <Input
                value={adversarialIterations}
                onChange={(event) => setAdversarialIterations(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Evolution Iterations</Label>
              <Input
                value={evolutionIterations}
                onChange={(event) => setEvolutionIterations(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Evaluation Iterations</Label>
              <Input
                value={evaluationIterations}
                onChange={(event) => setEvaluationIterations(event.target.value)}
              />
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-4">
            <div className="space-y-2">
              <Label>Temperature</Label>
              <Input value={temperature} onChange={(event) => setTemperature(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Top P</Label>
              <Input value={topP} onChange={(event) => setTopP(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Max Tokens</Label>
              <Input value={maxTokens} onChange={(event) => setMaxTokens(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>Confidence</Label>
              <Input
                value={confidenceThreshold}
                onChange={(event) => setConfidenceThreshold(event.target.value)}
              />
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <Label>Evaluator Threshold</Label>
              <Input
                value={evaluatorThreshold}
                onChange={(event) => setEvaluatorThreshold(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Enable Augmentation</Label>
              <Switch checked={enableAugmentation} onCheckedChange={setEnableAugmentation} />
            </div>
            <div className="space-y-2">
              <Label>Human Feedback</Label>
              <Switch checked={enableHumanFeedback} onCheckedChange={setEnableHumanFeedback} />
            </div>
          </div>

          <Button onClick={runWorkflow} disabled={loading}>
            Run Integrated Workflow
          </Button>
        </CardContent>
      </Card>

      {results ? (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary">Integrated Score: {String(results.integrated_score ?? "n/a")}</Badge>
              <Badge variant="outline">Total Cost: {String(results.total_cost_usd ?? "n/a")}</Badge>
            </div>
            <pre className="rounded border p-3 text-xs whitespace-pre-wrap">
              {JSON.stringify(results, null, 2)}
            </pre>
            <Button
              variant="outline"
              onClick={() => downloadHtml("integrated_report.html", buildReportHtml(results))}
            >
              Download HTML Report
            </Button>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
};
