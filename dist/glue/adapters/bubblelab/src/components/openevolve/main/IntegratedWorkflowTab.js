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
exports.IntegratedWorkflowTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const switch_1 = require("@/components/ui/switch");
const separator_1 = require("@/components/ui/separator");
const openevolveApi_1 = require("../../../lib/openevolveApi");
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
const parseList = (value) => value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
const downloadHtml = (filename, html) => {
    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
};
const buildReportHtml = (results) => {
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
const IntegratedWorkflowTab = () => {
    const [apiKey, setApiKey] = (0, react_1.useState)(() => {
        try {
            return globalThis.localStorage?.getItem("openevolve_api_key") ?? "";
        }
        catch {
            return "";
        }
    });
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey }), [apiKey]);
    const [content, setContent] = (0, react_1.useState)("");
    const [contentType, setContentType] = (0, react_1.useState)("text_general");
    const [baseUrl, setBaseUrl] = (0, react_1.useState)("https://openrouter.ai/api/v1");
    const [redModels, setRedModels] = (0, react_1.useState)("openai/gpt-4o, anthropic/claude-3-sonnet");
    const [blueModels, setBlueModels] = (0, react_1.useState)("openai/gpt-4o");
    const [evaluatorModels, setEvaluatorModels] = (0, react_1.useState)("openai/gpt-4o");
    const [systemPrompt, setSystemPrompt] = (0, react_1.useState)(defaultSystemPrompt);
    const [evaluatorPrompt, setEvaluatorPrompt] = (0, react_1.useState)(defaultEvaluatorPrompt);
    const [adversarialIterations, setAdversarialIterations] = (0, react_1.useState)("3");
    const [evolutionIterations, setEvolutionIterations] = (0, react_1.useState)("2");
    const [evaluationIterations, setEvaluationIterations] = (0, react_1.useState)("2");
    const [maxIterations, setMaxIterations] = (0, react_1.useState)("5");
    const [temperature, setTemperature] = (0, react_1.useState)("0.7");
    const [topP, setTopP] = (0, react_1.useState)("0.95");
    const [maxTokens, setMaxTokens] = (0, react_1.useState)("4096");
    const [confidenceThreshold, setConfidenceThreshold] = (0, react_1.useState)("0.7");
    const [evaluatorThreshold, setEvaluatorThreshold] = (0, react_1.useState)("90");
    const [enableAugmentation, setEnableAugmentation] = (0, react_1.useState)(false);
    const [enableHumanFeedback, setEnableHumanFeedback] = (0, react_1.useState)(false);
    const [results, setResults] = (0, react_1.useState)(null);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const [statusMessage, setStatusMessage] = (0, react_1.useState)(null);
    const [loading, setLoading] = (0, react_1.useState)(false);
    const runWorkflow = async () => {
        setErrorMessage(null);
        setStatusMessage(null);
        if (!content.trim()) {
            setErrorMessage("Content is required.");
            return;
        }
        setLoading(true);
        try {
            const payload = {
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
            const response = await openevolveApi_1.openevolveApi.runIntegratedWorkflow(payload, apiConfig);
            setResults(response);
            setStatusMessage("Integrated workflow completed.");
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Integrated workflow failed.");
        }
        finally {
            setLoading(false);
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Integrated Workflow</card_1.CardTitle>
          <card_1.CardDescription>Run adversarial testing + evolution + evaluation in one pass.</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent className="space-y-4">
          <div className="space-y-2">
            <label_1.Label>API Key</label_1.Label>
            <input_1.Input value={apiKey} type="password" onChange={(event) => {
            const value = event.target.value;
            setApiKey(value);
            try {
                globalThis.localStorage?.setItem("openevolve_api_key", value);
            }
            catch {
                // ignore
            }
        }}/>
          </div>

          {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}
          {statusMessage ? <div className="text-sm text-green-600">{statusMessage}</div> : null}

          <div className="space-y-2">
            <label_1.Label>Content</label_1.Label>
            <textarea_1.Textarea value={content} onChange={(event) => setContent(event.target.value)} rows={6}/>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>Content Type</label_1.Label>
              <select className="w-full rounded border border-input bg-background px-3 py-2 text-sm" value={contentType} onChange={(event) => setContentType(event.target.value)}>
                {CONTENT_TYPES.map((type) => (<option key={type} value={type}>
                    {type}
                  </option>))}
              </select>
            </div>
            <div className="space-y-2">
              <label_1.Label>Base URL</label_1.Label>
              <input_1.Input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)}/>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <label_1.Label>Red Team Models</label_1.Label>
              <input_1.Input value={redModels} onChange={(event) => setRedModels(event.target.value)}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Blue Team Models</label_1.Label>
              <input_1.Input value={blueModels} onChange={(event) => setBlueModels(event.target.value)}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Evaluator Models</label_1.Label>
              <input_1.Input value={evaluatorModels} onChange={(event) => setEvaluatorModels(event.target.value)}/>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-2">
              <label_1.Label>System Prompt</label_1.Label>
              <textarea_1.Textarea value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} rows={3}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Evaluator Prompt</label_1.Label>
              <textarea_1.Textarea value={evaluatorPrompt} onChange={(event) => setEvaluatorPrompt(event.target.value)} rows={3}/>
            </div>
          </div>

          <separator_1.Separator />

          <div className="grid gap-3 md:grid-cols-4">
            <div className="space-y-2">
              <label_1.Label>Max Iterations</label_1.Label>
              <input_1.Input value={maxIterations} onChange={(event) => setMaxIterations(event.target.value)}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Adversarial Iterations</label_1.Label>
              <input_1.Input value={adversarialIterations} onChange={(event) => setAdversarialIterations(event.target.value)}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Evolution Iterations</label_1.Label>
              <input_1.Input value={evolutionIterations} onChange={(event) => setEvolutionIterations(event.target.value)}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Evaluation Iterations</label_1.Label>
              <input_1.Input value={evaluationIterations} onChange={(event) => setEvaluationIterations(event.target.value)}/>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-4">
            <div className="space-y-2">
              <label_1.Label>Temperature</label_1.Label>
              <input_1.Input value={temperature} onChange={(event) => setTemperature(event.target.value)}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Top P</label_1.Label>
              <input_1.Input value={topP} onChange={(event) => setTopP(event.target.value)}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Max Tokens</label_1.Label>
              <input_1.Input value={maxTokens} onChange={(event) => setMaxTokens(event.target.value)}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Confidence</label_1.Label>
              <input_1.Input value={confidenceThreshold} onChange={(event) => setConfidenceThreshold(event.target.value)}/>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <div className="space-y-2">
              <label_1.Label>Evaluator Threshold</label_1.Label>
              <input_1.Input value={evaluatorThreshold} onChange={(event) => setEvaluatorThreshold(event.target.value)}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Enable Augmentation</label_1.Label>
              <switch_1.Switch checked={enableAugmentation} onCheckedChange={setEnableAugmentation}/>
            </div>
            <div className="space-y-2">
              <label_1.Label>Human Feedback</label_1.Label>
              <switch_1.Switch checked={enableHumanFeedback} onCheckedChange={setEnableHumanFeedback}/>
            </div>
          </div>

          <button_1.Button onClick={runWorkflow} disabled={loading}>
            Run Integrated Workflow
          </button_1.Button>
        </card_1.CardContent>
      </card_1.Card>

      {results ? (<card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="text-base">Results</card_1.CardTitle>
          </card_1.CardHeader>
          <card_1.CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <badge_1.Badge variant="secondary">Integrated Score: {String(results.integrated_score ?? "n/a")}</badge_1.Badge>
              <badge_1.Badge variant="outline">Total Cost: {String(results.total_cost_usd ?? "n/a")}</badge_1.Badge>
            </div>
            <pre className="rounded border p-3 text-xs whitespace-pre-wrap">
              {JSON.stringify(results, null, 2)}
            </pre>
            <button_1.Button variant="outline" onClick={() => downloadHtml("integrated_report.html", buildReportHtml(results))}>
              Download HTML Report
            </button_1.Button>
          </card_1.CardContent>
        </card_1.Card>) : null}
    </div>);
};
exports.IntegratedWorkflowTab = IntegratedWorkflowTab;
//# sourceMappingURL=IntegratedWorkflowTab.js.map