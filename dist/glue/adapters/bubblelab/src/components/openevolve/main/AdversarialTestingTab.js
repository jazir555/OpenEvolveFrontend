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
exports.AdversarialTestingTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const textarea_1 = require("@/components/ui/textarea");
const button_1 = require("@/components/ui/button");
const label_1 = require("@/components/ui/label");
const slider_1 = require("@/components/ui/slider");
const select_1 = require("@/components/ui/select");
const tabs_1 = require("@/components/ui/tabs");
const input_1 = require("@/components/ui/input");
const checkbox_1 = require("@/components/ui/checkbox");
const badge_1 = require("@/components/ui/badge");
const lucide_react_1 = require("lucide-react");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const readStorage = (key, fallback = "") => {
    try {
        return globalThis.localStorage?.getItem(key) ?? fallback;
    }
    catch {
        return fallback;
    }
};
const extractFinalContent = (result, fallback) => {
    if (!result)
        return fallback;
    const candidates = [
        result.final_content,
        result.best_content,
        result.best_code,
        result.best_program,
        result.output,
    ];
    return candidates.find((value) => typeof value === "string" && value.trim()) ?? fallback;
};
const AdversarialTestingTab = ({ state, updateState }) => {
    const [protocolText, setProtocolText] = (0, react_1.useState)(state.protocolText);
    const [apiKey, setApiKey] = (0, react_1.useState)(readStorage("openevolve_api_key"));
    const [apiBase, setApiBase] = (0, react_1.useState)(readStorage("openevolve_api_base"));
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey, baseUrl: apiBase || undefined }), [apiKey, apiBase]);
    const [contentType, setContentType] = (0, react_1.useState)("document_general");
    const [contentAnalysis, setContentAnalysis] = (0, react_1.useState)({
        length: 0,
        wordCount: 0,
        avgWordLength: 0,
    });
    const [redTeamModels, setRedTeamModels] = (0, react_1.useState)(["claude-3-sonnet"]);
    const [blueTeamModels, setBlueTeamModels] = (0, react_1.useState)(["gpt-4o"]);
    const [evaluatorModels, setEvaluatorModels] = (0, react_1.useState)(["gpt-4o", "claude-3-sonnet"]);
    const [rotationStrategy, setRotationStrategy] = (0, react_1.useState)("Round Robin");
    const [enablePerformanceTracking, setEnablePerformanceTracking] = (0, react_1.useState)(true);
    const [adversarialParams, setAdversarialParams] = (0, react_1.useState)({
        minIter: 1,
        maxIter: 5,
        confidence: 80,
        budgetLimit: 10.0,
        redTeamSampleSize: 2,
        blueTeamSampleSize: 2,
        evaluatorSampleSize: 2,
        evaluatorThreshold: 90.0,
        evaluatorConsecutiveRounds: 1,
        critiqueDepth: 5,
        patchQuality: 5,
    });
    const [advancedFeatures, setAdvancedFeatures] = (0, react_1.useState)({
        enableMultiObjective: false,
        featureDimensions: ["complexity", "diversity"],
        featureBins: 10,
        enableDataAugmentation: false,
        augmentationModel: "gpt-4o",
        augmentationTemperature: 0.7,
        eliteRatio: 0.1,
        explorationRatio: 0.2,
        archiveSize: 100,
    });
    const [qualityControl, setQualityControl] = (0, react_1.useState)({
        enableHumanFeedback: false,
        keywordAnalysisEnabled: true,
        keywordsToTarget: "",
        enableRealTimeMonitoring: true,
        enableComprehensiveReporting: true,
        enableEncryption: true,
        enableAuditTrail: true,
        complianceRequirements: "",
    });
    const [executionMode, setExecutionMode] = (0, react_1.useState)("Integrated Adversarial-Evolution");
    const [runId, setRunId] = (0, react_1.useState)(null);
    const [runStatus, setRunStatus] = (0, react_1.useState)(null);
    const [runLogs, setRunLogs] = (0, react_1.useState)([]);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const modelOptions = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "llama-3-70b",
        "llama-3-8b",
        "mistral-large",
        "mistral-medium",
        "mixtral-8x22b",
        "command-r-plus",
        "command-r",
        "pplx-7b-online",
        "pplx-70b-online",
        "openchat/openchat-3.5-0106",
        "microsoft/WizardLM-2-8x22B",
        "microsoft/WizardLM-2-7B",
    ];
    const updateContentAnalysis = (text) => {
        const length = text.length;
        const wordCount = text.split(/\s+/).filter((word) => word.length > 0).length;
        const avgWordLength = wordCount > 0 ? length / wordCount : 0;
        setContentAnalysis({
            length,
            wordCount,
            avgWordLength,
        });
    };
    const pollRun = async (activeRunId) => {
        const response = await openevolveApi_1.openevolveApi.getAdversarialRun(activeRunId, apiConfig);
        setRunStatus(response);
        setRunLogs(response.logs ?? []);
        updateState({ adversarialRunning: response.status === "running" || response.status === "queued" });
        if (response.result) {
            const finalContent = extractFinalContent(response.result, protocolText);
            updateState({
                adversarialResults: response.result,
                evolutionCurrentBest: finalContent,
            });
        }
    };
    (0, react_1.useEffect)(() => {
        if (!runId)
            return;
        let cancelled = false;
        const tick = async () => {
            try {
                await pollRun(runId);
            }
            catch (error) {
                if (!cancelled) {
                    setErrorMessage(error?.message ?? "Failed to fetch adversarial status.");
                }
            }
        };
        tick();
        const interval = setInterval(tick, 3000);
        return () => {
            cancelled = true;
            clearInterval(interval);
        };
    }, [runId, apiConfig.apiKey, apiConfig.baseUrl]);
    const handleRunAdversarial = async () => {
        setErrorMessage(null);
        if (!protocolText.trim()) {
            setErrorMessage("Provide content for adversarial testing.");
            return;
        }
        const evaluatorConfigs = evaluatorModels.map((model) => ({
            name: model,
            weight: 1.0,
            temperature: 0.3,
            max_tokens: 2048,
            role: "evaluator",
        }));
        const parameters = {
            api_key: apiKey || undefined,
            api_base: apiBase || undefined,
            adversarial_rounds: adversarialParams.maxIter,
            attack_strength: adversarialParams.critiqueDepth / 10,
            defense_strength: adversarialParams.patchQuality / 10,
            adversarial_budget: Math.round(adversarialParams.budgetLimit * 100),
            red_team_models: redTeamModels,
            blue_team_models: blueTeamModels,
            red_team_sample_size: adversarialParams.redTeamSampleSize,
            blue_team_sample_size: adversarialParams.blueTeamSampleSize,
            evaluator_models: evaluatorConfigs,
            cascade_evaluation: enablePerformanceTracking,
            use_llm_feedback: qualityControl.enableHumanFeedback,
            custom_constraints: qualityControl.keywordsToTarget
                ? qualityControl.keywordsToTarget.split(",").map((item) => item.trim())
                : [],
            regulatory_compliance: qualityControl.complianceRequirements
                ? qualityControl.complianceRequirements.split(",").map((item) => item.trim())
                : [],
            feature_dimensions: advancedFeatures.featureDimensions,
            feature_bins: advancedFeatures.featureBins,
            elite_ratio: advancedFeatures.eliteRatio,
            exploration_ratio: advancedFeatures.explorationRatio,
            archive_size: advancedFeatures.archiveSize,
        };
        try {
            const response = await openevolveApi_1.openevolveApi.startAdversarialRun({
                content: protocolText,
                content_type: contentType,
                parameters,
            }, apiConfig);
            setRunId(response.run_id);
            updateState({ adversarialRunning: true, adversarialStatusMessage: "Adversarial run started." });
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to start adversarial testing.");
        }
    };
    const handleStopAdversarial = async () => {
        if (!runId)
            return;
        try {
            await openevolveApi_1.openevolveApi.stopAdversarialRun(runId, apiConfig);
            updateState({ adversarialStatusMessage: "Cancellation requested." });
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to stop adversarial testing.");
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle className="flex items-center gap-2">
            <lucide_react_1.Sword className="h-5 w-5"/>
            Ultimate Adversarial Testing & Evolution
          </card_1.CardTitle>
          <card_1.CardDescription>Advanced AI-Powered Content Hardening with Multi-Model Consensus</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent>
          <div className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <label_1.Label>API Key</label_1.Label>
                <input_1.Input type="password" value={apiKey} onChange={(event) => {
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
              <div className="space-y-2">
                <label_1.Label>API Base URL</label_1.Label>
                <input_1.Input value={apiBase} onChange={(event) => {
            const value = event.target.value;
            setApiBase(value);
            try {
                globalThis.localStorage?.setItem("openevolve_api_base", value);
            }
            catch {
                // ignore storage errors
            }
        }}/>
              </div>
            </div>

            {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <label_1.Label htmlFor="content">Content to Test & Evolve</label_1.Label>
                <textarea_1.Textarea id="content" value={protocolText} onChange={(event) => {
            const newText = event.target.value;
            setProtocolText(newText);
            updateState({ protocolText: newText });
            updateContentAnalysis(newText);
        }} placeholder="Enter the content you want to harden through adversarial testing and evolution" className="min-h-[200px]"/>
              </div>

              <div className="space-y-4">
                <div>
                  <label_1.Label>Content Analysis</label_1.Label>
                  <div className="space-y-2 p-4 bg-muted rounded-lg">
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Content Length</span>
                      <span className="font-medium">{contentAnalysis.length.toLocaleString()} chars</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Word Count</span>
                      <span className="font-medium">{contentAnalysis.wordCount.toLocaleString()} words</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Avg Word Length</span>
                      <span className="font-medium">{contentAnalysis.avgWordLength.toFixed(1)} chars</span>
                    </div>
                  </div>
                </div>
                <div>
                  <label_1.Label>Content Type</label_1.Label>
                  <select_1.Select value={contentType} onValueChange={setContentType}>
                    <select_1.SelectTrigger>
                      <select_1.SelectValue />
                    </select_1.SelectTrigger>
                    <select_1.SelectContent>
                      {["document_general", "document_technical", "document_legal", "document_medical", "code_python", "code_javascript"].map((type) => (<select_1.SelectItem key={type} value={type}>
                          {type}
                        </select_1.SelectItem>))}
                    </select_1.SelectContent>
                  </select_1.Select>
                </div>
                <div>
                  <label_1.Label htmlFor="compliance">Compliance Requirements</label_1.Label>
                  <textarea_1.Textarea id="compliance" value={qualityControl.complianceRequirements} onChange={(event) => setQualityControl((prev) => ({
            ...prev,
            complianceRequirements: event.target.value,
        }))} placeholder="e.g., GDPR, HIPAA, SOC 2, ISO 27001 requirements..." className="min-h-[100px]"/>
                </div>
              </div>
            </div>

            <tabs_1.Tabs defaultValue="model-config" className="w-full">
              <tabs_1.TabsList className="grid w-full grid-cols-4">
                <tabs_1.TabsTrigger value="model-config">Model Config</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="process-params">Process Params</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="advanced">Advanced</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="quality-control">Quality Control</tabs_1.TabsTrigger>
              </tabs_1.TabsList>

              <tabs_1.TabsContent value="model-config" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label_1.Label className="flex items-center gap-1 mb-2">Red Team (Critics)</label_1.Label>
                    <select_1.Select value={redTeamModels[0]} onValueChange={(value) => setRedTeamModels([value])}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue />
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {modelOptions.map((model) => (<select_1.SelectItem key={model} value={model}>
                            {model}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>

                  <div>
                    <label_1.Label className="flex items-center gap-1 mb-2">Blue Team (Fixers)</label_1.Label>
                    <select_1.Select value={blueTeamModels[0]} onValueChange={(value) => setBlueTeamModels([value])}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue />
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {modelOptions.map((model) => (<select_1.SelectItem key={model} value={model}>
                            {model}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>

                  <div>
                    <label_1.Label className="flex items-center gap-1 mb-2">Evaluator Models</label_1.Label>
                    <select_1.Select value={evaluatorModels[0]} onValueChange={(value) => setEvaluatorModels([value])}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue />
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {modelOptions.map((model) => (<select_1.SelectItem key={model} value={model}>
                            {model}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label_1.Label>Rotation Strategy</label_1.Label>
                    <select_1.Select value={rotationStrategy} onValueChange={setRotationStrategy}>
                      <select_1.SelectTrigger>
                        <select_1.SelectValue />
                      </select_1.SelectTrigger>
                      <select_1.SelectContent>
                        {["Round Robin", "Performance", "Random"].map((strategy) => (<select_1.SelectItem key={strategy} value={strategy}>
                            {strategy}
                          </select_1.SelectItem>))}
                      </select_1.SelectContent>
                    </select_1.Select>
                  </div>
                  <div className="flex items-center space-x-2 mt-6">
                    <checkbox_1.Checkbox id="enablePerformanceTracking" checked={enablePerformanceTracking} onCheckedChange={(checked) => setEnablePerformanceTracking(Boolean(checked))}/>
                    <label_1.Label htmlFor="enablePerformanceTracking">Enable Performance Tracking</label_1.Label>
                  </div>
                </div>
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="process-params" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label_1.Label>Max Iterations</label_1.Label>
                    <input_1.Input type="number" value={adversarialParams.maxIter} onChange={(event) => setAdversarialParams((prev) => ({
            ...prev,
            maxIter: Number(event.target.value) || 0,
        }))}/>
                  </div>
                  <div>
                    <label_1.Label>Confidence Threshold</label_1.Label>
                    <input_1.Input type="number" value={adversarialParams.confidence} onChange={(event) => setAdversarialParams((prev) => ({
            ...prev,
            confidence: Number(event.target.value) || 0,
        }))}/>
                  </div>
                  <div>
                    <label_1.Label>Budget Limit (USD)</label_1.Label>
                    <input_1.Input type="number" value={adversarialParams.budgetLimit} onChange={(event) => setAdversarialParams((prev) => ({
            ...prev,
            budgetLimit: Number(event.target.value) || 0,
        }))}/>
                  </div>
                  <div>
                    <label_1.Label>Red Team Sample Size</label_1.Label>
                    <input_1.Input type="number" value={adversarialParams.redTeamSampleSize} onChange={(event) => setAdversarialParams((prev) => ({
            ...prev,
            redTeamSampleSize: Number(event.target.value) || 0,
        }))}/>
                  </div>
                  <div>
                    <label_1.Label>Blue Team Sample Size</label_1.Label>
                    <input_1.Input type="number" value={adversarialParams.blueTeamSampleSize} onChange={(event) => setAdversarialParams((prev) => ({
            ...prev,
            blueTeamSampleSize: Number(event.target.value) || 0,
        }))}/>
                  </div>
                  <div>
                    <label_1.Label>Critique Depth</label_1.Label>
                    <slider_1.Slider value={[adversarialParams.critiqueDepth]} onValueChange={(value) => setAdversarialParams((prev) => ({
            ...prev,
            critiqueDepth: value[0],
        }))} max={10} min={1}/>
                  </div>
                  <div>
                    <label_1.Label>Patch Quality</label_1.Label>
                    <slider_1.Slider value={[adversarialParams.patchQuality]} onValueChange={(value) => setAdversarialParams((prev) => ({
            ...prev,
            patchQuality: value[0],
        }))} max={10} min={1}/>
                  </div>
                </div>
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="advanced" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="flex items-center space-x-2">
                    <checkbox_1.Checkbox id="enableMultiObjective" checked={advancedFeatures.enableMultiObjective} onCheckedChange={(checked) => setAdvancedFeatures((prev) => ({
            ...prev,
            enableMultiObjective: Boolean(checked),
        }))}/>
                    <label_1.Label htmlFor="enableMultiObjective">Multi-Objective Mode</label_1.Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <checkbox_1.Checkbox id="enableDataAugmentation" checked={advancedFeatures.enableDataAugmentation} onCheckedChange={(checked) => setAdvancedFeatures((prev) => ({
            ...prev,
            enableDataAugmentation: Boolean(checked),
        }))}/>
                    <label_1.Label htmlFor="enableDataAugmentation">Data Augmentation</label_1.Label>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label_1.Label>Feature Bins</label_1.Label>
                    <input_1.Input type="number" value={advancedFeatures.featureBins} onChange={(event) => setAdvancedFeatures((prev) => ({
            ...prev,
            featureBins: Number(event.target.value) || 0,
        }))}/>
                  </div>
                  <div>
                    <label_1.Label>Archive Size</label_1.Label>
                    <input_1.Input type="number" value={advancedFeatures.archiveSize} onChange={(event) => setAdvancedFeatures((prev) => ({
            ...prev,
            archiveSize: Number(event.target.value) || 0,
        }))}/>
                  </div>
                </div>

                <div>
                  <label_1.Label>Feature Dimensions</label_1.Label>
                  <div className="flex flex-wrap gap-2">
                    {advancedFeatures.featureDimensions.map((dim) => (<badge_1.Badge key={dim} variant="secondary">
                        {dim}
                      </badge_1.Badge>))}
                  </div>
                </div>
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="quality-control" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center space-x-2">
                    <checkbox_1.Checkbox id="humanFeedback" checked={qualityControl.enableHumanFeedback} onCheckedChange={(checked) => setQualityControl((prev) => ({
            ...prev,
            enableHumanFeedback: Boolean(checked),
        }))}/>
                    <label_1.Label htmlFor="humanFeedback">Enable Human Feedback</label_1.Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <checkbox_1.Checkbox id="keywordAnalysis" checked={qualityControl.keywordAnalysisEnabled} onCheckedChange={(checked) => setQualityControl((prev) => ({
            ...prev,
            keywordAnalysisEnabled: Boolean(checked),
        }))}/>
                    <label_1.Label htmlFor="keywordAnalysis">Keyword Analysis</label_1.Label>
                  </div>
                </div>
                <div>
                  <label_1.Label>Keywords to Target</label_1.Label>
                  <textarea_1.Textarea value={qualityControl.keywordsToTarget} onChange={(event) => setQualityControl((prev) => ({
            ...prev,
            keywordsToTarget: event.target.value,
        }))} placeholder="Enter keywords separated by commas" className="min-h-[100px]"/>
                </div>
              </tabs_1.TabsContent>
            </tabs_1.Tabs>

            <div className="flex flex-wrap gap-2">
              <button_1.Button onClick={handleRunAdversarial} disabled={state.adversarialRunning}>
                Start Adversarial Run
              </button_1.Button>
              <button_1.Button variant="outline" onClick={handleStopAdversarial} disabled={!state.adversarialRunning}>
                Stop
              </button_1.Button>
            </div>

            {runLogs.length ? (<div className="rounded border p-3 text-xs whitespace-pre-wrap">
                {runLogs.slice(-200).join("\n")}
              </div>) : null}

            {state.adversarialResults ? (<card_1.Card>
                <card_1.CardHeader>
                  <card_1.CardTitle className="text-base">Results</card_1.CardTitle>
                </card_1.CardHeader>
                <card_1.CardContent>
                  <pre className="text-xs whitespace-pre-wrap">
                    {JSON.stringify(state.adversarialResults, null, 2)}
                  </pre>
                </card_1.CardContent>
              </card_1.Card>) : null}
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.AdversarialTestingTab = AdversarialTestingTab;
//# sourceMappingURL=AdversarialTestingTab.js.map