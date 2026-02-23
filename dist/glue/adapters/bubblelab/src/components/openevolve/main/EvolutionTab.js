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
exports.EvolutionTab = void 0;
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
const separator_1 = require("@/components/ui/separator");
const lucide_react_1 = require("lucide-react");
const openevolveApi_1 = require("../../../lib/openevolveApi");
const CONTENT_TYPES = [
    "document_general",
    "document_technical",
    "document_legal",
    "document_medical",
    "code_python",
    "code_javascript",
    "code_typescript",
];
const readStorage = (key, fallback = "") => {
    try {
        return globalThis.localStorage?.getItem(key) ?? fallback;
    }
    catch {
        return fallback;
    }
};
const extractFinalContent = (result) => {
    if (!result)
        return "";
    const candidates = [
        result.final_content,
        result.best_content,
        result.best_code,
        result.best_program,
        result.output,
    ];
    return candidates.find((value) => typeof value === "string" && value.trim()) ?? "";
};
const EvolutionTab = ({ state, updateState }) => {
    const [protocolText, setProtocolText] = (0, react_1.useState)(state.protocolText);
    const [apiKey, setApiKey] = (0, react_1.useState)(readStorage("openevolve_api_key"));
    const [apiBase, setApiBase] = (0, react_1.useState)(readStorage("openevolve_api_base"));
    const apiConfig = (0, react_1.useMemo)(() => ({ apiKey, baseUrl: apiBase || undefined }), [apiKey, apiBase]);
    const [contentType, setContentType] = (0, react_1.useState)("document_general");
    const [evolutionMode, setEvolutionMode] = (0, react_1.useState)("standard");
    const [maxIterations, setMaxIterations] = (0, react_1.useState)(20);
    const [populationSize, setPopulationSize] = (0, react_1.useState)(10);
    const [temperature, setTemperature] = (0, react_1.useState)(0.7);
    const [topP, setTopP] = (0, react_1.useState)(1.0);
    const [maxTokens, setMaxTokens] = (0, react_1.useState)(4096);
    const [systemPrompt, setSystemPrompt] = (0, react_1.useState)("You are an expert content generator.");
    const [evaluatorPrompt, setEvaluatorPrompt] = (0, react_1.useState)("Evaluate the quality of this content and provide a score from 0 to 100.");
    const [eliteRatio, setEliteRatio] = (0, react_1.useState)(0.1);
    const [explorationRatio, setExplorationRatio] = (0, react_1.useState)(0.2);
    const [exploitationRatio, setExploitationRatio] = (0, react_1.useState)(0.7);
    const [archiveSize, setArchiveSize] = (0, react_1.useState)(100);
    const [featureBins, setFeatureBins] = (0, react_1.useState)(10);
    const [featureDimensions, setFeatureDimensions] = (0, react_1.useState)(["complexity", "diversity"]);
    const [enableArtifacts, setEnableArtifacts] = (0, react_1.useState)(true);
    const [cascadeEval, setCascadeEval] = (0, react_1.useState)(true);
    const [llmFeedback, setLlmFeedback] = (0, react_1.useState)(false);
    const [enableTrace, setEnableTrace] = (0, react_1.useState)(false);
    const [diffBased, setDiffBased] = (0, react_1.useState)(true);
    const [parallelEval, setParallelEval] = (0, react_1.useState)(4);
    const [checkpointInterval, setCheckpointInterval] = (0, react_1.useState)(5);
    const [runId, setRunId] = (0, react_1.useState)(null);
    const [runStatus, setRunStatus] = (0, react_1.useState)(null);
    const [runLogs, setRunLogs] = (0, react_1.useState)([]);
    const [errorMessage, setErrorMessage] = (0, react_1.useState)(null);
    const evolutionModes = [
        { id: "standard", name: "Standard Evolution", desc: "Basic evolutionary optimization" },
        { id: "quality_diversity", name: "Quality-Diversity (MAP-Elites)", desc: "Maintains diverse, high-performing solutions" },
        { id: "multi_objective", name: "Multi-Objective", desc: "Optimizes for multiple competing objectives" },
        { id: "adversarial", name: "Adversarial Evolution", desc: "Red Team/Blue Team approach for robustness" },
        { id: "prompt_optimization", name: "Prompt Optimization", desc: "Optimizes LLM prompts for better performance" },
        { id: "algorithm_discovery", name: "Algorithm Discovery", desc: "Discovers novel algorithmic approaches" },
        { id: "symbolic_regression", name: "Symbolic Regression", desc: "Discovers mathematical expressions from data" },
        { id: "neuroevolution", name: "Neuroevolution", desc: "Evolves neural network architectures" },
    ];
    const toggleDimension = (dimension) => {
        setFeatureDimensions((prev) => prev.includes(dimension) ? prev.filter((item) => item !== dimension) : [...prev, dimension]);
    };
    const pollRun = async (activeRunId) => {
        const response = await openevolveApi_1.openevolveApi.getEvolutionRun(activeRunId, apiConfig);
        setRunStatus(response);
        setRunLogs(response.logs ?? []);
        if (response.result) {
            const finalContent = extractFinalContent(response.result);
            if (finalContent) {
                updateState({ evolutionCurrentBest: finalContent });
            }
        }
        updateState({ evolutionRunning: response.status === "running" || response.status === "queued" });
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
                    setErrorMessage(error?.message ?? "Failed to fetch evolution status.");
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
    const handleRunEvolution = async () => {
        setErrorMessage(null);
        if (!protocolText.trim()) {
            setErrorMessage("Provide content to evolve.");
            return;
        }
        const parameters = {
            max_iterations: maxIterations,
            population_size: populationSize,
            temperature,
            top_p: topP,
            max_tokens: maxTokens,
            system_prompt: systemPrompt,
            evaluator_system_message: evaluatorPrompt,
            elite_ratio: eliteRatio,
            exploration_ratio: explorationRatio,
            exploitation_ratio: exploitationRatio,
            archive_size: archiveSize,
            feature_bins: featureBins,
            feature_dimensions: featureDimensions,
            enable_artifacts: enableArtifacts,
            cascade_evaluation: cascadeEval,
            use_llm_feedback: llmFeedback,
            trace_enabled: enableTrace,
            diff_based_evolution: diffBased,
            parallel_evaluations: parallelEval,
            checkpoint_interval: checkpointInterval,
            api_key: apiKey || undefined,
            api_base: apiBase || undefined,
        };
        try {
            const response = await openevolveApi_1.openevolveApi.startEvolutionRun({
                content: protocolText,
                content_type: contentType,
                evolution_mode: evolutionMode,
                parameters,
            }, apiConfig);
            setRunId(response.run_id);
            updateState({ evolutionRunning: true, evolutionStatusMessage: "Evolution started." });
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to start evolution.");
        }
    };
    const handleStopEvolution = async () => {
        if (!runId)
            return;
        try {
            await openevolveApi_1.openevolveApi.stopEvolutionRun(runId, apiConfig);
            updateState({ evolutionStatusMessage: "Cancellation requested." });
        }
        catch (error) {
            setErrorMessage(error?.message ?? "Failed to stop evolution.");
        }
    };
    return (<div className="space-y-6">
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle className="flex items-center gap-2">
            <lucide_react_1.Code className="h-5 w-5"/>
            Evolution Engine
          </card_1.CardTitle>
          <card_1.CardDescription>Advanced Evolutionary Computing with OpenEvolve</card_1.CardDescription>
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
        }} placeholder="https://api.openai.com/v1"/>
              </div>
            </div>

            {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

            <div>
              <label_1.Label htmlFor="content">Content to Evolve</label_1.Label>
              <textarea_1.Textarea id="content" value={protocolText} onChange={(event) => {
            setProtocolText(event.target.value);
            updateState({ protocolText: event.target.value });
        }} placeholder="Enter the content you want to evolve..." className="min-h-[200px]"/>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <label_1.Label>Evolution Mode</label_1.Label>
                <select_1.Select value={evolutionMode} onValueChange={setEvolutionMode}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue />
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    {evolutionModes.map((mode) => (<select_1.SelectItem key={mode.id} value={mode.id}>
                        <div>
                          <div>{mode.name}</div>
                          <div className="text-xs text-muted-foreground">{mode.desc}</div>
                        </div>
                      </select_1.SelectItem>))}
                  </select_1.SelectContent>
                </select_1.Select>
              </div>
              <div className="space-y-2">
                <label_1.Label>Content Type</label_1.Label>
                <select_1.Select value={contentType} onValueChange={setContentType}>
                  <select_1.SelectTrigger>
                    <select_1.SelectValue />
                  </select_1.SelectTrigger>
                  <select_1.SelectContent>
                    {CONTENT_TYPES.map((type) => (<select_1.SelectItem key={type} value={type}>
                        {type}
                      </select_1.SelectItem>))}
                  </select_1.SelectContent>
                </select_1.Select>
              </div>
            </div>

            <tabs_1.Tabs defaultValue="config" className="w-full">
              <tabs_1.TabsList className="grid w-full grid-cols-4">
                <tabs_1.TabsTrigger value="config">Configuration</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="advanced">Advanced</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="prompts">Prompts</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="results">Results</tabs_1.TabsTrigger>
              </tabs_1.TabsList>

              <tabs_1.TabsContent value="config" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label_1.Label htmlFor="maxIterations">Max Iterations</label_1.Label>
                    <input_1.Input id="maxIterations" type="number" value={maxIterations} onChange={(event) => setMaxIterations(Number(event.target.value) || 0)} min="1"/>
                  </div>

                  <div>
                    <label_1.Label htmlFor="populationSize">Population Size</label_1.Label>
                    <input_1.Input id="populationSize" type="number" value={populationSize} onChange={(event) => setPopulationSize(Number(event.target.value) || 0)} min="1"/>
                  </div>

                  <div>
                    <label_1.Label>Temperature: {temperature.toFixed(2)}</label_1.Label>
                    <slider_1.Slider value={[temperature]} onValueChange={(value) => setTemperature(value[0])} max={2} min={0} step={0.1}/>
                  </div>

                  <div>
                    <label_1.Label>Top P: {topP.toFixed(2)}</label_1.Label>
                    <slider_1.Slider value={[topP]} onValueChange={(value) => setTopP(value[0])} max={1} min={0} step={0.05}/>
                  </div>

                  <div>
                    <label_1.Label htmlFor="maxTokens">Max Tokens</label_1.Label>
                    <input_1.Input id="maxTokens" type="number" value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value) || 0)} min="1"/>
                  </div>
                </div>

                <div className="flex justify-end space-x-2 pt-4">
                  <button_1.Button variant="outline" onClick={() => {
            setMaxIterations(20);
            setPopulationSize(10);
            setTemperature(0.7);
            setTopP(1.0);
            setMaxTokens(4096);
        }}>
                    Reset Defaults
                  </button_1.Button>
                  <button_1.Button onClick={handleRunEvolution} disabled={state.evolutionRunning}>
                    Run Evolution
                  </button_1.Button>
                  <button_1.Button variant="outline" onClick={handleStopEvolution} disabled={!state.evolutionRunning}>
                    Stop
                  </button_1.Button>
                </div>
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="advanced" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label_1.Label>Elite Ratio</label_1.Label>
                    <slider_1.Slider value={[eliteRatio]} onValueChange={(value) => setEliteRatio(value[0])} max={1} min={0} step={0.01}/>
                  </div>

                  <div>
                    <label_1.Label>Exploration Ratio</label_1.Label>
                    <slider_1.Slider value={[explorationRatio]} onValueChange={(value) => setExplorationRatio(value[0])} max={1} min={0} step={0.01}/>
                  </div>

                  <div>
                    <label_1.Label>Exploitation Ratio</label_1.Label>
                    <slider_1.Slider value={[exploitationRatio]} onValueChange={(value) => setExploitationRatio(value[0])} max={1} min={0} step={0.01}/>
                  </div>

                  <div>
                    <label_1.Label htmlFor="archiveSize">Archive Size</label_1.Label>
                    <input_1.Input id="archiveSize" type="number" value={archiveSize} onChange={(event) => setArchiveSize(Number(event.target.value) || 0)} min="10"/>
                  </div>

                  <div>
                    <label_1.Label htmlFor="featureBins">Feature Bins</label_1.Label>
                    <input_1.Input id="featureBins" type="number" value={featureBins} onChange={(event) => setFeatureBins(Number(event.target.value) || 0)} min="5"/>
                  </div>

                  <div className="space-y-2">
                    <label_1.Label>Feature Dimensions</label_1.Label>
                    <div className="flex flex-wrap gap-2">
                      {[
            "complexity",
            "diversity",
            "performance",
            "efficiency",
            "readability",
            "robustness",
        ].map((dimension) => (<badge_1.Badge key={dimension} variant={featureDimensions.includes(dimension) ? "default" : "secondary"} className="cursor-pointer" onClick={() => toggleDimension(dimension)}>
                          {dimension}
                        </badge_1.Badge>))}
                    </div>
                  </div>
                </div>

                <separator_1.Separator />

                <div className="space-y-4">
                  <h3 className="font-medium">Advanced OpenEvolve Features</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="flex items-center space-x-2">
                      <checkbox_1.Checkbox id="enableArtifacts" checked={enableArtifacts} onCheckedChange={(value) => setEnableArtifacts(Boolean(value))}/>
                      <label_1.Label htmlFor="enableArtifacts">Enable Artifacts</label_1.Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <checkbox_1.Checkbox id="cascadeEval" checked={cascadeEval} onCheckedChange={(value) => setCascadeEval(Boolean(value))}/>
                      <label_1.Label htmlFor="cascadeEval">Cascade Evaluation</label_1.Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <checkbox_1.Checkbox id="llmFeedback" checked={llmFeedback} onCheckedChange={(value) => setLlmFeedback(Boolean(value))}/>
                      <label_1.Label htmlFor="llmFeedback">LLM Feedback</label_1.Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <checkbox_1.Checkbox id="enableTrace" checked={enableTrace} onCheckedChange={(value) => setEnableTrace(Boolean(value))}/>
                      <label_1.Label htmlFor="enableTrace">Enable Trace</label_1.Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <checkbox_1.Checkbox id="diffBased" checked={diffBased} onCheckedChange={(value) => setDiffBased(Boolean(value))}/>
                      <label_1.Label htmlFor="diffBased">Diff-Based Evolution</label_1.Label>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label_1.Label htmlFor="parallelEval">Parallel Evaluations</label_1.Label>
                      <input_1.Input id="parallelEval" type="number" value={parallelEval} onChange={(event) => setParallelEval(Number(event.target.value) || 0)} min="1"/>
                    </div>

                    <div>
                      <label_1.Label htmlFor="checkpointInterval">Checkpoint Interval</label_1.Label>
                      <input_1.Input id="checkpointInterval" type="number" value={checkpointInterval} onChange={(event) => setCheckpointInterval(Number(event.target.value) || 0)} min="1"/>
                    </div>
                  </div>
                </div>
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="prompts" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label_1.Label htmlFor="systemPrompt">System Prompt</label_1.Label>
                    <textarea_1.Textarea id="systemPrompt" value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} className="min-h-[150px]"/>
                  </div>

                  <div>
                    <label_1.Label htmlFor="evaluatorPrompt">Evaluator System Prompt</label_1.Label>
                    <textarea_1.Textarea id="evaluatorPrompt" value={evaluatorPrompt} onChange={(event) => setEvaluatorPrompt(event.target.value)} className="min-h-[150px]"/>
                  </div>
                </div>
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="results" className="space-y-4 pt-4">
                {state.evolutionCurrentBest ? (<div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label_1.Label>Original Content</label_1.Label>
                        <textarea_1.Textarea value={protocolText} readOnly className="min-h-[150px] bg-muted"/>
                      </div>
                      <div>
                        <label_1.Label>Evolved Content</label_1.Label>
                        <textarea_1.Textarea value={state.evolutionCurrentBest} readOnly className="min-h-[150px] bg-muted"/>
                      </div>
                    </div>
                  </div>) : (<div className="text-center py-8 text-muted-foreground">
                    Run an evolution to see results here
                  </div>)}

                {runLogs.length ? (<div className="rounded border p-3 text-xs whitespace-pre-wrap">
                    {runLogs.slice(-200).join("\n")}
                  </div>) : null}
              </tabs_1.TabsContent>
            </tabs_1.Tabs>
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.EvolutionTab = EvolutionTab;
//# sourceMappingURL=EvolutionTab.js.map