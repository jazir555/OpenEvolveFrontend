import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Code } from "lucide-react";
import { openevolveApi } from "../../../lib/openevolveApi";
import type { EvolutionRunStatus } from "../../../lib/types";

interface EvolutionTabProps {
  state: any;
  updateState: (updates: any) => void;
}

const CONTENT_TYPES = [
  "document_general",
  "document_technical",
  "document_legal",
  "document_medical",
  "code_python",
  "code_javascript",
  "code_typescript",
];

const readStorage = (key: string, fallback = "") => {
  try {
    return globalThis.localStorage?.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
};

const extractFinalContent = (result: Record<string, unknown> | null) => {
  if (!result) return "";
  const candidates = [
    result.final_content,
    result.best_content,
    result.best_code,
    result.best_program,
    result.output,
  ] as Array<string | undefined>;
  return candidates.find((value) => typeof value === "string" && value.trim()) ?? "";
};

export const EvolutionTab: React.FC<EvolutionTabProps> = ({ state, updateState }) => {
  const [protocolText, setProtocolText] = useState(state.protocolText);
  const [apiKey, setApiKey] = useState(readStorage("openevolve_api_key"));
  const [apiBase, setApiBase] = useState(readStorage("openevolve_api_base"));
  const apiConfig = useMemo(() => ({ apiKey, baseUrl: apiBase || undefined }), [apiKey, apiBase]);

  const [contentType, setContentType] = useState("document_general");
  const [evolutionMode, setEvolutionMode] = useState("standard");
  const [maxIterations, setMaxIterations] = useState(20);
  const [populationSize, setPopulationSize] = useState(10);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [systemPrompt, setSystemPrompt] = useState("You are an expert content generator.");
  const [evaluatorPrompt, setEvaluatorPrompt] = useState(
    "Evaluate the quality of this content and provide a score from 0 to 100.",
  );

  const [eliteRatio, setEliteRatio] = useState(0.1);
  const [explorationRatio, setExplorationRatio] = useState(0.2);
  const [exploitationRatio, setExploitationRatio] = useState(0.7);
  const [archiveSize, setArchiveSize] = useState(100);
  const [featureBins, setFeatureBins] = useState(10);
  const [featureDimensions, setFeatureDimensions] = useState<string[]>(["complexity", "diversity"]);

  const [enableArtifacts, setEnableArtifacts] = useState(true);
  const [cascadeEval, setCascadeEval] = useState(true);
  const [llmFeedback, setLlmFeedback] = useState(false);
  const [enableTrace, setEnableTrace] = useState(false);
  const [diffBased, setDiffBased] = useState(true);
  const [parallelEval, setParallelEval] = useState(4);
  const [checkpointInterval, setCheckpointInterval] = useState(5);

  const [runId, setRunId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<EvolutionRunStatus | null>(null);
  const [runLogs, setRunLogs] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

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

  const toggleDimension = (dimension: string) => {
    setFeatureDimensions((prev) =>
      prev.includes(dimension) ? prev.filter((item) => item !== dimension) : [...prev, dimension],
    );
  };

  const pollRun = async (activeRunId: string) => {
    const response = await openevolveApi.getEvolutionRun(activeRunId, apiConfig);
    setRunStatus(response);
    setRunLogs(response.logs ?? []);

    if (response.result) {
      const finalContent = extractFinalContent(response.result as Record<string, unknown>);
      if (finalContent) {
        updateState({ evolutionCurrentBest: finalContent });
      }
    }
    updateState({ evolutionRunning: response.status === "running" || response.status === "queued" });
  };

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;

    const tick = async () => {
      try {
        await pollRun(runId);
      } catch (error: any) {
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

    const parameters: Record<string, unknown> = {
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
      const response = await openevolveApi.startEvolutionRun(
        {
          content: protocolText,
          content_type: contentType,
          evolution_mode: evolutionMode,
          parameters,
        },
        apiConfig,
      );
      setRunId(response.run_id);
      updateState({ evolutionRunning: true, evolutionStatusMessage: "Evolution started." });
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to start evolution.");
    }
  };

  const handleStopEvolution = async () => {
    if (!runId) return;
    try {
      await openevolveApi.stopEvolutionRun(runId, apiConfig);
      updateState({ evolutionStatusMessage: "Cancellation requested." });
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to stop evolution.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="h-5 w-5" />
            Evolution Engine
          </CardTitle>
          <CardDescription>Advanced Evolutionary Computing with OpenEvolve</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2">
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
              <div className="space-y-2">
                <Label>API Base URL</Label>
                <Input
                  value={apiBase}
                  onChange={(event) => {
                    const value = event.target.value;
                    setApiBase(value);
                    try {
                      globalThis.localStorage?.setItem("openevolve_api_base", value);
                    } catch {
                      // ignore storage errors
                    }
                  }}
                  placeholder="https://api.openai.com/v1"
                />
              </div>
            </div>

            {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

            <div>
              <Label htmlFor="content">Content to Evolve</Label>
              <Textarea
                id="content"
                value={protocolText}
                onChange={(event) => {
                  setProtocolText(event.target.value);
                  updateState({ protocolText: event.target.value });
                }}
                placeholder="Enter the content you want to evolve..."
                className="min-h-[200px]"
              />
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label>Evolution Mode</Label>
                <Select value={evolutionMode} onValueChange={setEvolutionMode}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {evolutionModes.map((mode) => (
                      <SelectItem key={mode.id} value={mode.id}>
                        <div>
                          <div>{mode.name}</div>
                          <div className="text-xs text-muted-foreground">{mode.desc}</div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Content Type</Label>
                <Select value={contentType} onValueChange={setContentType}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {CONTENT_TYPES.map((type) => (
                      <SelectItem key={type} value={type}>
                        {type}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <Tabs defaultValue="config" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="config">Configuration</TabsTrigger>
                <TabsTrigger value="advanced">Advanced</TabsTrigger>
                <TabsTrigger value="prompts">Prompts</TabsTrigger>
                <TabsTrigger value="results">Results</TabsTrigger>
              </TabsList>

              <TabsContent value="config" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="maxIterations">Max Iterations</Label>
                    <Input
                      id="maxIterations"
                      type="number"
                      value={maxIterations}
                      onChange={(event) => setMaxIterations(Number(event.target.value) || 0)}
                      min="1"
                    />
                  </div>

                  <div>
                    <Label htmlFor="populationSize">Population Size</Label>
                    <Input
                      id="populationSize"
                      type="number"
                      value={populationSize}
                      onChange={(event) => setPopulationSize(Number(event.target.value) || 0)}
                      min="1"
                    />
                  </div>

                  <div>
                    <Label>Temperature: {temperature.toFixed(2)}</Label>
                    <Slider
                      value={[temperature]}
                      onValueChange={(value) => setTemperature(value[0])}
                      max={2}
                      min={0}
                      step={0.1}
                    />
                  </div>

                  <div>
                    <Label>Top P: {topP.toFixed(2)}</Label>
                    <Slider
                      value={[topP]}
                      onValueChange={(value) => setTopP(value[0])}
                      max={1}
                      min={0}
                      step={0.05}
                    />
                  </div>

                  <div>
                    <Label htmlFor="maxTokens">Max Tokens</Label>
                    <Input
                      id="maxTokens"
                      type="number"
                      value={maxTokens}
                      onChange={(event) => setMaxTokens(Number(event.target.value) || 0)}
                      min="1"
                    />
                  </div>
                </div>

                <div className="flex justify-end space-x-2 pt-4">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setMaxIterations(20);
                      setPopulationSize(10);
                      setTemperature(0.7);
                      setTopP(1.0);
                      setMaxTokens(4096);
                    }}
                  >
                    Reset Defaults
                  </Button>
                  <Button onClick={handleRunEvolution} disabled={state.evolutionRunning}>
                    Run Evolution
                  </Button>
                  <Button variant="outline" onClick={handleStopEvolution} disabled={!state.evolutionRunning}>
                    Stop
                  </Button>
                </div>
              </TabsContent>

              <TabsContent value="advanced" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Elite Ratio</Label>
                    <Slider
                      value={[eliteRatio]}
                      onValueChange={(value) => setEliteRatio(value[0])}
                      max={1}
                      min={0}
                      step={0.01}
                    />
                  </div>

                  <div>
                    <Label>Exploration Ratio</Label>
                    <Slider
                      value={[explorationRatio]}
                      onValueChange={(value) => setExplorationRatio(value[0])}
                      max={1}
                      min={0}
                      step={0.01}
                    />
                  </div>

                  <div>
                    <Label>Exploitation Ratio</Label>
                    <Slider
                      value={[exploitationRatio]}
                      onValueChange={(value) => setExploitationRatio(value[0])}
                      max={1}
                      min={0}
                      step={0.01}
                    />
                  </div>

                  <div>
                    <Label htmlFor="archiveSize">Archive Size</Label>
                    <Input
                      id="archiveSize"
                      type="number"
                      value={archiveSize}
                      onChange={(event) => setArchiveSize(Number(event.target.value) || 0)}
                      min="10"
                    />
                  </div>

                  <div>
                    <Label htmlFor="featureBins">Feature Bins</Label>
                    <Input
                      id="featureBins"
                      type="number"
                      value={featureBins}
                      onChange={(event) => setFeatureBins(Number(event.target.value) || 0)}
                      min="5"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Feature Dimensions</Label>
                    <div className="flex flex-wrap gap-2">
                      {[
                        "complexity",
                        "diversity",
                        "performance",
                        "efficiency",
                        "readability",
                        "robustness",
                      ].map((dimension) => (
                        <Badge
                          key={dimension}
                          variant={featureDimensions.includes(dimension) ? "default" : "secondary"}
                          className="cursor-pointer"
                          onClick={() => toggleDimension(dimension)}
                        >
                          {dimension}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>

                <Separator />

                <div className="space-y-4">
                  <h3 className="font-medium">Advanced OpenEvolve Features</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="flex items-center space-x-2">
                      <Checkbox id="enableArtifacts" checked={enableArtifacts} onCheckedChange={(value) => setEnableArtifacts(Boolean(value))} />
                      <Label htmlFor="enableArtifacts">Enable Artifacts</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox id="cascadeEval" checked={cascadeEval} onCheckedChange={(value) => setCascadeEval(Boolean(value))} />
                      <Label htmlFor="cascadeEval">Cascade Evaluation</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox id="llmFeedback" checked={llmFeedback} onCheckedChange={(value) => setLlmFeedback(Boolean(value))} />
                      <Label htmlFor="llmFeedback">LLM Feedback</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox id="enableTrace" checked={enableTrace} onCheckedChange={(value) => setEnableTrace(Boolean(value))} />
                      <Label htmlFor="enableTrace">Enable Trace</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Checkbox id="diffBased" checked={diffBased} onCheckedChange={(value) => setDiffBased(Boolean(value))} />
                      <Label htmlFor="diffBased">Diff-Based Evolution</Label>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="parallelEval">Parallel Evaluations</Label>
                      <Input
                        id="parallelEval"
                        type="number"
                        value={parallelEval}
                        onChange={(event) => setParallelEval(Number(event.target.value) || 0)}
                        min="1"
                      />
                    </div>

                    <div>
                      <Label htmlFor="checkpointInterval">Checkpoint Interval</Label>
                      <Input
                        id="checkpointInterval"
                        type="number"
                        value={checkpointInterval}
                        onChange={(event) => setCheckpointInterval(Number(event.target.value) || 0)}
                        min="1"
                      />
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="prompts" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="systemPrompt">System Prompt</Label>
                    <Textarea
                      id="systemPrompt"
                      value={systemPrompt}
                      onChange={(event) => setSystemPrompt(event.target.value)}
                      className="min-h-[150px]"
                    />
                  </div>

                  <div>
                    <Label htmlFor="evaluatorPrompt">Evaluator System Prompt</Label>
                    <Textarea
                      id="evaluatorPrompt"
                      value={evaluatorPrompt}
                      onChange={(event) => setEvaluatorPrompt(event.target.value)}
                      className="min-h-[150px]"
                    />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="results" className="space-y-4 pt-4">
                {state.evolutionCurrentBest ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <Label>Original Content</Label>
                        <Textarea value={protocolText} readOnly className="min-h-[150px] bg-muted" />
                      </div>
                      <div>
                        <Label>Evolved Content</Label>
                        <Textarea
                          value={state.evolutionCurrentBest}
                          readOnly
                          className="min-h-[150px] bg-muted"
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    Run an evolution to see results here
                  </div>
                )}

                {runLogs.length ? (
                  <div className="rounded border p-3 text-xs whitespace-pre-wrap">
                    {runLogs.slice(-200).join("\n")}
                  </div>
                ) : null}
              </TabsContent>
            </Tabs>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
