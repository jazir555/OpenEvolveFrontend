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
import { Sword } from "lucide-react";
import { openevolveApi } from "@/lib/openevolveApi";
import type { AdversarialRunStatus } from "@/lib/types";

interface AdversarialTestingTabProps {
  state: any;
  updateState: (updates: any) => void;
}

const readStorage = (key: string, fallback = "") => {
  try {
    return globalThis.localStorage?.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
};

const extractFinalContent = (result: Record<string, unknown> | null, fallback: string) => {
  if (!result) return fallback;
  const candidates = [
    result.final_content,
    result.best_content,
    result.best_code,
    result.best_program,
    result.output,
  ] as Array<string | undefined>;
  return candidates.find((value) => typeof value === "string" && value.trim()) ?? fallback;
};

export const AdversarialTestingTab: React.FC<AdversarialTestingTabProps> = ({ state, updateState }) => {
  const [protocolText, setProtocolText] = useState(state.protocolText);
  const [apiKey, setApiKey] = useState(readStorage("openevolve_api_key"));
  const [apiBase, setApiBase] = useState(readStorage("openevolve_api_base"));
  const apiConfig = useMemo(() => ({ apiKey, baseUrl: apiBase || undefined }), [apiKey, apiBase]);

  const [contentType, setContentType] = useState("document_general");
  const [contentAnalysis, setContentAnalysis] = useState({
    length: 0,
    wordCount: 0,
    avgWordLength: 0,
  });

  const [redTeamModels, setRedTeamModels] = useState(["claude-3-sonnet"]);
  const [blueTeamModels, setBlueTeamModels] = useState(["gpt-4o"]);
  const [evaluatorModels, setEvaluatorModels] = useState(["gpt-4o", "claude-3-sonnet"]);
  const [rotationStrategy, setRotationStrategy] = useState("Round Robin");
  const [enablePerformanceTracking, setEnablePerformanceTracking] = useState(true);

  const [adversarialParams, setAdversarialParams] = useState({
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

  const [advancedFeatures, setAdvancedFeatures] = useState({
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

  const [qualityControl, setQualityControl] = useState({
    enableHumanFeedback: false,
    keywordAnalysisEnabled: true,
    keywordsToTarget: "",
    enableRealTimeMonitoring: true,
    enableComprehensiveReporting: true,
    enableEncryption: true,
    enableAuditTrail: true,
    complianceRequirements: "",
  });

  const [executionMode, setExecutionMode] = useState("Integrated Adversarial-Evolution");
  const [runId, setRunId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<AdversarialRunStatus | null>(null);
  const [runLogs, setRunLogs] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

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

  const updateContentAnalysis = (text: string) => {
    const length = text.length;
    const wordCount = text.split(/\s+/).filter((word) => word.length > 0).length;
    const avgWordLength = wordCount > 0 ? length / wordCount : 0;

    setContentAnalysis({
      length,
      wordCount,
      avgWordLength,
    });
  };

  const pollRun = async (activeRunId: string) => {
    const response = await openevolveApi.getAdversarialRun(activeRunId, apiConfig);
    setRunStatus(response);
    setRunLogs(response.logs ?? []);
    updateState({ adversarialRunning: response.status === "running" || response.status === "queued" });
    if (response.result) {
      const finalContent = extractFinalContent(response.result as Record<string, unknown>, protocolText);
      updateState({
        adversarialResults: response.result,
        evolutionCurrentBest: finalContent,
      });
    }
  };

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        await pollRun(runId);
      } catch (error: any) {
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

    const parameters: Record<string, unknown> = {
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
      const response = await openevolveApi.startAdversarialRun(
        {
          content: protocolText,
          content_type: contentType,
          parameters,
        },
        apiConfig,
      );
      setRunId(response.run_id);
      updateState({ adversarialRunning: true, adversarialStatusMessage: "Adversarial run started." });
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to start adversarial testing.");
    }
  };

  const handleStopAdversarial = async () => {
    if (!runId) return;
    try {
      await openevolveApi.stopAdversarialRun(runId, apiConfig);
      updateState({ adversarialStatusMessage: "Cancellation requested." });
    } catch (error: any) {
      setErrorMessage(error?.message ?? "Failed to stop adversarial testing.");
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sword className="h-5 w-5" />
            Ultimate Adversarial Testing & Evolution
          </CardTitle>
          <CardDescription>Advanced AI-Powered Content Hardening with Multi-Model Consensus</CardDescription>
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
                />
              </div>
            </div>

            {errorMessage ? <div className="text-sm text-red-500">{errorMessage}</div> : null}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <Label htmlFor="content">Content to Test & Evolve</Label>
                <Textarea
                  id="content"
                  value={protocolText}
                  onChange={(event) => {
                    const newText = event.target.value;
                    setProtocolText(newText);
                    updateState({ protocolText: newText });
                    updateContentAnalysis(newText);
                  }}
                  placeholder="Enter the content you want to harden through adversarial testing and evolution"
                  className="min-h-[200px]"
                />
              </div>

              <div className="space-y-4">
                <div>
                  <Label>Content Analysis</Label>
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
                  <Label>Content Type</Label>
                  <Select value={contentType} onValueChange={setContentType}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {["document_general", "document_technical", "document_legal", "document_medical", "code_python", "code_javascript"].map((type) => (
                        <SelectItem key={type} value={type}>
                          {type}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label htmlFor="compliance">Compliance Requirements</Label>
                  <Textarea
                    id="compliance"
                    value={qualityControl.complianceRequirements}
                    onChange={(event) =>
                      setQualityControl((prev) => ({
                        ...prev,
                        complianceRequirements: event.target.value,
                      }))
                    }
                    placeholder="e.g., GDPR, HIPAA, SOC 2, ISO 27001 requirements..."
                    className="min-h-[100px]"
                  />
                </div>
              </div>
            </div>

            <Tabs defaultValue="model-config" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="model-config">Model Config</TabsTrigger>
                <TabsTrigger value="process-params">Process Params</TabsTrigger>
                <TabsTrigger value="advanced">Advanced</TabsTrigger>
                <TabsTrigger value="quality-control">Quality Control</TabsTrigger>
              </TabsList>

              <TabsContent value="model-config" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <Label className="flex items-center gap-1 mb-2">Red Team (Critics)</Label>
                    <Select value={redTeamModels[0]} onValueChange={(value) => setRedTeamModels([value])}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {modelOptions.map((model) => (
                          <SelectItem key={model} value={model}>
                            {model}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label className="flex items-center gap-1 mb-2">Blue Team (Fixers)</Label>
                    <Select value={blueTeamModels[0]} onValueChange={(value) => setBlueTeamModels([value])}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {modelOptions.map((model) => (
                          <SelectItem key={model} value={model}>
                            {model}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label className="flex items-center gap-1 mb-2">Evaluator Models</Label>
                    <Select
                      value={evaluatorModels[0]}
                      onValueChange={(value) => setEvaluatorModels([value])}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {modelOptions.map((model) => (
                          <SelectItem key={model} value={model}>
                            {model}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Rotation Strategy</Label>
                    <Select value={rotationStrategy} onValueChange={setRotationStrategy}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {["Round Robin", "Performance", "Random"].map((strategy) => (
                          <SelectItem key={strategy} value={strategy}>
                            {strategy}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-center space-x-2 mt-6">
                    <Checkbox
                      id="enablePerformanceTracking"
                      checked={enablePerformanceTracking}
                      onCheckedChange={(checked) => setEnablePerformanceTracking(Boolean(checked))}
                    />
                    <Label htmlFor="enablePerformanceTracking">Enable Performance Tracking</Label>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="process-params" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <Label>Max Iterations</Label>
                    <Input
                      type="number"
                      value={adversarialParams.maxIter}
                      onChange={(event) =>
                        setAdversarialParams((prev) => ({
                          ...prev,
                          maxIter: Number(event.target.value) || 0,
                        }))
                      }
                    />
                  </div>
                  <div>
                    <Label>Confidence Threshold</Label>
                    <Input
                      type="number"
                      value={adversarialParams.confidence}
                      onChange={(event) =>
                        setAdversarialParams((prev) => ({
                          ...prev,
                          confidence: Number(event.target.value) || 0,
                        }))
                      }
                    />
                  </div>
                  <div>
                    <Label>Budget Limit (USD)</Label>
                    <Input
                      type="number"
                      value={adversarialParams.budgetLimit}
                      onChange={(event) =>
                        setAdversarialParams((prev) => ({
                          ...prev,
                          budgetLimit: Number(event.target.value) || 0,
                        }))
                      }
                    />
                  </div>
                  <div>
                    <Label>Red Team Sample Size</Label>
                    <Input
                      type="number"
                      value={adversarialParams.redTeamSampleSize}
                      onChange={(event) =>
                        setAdversarialParams((prev) => ({
                          ...prev,
                          redTeamSampleSize: Number(event.target.value) || 0,
                        }))
                      }
                    />
                  </div>
                  <div>
                    <Label>Blue Team Sample Size</Label>
                    <Input
                      type="number"
                      value={adversarialParams.blueTeamSampleSize}
                      onChange={(event) =>
                        setAdversarialParams((prev) => ({
                          ...prev,
                          blueTeamSampleSize: Number(event.target.value) || 0,
                        }))
                      }
                    />
                  </div>
                  <div>
                    <Label>Critique Depth</Label>
                    <Slider
                      value={[adversarialParams.critiqueDepth]}
                      onValueChange={(value) =>
                        setAdversarialParams((prev) => ({
                          ...prev,
                          critiqueDepth: value[0],
                        }))
                      }
                      max={10}
                      min={1}
                    />
                  </div>
                  <div>
                    <Label>Patch Quality</Label>
                    <Slider
                      value={[adversarialParams.patchQuality]}
                      onValueChange={(value) =>
                        setAdversarialParams((prev) => ({
                          ...prev,
                          patchQuality: value[0],
                        }))
                      }
                      max={10}
                      min={1}
                    />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="advanced" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="enableMultiObjective"
                      checked={advancedFeatures.enableMultiObjective}
                      onCheckedChange={(checked) =>
                        setAdvancedFeatures((prev) => ({
                          ...prev,
                          enableMultiObjective: Boolean(checked),
                        }))
                      }
                    />
                    <Label htmlFor="enableMultiObjective">Multi-Objective Mode</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="enableDataAugmentation"
                      checked={advancedFeatures.enableDataAugmentation}
                      onCheckedChange={(checked) =>
                        setAdvancedFeatures((prev) => ({
                          ...prev,
                          enableDataAugmentation: Boolean(checked),
                        }))
                      }
                    />
                    <Label htmlFor="enableDataAugmentation">Data Augmentation</Label>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Feature Bins</Label>
                    <Input
                      type="number"
                      value={advancedFeatures.featureBins}
                      onChange={(event) =>
                        setAdvancedFeatures((prev) => ({
                          ...prev,
                          featureBins: Number(event.target.value) || 0,
                        }))
                      }
                    />
                  </div>
                  <div>
                    <Label>Archive Size</Label>
                    <Input
                      type="number"
                      value={advancedFeatures.archiveSize}
                      onChange={(event) =>
                        setAdvancedFeatures((prev) => ({
                          ...prev,
                          archiveSize: Number(event.target.value) || 0,
                        }))
                      }
                    />
                  </div>
                </div>

                <div>
                  <Label>Feature Dimensions</Label>
                  <div className="flex flex-wrap gap-2">
                    {advancedFeatures.featureDimensions.map((dim) => (
                      <Badge key={dim} variant="secondary">
                        {dim}
                      </Badge>
                    ))}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="quality-control" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="humanFeedback"
                      checked={qualityControl.enableHumanFeedback}
                      onCheckedChange={(checked) =>
                        setQualityControl((prev) => ({
                          ...prev,
                          enableHumanFeedback: Boolean(checked),
                        }))
                      }
                    />
                    <Label htmlFor="humanFeedback">Enable Human Feedback</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="keywordAnalysis"
                      checked={qualityControl.keywordAnalysisEnabled}
                      onCheckedChange={(checked) =>
                        setQualityControl((prev) => ({
                          ...prev,
                          keywordAnalysisEnabled: Boolean(checked),
                        }))
                      }
                    />
                    <Label htmlFor="keywordAnalysis">Keyword Analysis</Label>
                  </div>
                </div>
                <div>
                  <Label>Keywords to Target</Label>
                  <Textarea
                    value={qualityControl.keywordsToTarget}
                    onChange={(event) =>
                      setQualityControl((prev) => ({
                        ...prev,
                        keywordsToTarget: event.target.value,
                      }))
                    }
                    placeholder="Enter keywords separated by commas"
                    className="min-h-[100px]"
                  />
                </div>
              </TabsContent>
            </Tabs>

            <div className="flex flex-wrap gap-2">
              <Button onClick={handleRunAdversarial} disabled={state.adversarialRunning}>
                Start Adversarial Run
              </Button>
              <Button variant="outline" onClick={handleStopAdversarial} disabled={!state.adversarialRunning}>
                Stop
              </Button>
            </div>

            {runLogs.length ? (
              <div className="rounded border p-3 text-xs whitespace-pre-wrap">
                {runLogs.slice(-200).join("\n")}
              </div>
            ) : null}

            {state.adversarialResults ? (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <pre className="text-xs whitespace-pre-wrap">
                    {JSON.stringify(state.adversarialResults, null, 2)}
                  </pre>
                </CardContent>
              </Card>
            ) : null}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
