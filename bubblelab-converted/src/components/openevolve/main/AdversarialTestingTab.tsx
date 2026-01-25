import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  Sword, 
  Shield, 
  Scale, 
  Play, 
  Square, 
  RotateCcw, 
  FileText, 
  Settings, 
  BarChart3, 
  MessageSquare, 
  Eye,
  Download,
  Upload
} from 'lucide-react';

interface AdversarialTestingTabProps {
  state: any;
  updateState: (updates: any) => void;
}

export const AdversarialTestingTab: React.FC<AdversarialTestingTabProps> = ({ state, updateState }) => {
  const [protocolText, setProtocolText] = useState(state.protocolText);
  const [contentAnalysis, setContentAnalysis] = useState({
    length: 0,
    wordCount: 0,
    avgWordLength: 0
  });
  
  const [redTeamModels, setRedTeamModels] = useState(['claude-3-sonnet']);
  const [blueTeamModels, setBlueTeamModels] = useState(['gpt-4o']);
  const [evaluatorModels, setEvaluatorModels] = useState(['gpt-4o', 'claude-3-sonnet']);
  const [rotationStrategy, setRotationStrategy] = useState('Round Robin');
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
    patchQuality: 5
  });
  
  const [advancedFeatures, setAdvancedFeatures] = useState({
    enableMultiObjective: false,
    featureDimensions: ['complexity', 'diversity'],
    featureBins: 10,
    enableDataAugmentation: false,
    augmentationModel: 'gpt-4o',
    augmentationTemperature: 0.7,
    eliteRatio: 0.1,
    explorationRatio: 0.2,
    archiveSize: 100
  });
  
  const [qualityControl, setQualityControl] = useState({
    enableHumanFeedback: false,
    keywordAnalysisEnabled: true,
    keywordsToTarget: '',
    enableRealTimeMonitoring: true,
    enableComprehensiveReporting: true,
    enableEncryption: true,
    enableAuditTrail: true
  });
  
  const [executionMode, setExecutionMode] = useState('Integrated Adversarial-Evolution');
  
  const modelOptions = [
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
    "gemini-1.5-pro", "gemini-1.5-flash",
    "llama-3-70b", "llama-3-8b",
    "mistral-large", "mistral-medium", "mixtral-8x22b",
    "command-r-plus", "command-r",
    "pplx-7b-online", "pplx-70b-online",
    "openchat/openchat-3.5-0106",
    "microsoft/WizardLM-2-8x22B", "microsoft/WizardLM-2-7B",
  ];
  
  const updateContentAnalysis = (text: string) => {
    const length = text.length;
    const wordCount = text.split(/\s+/).filter(word => word.length > 0).length;
    const avgWordLength = wordCount > 0 ? length / wordCount : 0;
    
    setContentAnalysis({
      length,
      wordCount,
      avgWordLength
    });
  };

  const handleRunAdversarial = () => {
    updateState({ 
      adversarialRunning: true, 
      adversarialStatusMessage: "Starting ultimate adversarial testing & evolution..." 
    });
    
    // Simulate adversarial process
    setTimeout(() => {
      updateState({ 
        adversarialRunning: false, 
        adversarialResults: {
          final_content: `Hardened content based on: ${protocolText.substring(0, 50)}...`,
          initial_content: protocolText,
          integrated_score: 0.85,
          total_cost_usd: 2.45,
          total_tokens: { prompt: 1200, completion: 800 }
        },
        adversarialStatusMessage: "Ultimate adversarial testing & evolution completed successfully!"
      });
    }, 5000);
  };

  const handleStopAdversarial = () => {
    updateState({ 
      adversarialRunning: false,
      adversarialStatusMessage: "Adversarial testing stopped by user."
    });
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sword className="h-5 w-5" />
            Ultimate Adversarial Testing & Evolution
          </CardTitle>
          <CardDescription>
            Advanced AI-Powered Content Hardening with Multi-Model Consensus
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <Label htmlFor="content">Content to Test & Evolve</Label>
                <Textarea
                  id="content"
                  value={protocolText}
                  onChange={(e) => {
                    const newText = e.target.value;
                    setProtocolText(newText);
                    updateState({ protocolText: newText });
                    updateContentAnalysis(newText);
                  }}
                  placeholder="Enter the content you want to harden through adversarial testing and evolution"
                  className="min-h-[200px]"
                />
              </div>
              
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
                
                <div className="mt-4">
                  <Label htmlFor="compliance">Compliance Requirements</Label>
                  <Textarea
                    id="compliance"
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
                    <Label className="flex items-center gap-1 mb-2">
                      <Sword className="h-4 w-4 text-red-500" />
                      Red Team (Critics)
                    </Label>
                    <Select value={redTeamModels[0]} onValueChange={(value) => setRedTeamModels([value])}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {modelOptions.map(model => (
                          <SelectItem key={model} value={model}>{model}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <div className="mt-2">
                      <Label htmlFor="red-team-sample">Red Team Sample Size</Label>
                      <Input
                        id="red-team-sample"
                        type="number"
                        value={adversarialParams.redTeamSampleSize}
                        onChange={(e) => setAdversarialParams({...adversarialParams, redTeamSampleSize: parseInt(e.target.value) || 1})}
                        min="1"
                        max={redTeamModels.length || 1}
                      />
                    </div>
                  </div>
                  
                  <div>
                    <Label className="flex items-center gap-1 mb-2">
                      <Shield className="h-4 w-4 text-blue-500" />
                      Blue Team (Fixers)
                    </Label>
                    <Select value={blueTeamModels[0]} onValueChange={(value) => setBlueTeamModels([value])}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {modelOptions.map(model => (
                          <SelectItem key={model} value={model}>{model}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <div className="mt-2">
                      <Label htmlFor="blue-team-sample">Blue Team Sample Size</Label>
                      <Input
                        id="blue-team-sample"
                        type="number"
                        value={adversarialParams.blueTeamSampleSize}
                        onChange={(e) => setAdversarialParams({...adversarialParams, blueTeamSampleSize: parseInt(e.target.value) || 1})}
                        min="1"
                        max={blueTeamModels.length || 1}
                      />
                    </div>
                  </div>
                  
                  <div>
                    <Label className="flex items-center gap-1 mb-2">
                      <Scale className="h-4 w-4 text-purple-500" />
                      Evaluator Team (Judges)
                    </Label>
                    <Select value={evaluatorModels[0]} onValueChange={(value) => setEvaluatorModels([value])}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {modelOptions.map(model => (
                          <SelectItem key={model} value={model}>{model}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <div className="mt-2">
                      <Label htmlFor="evaluator-sample">Evaluator Sample Size</Label>
                      <Input
                        id="evaluator-sample"
                        type="number"
                        value={adversarialParams.evaluatorSampleSize}
                        onChange={(e) => setAdversarialParams({...adversarialParams, evaluatorSampleSize: parseInt(e.target.value) || 1})}
                        min="1"
                        max={evaluatorModels.length || 1}
                      />
                    </div>
                  </div>
                </div>
                
                <Separator />
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="rotation-strategy">Rotation Strategy</Label>
                    <Select value={rotationStrategy} onValueChange={setRotationStrategy}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Round Robin">Round Robin</SelectItem>
                        <SelectItem value="Random Sampling">Random Sampling</SelectItem>
                        <SelectItem value="Performance-Based">Performance-Based</SelectItem>
                        <SelectItem value="Staged">Staged</SelectItem>
                        <SelectItem value="Adaptive">Adaptive</SelectItem>
                        <SelectItem value="Focus-Category">Focus-Category</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="flex items-center space-x-2 pt-6">
                    <Checkbox 
                      id="perf-tracking" 
                      checked={enablePerformanceTracking}
                      onCheckedChange={(checked) => setEnablePerformanceTracking(!!checked)}
                    />
                    <Label htmlFor="perf-tracking">Enable Performance Tracking</Label>
                  </div>
                </div>
                
                <Separator />
                
                <div className="space-y-4">
                  <h3 className="font-medium flex items-center gap-2">
                    <MessageSquare className="h-4 w-4" />
                    Custom Prompts & Templates
                  </h3>
                  <div className="flex items-center space-x-2">
                    <Checkbox id="custom-prompts" />
                    <Label htmlFor="custom-prompts">Enable Custom Prompts</Label>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="red-prompt">Red Team Critique Prompt</Label>
                      <Textarea
                        id="red-prompt"
                        placeholder="Custom prompt for Red Team vulnerability analysis..."
                        className="min-h-[100px]"
                      />
                    </div>
                    
                    <div>
                      <Label htmlFor="blue-prompt">Blue Team Patch Prompt</Label>
                      <Textarea
                        id="blue-prompt"
                        placeholder="Custom prompt for Blue Team issue resolution..."
                        className="min-h-[100px]"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <Label htmlFor="approval-prompt">Approval Prompt</Label>
                    <Textarea
                      id="approval-prompt"
                      placeholder="Custom prompt for final content approval assessment..."
                      className="min-h-[80px]"
                    />
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="process-params" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <Label htmlFor="min-iter">Minimum Iterations</Label>
                    <Input
                      id="min-iter"
                      type="number"
                      value={adversarialParams.minIter}
                      onChange={(e) => setAdversarialParams({...adversarialParams, minIter: parseInt(e.target.value) || 1})}
                      min="1"
                      max="100"
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="max-iter">Maximum Iterations</Label>
                    <Input
                      id="max-iter"
                      type="number"
                      value={adversarialParams.maxIter}
                      onChange={(e) => setAdversarialParams({...adversarialParams, maxIter: Math.max(parseInt(e.target.value) || 1, adversarialParams.minIter)})} 
                      min={adversarialParams.minIter}
                      max="200"
                    />
                  </div>
                  
                  <div>
                    <Label>Confidence Threshold: {adversarialParams.confidence}%</Label>
                    <Slider
                      value={[adversarialParams.confidence]}
                      onValueChange={(value) => setAdversarialParams({...adversarialParams, confidence: value[0]})}
                      max={100}
                      min={50}
                      step={1}
                    />
                  </div>
                  
                  <div>
                    <Label>Evaluator Threshold: {adversarialParams.evaluatorThreshold.toFixed(1)}</Label>
                    <Slider
                      value={[adversarialParams.evaluatorThreshold]}
                      onValueChange={(value) => setAdversarialParams({...adversarialParams, evaluatorThreshold: value[0]})}
                      max={100}
                      min={50}
                      step={0.5}
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="consecutive-rounds">Consecutive Rounds Required</Label>
                    <Input
                      id="consecutive-rounds"
                      type="number"
                      value={adversarialParams.evaluatorConsecutiveRounds}
                      onChange={(e) => setAdversarialParams({...adversarialParams, evaluatorConsecutiveRounds: parseInt(e.target.value) || 1})}
                      min="1"
                      max="10"
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="budget-limit">Budget Limit (USD)</Label>
                    <Input
                      id="budget-limit"
                      type="number"
                      value={adversarialParams.budgetLimit}
                      onChange={(e) => setAdversarialParams({...adversarialParams, budgetLimit: parseFloat(e.target.value) || 0})}
                      min="0"
                      step="0.01"
                    />
                  </div>
                </div>
                
                <Separator />
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Critique Depth Level: {adversarialParams.critiqueDepth}</Label>
                    <Slider
                      value={[adversarialParams.critiqueDepth]}
                      onValueChange={(value) => setAdversarialParams({...adversarialParams, critiqueDepth: value[0]})}
                      max={10}
                      min={1}
                      step={1}
                    />
                    <p className="text-xs text-muted-foreground mt-1">Controls thoroughness of Red Team analysis (1=surface, 10=deep)</p>
                  </div>
                  
                  <div>
                    <Label>Patch Quality Level: {adversarialParams.patchQuality}</Label>
                    <Slider
                      value={[adversarialParams.patchQuality]}
                      onValueChange={(value) => setAdversarialParams({...adversarialParams, patchQuality: value[0]})}
                      max={10}
                      min={1}
                      step={1}
                    />
                    <p className="text-xs text-muted-foreground mt-1">Governs thoroughness of Blue Team fix implementation (1=basic, 10=comprehensive)</p>
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="advanced" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox 
                        id="multi-objective" 
                        checked={advancedFeatures.enableMultiObjective}
                        onCheckedChange={(checked) => setAdvancedFeatures({...advancedFeatures, enableMultiObjective: !!checked})}
                      />
                      <Label htmlFor="multi-objective">Enable Multi-Objective Optimization</Label>
                    </div>
                    
                    {advancedFeatures.enableMultiObjective && (
                      <>
                        <div>
                          <Label>Feature Dimensions</Label>
                          <div className="flex flex-wrap gap-2 mt-2">
                            {advancedFeatures.featureDimensions.map(dim => (
                              <Badge key={dim} variant="secondary" className="cursor-pointer">
                                {dim}
                              </Badge>
                            ))}
                          </div>
                        </div>
                        
                        <div>
                          <Label htmlFor="feature-bins">Feature Bins</Label>
                          <Input
                            id="feature-bins"
                            type="number"
                            value={advancedFeatures.featureBins}
                            onChange={(e) => setAdvancedFeatures({...advancedFeatures, featureBins: parseInt(e.target.value) || 10})}
                            min="5"
                            max="50"
                          />
                        </div>
                      </>
                    )}
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox 
                        id="data-augmentation" 
                        checked={advancedFeatures.enableDataAugmentation}
                        onCheckedChange={(checked) => setAdvancedFeatures({...advancedFeatures, enableDataAugmentation: !!checked})}
                      />
                      <Label htmlFor="data-augmentation">Enable Data Augmentation</Label>
                    </div>
                    
                    {advancedFeatures.enableDataAugmentation && (
                      <>
                        <div>
                          <Label htmlFor="augmentation-model">Augmentation Model</Label>
                          <Select value={advancedFeatures.augmentationModel} onValueChange={(value) => setAdvancedFeatures({...advancedFeatures, augmentationModel: value})}>
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {modelOptions.map(model => (
                                <SelectItem key={model} value={model}>{model}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        
                        <div>
                          <Label>Augmentation Temperature: {advancedFeatures.augmentationTemperature.toFixed(1)}</Label>
                          <Slider
                            value={[advancedFeatures.augmentationTemperature]}
                            onValueChange={(value) => setAdvancedFeatures({...advancedFeatures, augmentationTemperature: value[0]})}
                            max={2}
                            min={0}
                            step={0.1}
                          />
                        </div>
                      </>
                    )}
                  </div>
                </div>
                
                <Separator />
                
                <div>
                  <h3 className="font-medium mb-2">Evolution Parameters</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <Label>Elite Ratio: {(advancedFeatures.eliteRatio * 100).toFixed(0)}%</Label>
                      <Slider
                        value={[advancedFeatures.eliteRatio]}
                        onValueChange={(value) => setAdvancedFeatures({...advancedFeatures, eliteRatio: value[0]})}
                        max={1}
                        min={0}
                        step={0.01}
                      />
                    </div>
                    
                    <div>
                      <Label>Exploration Ratio: {(advancedFeatures.explorationRatio * 100).toFixed(0)}%</Label>
                      <Slider
                        value={[advancedFeatures.explorationRatio]}
                        onValueChange={(value) => setAdvancedFeatures({...advancedFeatures, explorationRatio: value[0]})}
                        max={1}
                        min={0}
                        step={0.01}
                      />
                    </div>
                    
                    <div>
                      <Label htmlFor="archive-size">Archive Size</Label>
                      <Input
                        id="archive-size"
                        type="number"
                        value={advancedFeatures.archiveSize}
                        onChange={(e) => setAdvancedFeatures({...advancedFeatures, archiveSize: parseInt(e.target.value) || 100})}
                        min="10"
                        max="1000"
                      />
                    </div>
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="quality-control" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox 
                        id="human-feedback" 
                        checked={qualityControl.enableHumanFeedback}
                        onCheckedChange={(checked) => setQualityControl({...qualityControl, enableHumanFeedback: !!checked})}
                      />
                      <Label htmlFor="human-feedback">Enable Human Feedback Integration</Label>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Checkbox 
                        id="keyword-analysis" 
                        checked={qualityControl.keywordAnalysisEnabled}
                        onCheckedChange={(checked) => setQualityControl({...qualityControl, keywordAnalysisEnabled: !!checked})}
                      />
                      <Label htmlFor="keyword-analysis">Enable Keyword Analysis</Label>
                    </div>
                    
                    {qualityControl.keywordAnalysisEnabled && (
                      <div>
                        <Label htmlFor="keywords-target">Keywords to Target</Label>
                        <Textarea
                          id="keywords-target"
                          value={qualityControl.keywordsToTarget}
                          onChange={(e) => setQualityControl({...qualityControl, keywordsToTarget: e.target.value})}
                          placeholder="Enter keywords separated by commas..."
                          className="min-h-[80px]"
                        />
                      </div>
                    )}
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox 
                        id="real-time-monitoring" 
                        checked={qualityControl.enableRealTimeMonitoring}
                        onCheckedChange={(checked) => setQualityControl({...qualityControl, enableRealTimeMonitoring: !!checked})}
                      />
                      <Label htmlFor="real-time-monitoring">Enable Real-Time Monitoring</Label>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Checkbox 
                        id="comprehensive-reporting" 
                        checked={qualityControl.enableComprehensiveReporting}
                        onCheckedChange={(checked) => setQualityControl({...qualityControl, enableComprehensiveReporting: !!checked})}
                      />
                      <Label htmlFor="comprehensive-reporting">Enable Comprehensive Reporting</Label>
                    </div>
                  </div>
                </div>
                
                <Separator />
                
                <div>
                  <h3 className="font-medium mb-2">Security & Compliance</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center space-x-2">
                      <Checkbox 
                        id="encryption" 
                        checked={qualityControl.enableEncryption}
                        onCheckedChange={(checked) => setQualityControl({...qualityControl, enableEncryption: !!checked})}
                      />
                      <Label htmlFor="encryption">Enable Data Encryption</Label>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Checkbox 
                        id="audit-trail" 
                        checked={qualityControl.enableAuditTrail}
                        onCheckedChange={(checked) => setQualityControl({...qualityControl, enableAuditTrail: !!checked})}
                      />
                      <Label htmlFor="audit-trail">Enable Audit Trail</Label>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
            
            <div className="flex justify-end space-x-2">
              <div className="flex-1">
                <Label htmlFor="execution-mode">Execution Mode</Label>
                <Select value={executionMode} onValueChange={setExecutionMode}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Adversarial Testing Only">Adversarial Testing Only</SelectItem>
                    <SelectItem value="Integrated Adversarial-Evolution">Integrated Adversarial-Evolution</SelectItem>
                    <SelectItem value="Full Tripartite Workflow">Full Tripartite Workflow</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <Button 
                variant="outline" 
                onClick={() => {
                  // Reset to defaults
                }}
              >
                <RotateCcw className="mr-2 h-4 w-4" />
                Reset
              </Button>
              
              <Button 
                onClick={handleRunAdversarial}
                disabled={state.adversarialRunning}
                className="bg-red-600 hover:bg-red-700"
              >
                <Sword className="mr-2 h-4 w-4" />
                {state.adversarialRunning ? 'Running...' : 'Run Ultimate Testing'}
              </Button>
              
              {state.adversarialRunning && (
                <Button 
                  variant="destructive" 
                  onClick={handleStopAdversarial}
                >
                  <Square className="mr-2 h-4 w-4" />
                  Stop Testing
                </Button>
              )}
            </div>
            
            {state.adversarialResults && (
              <Card>
                <CardHeader>
                  <CardTitle>🎯 Ultimate Testing Results</CardTitle>
                  <CardDescription>Final hardened content and analytics</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <Label>Final Hardened Content</Label>
                      <Textarea
                        value={state.adversarialResults.final_content}
                        readOnly
                        className="min-h-[200px] bg-muted"
                      />
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <Label>Original Content</Label>
                        <Textarea
                          value={state.adversarialResults.initial_content}
                          readOnly
                          className="min-h-[150px] bg-muted"
                        />
                      </div>
                      
                      <div>
                        <Label>Improved Content</Label>
                        <Textarea
                          value={state.adversarialResults.final_content}
                          readOnly
                          className="min-h-[150px] bg-muted"
                        />
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="p-4 bg-blue-50 rounded-lg">
                        <p className="text-sm text-blue-800">Integrated Score</p>
                        <p className="text-2xl font-bold text-blue-800">{(state.adversarialResults.integrated_score * 100).toFixed(2)}%</p>
                      </div>
                      
                      <div className="p-4 bg-green-50 rounded-lg">
                        <p className="text-sm text-green-800">Total Cost</p>
                        <p className="text-2xl font-bold text-green-800">${state.adversarialResults.total_cost_usd.toFixed(4)}</p>
                      </div>
                      
                      <div className="p-4 bg-purple-50 rounded-lg">
                        <p className="text-sm text-purple-800">Tokens Used</p>
                        <p className="text-lg font-bold text-purple-800">
                          Prompt: {state.adversarialResults.total_tokens.prompt.toLocaleString()}, 
                          Completion: {state.adversarialResults.total_tokens.completion.toLocaleString()}
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
            
            {state.adversarialStatusMessage && (
              <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                <p className="text-blue-800">{state.adversarialStatusMessage}</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};