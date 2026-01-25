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
  Play, 
  Square, 
  RotateCcw, 
  Code, 
  FileText, 
  Settings, 
  BarChart3, 
  MessageSquare, 
  Eye,
  Download,
  Upload
} from 'lucide-react';

interface EvolutionTabProps {
  state: any;
  updateState: (updates: any) => void;
}

export const EvolutionTab: React.FC<EvolutionTabProps> = ({ state, updateState }) => {
  const [protocolText, setProtocolText] = useState(state.protocolText);
  const [evolutionMode, setEvolutionMode] = useState('standard');
  const [maxIterations, setMaxIterations] = useState(20);
  const [populationSize, setPopulationSize] = useState(10);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState(4096);
  
  const evolutionModes = [
    { id: 'standard', name: 'Standard Evolution', desc: 'Basic evolutionary optimization' },
    { id: 'quality_diversity', name: 'Quality-Diversity (MAP-Elites)', desc: 'Maintains diverse, high-performing solutions' },
    { id: 'multi_objective', name: 'Multi-Objective', desc: 'Optimizes for multiple competing objectives' },
    { id: 'adversarial', name: 'Adversarial Evolution', desc: 'Red Team/Blue Team approach for robustness' },
    { id: 'prompt_optimization', name: 'Prompt Optimization', desc: 'Optimizes LLM prompts for better performance' },
    { id: 'algorithm_discovery', name: 'Algorithm Discovery', desc: 'Discovers novel algorithmic approaches' },
    { id: 'symbolic_regression', name: 'Symbolic Regression', desc: 'Discovers mathematical expressions from data' },
    { id: 'neuroevolution', name: 'Neuroevolution', desc: 'Evolves neural network architectures' },
  ];

  const handleRunEvolution = () => {
    updateState({ 
      evolutionRunning: true, 
      evolutionStatusMessage: `Starting ${evolutionMode} evolution...` 
    });
    
    // Simulate evolution process
    setTimeout(() => {
      updateState({ 
        evolutionRunning: false, 
        evolutionCurrentBest: `Evolved content based on: ${protocolText.substring(0, 50)}...`,
        evolutionStatusMessage: `${evolutionMode} evolution completed successfully!`
      });
    }, 3000);
  };

  const handleStopEvolution = () => {
    updateState({ 
      evolutionRunning: false,
      evolutionStatusMessage: "Evolution stopped by user."
    });
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="h-5 w-5" />
            Evolution Engine
          </CardTitle>
          <CardDescription>
            Advanced Evolutionary Computing with OpenEvolve
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div>
              <Label htmlFor="content">Content to Evolve</Label>
              <Textarea
                id="content"
                value={protocolText}
                onChange={(e) => {
                  setProtocolText(e.target.value);
                  updateState({ protocolText: e.target.value });
                }}
                placeholder="Enter the content you want to evolve..."
                className="min-h-[200px]"
              />
            </div>

            <div>
              <Label>Evolution Mode</Label>
              <Select value={evolutionMode} onValueChange={setEvolutionMode}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {evolutionModes.map(mode => (
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
                      onChange={(e) => setMaxIterations(parseInt(e.target.value) || 0)}
                      min="1"
                      max="10000"
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="populationSize">Population Size</Label>
                    <Input
                      id="populationSize"
                      type="number"
                      value={populationSize}
                      onChange={(e) => setPopulationSize(parseInt(e.target.value) || 0)}
                      min="1"
                      max="10000"
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
                      onChange={(e) => setMaxTokens(parseInt(e.target.value) || 0)}
                      min="1"
                      max="4096"
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
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Reset Defaults
                  </Button>
                  <Button 
                    onClick={handleRunEvolution}
                    disabled={state.evolutionRunning}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    <Play className="mr-2 h-4 w-4" />
                    {state.evolutionRunning ? 'Running...' : `Run ${evolutionModes.find(m => m.id === evolutionMode)?.name}`}
                  </Button>
                  {state.evolutionRunning && (
                    <Button 
                      variant="destructive" 
                      onClick={handleStopEvolution}
                    >
                      <Square className="mr-2 h-4 w-4" />
                      Stop Evolution
                    </Button>
                  )}
                </div>
              </TabsContent>
              
              <TabsContent value="advanced" className="space-y-4 pt-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Elite Ratio</Label>
                    <Slider
                      defaultValue={[0.1]}
                      max={1}
                      min={0}
                      step={0.01}
                    />
                  </div>
                  
                  <div>
                    <Label>Exploration Ratio</Label>
                    <Slider
                      defaultValue={[0.2]}
                      max={1}
                      min={0}
                      step={0.01}
                    />
                  </div>
                  
                  <div>
                    <Label>Exploitation Ratio</Label>
                    <Slider
                      defaultValue={[0.7]}
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
                      defaultValue="100"
                      min="10"
                      max="1000"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label>Feature Dimensions</Label>
                    <div className="flex flex-wrap gap-2">
                      {['complexity', 'diversity', 'performance', 'efficiency', 'readability', 'robustness'].map(dim => (
                        <Badge key={dim} variant="secondary" className="cursor-pointer">
                          {dim}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <Label htmlFor="featureBins">Feature Bins</Label>
                    <Input
                      id="featureBins"
                      type="number"
                      defaultValue="10"
                      min="5"
                      max="50"
                    />
                  </div>
                </div>
                
                <Separator />
                
                <div className="space-y-4">
                  <h3 className="font-medium">Advanced OpenEvolve Features</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="flex items-center space-x-2">
                      <Checkbox id="enableArtifacts" defaultChecked />
                      <Label htmlFor="enableArtifacts">Enable Artifacts</Label>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Checkbox id="cascadeEval" defaultChecked />
                      <Label htmlFor="cascadeEval">Cascade Evaluation</Label>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Checkbox id="llmFeedback" />
                      <Label htmlFor="llmFeedback">LLM Feedback</Label>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Checkbox id="includeArtifacts" defaultChecked />
                      <Label htmlFor="includeArtifacts">Include Artifacts</Label>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Checkbox id="enableTrace" />
                      <Label htmlFor="enableTrace">Enable Trace</Label>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Checkbox id="diffBased" defaultChecked />
                      <Label htmlFor="diffBased">Diff-Based Evolution</Label>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="parallelEval">Parallel Evaluations</Label>
                      <Input
                        id="parallelEval"
                        type="number"
                        defaultValue="4"
                        min="1"
                        max="16"
                      />
                    </div>
                    
                    <div>
                      <Label htmlFor="checkpointInterval">Checkpoint Interval</Label>
                      <Input
                        id="checkpointInterval"
                        type="number"
                        defaultValue="5"
                        min="1"
                        max="100"
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
                      defaultValue="You are an expert content generator."
                      placeholder="Enter system prompt..."
                      className="min-h-[150px]"
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="evaluatorPrompt">Evaluator System Prompt</Label>
                    <Textarea
                      id="evaluatorPrompt"
                      defaultValue="Evaluate the quality of this content and provide a score from 0 to 100."
                      placeholder="Enter evaluator prompt..."
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
                        <Textarea
                          value={protocolText}
                          readOnly
                          className="min-h-[150px] bg-muted"
                        />
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
                    
                    <div>
                      <Label>Content Comparison</Label>
                      <div className="border rounded-lg p-4 bg-muted">
                        <p className="text-sm text-muted-foreground">
                          Original: {protocolText.substring(0, 100)}...
                        </p>
                        <p className="text-sm text-muted-foreground mt-2">
                          Evolved: {state.evolutionCurrentBest.substring(0, 100)}...
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    Run an evolution to see results here
                  </div>
                )}
                
                {state.evolutionStatusMessage && (
                  <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <p className="text-blue-800">{state.evolutionStatusMessage}</p>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};