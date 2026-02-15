/**
 * Workflow Execution Tab
 *
 * UI for executing and managing workflows across multiple plugins.
 * Provides workflow selection, parameter input, execution monitoring, and result visualization.
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Play, RefreshCw, CheckCircle, XCircle, Clock, FileText, Settings } from 'lucide-react';
import { getWorkflowOrchestrator } from '../../../../../../../orchestration/workflow-system/workflow-orchestrator';
import { getPluginRegistry } from '../../../../../../../orchestration/workflow-system/plugin-registry';
import { getAllWorkflowTemplates, getWorkflowTemplate } from '../../../../../../../orchestration/workflow-system/workflow-templates';
import type { WorkflowDefinition, WorkflowExecutionResult } from '../../../../../../../orchestration/workflow-system/workflow-orchestrator';

interface WorkflowExecutionState {
  selectedWorkflow: string | null;
  parameters: Record<string, unknown>;
  executionResult: WorkflowExecutionResult | null;
  isExecuting: boolean;
  activeExecutions: string[];
  executionHistory: Array<{ id: string; workflowId: string; timestamp: Date; status: string }>;
}

export const WorkflowExecutionTab: React.FC = () => {
  const [state, setState] = useState<WorkflowExecutionState>({
    selectedWorkflow: null,
    parameters: {},
    executionResult: null,
    isExecuting: false,
    activeExecutions: [],
    executionHistory: []
  });

  const [workflowTemplates] = useState(() => getAllWorkflowTemplates());
  const [selectedTemplate, setSelectedTemplate] = useState<WorkflowDefinition | null>(null);

  const orchestrator = getWorkflowOrchestrator();
  const registry = getPluginRegistry();

  // Load execution history from localStorage
  useEffect(() => {
    try {
      const savedHistory = localStorage.getItem('workflow-execution-history');
      if (savedHistory) {
        const parsed = JSON.parse(savedHistory);
        setState(prev => ({
          ...prev,
          executionHistory: parsed.map((h: any) => ({
            ...h,
            timestamp: new Date(h.timestamp)
          }))
        }));
      }
    } catch (error) {
      console.error('Failed to load execution history:', error);
    }
  }, []);

  // Save execution history to localStorage
  useEffect(() => {
    if (state.executionHistory.length > 0) {
      localStorage.setItem('workflow-execution-history', JSON.stringify(state.executionHistory));
    }
  }, [state.executionHistory]);

  const handleWorkflowSelect = (workflowId: string) => {
    const template = getWorkflowTemplate(workflowId);
    if (template) {
      setSelectedTemplate(template);
      setState(prev => ({
        ...prev,
        selectedWorkflow: workflowId,
        parameters: {},
        executionResult: null
      }));
    }
  };

  const handleParameterChange = (key: string, value: unknown) => {
    setState(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [key]: value
      }
    }));
  };

  const executeWorkflow = async () => {
    if (!selectedTemplate || state.isExecuting) {
      return;
    }

    setState(prev => ({ ...prev, isExecuting: true, executionResult: null }));

    try {
      const result = await orchestrator.executeWorkflow(
        selectedTemplate,
        state.parameters
      );

      // Add to history
      const historyEntry = {
        id: result.executionId,
        workflowId: result.workflowId,
        timestamp: new Date(),
        status: result.status
      };

      setState(prev => ({
        ...prev,
        executionResult: result,
        isExecuting: false,
        executionHistory: [historyEntry, ...prev.executionHistory].slice(0, 50) // Keep last 50
      }));
    } catch (error) {
      console.error('Workflow execution failed:', error);
      setState(prev => ({
        ...prev,
        isExecuting: false
      }));
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'partial':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getWorkflowParameters = () => {
    if (!selectedTemplate) {
      return [];
    }

    // Extract parameters from workflow steps
    const parameters = new Set<string>();
    for (const step of selectedTemplate.steps) {
      for (const [key, value] of Object.entries(step.input)) {
        if (typeof value === 'string' && value.startsWith('$')) {
          parameters.add(value.slice(1));
        }
      }
    }

    return Array.from(parameters);
  };

  const workflowParameters = getWorkflowParameters();

  return (
    <div className="space-y-6">
      {/* Workflow Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Workflow Selection
          </CardTitle>
          <CardDescription>Choose a workflow template to execute</CardDescription>
        </CardHeader>
        <CardContent>
          <Select value={state.selectedWorkflow || ''} onValueChange={handleWorkflowSelect}>
            <SelectTrigger>
              <SelectValue placeholder="Select a workflow..." />
            </SelectTrigger>
            <SelectContent>
              {workflowTemplates.map(template => (
                <SelectItem key={template.id} value={template.id}>
                  <div className="flex flex-col">
                    <span className="font-medium">{template.name}</span>
                    <span className="text-xs text-muted-foreground">{template.description}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {selectedTemplate && (
            <div className="mt-4 p-4 rounded border bg-muted/50">
              <h4 className="font-medium mb-2">{selectedTemplate.name}</h4>
              <p className="text-sm text-muted-foreground mb-3">{selectedTemplate.description}</p>
              <div className="flex gap-2 flex-wrap">
                <Badge variant="secondary">{selectedTemplate.steps.length} Steps</Badge>
                <Badge variant="secondary">v{selectedTemplate.version}</Badge>
                <Badge variant="secondary">On Error: {selectedTemplate.onError}</Badge>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Parameters */}
      {selectedTemplate && workflowParameters.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Parameters
            </CardTitle>
            <CardDescription>Configure workflow parameters</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {workflowParameters.map(param => (
              <div key={param} className="space-y-2">
                <Label htmlFor={param}>{param.charAt(0).toUpperCase() + param.slice(1)}</Label>
                {param.includes('query') || param.includes('problem') || param.includes('theorem') ? (
                  <Textarea
                    id={param}
                    placeholder={`Enter ${param}...`}
                    value={(state.parameters[param] as string) || ''}
                    onChange={(e) => handleParameterChange(param, e.target.value)}
                    rows={3}
                  />
                ) : (
                  <Input
                    id={param}
                    placeholder={`Enter ${param}...`}
                    value={(state.parameters[param] as string) || ''}
                    onChange={(e) => handleParameterChange(param, e.target.value)}
                  />
                )}
              </div>
            ))}

            <div className="flex gap-2">
              <Button
                onClick={executeWorkflow}
                disabled={state.isExecuting}
                className="flex-1"
              >
                {state.isExecuting ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Executing...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Execute Workflow
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {state.executionResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {getStatusIcon(state.executionResult.status)}
              Execution Results
            </CardTitle>
            <CardDescription>
              {state.executionResult.status === 'completed'
                ? 'Workflow completed successfully'
                : state.executionResult.status === 'failed'
                ? 'Workflow execution failed'
                : 'Workflow partially completed'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="summary">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="summary">Summary</TabsTrigger>
                <TabsTrigger value="steps">Step Results</TabsTrigger>
                <TabsTrigger value="output">Output</TabsTrigger>
              </TabsList>

              <TabsContent value="summary" className="space-y-4">
                <div className="grid gap-3 md:grid-cols-3">
                  <div className="rounded border p-3">
                    <div className="text-xs text-muted-foreground">Status</div>
                    <div className="text-lg font-semibold capitalize">{state.executionResult.status}</div>
                  </div>
                  <div className="rounded border p-3">
                    <div className="text-xs text-muted-foreground">Duration</div>
                    <div className="text-lg font-semibold">{state.executionResult.duration}ms</div>
                  </div>
                  <div className="rounded border p-3">
                    <div className="text-xs text-muted-foreground">Steps</div>
                    <div className="text-lg font-semibold">{state.executionResult.stepResults.size}</div>
                  </div>
                </div>

                {state.executionResult.errors.length > 0 && (
                  <div className="rounded border border-red-200 bg-red-50 p-3">
                    <div className="text-sm font-medium text-red-800 mb-2">Errors</div>
                    <ul className="text-sm text-red-700 space-y-1">
                      {state.executionResult.errors.map((error, idx) => (
                        <li key={idx}>
                          <span className="font-medium">{error.stepId}:</span> {error.error}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="steps" className="space-y-3">
                {Array.from(state.executionResult.stepResults.entries()).map(([stepId, result]) => {
                  const step = selectedTemplate?.steps.find(s => s.id === stepId);
                  const resultAny = result as any;
                  return (
                    <Card key={stepId}>
                      <CardHeader>
                        <CardTitle className="text-sm flex items-center gap-2">
                          {resultAny.success ? (
                            <CheckCircle className="h-4 w-4 text-green-500" />
                          ) : (
                            <XCircle className="h-4 w-4 text-red-500" />
                          )}
                          {step?.name || stepId}
                        </CardTitle>
                        {resultAny.duration && (
                          <CardDescription>{resultAny.duration}ms</CardDescription>
                        )}
                      </CardHeader>
                      <CardContent>
                        <pre className="text-xs overflow-auto max-h-40">
                          {JSON.stringify(resultAny.output || resultAny.error, null, 2)}
                        </pre>
                      </CardContent>
                    </Card>
                  );
                })}
              </TabsContent>

              <TabsContent value="output">
                <pre className="rounded border p-4 text-xs overflow-auto max-h-96">
                  {JSON.stringify(state.executionResult.results, null, 2)}
                </pre>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}

      {/* Execution History */}
      <Card>
        <CardHeader>
          <CardTitle>Execution History</CardTitle>
          <CardDescription>Recent workflow executions</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {state.executionHistory.length === 0 ? (
              <p className="text-sm text-muted-foreground">No executions yet</p>
            ) : (
              state.executionHistory.map(entry => {
                const template = getWorkflowTemplate(entry.workflowId);
                return (
                  <div key={entry.id} className="flex items-center justify-between p-3 rounded border">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(entry.status)}
                      <div>
                        <div className="font-medium">{template?.name || entry.workflowId}</div>
                        <div className="text-xs text-muted-foreground">
                          {entry.timestamp.toLocaleString()}
                        </div>
                      </div>
                    </div>
                    <Badge variant="outline" className="capitalize">
                      {entry.status}
                    </Badge>
                  </div>
                );
              })
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
