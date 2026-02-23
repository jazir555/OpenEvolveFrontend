"use strict";
/**
 * Workflow Execution Tab
 *
 * UI for executing and managing workflows across multiple plugins.
 * Provides workflow selection, parameter input, execution monitoring, and result visualization.
 */
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
exports.WorkflowExecutionTab = void 0;
const react_1 = __importStar(require("react"));
const card_1 = require("@/components/ui/card");
const button_1 = require("@/components/ui/button");
const input_1 = require("@/components/ui/input");
const textarea_1 = require("@/components/ui/textarea");
const label_1 = require("@/components/ui/label");
const badge_1 = require("@/components/ui/badge");
const tabs_1 = require("@/components/ui/tabs");
const select_1 = require("@/components/ui/select");
const lucide_react_1 = require("lucide-react");
const workflow_orchestrator_1 = require("../../../../../../../orchestration/workflow-system/workflow-orchestrator");
const plugin_registry_1 = require("../../../../../../../orchestration/workflow-system/plugin-registry");
const workflow_templates_1 = require("../../../../../../../orchestration/workflow-system/workflow-templates");
const WorkflowExecutionTab = () => {
    const [state, setState] = (0, react_1.useState)({
        selectedWorkflow: null,
        parameters: {},
        executionResult: null,
        isExecuting: false,
        activeExecutions: [],
        executionHistory: []
    });
    const [workflowTemplates] = (0, react_1.useState)(() => (0, workflow_templates_1.getAllWorkflowTemplates)());
    const [selectedTemplate, setSelectedTemplate] = (0, react_1.useState)(null);
    const orchestrator = (0, workflow_orchestrator_1.getWorkflowOrchestrator)();
    const registry = (0, plugin_registry_1.getPluginRegistry)();
    // Load execution history from localStorage
    (0, react_1.useEffect)(() => {
        try {
            const savedHistory = localStorage.getItem('workflow-execution-history');
            if (savedHistory) {
                const parsed = JSON.parse(savedHistory);
                setState(prev => ({
                    ...prev,
                    executionHistory: parsed.map((h) => ({
                        ...h,
                        timestamp: new Date(h.timestamp)
                    }))
                }));
            }
        }
        catch (error) {
            console.error('Failed to load execution history:', error);
        }
    }, []);
    // Save execution history to localStorage
    (0, react_1.useEffect)(() => {
        if (state.executionHistory.length > 0) {
            localStorage.setItem('workflow-execution-history', JSON.stringify(state.executionHistory));
        }
    }, [state.executionHistory]);
    const handleWorkflowSelect = (workflowId) => {
        const template = (0, workflow_templates_1.getWorkflowTemplate)(workflowId);
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
    const handleParameterChange = (key, value) => {
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
            const result = await orchestrator.executeWorkflow(selectedTemplate, state.parameters);
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
        }
        catch (error) {
            console.error('Workflow execution failed:', error);
            setState(prev => ({
                ...prev,
                isExecuting: false
            }));
        }
    };
    const getStatusIcon = (status) => {
        switch (status) {
            case 'completed':
                return <lucide_react_1.CheckCircle className="h-4 w-4 text-green-500"/>;
            case 'failed':
                return <lucide_react_1.XCircle className="h-4 w-4 text-red-500"/>;
            case 'partial':
                return <lucide_react_1.Clock className="h-4 w-4 text-yellow-500"/>;
            default:
                return <lucide_react_1.Clock className="h-4 w-4 text-gray-500"/>;
        }
    };
    const getWorkflowParameters = () => {
        if (!selectedTemplate) {
            return [];
        }
        // Extract parameters from workflow steps
        const parameters = new Set();
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
    return (<div className="space-y-6">
      {/* Workflow Selection */}
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle className="flex items-center gap-2">
            <lucide_react_1.FileText className="h-5 w-5"/>
            Workflow Selection
          </card_1.CardTitle>
          <card_1.CardDescription>Choose a workflow template to execute</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent>
          <select_1.Select value={state.selectedWorkflow || ''} onValueChange={handleWorkflowSelect}>
            <select_1.SelectTrigger>
              <select_1.SelectValue placeholder="Select a workflow..."/>
            </select_1.SelectTrigger>
            <select_1.SelectContent>
              {workflowTemplates.map(template => (<select_1.SelectItem key={template.id} value={template.id}>
                  <div className="flex flex-col">
                    <span className="font-medium">{template.name}</span>
                    <span className="text-xs text-muted-foreground">{template.description}</span>
                  </div>
                </select_1.SelectItem>))}
            </select_1.SelectContent>
          </select_1.Select>

          {selectedTemplate && (<div className="mt-4 p-4 rounded border bg-muted/50">
              <h4 className="font-medium mb-2">{selectedTemplate.name}</h4>
              <p className="text-sm text-muted-foreground mb-3">{selectedTemplate.description}</p>
              <div className="flex gap-2 flex-wrap">
                <badge_1.Badge variant="secondary">{selectedTemplate.steps.length} Steps</badge_1.Badge>
                <badge_1.Badge variant="secondary">v{selectedTemplate.version}</badge_1.Badge>
                <badge_1.Badge variant="secondary">On Error: {selectedTemplate.onError}</badge_1.Badge>
              </div>
            </div>)}
        </card_1.CardContent>
      </card_1.Card>

      {/* Parameters */}
      {selectedTemplate && workflowParameters.length > 0 && (<card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="flex items-center gap-2">
              <lucide_react_1.Settings className="h-5 w-5"/>
              Parameters
            </card_1.CardTitle>
            <card_1.CardDescription>Configure workflow parameters</card_1.CardDescription>
          </card_1.CardHeader>
          <card_1.CardContent className="space-y-4">
            {workflowParameters.map(param => (<div key={param} className="space-y-2">
                <label_1.Label htmlFor={param}>{param.charAt(0).toUpperCase() + param.slice(1)}</label_1.Label>
                {param.includes('query') || param.includes('problem') || param.includes('theorem') ? (<textarea_1.Textarea id={param} placeholder={`Enter ${param}...`} value={state.parameters[param] || ''} onChange={(e) => handleParameterChange(param, e.target.value)} rows={3}/>) : (<input_1.Input id={param} placeholder={`Enter ${param}...`} value={state.parameters[param] || ''} onChange={(e) => handleParameterChange(param, e.target.value)}/>)}
              </div>))}

            <div className="flex gap-2">
              <button_1.Button onClick={executeWorkflow} disabled={state.isExecuting} className="flex-1">
                {state.isExecuting ? (<>
                    <lucide_react_1.RefreshCw className="h-4 w-4 mr-2 animate-spin"/>
                    Executing...
                  </>) : (<>
                    <lucide_react_1.Play className="h-4 w-4 mr-2"/>
                    Execute Workflow
                  </>)}
              </button_1.Button>
            </div>
          </card_1.CardContent>
        </card_1.Card>)}

      {/* Results */}
      {state.executionResult && (<card_1.Card>
          <card_1.CardHeader>
            <card_1.CardTitle className="flex items-center gap-2">
              {getStatusIcon(state.executionResult.status)}
              Execution Results
            </card_1.CardTitle>
            <card_1.CardDescription>
              {state.executionResult.status === 'completed'
                ? 'Workflow completed successfully'
                : state.executionResult.status === 'failed'
                    ? 'Workflow execution failed'
                    : 'Workflow partially completed'}
            </card_1.CardDescription>
          </card_1.CardHeader>
          <card_1.CardContent>
            <tabs_1.Tabs defaultValue="summary">
              <tabs_1.TabsList className="grid w-full grid-cols-3">
                <tabs_1.TabsTrigger value="summary">Summary</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="steps">Step Results</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="output">Output</tabs_1.TabsTrigger>
              </tabs_1.TabsList>

              <tabs_1.TabsContent value="summary" className="space-y-4">
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

                {state.executionResult.errors.length > 0 && (<div className="rounded border border-red-200 bg-red-50 p-3">
                    <div className="text-sm font-medium text-red-800 mb-2">Errors</div>
                    <ul className="text-sm text-red-700 space-y-1">
                      {state.executionResult.errors.map((error, idx) => (<li key={idx}>
                          <span className="font-medium">{error.stepId}:</span> {error.error}
                        </li>))}
                    </ul>
                  </div>)}
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="steps" className="space-y-3">
                {Array.from(state.executionResult.stepResults.entries()).map(([stepId, result]) => {
                const step = selectedTemplate?.steps.find(s => s.id === stepId);
                const resultAny = result;
                return (<card_1.Card key={stepId}>
                      <card_1.CardHeader>
                        <card_1.CardTitle className="text-sm flex items-center gap-2">
                          {resultAny.success ? (<lucide_react_1.CheckCircle className="h-4 w-4 text-green-500"/>) : (<lucide_react_1.XCircle className="h-4 w-4 text-red-500"/>)}
                          {step?.name || stepId}
                        </card_1.CardTitle>
                        {resultAny.duration && (<card_1.CardDescription>{resultAny.duration}ms</card_1.CardDescription>)}
                      </card_1.CardHeader>
                      <card_1.CardContent>
                        <pre className="text-xs overflow-auto max-h-40">
                          {JSON.stringify(resultAny.output || resultAny.error, null, 2)}
                        </pre>
                      </card_1.CardContent>
                    </card_1.Card>);
            })}
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="output">
                <pre className="rounded border p-4 text-xs overflow-auto max-h-96">
                  {JSON.stringify(state.executionResult.results, null, 2)}
                </pre>
              </tabs_1.TabsContent>
            </tabs_1.Tabs>
          </card_1.CardContent>
        </card_1.Card>)}

      {/* Execution History */}
      <card_1.Card>
        <card_1.CardHeader>
          <card_1.CardTitle>Execution History</card_1.CardTitle>
          <card_1.CardDescription>Recent workflow executions</card_1.CardDescription>
        </card_1.CardHeader>
        <card_1.CardContent>
          <div className="space-y-2">
            {state.executionHistory.length === 0 ? (<p className="text-sm text-muted-foreground">No executions yet</p>) : (state.executionHistory.map(entry => {
            const template = (0, workflow_templates_1.getWorkflowTemplate)(entry.workflowId);
            return (<div key={entry.id} className="flex items-center justify-between p-3 rounded border">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(entry.status)}
                      <div>
                        <div className="font-medium">{template?.name || entry.workflowId}</div>
                        <div className="text-xs text-muted-foreground">
                          {entry.timestamp.toLocaleString()}
                        </div>
                      </div>
                    </div>
                    <badge_1.Badge variant="outline" className="capitalize">
                      {entry.status}
                    </badge_1.Badge>
                  </div>);
        }))}
          </div>
        </card_1.CardContent>
      </card_1.Card>
    </div>);
};
exports.WorkflowExecutionTab = WorkflowExecutionTab;
//# sourceMappingURL=WorkflowExecutionTab.js.map