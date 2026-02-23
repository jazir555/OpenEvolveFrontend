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
exports.OpenEvolveApp = void 0;
const react_1 = __importStar(require("react"));
const tabs_1 = require("@/components/ui/tabs");
const EvolutionTab_1 = require("./EvolutionTab");
const AdversarialTestingTab_1 = require("./AdversarialTestingTab");
const GithubIntegrationTab_1 = require("./GithubIntegrationTab");
const ActivityFeedTab_1 = require("./ActivityFeedTab");
const ReportTemplatesTab_1 = require("./ReportTemplatesTab");
const ReportingDashboardTab_1 = require("./ReportingDashboardTab");
const ModelDashboardTab_1 = require("./ModelDashboardTab");
const TasksTab_1 = require("./TasksTab");
const AdminTab_1 = require("./AdminTab");
const AnalyticsDashboardTab_1 = require("./AnalyticsDashboardTab");
const AnalyticsMonitoringTab_1 = require("./AnalyticsMonitoringTab");
const OpenEvolveDashboardTab_1 = require("./OpenEvolveDashboardTab");
const OpenEvolveVisualizationTab_1 = require("./OpenEvolveVisualizationTab");
const OrchestratorTab_1 = require("./OrchestratorTab");
const MonitoringTab_1 = require("./MonitoringTab");
const SystemMonitoringTab_1 = require("./SystemMonitoringTab");
const SgdMonitoringTab_1 = require("./SgdMonitoringTab");
const TeamManagerTab_1 = require("./TeamManagerTab");
const GauntletDesignerTab_1 = require("./GauntletDesignerTab");
const KnowledgeBaseTab_1 = require("./KnowledgeBaseTab");
const AutoApprovalTab_1 = require("./AutoApprovalTab");
const WorkflowTemplatesTab_1 = require("./WorkflowTemplatesTab");
const WorkflowVisualizationTab_1 = require("./WorkflowVisualizationTab");
const NotificationsTab_1 = require("./NotificationsTab");
const SuggestionsTab_1 = require("./SuggestionsTab");
const SettingsTab_1 = require("./SettingsTab");
const VersionControlTab_1 = require("./VersionControlTab");
const ValidationManagerTab_1 = require("./ValidationManagerTab");
const SovereignDashboardTab_1 = require("./SovereignDashboardTab");
const CollaborationTab_1 = require("./CollaborationTab");
const DependencyGraphTab_1 = require("./DependencyGraphTab");
const PromptManagerTab_1 = require("./PromptManagerTab");
const ContentManagerTab_1 = require("./ContentManagerTab");
const ExportImportTab_1 = require("./ExportImportTab");
const EvaluatorHubTab_1 = require("./EvaluatorHubTab");
const DecompositionReviewTab_1 = require("./DecompositionReviewTab");
const IntegratedWorkflowTab_1 = require("./IntegratedWorkflowTab");
const ConfigurationTab_1 = require("./ConfigurationTab");
const RbacTab_1 = require("./RbacTab");
const ModelOrchestrationTab_1 = require("./ModelOrchestrationTab");
const ResourceManagerTab_1 = require("./ResourceManagerTab");
const BubbleLabsIntegrationTab_1 = require("./BubbleLabsIntegrationTab");
const WorkflowLifecycleTab_1 = require("./WorkflowLifecycleTab");
const MakerStudioTab_1 = require("./MakerStudioTab");
const KnowledgeExplorerTab_1 = require("./KnowledgeExplorerTab");
const LeanAideTab_1 = require("./LeanAideTab");
const WorkflowExecutionTab_1 = require("./WorkflowExecutionTab");
const Header_1 = require("./Header");
const Sidebar_1 = require("./Sidebar");
const structuredLogger_1 = require("../../../../../../lib/structuredLogger");
const useBubbleLabIntegration_1 = require("../../hooks/useBubbleLabIntegration");
const OpenEvolveApp = () => {
    const [activeTab, setActiveTab] = (0, react_1.useState)('evolution');
    // Initialize BubbleLab integration
    const { isInitialized: integrationReady, error: integrationError } = (0, useBubbleLabIntegration_1.useBubbleLabIntegration)({
        ragbits: {
            serverUrl: process.env.NEXT_PUBLIC_RAGBITS_URL || 'http://localhost:3000/ragbits',
            enabled: true
        },
        datapizza: {
            serverUrl: process.env.NEXT_PUBLIC_DATAPIZZA_URL || 'http://localhost:3000/datapizza',
            enabled: true
        },
        autoStart: true
    });
    const [state, setState] = (0, react_1.useState)({
        protocolText: '# Sample Protocol\n\nThis is a sample protocol for testing purposes.',
        evolutionRunning: false,
        adversarialRunning: false,
        evolutionHistory: [],
        adversarialResults: null,
        evolutionCurrentBest: '',
        evolutionStatusMessage: '',
        adversarialStatusMessage: '',
        evolutionBestScore: 0,
    });
    // Initialize state from localStorage or default values
    (0, react_1.useEffect)(() => {
        const savedState = localStorage.getItem('openevolve-state');
        if (savedState) {
            try {
                setState(JSON.parse(savedState));
            }
            catch (e) {
                structuredLogger_1.apiLogger.error('Failed to parse saved state', e, {
                    component: 'OpenEvolveApp',
                    action: 'initialize_state'
                });
            }
        }
    }, []);
    // Save state to localStorage when it changes
    (0, react_1.useEffect)(() => {
        localStorage.setItem('openevolve-state', JSON.stringify(state));
    }, [state]);
    const updateState = (updates) => {
        setState(prev => ({ ...prev, ...updates }));
    };
    // Show loading state while integration initializes
    if (!integrationReady) {
        return (<div className="flex h-screen items-center justify-center bg-background">
        <div className="text-center space-y-4">
          <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto"/>
          <p className="text-sm text-muted-foreground">Initializing OpenEvolve...</p>
          {integrationError && (<p className="text-sm text-red-500">Error: {integrationError.message}</p>)}
        </div>
      </div>);
    }
    return (<div className="flex h-screen bg-background">
      <Sidebar_1.Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header_1.Header />
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-7xl mx-auto">
            <tabs_1.Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <tabs_1.TabsList className="flex flex-wrap gap-2">
                <tabs_1.TabsTrigger value="evolution">Evolution</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="adversarial">Adversarial Testing</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="github">GitHub</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="activity">Activity Feed</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="reports">Report Templates</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="reporting-dashboard">Reporting Dashboard</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="models">Model Dashboard</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="tasks">Tasks</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="admin">Admin</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="teams">Teams</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="gauntlets">Gauntlets</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="analytics">Analytics</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="analytics-monitoring">Analytics Monitoring</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="openevolve">OpenEvolve</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="openevolve-visualization">OpenEvolve Visualization</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="orchestrator">Orchestrator</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="monitoring">Monitoring</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="system-monitoring">System Monitoring</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="sgd-monitoring">SGD Monitoring</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="collaboration">Collaboration</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="dependencies">Dependencies</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="knowledge">Knowledge Base</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="auto-approval">Auto Approval</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="workflow-templates">Workflow Templates</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="workflow-visualization">Workflow Visualization</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="notifications">Notifications</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="suggestions">Suggestions</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="settings">Settings</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="version-control">Version Control</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="validation-manager">Validation Manager</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="sovereign">Sovereign</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="prompts">Prompts</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="content">Content Tools</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="export-import">Export / Import</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="evaluators">Evaluator Hub</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="decomposition-review">Decomposition Review</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="integrated-workflow">Integrated Workflow</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="configuration">Configuration</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="rbac">RBAC</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="model-orchestration">Model Orchestration</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="resource-manager">Resource Manager</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="bubblelabs">BubbleLabs</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="workflow-lifecycle">Workflow Lifecycle</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="maker-studio">Maker Studio</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="knowledge-explorer">Knowledge Explorer</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="leanaide">LeanAide</tabs_1.TabsTrigger>
                <tabs_1.TabsTrigger value="workflow-execution">Workflow Executor</tabs_1.TabsTrigger>
              </tabs_1.TabsList>
              
              <tabs_1.TabsContent value="evolution" className="mt-6">
                <EvolutionTab_1.EvolutionTab state={state} updateState={updateState}/>
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="adversarial" className="mt-6">
                <AdversarialTestingTab_1.AdversarialTestingTab state={state} updateState={updateState}/>
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="github" className="mt-6">
                <GithubIntegrationTab_1.GithubIntegrationTab />
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="activity" className="mt-6">
                <ActivityFeedTab_1.ActivityFeedTab />
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="reports" className="mt-6">
                <ReportTemplatesTab_1.ReportTemplatesTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="reporting-dashboard" className="mt-6">
                <ReportingDashboardTab_1.ReportingDashboardTab />
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="models" className="mt-6">
                <ModelDashboardTab_1.ModelDashboardTab />
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="tasks" className="mt-6">
                <TasksTab_1.TasksTab />
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="admin" className="mt-6">
                <AdminTab_1.AdminTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="teams" className="mt-6">
                <TeamManagerTab_1.TeamManagerTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="gauntlets" className="mt-6">
                <GauntletDesignerTab_1.GauntletDesignerTab />
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="analytics" className="mt-6">
                <AnalyticsDashboardTab_1.AnalyticsDashboardTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="analytics-monitoring" className="mt-6">
                <AnalyticsMonitoringTab_1.AnalyticsMonitoringTab />
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="openevolve" className="mt-6">
                <OpenEvolveDashboardTab_1.OpenEvolveDashboardTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="openevolve-visualization" className="mt-6">
                <OpenEvolveVisualizationTab_1.OpenEvolveVisualizationTab />
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="orchestrator" className="mt-6">
                <OrchestratorTab_1.OrchestratorTab />
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="monitoring" className="mt-6">
                <MonitoringTab_1.MonitoringTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="system-monitoring" className="mt-6">
                <SystemMonitoringTab_1.SystemMonitoringTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="sgd-monitoring" className="mt-6">
                <SgdMonitoringTab_1.SgdMonitoringTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="collaboration" className="mt-6">
                <CollaborationTab_1.CollaborationTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="dependencies" className="mt-6">
                <DependencyGraphTab_1.DependencyGraphTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="knowledge" className="mt-6">
                <KnowledgeBaseTab_1.KnowledgeBaseTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="auto-approval" className="mt-6">
                <AutoApprovalTab_1.AutoApprovalTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="workflow-templates" className="mt-6">
                <WorkflowTemplatesTab_1.WorkflowTemplatesTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="workflow-visualization" className="mt-6">
                <WorkflowVisualizationTab_1.WorkflowVisualizationTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="notifications" className="mt-6">
                <NotificationsTab_1.NotificationsTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="suggestions" className="mt-6">
                <SuggestionsTab_1.SuggestionsTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="settings" className="mt-6">
                <SettingsTab_1.SettingsTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="version-control" className="mt-6">
                <VersionControlTab_1.VersionControlTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="validation-manager" className="mt-6">
                <ValidationManagerTab_1.ValidationManagerTab />
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="sovereign" className="mt-6">
                <SovereignDashboardTab_1.SovereignDashboardTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="prompts" className="mt-6">
                <PromptManagerTab_1.PromptManagerTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="content" className="mt-6">
                <ContentManagerTab_1.ContentManagerTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="export-import" className="mt-6">
                <ExportImportTab_1.ExportImportTab state={state} updateState={updateState}/>
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="evaluators" className="mt-6">
                <EvaluatorHubTab_1.EvaluatorHubTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="decomposition-review" className="mt-6">
                <DecompositionReviewTab_1.DecompositionReviewTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="integrated-workflow" className="mt-6">
                <IntegratedWorkflowTab_1.IntegratedWorkflowTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="configuration" className="mt-6">
                <ConfigurationTab_1.ConfigurationTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="rbac" className="mt-6">
                <RbacTab_1.RbacTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="model-orchestration" className="mt-6">
                <ModelOrchestrationTab_1.ModelOrchestrationTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="resource-manager" className="mt-6">
                <ResourceManagerTab_1.ResourceManagerTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="bubblelabs" className="mt-6">
                <BubbleLabsIntegrationTab_1.BubbleLabsIntegrationTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="workflow-lifecycle" className="mt-6">
                <WorkflowLifecycleTab_1.WorkflowLifecycleTab />
              </tabs_1.TabsContent>
              
              <tabs_1.TabsContent value="maker-studio" className="mt-6">
                <MakerStudioTab_1.MakerStudioTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="knowledge-explorer" className="mt-6">
                <KnowledgeExplorerTab_1.KnowledgeExplorerTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="leanaide" className="mt-6">
                <LeanAideTab_1.LeanAideTab />
              </tabs_1.TabsContent>

              <tabs_1.TabsContent value="workflow-execution" className="mt-6">
                <WorkflowExecutionTab_1.WorkflowExecutionTab />
              </tabs_1.TabsContent>
            </tabs_1.Tabs>
          </div>
        </main>
      </div>
    </div>);
};
exports.OpenEvolveApp = OpenEvolveApp;
//# sourceMappingURL=OpenEvolveApp.js.map