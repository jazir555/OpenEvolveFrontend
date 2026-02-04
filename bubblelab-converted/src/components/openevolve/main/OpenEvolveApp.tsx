import React, { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { EvolutionTab } from './EvolutionTab';
import { AdversarialTestingTab } from './AdversarialTestingTab';
import { GithubIntegrationTab } from './GithubIntegrationTab';
import { ActivityFeedTab } from './ActivityFeedTab';
import { ReportTemplatesTab } from './ReportTemplatesTab';
import { ReportingDashboardTab } from './ReportingDashboardTab';
import { ModelDashboardTab } from './ModelDashboardTab';
import { TasksTab } from './TasksTab';
import { AdminTab } from './AdminTab';
import { AnalyticsDashboardTab } from './AnalyticsDashboardTab';
import { AnalyticsMonitoringTab } from './AnalyticsMonitoringTab';
import { OpenEvolveDashboardTab } from './OpenEvolveDashboardTab';
import { OpenEvolveVisualizationTab } from './OpenEvolveVisualizationTab';
import { OrchestratorTab } from './OrchestratorTab';
import { MonitoringTab } from './MonitoringTab';
import { SystemMonitoringTab } from './SystemMonitoringTab';
import { SgdMonitoringTab } from './SgdMonitoringTab';
import { TeamManagerTab } from './TeamManagerTab';
import { GauntletDesignerTab } from './GauntletDesignerTab';
import { KnowledgeBaseTab } from './KnowledgeBaseTab';
import { AutoApprovalTab } from './AutoApprovalTab';
import { WorkflowTemplatesTab } from './WorkflowTemplatesTab';
import { WorkflowVisualizationTab } from './WorkflowVisualizationTab';
import { NotificationsTab } from './NotificationsTab';
import { SuggestionsTab } from './SuggestionsTab';
import { SettingsTab } from './SettingsTab';
import { SovereignDashboardTab } from './SovereignDashboardTab';
import { CollaborationTab } from './CollaborationTab';
import { DependencyGraphTab } from './DependencyGraphTab';
import { PromptManagerTab } from './PromptManagerTab';
import { ContentManagerTab } from './ContentManagerTab';
import { ExportImportTab } from './ExportImportTab';
import { EvaluatorHubTab } from './EvaluatorHubTab';
import { DecompositionReviewTab } from './DecompositionReviewTab';
import { IntegratedWorkflowTab } from './IntegratedWorkflowTab';
import { ConfigurationTab } from './ConfigurationTab';
import { RbacTab } from './RbacTab';
import { ModelOrchestrationTab } from './ModelOrchestrationTab';
import { ResourceManagerTab } from './ResourceManagerTab';
import { BubbleLabsIntegrationTab } from './BubbleLabsIntegrationTab';
import { MakerStudioTab } from './MakerStudioTab';
import { KnowledgeExplorerTab } from './KnowledgeExplorerTab';
import { LeanAideTab } from './LeanAideTab';
import { Header } from './Header';
import { Sidebar } from './Sidebar';

interface OpenEvolveAppState {
  protocolText: string;
  evolutionRunning: boolean;
  adversarialRunning: boolean;
  evolutionHistory: any[];
  adversarialResults: any;
  evolutionCurrentBest: string;
  evolutionStatusMessage: string;
  adversarialStatusMessage: string;
  evolutionBestScore: number;
  // Add other state properties as needed
}

export const OpenEvolveApp: React.FC = () => {
  const [activeTab, setActiveTab] = useState('evolution');
  const [state, setState] = useState<OpenEvolveAppState>({
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
  useEffect(() => {
    const savedState = localStorage.getItem('openevolve-state');
    if (savedState) {
      try {
        setState(JSON.parse(savedState));
      } catch (e) {
        console.error('Failed to parse saved state', e);
      }
    }
  }, []);

  // Save state to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('openevolve-state', JSON.stringify(state));
  }, [state]);

  const updateState = (updates: Partial<OpenEvolveAppState>) => {
    setState(prev => ({ ...prev, ...updates }));
  };

  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-7xl mx-auto">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="flex flex-wrap gap-2">
                <TabsTrigger value="evolution">Evolution</TabsTrigger>
                <TabsTrigger value="adversarial">Adversarial Testing</TabsTrigger>
                <TabsTrigger value="github">GitHub</TabsTrigger>
                <TabsTrigger value="activity">Activity Feed</TabsTrigger>
                <TabsTrigger value="reports">Report Templates</TabsTrigger>
                <TabsTrigger value="reporting-dashboard">Reporting Dashboard</TabsTrigger>
                <TabsTrigger value="models">Model Dashboard</TabsTrigger>
                <TabsTrigger value="tasks">Tasks</TabsTrigger>
                <TabsTrigger value="admin">Admin</TabsTrigger>
                <TabsTrigger value="teams">Teams</TabsTrigger>
                <TabsTrigger value="gauntlets">Gauntlets</TabsTrigger>
                <TabsTrigger value="analytics">Analytics</TabsTrigger>
                <TabsTrigger value="analytics-monitoring">Analytics Monitoring</TabsTrigger>
                <TabsTrigger value="openevolve">OpenEvolve</TabsTrigger>
                <TabsTrigger value="openevolve-visualization">OpenEvolve Visualization</TabsTrigger>
                <TabsTrigger value="orchestrator">Orchestrator</TabsTrigger>
                <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
                <TabsTrigger value="system-monitoring">System Monitoring</TabsTrigger>
                <TabsTrigger value="sgd-monitoring">SGD Monitoring</TabsTrigger>
                <TabsTrigger value="collaboration">Collaboration</TabsTrigger>
                <TabsTrigger value="dependencies">Dependencies</TabsTrigger>
                <TabsTrigger value="knowledge">Knowledge Base</TabsTrigger>
                <TabsTrigger value="auto-approval">Auto Approval</TabsTrigger>
                <TabsTrigger value="workflow-templates">Workflow Templates</TabsTrigger>
                <TabsTrigger value="workflow-visualization">Workflow Visualization</TabsTrigger>
                <TabsTrigger value="notifications">Notifications</TabsTrigger>
                <TabsTrigger value="suggestions">Suggestions</TabsTrigger>
                <TabsTrigger value="settings">Settings</TabsTrigger>
                <TabsTrigger value="sovereign">Sovereign</TabsTrigger>
                <TabsTrigger value="prompts">Prompts</TabsTrigger>
                <TabsTrigger value="content">Content Tools</TabsTrigger>
                <TabsTrigger value="export-import">Export / Import</TabsTrigger>
                <TabsTrigger value="evaluators">Evaluator Hub</TabsTrigger>
                <TabsTrigger value="decomposition-review">Decomposition Review</TabsTrigger>
                <TabsTrigger value="integrated-workflow">Integrated Workflow</TabsTrigger>
                <TabsTrigger value="configuration">Configuration</TabsTrigger>
                <TabsTrigger value="rbac">RBAC</TabsTrigger>
                <TabsTrigger value="model-orchestration">Model Orchestration</TabsTrigger>
                <TabsTrigger value="resource-manager">Resource Manager</TabsTrigger>
                <TabsTrigger value="bubblelabs">BubbleLabs</TabsTrigger>
                <TabsTrigger value="maker-studio">Maker Studio</TabsTrigger>
                <TabsTrigger value="knowledge-explorer">Knowledge Explorer</TabsTrigger>
                <TabsTrigger value="leanaide">LeanAide</TabsTrigger>
              </TabsList>
              
              <TabsContent value="evolution" className="mt-6">
                <EvolutionTab 
                  state={state} 
                  updateState={updateState} 
                />
              </TabsContent>
              
              <TabsContent value="adversarial" className="mt-6">
                <AdversarialTestingTab 
                  state={state} 
                  updateState={updateState} 
                />
              </TabsContent>
              
              <TabsContent value="github" className="mt-6">
                <GithubIntegrationTab />
              </TabsContent>
              
              <TabsContent value="activity" className="mt-6">
                <ActivityFeedTab />
              </TabsContent>
              
              <TabsContent value="reports" className="mt-6">
                <ReportTemplatesTab />
              </TabsContent>

              <TabsContent value="reporting-dashboard" className="mt-6">
                <ReportingDashboardTab />
              </TabsContent>
              
              <TabsContent value="models" className="mt-6">
                <ModelDashboardTab />
              </TabsContent>
              
              <TabsContent value="tasks" className="mt-6">
                <TasksTab />
              </TabsContent>
              
              <TabsContent value="admin" className="mt-6">
                <AdminTab />
              </TabsContent>

              <TabsContent value="teams" className="mt-6">
                <TeamManagerTab />
              </TabsContent>

              <TabsContent value="gauntlets" className="mt-6">
                <GauntletDesignerTab />
              </TabsContent>
              
              <TabsContent value="analytics" className="mt-6">
                <AnalyticsDashboardTab />
              </TabsContent>

              <TabsContent value="analytics-monitoring" className="mt-6">
                <AnalyticsMonitoringTab />
              </TabsContent>
              
              <TabsContent value="openevolve" className="mt-6">
                <OpenEvolveDashboardTab />
              </TabsContent>

              <TabsContent value="openevolve-visualization" className="mt-6">
                <OpenEvolveVisualizationTab />
              </TabsContent>
              
              <TabsContent value="orchestrator" className="mt-6">
                <OrchestratorTab />
              </TabsContent>
              
              <TabsContent value="monitoring" className="mt-6">
                <MonitoringTab />
              </TabsContent>

              <TabsContent value="system-monitoring" className="mt-6">
                <SystemMonitoringTab />
              </TabsContent>

              <TabsContent value="sgd-monitoring" className="mt-6">
                <SgdMonitoringTab />
              </TabsContent>

              <TabsContent value="collaboration" className="mt-6">
                <CollaborationTab />
              </TabsContent>

              <TabsContent value="dependencies" className="mt-6">
                <DependencyGraphTab />
              </TabsContent>

              <TabsContent value="knowledge" className="mt-6">
                <KnowledgeBaseTab />
              </TabsContent>

              <TabsContent value="auto-approval" className="mt-6">
                <AutoApprovalTab />
              </TabsContent>

              <TabsContent value="workflow-templates" className="mt-6">
                <WorkflowTemplatesTab />
              </TabsContent>

              <TabsContent value="workflow-visualization" className="mt-6">
                <WorkflowVisualizationTab />
              </TabsContent>

              <TabsContent value="notifications" className="mt-6">
                <NotificationsTab />
              </TabsContent>

              <TabsContent value="suggestions" className="mt-6">
                <SuggestionsTab />
              </TabsContent>

              <TabsContent value="settings" className="mt-6">
                <SettingsTab />
              </TabsContent>

              <TabsContent value="sovereign" className="mt-6">
                <SovereignDashboardTab />
              </TabsContent>

              <TabsContent value="prompts" className="mt-6">
                <PromptManagerTab />
              </TabsContent>

              <TabsContent value="content" className="mt-6">
                <ContentManagerTab />
              </TabsContent>

              <TabsContent value="export-import" className="mt-6">
                <ExportImportTab state={state} updateState={updateState} />
              </TabsContent>

              <TabsContent value="evaluators" className="mt-6">
                <EvaluatorHubTab />
              </TabsContent>

              <TabsContent value="decomposition-review" className="mt-6">
                <DecompositionReviewTab />
              </TabsContent>

              <TabsContent value="integrated-workflow" className="mt-6">
                <IntegratedWorkflowTab />
              </TabsContent>

              <TabsContent value="configuration" className="mt-6">
                <ConfigurationTab />
              </TabsContent>

              <TabsContent value="rbac" className="mt-6">
                <RbacTab />
              </TabsContent>

              <TabsContent value="model-orchestration" className="mt-6">
                <ModelOrchestrationTab />
              </TabsContent>

              <TabsContent value="resource-manager" className="mt-6">
                <ResourceManagerTab />
              </TabsContent>

              <TabsContent value="bubblelabs" className="mt-6">
                <BubbleLabsIntegrationTab />
              </TabsContent>

              <TabsContent value="maker-studio" className="mt-6">
                <MakerStudioTab />
              </TabsContent>

              <TabsContent value="knowledge-explorer" className="mt-6">
                <KnowledgeExplorerTab />
              </TabsContent>

              <TabsContent value="leanaide" className="mt-6">
                <LeanAideTab />
              </TabsContent>
            </Tabs>
          </div>
        </main>
      </div>
    </div>
  );
};
