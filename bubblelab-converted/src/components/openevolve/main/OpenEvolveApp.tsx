import React, { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { EvolutionTab } from './EvolutionTab';
import { AdversarialTestingTab } from './AdversarialTestingTab';
import { GithubIntegrationTab } from './GithubIntegrationTab';
import { ActivityFeedTab } from './ActivityFeedTab';
import { ReportTemplatesTab } from './ReportTemplatesTab';
import { ModelDashboardTab } from './ModelDashboardTab';
import { TasksTab } from './TasksTab';
import { AdminTab } from './AdminTab';
import { AnalyticsDashboardTab } from './AnalyticsDashboardTab';
import { OpenEvolveDashboardTab } from './OpenEvolveDashboardTab';
import { OrchestratorTab } from './OrchestratorTab';
import { MonitoringTab } from './MonitoringTab';
import { KnowledgeBaseTab } from './KnowledgeBaseTab';
import { AutoApprovalTab } from './AutoApprovalTab';
import { WorkflowTemplatesTab } from './WorkflowTemplatesTab';
import { NotificationsTab } from './NotificationsTab';
import { SuggestionsTab } from './SuggestionsTab';
import { SettingsTab } from './SettingsTab';
import { SovereignDashboardTab } from './SovereignDashboardTab';
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
                <TabsTrigger value="models">Model Dashboard</TabsTrigger>
                <TabsTrigger value="tasks">Tasks</TabsTrigger>
                <TabsTrigger value="admin">Admin</TabsTrigger>
                <TabsTrigger value="analytics">Analytics</TabsTrigger>
                <TabsTrigger value="openevolve">OpenEvolve</TabsTrigger>
                <TabsTrigger value="orchestrator">Orchestrator</TabsTrigger>
                <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
                <TabsTrigger value="knowledge">Knowledge Base</TabsTrigger>
                <TabsTrigger value="auto-approval">Auto Approval</TabsTrigger>
                <TabsTrigger value="workflow-templates">Workflow Templates</TabsTrigger>
                <TabsTrigger value="notifications">Notifications</TabsTrigger>
                <TabsTrigger value="suggestions">Suggestions</TabsTrigger>
                <TabsTrigger value="settings">Settings</TabsTrigger>
                <TabsTrigger value="sovereign">Sovereign</TabsTrigger>
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
              
              <TabsContent value="models" className="mt-6">
                <ModelDashboardTab />
              </TabsContent>
              
              <TabsContent value="tasks" className="mt-6">
                <TasksTab />
              </TabsContent>
              
              <TabsContent value="admin" className="mt-6">
                <AdminTab />
              </TabsContent>
              
              <TabsContent value="analytics" className="mt-6">
                <AnalyticsDashboardTab />
              </TabsContent>
              
              <TabsContent value="openevolve" className="mt-6">
                <OpenEvolveDashboardTab />
              </TabsContent>
              
              <TabsContent value="orchestrator" className="mt-6">
                <OrchestratorTab />
              </TabsContent>
              
              <TabsContent value="monitoring" className="mt-6">
                <MonitoringTab />
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
            </Tabs>
          </div>
        </main>
      </div>
    </div>
  );
};
