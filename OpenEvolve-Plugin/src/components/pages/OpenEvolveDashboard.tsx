// @ts-nocheck
import { useMemo, useState } from 'react';
import { useWorkflows } from '@/services/hooks/useWorkflows';
import { useAuthStore } from '@/stores/authStore';
import { useWorkflowStore } from '@/stores/workflowStore';
import { WorkflowList } from '@/components/workflow/WorkflowList';
import { ExecutionMonitor } from '@/components/workflow/ExecutionMonitor';
import { WorkflowTabs } from '@/components/workflow/WorkflowTabs';
import { ProviderSettingsPanel } from '@/components/config/ProviderSettingsPanel';
import { EvolutionConfigPanel } from '@/components/config/EvolutionConfigPanel';
import { AdversarialConfigPanel } from '@/components/config/AdversarialConfigPanel';
import { DecompositionConfigPanel } from '@/components/config/DecompositionConfigPanel';
import { IntegrationConfigPanel } from '@/components/config/IntegrationConfigPanel';
import { KnowledgeConfigPanel } from '@/components/config/KnowledgeConfigPanel';
import { OpenEvolveConfigPanel } from '@/components/config/OpenEvolveConfigPanel';
import { SystemHealthPanel } from '@/components/shared/SystemHealthPanel';
import { WelcomeBanner } from '@/components/shared/WelcomeBanner';
import { VizErrorBoundary } from '@/components/shared/VizErrorBoundary';
import { PageErrorBoundary } from '@/components/shared/PageErrorBoundary';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';
import { useDecompositionHistory } from '@/hooks/useDecomposition';
import {
  ArtifactViz,
  DependencyViz,
  MakerViz,
  ProblemAnalysisViz,
  WorkflowMonitorViz,
} from 'openevolve-pygraphistry-plugin';
// @ts-nocheck

function OpenEvolveDashboardBase() {
  const { user } = useAuthStore();
  const { currentWorkflow, setCurrentWorkflow } = useWorkflowStore();
  const {
    workflows,
    isLoading,
    startWorkflow,
    pauseWorkflow,
    resumeWorkflow,
    stopWorkflow,
    deleteWorkflow,
    error: workflowsError,
  } = useWorkflows();

  const [showMonitor, setShowMonitor] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);
  const [decompositionConfig, setDecompositionConfig] = useState({
    strategy: 'hierarchical',
    maxDepth: 3,
    recursionDepthLimit: 1,
    maxSubProblems: 3,
    granularity: 'medium',
    minSubProblemSize: 50,
    maxSubProblemSize: 500,
    targetSubProblemSize: 200,
    parallelDecomposition: true,
    maxParallelTasks: 5,
    asyncDecomposition: false,
    pruningEnabled: true,
    pruningThreshold: 0.1,
    similarityThreshold: 0.8,
    mergeThreshold: 0.9,
    semanticSimilarity: 'cosine',
    embeddingModel: 'text-embedding-ada-002',
    clusteringAlgorithm: 'hierarchical',
    dependencyDetection: 'hybrid',
    circularDependencyHandling: 'merge',
    dependencyVisualization: false,
    timeHorizon: 24,
    timeGranularity: 'hours',
    temporalDependencies: true,
    cohesivenessTarget: 0.7,
    couplingLimit: 0.3,
    complexityThreshold: 0.8,
    validateDecomposition: true,
    testDecomposition: false,
    feedbackLoop: true,
    adaptiveDecomposition: false,
    learningEnabled: false,
    historicalData: false,
  });

  const { data: decompositionHistory } = useDecompositionHistory({ limit: 1, offset: 0 });

  const decompositionSummary = useMemo(() => {
    const latest = decompositionHistory?.[0];
    if (!latest) {
      return { subProblems: [], problemStatement: '' };
    }

    const subProblems = (latest.tasks || []).map((task) => ({
      id: task.task_id,
      title: task.title,
      description: task.description,
      status: task.status,
      complexity: task.estimated_effort,
      dependencies: task.dependencies,
    }));

    return { subProblems, problemStatement: latest.problem_statement || '' };
  }, [decompositionHistory]);

  const handleWorkflowSelect = (workflow: any) => {
    setCurrentWorkflow(workflow);
    setShowMonitor(true);
  };

  const handleExecute = async (evolutionId: string) => {
    const workflow = workflows.find((w) => w.evolution_id === evolutionId);
    if (!workflow) {
      return;
    }

    try {
      setPageError(null);
      await startWorkflow({
        content: workflow.initialContent || '',
        mode: 'standard',
        parameters: {},
        models: [],
      });
      setShowMonitor(true);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : 'Failed to start workflow.');
    }
  };

  const handlePause = async (evolutionId: string) => {
    try {
      setPageError(null);
      await pauseWorkflow(evolutionId);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : 'Failed to pause workflow.');
    }
  };

  const handleResume = async (evolutionId: string) => {
    try {
      setPageError(null);
      await resumeWorkflow(evolutionId);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : 'Failed to resume workflow.');
    }
  };

  const handleStop = async (evolutionId: string) => {
    try {
      setPageError(null);
      await stopWorkflow(evolutionId);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : 'Failed to stop workflow.');
    }
  };

  const handleDelete = async (evolutionId: string) => {
    try {
      setPageError(null);
      await deleteWorkflow(evolutionId);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : 'Failed to delete workflow.');
    }
  };

  const tabs = [
    {
      id: 'workflows',
      label: 'Workflows',
      content: (
        <div>
          <WorkflowList
            workflows={workflows}
            onWorkflowSelect={handleWorkflowSelect}
            onExecute={handleExecute}
            onPause={handlePause}
            onResume={handleResume}
            onStop={handleStop}
            onDelete={handleDelete}
          />
        </div>
      ),
    },
    {
      id: 'evolution',
      label: 'Evolution',
      content: (
        <EvolutionConfigPanel />
      ),
    },
    {
      id: 'adversarial',
      label: 'Adversarial',
      content: (
        <AdversarialConfigPanel />
      ),
    },
    {
      id: 'decomposition',
      label: 'Decomposition',
      content: (
        <div className="space-y-6">
          <DecompositionConfigPanel
            config={decompositionConfig}
            onConfigChange={setDecompositionConfig}
          />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <VizErrorBoundary label="Problem analysis visualization">
              <ProblemAnalysisViz initialText={decompositionSummary.problemStatement} />
            </VizErrorBoundary>
            <VizErrorBoundary label="Dependency visualization">
              <DependencyViz subProblems={decompositionSummary.subProblems} />
            </VizErrorBoundary>
          </div>
        </div>
      ),
    },
    {
      id: 'integrations',
      label: 'Integrations',
      content: (
        <IntegrationConfigPanel />
      ),
    },
    {
      id: 'knowledge',
      label: 'Knowledge',
      content: (
        <div className="space-y-6">
          <KnowledgeConfigPanel />
          <VizErrorBoundary label="Knowledge visualization">
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm min-h-[400px]">
               <ArtifactViz />
            </div>
          </VizErrorBoundary>
        </div>
      ),
    },
    {
      id: 'mdap-maker',
      label: 'MDAP/MAKER',
      content: (
        <div className="space-y-6">
          <OpenEvolveConfigPanel />
          <VizErrorBoundary label="MDAP/MAKER visualization">
            <MakerViz />
          </VizErrorBoundary>
        </div>
      ),
    },
  ];

  return (
    <PageErrorBoundary label="OpenEvolve dashboard">
      <div className="flex h-screen bg-slate-50">
      <ProviderSettingsPanel />

      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="border-b border-slate-200 bg-white px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">OpenEvolve Dashboard</h1>
              <p className="text-sm text-slate-600">Welcome back, {user?.name || 'User'}</p>
            </div>
            <div className="text-sm text-slate-500">{workflows.length} workflows</div>
          </div>
        </header>

        <div className="flex-1 overflow-auto p-6 space-y-6">
          <WelcomeBanner userName={user?.name} workflowCount={workflows.length} />

          {(pageError || workflowsError) && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {pageError || (workflowsError instanceof Error ? workflowsError.message : 'Failed to load workflows.')}
            </div>
          )}

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2 space-y-6">
              {showMonitor && currentWorkflow && (
                <ExecutionMonitor executionId={currentWorkflow.evolution_id} />
              )}
              {showMonitor && currentWorkflow && (
                <div className="rounded-2xl border border-slate-200 bg-white p-4">
                  <VizErrorBoundary label="Workflow monitor visualization">
                    <WorkflowMonitorViz workflowId={currentWorkflow.evolution_id} />
                  </VizErrorBoundary>
                </div>
              )}
            </div>
            <SystemHealthPanel />
          </div>

          {isLoading ? (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
              <p className="mt-4 text-gray-600">Loading workflows...</p>
            </div>
          ) : (
            <WorkflowTabs tabs={tabs} defaultTab="workflows" />
          )}
        </div>
      </main>
    </div>
    </PageErrorBoundary>
  );
}

export const OpenEvolveDashboard = withComponentBoundary(
  OpenEvolveDashboardBase,
  'OpenEvolveDashboard'
);
