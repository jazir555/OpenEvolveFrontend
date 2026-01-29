// @ts-nocheck
import { useEffect, useRef, useState } from 'react';
import {
  useWorkflowConfig,
  useWorkflowModels,
  useWorkflow,
  useWorkflows,
} from '@/services/hooks/useWorkflows';
import { ProviderSettingsPanel } from '@/components/config/ProviderSettingsPanel';
import { LiveLogViewer, LogEntry } from '@/components/shared/LiveLogViewer';
import { ProgressBar } from '@/components/shared/ProgressBar';
import { StatusBadge } from '@/components/shared/StatusBadge';
import { BubbleButton, BubbleCard, BubbleField, BubbleInput, BubbleTextArea } from '@/components/bubblelab';
import { PageErrorBoundary } from '@/components/shared/PageErrorBoundary';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';
// @ts-nocheck

function WorkflowBuilderBase() {
  const { config } = useWorkflowConfig();
  const { models } = useWorkflowModels();
  const { startWorkflow } = useWorkflows();
  const [workflowName, setWorkflowName] = useState('');
  const [workflowDescription, setWorkflowDescription] = useState('');
  const [initialContent, setInitialContent] = useState('');
  const [executionId, setExecutionId] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [pageError, setPageError] = useState<string | null>(null);
  const lastStatusRef = useRef<string | null>(null);

  const { workflow, pause, resume, stop, isLoading, error } = useWorkflow(executionId || undefined);
  const status = workflow?.status || 'idle';
  const progress = workflow?.progress?.percentage || 0;
  const isRunning = status === 'running';

  useEffect(() => {
    if (!status || status === lastStatusRef.current) {
      return;
    }
    lastStatusRef.current = status;
    setLogs((prev) => [
      ...prev,
      {
        timestamp: new Date().toISOString(),
        level: 'info',
        message: `Workflow status: ${status}`,
      },
    ]);
  }, [status]);

  const handleStart = async () => {
    if (!initialContent.trim()) {
      alert('Please enter initial content');
      return;
    }

    setLogs([
      {
        timestamp: new Date().toISOString(),
        level: 'info',
        message: 'Starting workflow...',
      },
    ]);

    try {
      setPageError(null);
      const response = await startWorkflow({
        content: initialContent,
        mode: config.mode,
        parameters: {
          max_iterations: config.max_iterations,
          population_size: config.population_size,
          temperature: config.temperature,
          top_p: config.top_p,
        },
        models: models.map((item) => ({
          provider: item.provider,
          model: item.model,
          api_key: item.api_key || '',
        })),
      });

      if (response?.evolution_id) {
        setExecutionId(response.evolution_id);
      }
    } catch (err) {
      setPageError(err instanceof Error ? err.message : 'Failed to start workflow.');
    }
  };

  const handlePause = async () => {
    if (!executionId) return;
    try {
      setPageError(null);
      await pause();
    } catch (err) {
      setPageError(err instanceof Error ? err.message : 'Failed to pause workflow.');
    }
  };

  const handleReset = () => {
    setExecutionId(null);
    setLogs([]);
  };

  const handleStop = async () => {
    if (!executionId) return;
    try {
      setPageError(null);
      await stop();
    } catch (err) {
      setPageError(err instanceof Error ? err.message : 'Failed to stop workflow.');
    }
  };

  return (
    <PageErrorBoundary label="Workflow builder">
      <div className="workflow-builder flex h-screen bg-slate-50">
      <ProviderSettingsPanel />

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b border-slate-200 px-6 py-4">
          <h1 className="text-2xl font-bold text-gray-900">Workflow Builder</h1>
          <p className="text-sm text-gray-600">Create and manage evolutionary workflows</p>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {(pageError || error) && (
            <div className="mb-6 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {pageError || (error instanceof Error ? error.message : 'Failed to load workflow status.')}
            </div>
          )}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Workflow Configuration */}
            <div className="lg:col-span-2 space-y-6">
              <BubbleCard title="Workflow Details">
                <div className="space-y-4">
                  <BubbleField label="Workflow Name">
                    <BubbleInput
                      type="text"
                      value={workflowName}
                      onChange={(e) => setWorkflowName(e.target.value)}
                      placeholder="Enter workflow name..."
                    />
                  </BubbleField>

                  <BubbleField label="Description">
                    <BubbleTextArea
                      value={workflowDescription}
                      onChange={(e) => setWorkflowDescription(e.target.value)}
                      placeholder="Describe the workflow purpose..."
                      rows={3}
                    />
                  </BubbleField>

                  <BubbleField label="Initial Content">
                    <BubbleTextArea
                      value={initialContent}
                      onChange={(e) => setInitialContent(e.target.value)}
                      placeholder="Enter initial content to evolve..."
                      rows={10}
                      className="font-mono text-sm"
                    />
                  </BubbleField>
                </div>
              </BubbleCard>

              {/* Controls */}
              <BubbleCard title="Execution Controls">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <StatusBadge status={status} />
                    {isRunning && (
                      <span className="text-sm text-gray-600">Running...</span>
                    )}
                  </div>

                  <div className="flex gap-2">
                    {!isRunning && status === 'idle' && (
                      <BubbleButton
                        onClick={handleStart}
                        variant="primary"
                      >
                        Start Workflow
                      </BubbleButton>
                    )}
                    {isRunning && (
                      <BubbleButton
                        onClick={handlePause}
                        variant="secondary"
                      >
                        Pause
                      </BubbleButton>
                    )}
                    {!isRunning && status === 'paused' && (
                      <BubbleButton
                        onClick={async () => {
                          try {
                            setPageError(null);
                            await resume();
                          } catch (err) {
                            setPageError(err instanceof Error ? err.message : 'Failed to resume workflow.');
                          }
                        }}
                        variant="primary"
                      >
                        Resume
                      </BubbleButton>
                    )}
                    {executionId && status !== 'stopped' && status !== 'completed' && (
                      <BubbleButton
                        onClick={handleStop}
                        variant="secondary"
                      >
                        Stop
                      </BubbleButton>
                    )}
                    <BubbleButton
                      onClick={handleReset}
                      variant="secondary"
                    >
                      Reset
                    </BubbleButton>
                  </div>
                </div>

                {progress > 0 && (
                  <ProgressBar progress={progress} label="Workflow Progress" />
                )}
              </BubbleCard>
            </div>

            {/* Right Sidebar - Monitoring */}
            <div className="space-y-6">
              {/* Status Panel */}
              <BubbleCard title="Status">
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Mode</span>
                    <span className="font-medium">{config.mode}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Iterations</span>
                    <span className="font-medium">
                      {workflow?.progress?.current_iteration || 0}/{config.max_iterations}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Population</span>
                    <span className="font-medium">{config.population_size}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Temperature</span>
                    <span className="font-medium">{config.temperature}</span>
                  </div>
                </div>
              </BubbleCard>

              {/* Live Logs */}
              <BubbleCard title="Live Logs">
                <LiveLogViewer logs={logs} maxHeight="300px" />
                {isLoading && (
                  <p className="mt-3 text-xs text-slate-400">Fetching workflow status...</p>
                )}
              </BubbleCard>
            </div>
          </div>
        </div>
      </main>
    </div>
    </PageErrorBoundary>
  );
}

export const WorkflowBuilder = withComponentBoundary(WorkflowBuilderBase, 'WorkflowBuilder');
