// @ts-nocheck
import { useEffect, useRef, useState } from 'react';
import { ProviderSettingsPanel } from '@/components/config/ProviderSettingsPanel';
import { BubbleButton, BubbleCard, BubbleField, BubbleTextArea } from '@/components/bubblelab';
import { ProgressBar } from '@/components/shared/ProgressBar';
import { StatusBadge } from '@/components/shared/StatusBadge';
import { LiveLogViewer, LogEntry } from '@/components/shared/LiveLogViewer';
import { ExecutionMonitor } from '@/components/workflow/ExecutionMonitor';
import { useEvolution } from '@/services/hooks/useApi';
import { useWorkflowConfig, useWorkflowModels } from '@/services/hooks/useWorkflows';
import { VizErrorBoundary } from '@/components/shared/VizErrorBoundary';
import { PageErrorBoundary } from '@/components/shared/PageErrorBoundary';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';
import { gracefulErrorHandler } from '@/utils/gracefulErrorHandler';
import { toast } from 'react-toastify';
import {
  LineageViz,
  MAPElitesViz,
  OptimizationViz,
} from 'openevolve-pygraphistry-plugin';

function EvolutionPageBase() {
  const { config } = useWorkflowConfig();
  const { models } = useWorkflowModels();
  const [initialContent, setInitialContent] = useState('');
  const [evolutionId, setEvolutionId] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [pageError, setPageError] = useState<string | null>(null);
  const lastStatusRef = useRef<string | null>(null);

  const {
    evolution,
    startEvolution,
    pauseEvolution,
    resumeEvolution,
    stopEvolution,
    isLoading,
    error,
  } = useEvolution(evolutionId || undefined);

  const status = evolution?.status || 'idle';
  const progress = evolution?.progress?.percentage || 0;
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
        message: `Evolution status: ${status}`,
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
        message: 'Starting evolution...',
      },
    ]);

    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      setPageError(null);
      const response = await startEvolution({
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
        setEvolutionId(response.evolution_id);
      }

      return response;
    }, {
      strategy: 'retry',
      maxRetries: 3,
      retryDelay: 1000,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'EvolutionPage',
        function: 'handleStart',
        operation: 'START_EVOLUTION',
        additionalData: {
          contentLength: initialContent.length,
          mode: config.mode,
          modelCount: models.length
        }
      }
    });

    if (!result.success) {
      const errorMessage = result.error?.message || 'Failed to start evolution.';
      setPageError(errorMessage);
      toast.error(`Failed to start evolution: ${errorMessage}`);
    }
  };

  const handlePause = async () => {
    if (!evolutionId) return;

    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      setPageError(null);
      await pauseEvolution(evolutionId);
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'EvolutionPage',
        function: 'handlePause',
        operation: 'PAUSE_EVOLUTION',
        additionalData: { evolutionId }
      }
    });

    if (!result.success) {
      const errorMessage = result.error?.message || 'Failed to pause evolution.';
      setPageError(errorMessage);
      toast.error(`Failed to pause evolution: ${errorMessage}`);
    }
  };

  const handleResume = async () => {
    if (!evolutionId) return;

    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      setPageError(null);
      await resumeEvolution(evolutionId);
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'EvolutionPage',
        function: 'handleResume',
        operation: 'RESUME_EVOLUTION',
        additionalData: { evolutionId }
      }
    });

    if (!result.success) {
      const errorMessage = result.error?.message || 'Failed to resume evolution.';
      setPageError(errorMessage);
      toast.error(`Failed to resume evolution: ${errorMessage}`);
    }
  };

  const handleStop = async () => {
    if (!evolutionId) return;

    const result = await gracefulErrorHandler.executeWithErrorHandling(async () => {
      setPageError(null);
      await stopEvolution(evolutionId);
    }, {
      strategy: 'retry',
      maxRetries: 2,
      retryDelay: 500,
      showUserNotification: true,
      logError: true,
      context: {
        component: 'EvolutionPage',
        function: 'handleStop',
        operation: 'STOP_EVOLUTION',
        additionalData: { evolutionId }
      }
    });

    if (!result.success) {
      const errorMessage = result.error?.message || 'Failed to stop evolution.';
      setPageError(errorMessage);
      toast.error(`Failed to stop evolution: ${errorMessage}`);
    }
  };

  const handleReset = () => {
    setEvolutionId(null);
    setLogs([]);
  };

  return (
    <PageErrorBoundary label="Evolution workflow">
      <div className="flex h-screen bg-slate-50">
      <ProviderSettingsPanel />

      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-white border-b border-slate-200 px-6 py-4">
          <h1 className="text-2xl font-bold text-slate-900">Evolution Engine</h1>
          <p className="text-sm text-slate-600">Run evolutionary optimization workflows.</p>
        </header>

        <div className="flex-1 overflow-auto p-6 space-y-6">
          {(pageError || error) && (
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {pageError || (error instanceof Error ? error.message : 'Failed to load evolution status.')}
            </div>
          )}

          <BubbleCard title="Evolution Setup">
            <div className="space-y-4">
              <BubbleField label="Initial Content">
                <BubbleTextArea
                  rows={8}
                  value={initialContent}
                  onChange={(event) => setInitialContent(event.target.value)}
                  placeholder="Enter initial content to evolve..."
                />
              </BubbleField>
            </div>
          </BubbleCard>

          <BubbleCard title="Execution Controls">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <StatusBadge status={status} />
                {isRunning && <span className="text-sm text-slate-500">Running...</span>}
              </div>
              <div className="flex flex-wrap gap-2">
                {!isRunning && status === 'idle' && (
                  <BubbleButton onClick={handleStart}>Start Evolution</BubbleButton>
                )}
                {isRunning && (
                  <BubbleButton onClick={handlePause} variant="secondary">
                    Pause
                  </BubbleButton>
                )}
                {!isRunning && status === 'paused' && (
                  <BubbleButton onClick={handleResume}>Resume</BubbleButton>
                )}
                {evolutionId && status !== 'stopped' && status !== 'completed' && (
                  <BubbleButton onClick={handleStop} variant="secondary">
                    Stop
                  </BubbleButton>
                )}
                <BubbleButton onClick={handleReset} variant="secondary">
                  Reset
                </BubbleButton>
              </div>
            </div>
            {progress > 0 && <ProgressBar progress={progress} label="Evolution Progress" />}
          </BubbleCard>

          {evolutionId && (
            <ExecutionMonitor executionId={evolutionId} />
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <BubbleCard title="Evolution Lineage" description="Trace ancestry across generations.">
              <VizErrorBoundary label="Evolution lineage visualization">
                <LineageViz />
              </VizErrorBoundary>
            </BubbleCard>
            <BubbleCard title="MAP-Elites Quality Diversity" description="Run QD evolution on the current seed.">
              <VizErrorBoundary label="MAP-Elites visualization">
                <MAPElitesViz initialContent={initialContent} iterations={50} />
              </VizErrorBoundary>
            </BubbleCard>
          </div>

          <BubbleCard title="Optimization Insights" description="Optimization run aligned with evolution objectives.">
            <VizErrorBoundary label="Optimization visualization">
              <OptimizationViz problemType="evolution" initialValue={10.0} />
            </VizErrorBoundary>
          </BubbleCard>

          <BubbleCard title="Live Logs">
            <LiveLogViewer logs={logs} maxHeight="260px" />
            {isLoading && (
              <p className="mt-3 text-xs text-slate-400">Fetching evolution status...</p>
            )}
          </BubbleCard>
        </div>
      </main>
    </div>
    </PageErrorBoundary>
  );
}

export const EvolutionPage = withComponentBoundary(EvolutionPageBase, 'EvolutionPage');
