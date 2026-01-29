import { useRealtimeEvolution } from '@/services/hooks/useRealtime';
import { ProgressBar } from '@/components/shared/ProgressBar';
import { LiveLogViewer, LogEntry } from '@/components/shared/LiveLogViewer';
import { StatusBadge } from '@/components/shared/StatusBadge';
import { cn } from '@/lib/utils';
import { useState, useEffect } from 'react';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';

interface ExecutionMonitorProps {
  executionId: string;
  className?: string;
}

function ExecutionMonitorBase({ executionId, className }: ExecutionMonitorProps) {
  const { data, isConnected, error } = useRealtimeEvolution(executionId);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  useEffect(() => {
    if (Array.isArray(data?.logs) && data.logs.length) {
      setLogs((prev) => [...prev, ...data.logs]);
    }
  }, [data?.logs]);

  const progress = data?.progress && data.progress.max_iterations > 0
    ? (data.progress.current_iteration / data.progress.max_iterations) * 100
    : 0;

  return (
    <div className={cn('execution-monitor bg-white border border-gray-200 rounded-lg p-6', className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">Execution Monitor</h2>
        <div className="flex items-center gap-3">
          <StatusBadge status={data?.status} isConnected={isConnected} />
          {error && (
            <span className="text-xs text-red-600">
              {error.message}
            </span>
          )}
        </div>
      </div>

      {/* Progress */}
      {data?.progress && (
        <div className="mb-6">
          <ProgressBar
            progress={progress}
            label={`Iteration ${data.progress.current_iteration} of ${data.progress.max_iterations}`}
            showPercentage
          />
        </div>
      )}

      {/* Metrics */}
      {data?.metrics && (
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="text-sm text-blue-600 mb-1">Avg Fitness</div>
            <div className="text-2xl font-bold text-blue-900">
              {Number.isFinite(Number(data.metrics.average_fitness))
                ? Number(data.metrics.average_fitness).toFixed(4)
                : 'N/A'}
            </div>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <div className="text-sm text-green-600 mb-1">Diversity Score</div>
            <div className="text-2xl font-bold text-green-900">
              {Number.isFinite(Number(data.metrics.diversity_score))
                ? Number(data.metrics.diversity_score).toFixed(4)
                : 'N/A'}
            </div>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg">
            <div className="text-sm text-purple-600 mb-1">Convergence Rate</div>
            <div className="text-2xl font-bold text-purple-900">
              {Number.isFinite(Number(data.metrics.convergence_rate))
                ? Number(data.metrics.convergence_rate).toFixed(4)
                : 'N/A'}
            </div>
          </div>
        </div>
      )}

      {/* Logs */}
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-2">Live Logs</h3>
        <LiveLogViewer logs={logs} />
      </div>

      {/* Best Individual */}
      {data?.best_individual && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Best Individual</h3>
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="text-sm text-gray-600 mb-2">
              Fitness:{' '}
              <span className="font-semibold">
                {Number.isFinite(Number(data.best_individual.fitness))
                  ? Number(data.best_individual.fitness).toFixed(4)
                  : 'N/A'}
              </span>
            </div>
            <div className="text-sm text-gray-800 line-clamp-3">
              {data.best_individual.content}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export const ExecutionMonitor = withComponentBoundary(ExecutionMonitorBase, 'ExecutionMonitor');
