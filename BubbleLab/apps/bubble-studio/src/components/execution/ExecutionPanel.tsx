/**
 * Execution Panel Component
 * Real-time workflow execution display with streaming
 */

import { useEffect, useState } from 'react';
import { useExecutionStream } from '../../hooks/use-execution-stream';
import { ExecutionProgressBar } from './ExecutionProgressBar';
import { ExecutionControls } from './ExecutionControls';
import { ExecutionLogs } from './ExecutionLogs';

interface ExecutionPanelProps {
  workflowId: string;
}

export function ExecutionPanel({ workflowId }: ExecutionPanelProps) {
  const [currentStep, setCurrentStep] = useState('');
  const [progress, setProgress] = useState(0);

  const { connected, error } = useExecutionStream(workflowId, {
    onEvent: (event) => {
      if (event.event === 'execution.progress') {
        setCurrentStep(event.data.current_step || '');
        setProgress(event.data.progress || 0);
      }
    },
    onError: (err) => {
      console.error('Execution stream error:', err);
    },
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Execution Monitor
          </h2>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            {connected ? (
              <span className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                Live
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-gray-400" />
                Disconnected
              </span>
            )}
          </p>
        </div>

        <ExecutionControls workflowId={workflowId} />
      </div>

      {/* Progress Bar */}
      <ExecutionProgressBar progress={progress} currentStep={currentStep} />

      {/* Execution Logs */}
      <ExecutionLogs workflowId={workflowId} />

      {/* Error Display */}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-900/20">
          <p className="text-sm font-medium text-red-800 dark:text-red-400">
            Connection Error: {error.message}
          </p>
          <p className="mt-1 text-xs text-red-600 dark:text-red-400">
            Attempting to reconnect...
          </p>
        </div>
      )}
    </div>
  );
}
