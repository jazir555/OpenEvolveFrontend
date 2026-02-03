/**
 * Execution Controls Component
 * Start, pause, resume, stop workflow execution
 */

import { useStartWorkflow, usePauseWorkflow, useResumeWorkflow, useStopWorkflow } from '../../hooks/use-workflows-api';

interface ExecutionControlsProps {
  workflowId: string;
}

export function ExecutionControls({ workflowId }: ExecutionControlsProps) {
  const startWorkflow = useStartWorkflow();
  const pauseWorkflow = usePauseWorkflow();
  const resumeWorkflow = useResumeWorkflow();
  const stopWorkflow = useStopWorkflow();

  return (
    <div className="flex gap-2">
      <button
        onClick={() => startWorkflow.mutate(workflowId)}
        disabled={startWorkflow.isPending}
        className="inline-flex items-center rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {startWorkflow.isPending ? (
          <>
            <svg className="mr-2 h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Starting...
          </>
        ) : (
          <>
            <svg className="mr-2 h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                clipRule="evenodd"
              />
            </svg>
            Start
          </>
        )}
      </button>

      <button
        onClick={() => pauseWorkflow.mutate(workflowId)}
        disabled={pauseWorkflow.isPending}
        className="inline-flex items-center rounded-lg bg-yellow-600 px-4 py-2 text-sm font-medium text-white hover:bg-yellow-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {pauseWorkflow.isPending ? (
          'Pausing...'
        ) : (
          <>
            <svg className="mr-2 h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
            Pause
          </>
        )}
      </button>

      <button
        onClick={() => resumeWorkflow.mutate(workflowId)}
        disabled={resumeWorkflow.isPending}
        className="inline-flex items-center rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {resumeWorkflow.isPending ? (
          'Resuming...'
        ) : (
          <>
            <svg className="mr-2 h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                clipRule="evenodd"
              />
            </svg>
            Resume
          </>
        )}
      </button>

      <button
        onClick={() => stopWorkflow.mutate(workflowId)}
        disabled={stopWorkflow.isPending}
        className="inline-flex items-center rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {stopWorkflow.isPending ? (
          'Stopping...'
        ) : (
          <>
            <svg className="mr-2 h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z"
                clipRule="evenodd"
              />
            </svg>
            Stop
          </>
        )}
      </button>
    </div>
  );
}
