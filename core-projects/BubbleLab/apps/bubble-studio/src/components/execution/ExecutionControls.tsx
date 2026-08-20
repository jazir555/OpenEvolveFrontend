/**
 * Execution Controls Component
 * Pause / resume workflow execution.
 *
 * Start and Stop are rendered disabled: the OpenEvolve backend
 * (`engines/other/api_server.py`) has no `POST /workflows/{id}/start` or
 * `POST /workflows/{id}/stop` route. Only `pause` and `resume` exist for a
 * workflow; a run is launched through `POST /executions`
 * (`openevolveApi.executeWorkflow`) and cancelled through
 * `POST /executions/{id}/cancel` (`openevolveApi.cancelExecution`).
 */

import { usePauseWorkflow, useResumeWorkflow } from '../../hooks/use-workflows-api';
import {
  WORKFLOW_START_SUPPORTED,
  WORKFLOW_START_UNAVAILABLE_MESSAGE,
  WORKFLOW_STOP_SUPPORTED,
  WORKFLOW_STOP_UNAVAILABLE_MESSAGE,
} from '../../lib/api-client';

interface ExecutionControlsProps {
  workflowId: string;
}

export function ExecutionControls({ workflowId }: ExecutionControlsProps) {
  const pauseWorkflow = usePauseWorkflow();
  const resumeWorkflow = useResumeWorkflow();

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <button
          type="button"
          disabled={!WORKFLOW_START_SUPPORTED}
          title={WORKFLOW_START_UNAVAILABLE_MESSAGE}
          className="inline-flex items-center rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <svg className="mr-2 h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
              clipRule="evenodd"
            />
          </svg>
          Start
        </button>

        <button
          type="button"
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
          type="button"
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
          type="button"
          disabled={!WORKFLOW_STOP_SUPPORTED}
          title={WORKFLOW_STOP_UNAVAILABLE_MESSAGE}
          className="inline-flex items-center rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <svg className="mr-2 h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z"
              clipRule="evenodd"
            />
          </svg>
          Stop
        </button>
      </div>

      {(!WORKFLOW_START_SUPPORTED || !WORKFLOW_STOP_SUPPORTED) && (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          Start and Stop are not supported by the backend (no{' '}
          <code>/api/workflows/{'{id}'}/start</code> or{' '}
          <code>/api/workflows/{'{id}'}/stop</code> route). Launch a run with
          Execute (<code>POST /api/executions</code>) and cancel it from the execution
          view; Pause and Resume act on the workflow.
        </p>
      )}
    </div>
  );
}
