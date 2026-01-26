/**
 * Workflow Execution Page
 * Real-time workflow execution monitoring
 */

import { createFileRoute } from '@tanstack/react-router';
import { ExecutionPanel } from '../../components/execution/ExecutionPanel';
import { ResultsView } from '../../components/execution/ResultsView';

export const Route = createFileRoute('/oe-workflows/$workflowId/execute')({
  component: WorkflowExecutionPage,
});

function WorkflowExecutionPage() {
  const { workflowId } = Route.useParams();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Workflow Execution
        </h1>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
          Monitor and control workflow execution in real-time
        </p>
      </div>

      {/* Execution Panel */}
      <ExecutionPanel workflowId={workflowId} />

      {/* Results Section (shows when execution completes) */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Execution Results
        </h2>
        <ResultsView workflowId={workflowId} />
      </div>
    </div>
  );
}
