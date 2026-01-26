/**
 * Workflow Detail Page
 * Display workflow details and configuration
 */

import { createFileRoute } from '@tanstack/react-router';
import { useWorkflow } from '../../hooks/use-workflows-api';
import { WorkflowStatus } from '../../types/api';

export const Route = createFileRoute('/oe-workflows/$workflowId')({
  component: WorkflowDetailPage,
});

function WorkflowDetailPage() {
  const { workflowId } = Route.useParams();
  const { data: workflow, isLoading, error } = useWorkflow(workflowId);

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <svg className="h-12 w-12 animate-spin text-blue-600" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
      </div>
    );
  }

  if (error || !workflow) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-6 dark:border-red-900 dark:bg-red-900/20">
        <p className="text-red-800 dark:text-red-400">
          Error loading workflow: {error?.message || 'Workflow not found'}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              {workflow.name}
            </h1>
            <StatusBadge status={workflow.status} />
          </div>
          {workflow.description && (
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              {workflow.description}
            </p>
          )}
        </div>

        <div className="flex gap-2">
          <a
            href={`/oe-workflows/${workflow.id}/execute`}
            className="inline-flex items-center rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700"
          >
            Execute
          </a>
          <a
            href={`/oe-workflows/${workflow.id}/edit`}
            className="inline-flex items-center rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
          >
            Edit
          </a>
        </div>
      </div>

      {/* Workflow Details */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Problem Statement */}
        <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
            Problem Statement
          </h3>
          <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
            {workflow.problem_statement}
          </p>
        </div>

        {/* Configuration */}
        <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
            Configuration
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Content Type:</span>
              <span className="font-medium text-gray-900 dark:text-white">{workflow.content_type}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Teams:</span>
              <span className="font-medium text-gray-900 dark:text-white">{workflow.teams.length} selected</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Gauntlets:</span>
              <span className="font-medium text-gray-900 dark:text-white">{workflow.gauntlets.length} selected</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Created:</span>
              <span className="font-medium text-gray-900 dark:text-white">
                {new Date(workflow.created_at).toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Metadata */}
      {workflow.metadata && (
        <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
            Advanced Settings
          </h3>
          <div className="grid gap-4 sm:grid-cols-2">
            {workflow.metadata.mdap_enabled !== undefined && (
              <div>
                <span className="text-sm text-gray-600 dark:text-gray-400">MDAP:</span>
                <span className="ml-2 text-sm font-medium text-gray-900 dark:text-white">
                  {workflow.metadata.mdap_enabled ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            )}
            {workflow.metadata.maker_enabled !== undefined && (
              <div>
                <span className="text-sm text-gray-600 dark:text-gray-400">MAKER:</span>
                <span className="ml-2 text-sm font-medium text-gray-900 dark:text-white">
                  {workflow.metadata.maker_enabled ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: WorkflowStatus }) {
  const styles = {
    [WorkflowStatus.CREATED]: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
    [WorkflowStatus.RUNNING]: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    [WorkflowStatus.PAUSED]: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400',
    [WorkflowStatus.COMPLETED]: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    [WorkflowStatus.FAILED]: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    [WorkflowStatus.CANCELLED]: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
  };

  return (
    <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${styles[status]}`}>
      {status}
    </span>
  );
}
