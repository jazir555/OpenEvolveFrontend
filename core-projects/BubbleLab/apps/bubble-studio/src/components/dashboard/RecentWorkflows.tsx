/**
 * Recent Workflows Component
 * Displays list of recent workflows with status indicators
 */

import { Link } from '@tanstack/react-router';
import { Workflow, WorkflowStatus } from '../../types/api';

interface RecentWorkflowsProps {
  workflows: Workflow[];
  isLoading: boolean;
  error: Error | null;
}

export function RecentWorkflows({ workflows, isLoading, error }: RecentWorkflowsProps) {
  if (isLoading) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h2 className="mb-4 text-xl font-semibold text-gray-900 dark:text-white">
          Recent Workflows
        </h2>
        <div className="space-y-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <div
              key={i}
              className="h-20 animate-pulse rounded bg-gray-200 dark:bg-gray-700"
            />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-6 dark:border-red-900 dark:bg-red-900/20">
        <p className="text-red-800 dark:text-red-400">
          Error loading workflows: {error.message}
        </p>
      </div>
    );
  }

  if (workflows.length === 0) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-12 text-center shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <svg
          className="mx-auto h-12 w-12 text-gray-400"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path
            fillRule="evenodd"
            d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z"
            clipRule="evenodd"
          />
        </svg>
        <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
          No workflows
        </h3>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Get started by creating a new workflow.
        </p>
        <div className="mt-6">
          <Link
            to="/oe-workflows/create"
            className="inline-flex items-center rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
          >
            Create Workflow
          </Link>
        </div>
      </div>
    );
  }

  const recentWorkflows = workflows.slice(0, 10);

  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
      <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4 dark:border-gray-700">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Recent Workflows
        </h2>
        <Link
          to="/oe-workflows"
          className="text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
        >
          View all →
        </Link>
      </div>

      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        {recentWorkflows.map((workflow) => (
          <Link
            key={workflow.id}
            to="/oe-workflows/$workflowId"
            params={{ workflowId: workflow.id }}
            className="block hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            <div className="px-6 py-4">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <p className="text-sm font-medium text-gray-900 dark:text-white">
                      {workflow.name}
                    </p>
                    <StatusBadge status={workflow.status} />
                  </div>
                  <p className="mt-1 truncate text-sm text-gray-600 dark:text-gray-400">
                    {workflow.description || workflow.problem_statement}
                  </p>
                </div>
                <div className="ml-4 flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                  <div>
                    <p className="text-xs">Created</p>
                    <p className="font-medium">
                      {new Date(workflow.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  {workflow.completed_at && (
                    <div>
                      <p className="text-xs">Duration</p>
                      <p className="font-medium">
                        {formatDuration(workflow.created_at, workflow.completed_at)}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </Link>
        ))}
      </div>
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
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
        styles[status]
      }`}
    >
      {status}
    </span>
  );
}

function formatDuration(start: string, end: string): string {
  const startTime = new Date(start).getTime();
  const endTime = new Date(end).getTime();
  const duration = endTime - startTime;

  const seconds = Math.floor(duration / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}
