/**
 * WorkflowCard Component
 * Display workflow summary with actions
 */

import { Link } from '@tanstack/react-router';
import { StatusBadge } from '../common/StatusBadge';
import { TimeAgo } from '../common/TimeAgo';
import { formatDuration } from '../../utils/date';

interface WorkflowCardProps {
  workflow: {
    id: string;
    name: string;
    description?: string;
    status: 'created' | 'running' | 'paused' | 'completed' | 'failed';
    created_at: string;
    started_at?: string;
    completed_at?: string;
    content_type: string;
    teams: string[];
    gauntlets: string[];
  };
  onDelete?: (id: string) => void;
  onDuplicate?: (id: string) => void;
}

export function WorkflowCard({ workflow, onDelete, onDuplicate }: WorkflowCardProps) {
  const duration =
    workflow.started_at && workflow.completed_at
      ? formatDuration(
          new Date(workflow.completed_at).getTime() -
            new Date(workflow.started_at).getTime()
        )
      : undefined;

  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <Link
            to="/oe-workflows/$workflowId"
            params={{ workflowId: workflow.id }}
            className="text-lg font-semibold text-gray-900 dark:text-white hover:text-blue-600 dark:hover:text-blue-400"
          >
            {workflow.name}
          </Link>
          {workflow.description && (
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 line-clamp-2">
              {workflow.description}
            </p>
          )}
        </div>
        <StatusBadge status={workflow.status} />
      </div>

      {/* Metadata */}
      <div className="flex flex-wrap gap-2 mb-3">
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300">
          {workflow.content_type}
        </span>
        {workflow.teams.length > 0 && (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400">
            {workflow.teams.length} {workflow.teams.length === 1 ? 'team' : 'teams'}
          </span>
        )}
        {workflow.gauntlets.length > 0 && (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400">
            {workflow.gauntlets.length} {workflow.gauntlets.length === 1 ? 'gauntlet' : 'gauntlets'}
          </span>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-3 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <TimeAgo date={workflow.created_at} />
          </span>
          {duration && (
            <span className="flex items-center gap-1">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              {duration}
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <Link
            to="/oe-workflows/$workflowId/execute"
            params={{ workflowId: workflow.id }}
            className="px-3 py-1 text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
          >
            Execute
          </Link>
          {onDuplicate && (
            <button
              onClick={() => onDuplicate(workflow.id)}
              className="px-3 py-1 text-sm font-medium text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200"
            >
              Duplicate
            </button>
          )}
          {onDelete && (
            <button
              onClick={() => onDelete(workflow.id)}
              className="px-3 py-1 text-sm font-medium text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
            >
              Delete
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
