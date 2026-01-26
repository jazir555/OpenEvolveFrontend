/**
 * OpenEvolve Workflows List Page
 * New route for OpenEvolve workflow management
 */

import { createFileRoute, Link } from '@tanstack/react-router';
import { useWorkflows } from '../hooks/use-workflows-api';
import { useWorkflowStats } from '../stores/workflowStore';
import { QuickStats } from '../components/dashboard/QuickStats';
import { RecentWorkflows } from '../components/dashboard/RecentWorkflows';
import { QuickActions } from '../components/dashboard/QuickActions';

export const Route = createFileRoute('/oe-workflows')({
  component: WorkflowsListPage,
});

function WorkflowsListPage() {
  const { data: workflows, isLoading, error } = useWorkflows();
  const stats = useWorkflowStats();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            OpenEvolve Workflows
          </h1>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Manage and monitor your OpenEvolve workflows
          </p>
        </div>
      </div>

      {/* Quick Stats */}
      <QuickStats stats={stats} isLoading={isLoading} />

      {/* Quick Actions */}
      <QuickActions />

      {/* Recent Workflows */}
      <RecentWorkflows
        workflows={workflows || []}
        isLoading={isLoading}
        error={error}
      />
    </div>
  );
}
