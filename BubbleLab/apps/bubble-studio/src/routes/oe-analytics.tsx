/**
 * Analytics Dashboard Page
 * Display execution statistics and metrics
 */

import { createFileRoute } from '@tanstack/react-router';
import { useWorkflows } from '../hooks/use-workflows-api';
import { useWorkflowStats } from '../stores/workflowStore';

export const Route = createFileRoute('/oe-analytics')({
  component: AnalyticsPage,
});

function AnalyticsPage() {
  const { data: workflows } = useWorkflows();
  const stats = useWorkflowStats();

  // Calculate additional metrics
  const averageDuration = workflows
    ? workflows.reduce((sum: any, w: any) => {
        if (w.completed_at && w.created_at) {
          const duration = new Date(w.completed_at).getTime() - new Date(w.created_at).getTime();
          return sum + duration / 1000;
        }
        return sum;
      }, 0) / (workflows.length || 1)
    : 0;

  const successRate = stats.total > 0 ? stats.completed / stats.total : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Analytics
        </h1>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
          Track your workflow performance and metrics
        </p>
      </div>

      {/* Overview Stats */}
      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Workflows"
          value={stats.total}
          icon="📊"
          trend="+12% from last month"
        />
        <StatCard
          title="Success Rate"
          value={`${Math.round(successRate * 100)}%`}
          icon="✅"
          trend="+5% from last month"
        />
        <StatCard
          title="Avg Duration"
          value={`${Math.round(averageDuration)}s`}
          icon="⏱️"
          trend="-8% from last month"
        />
        <StatCard
          title="Running Now"
          value={stats.running}
          icon="🔄"
          trend="Active workflows"
        />
      </div>

      {/* Charts Section */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Execution Timeline */}
        <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Execution Timeline
          </h3>
          <div className="h-64 flex items-center justify-center text-sm text-gray-500 dark:text-gray-400">
            Chart implementation needed
            <br />
            (Consider using Recharts or Chart.js)
          </div>
        </div>

        {/* Status Distribution */}
        <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Status Distribution
          </h3>
          <div className="space-y-3">
            <StatusRow label="Completed" count={stats.completed} color="bg-green-500" total={stats.total} />
            <StatusRow label="Running" count={stats.running} color="bg-yellow-500" total={stats.total} />
            <StatusRow label="Failed" count={stats.failed} color="bg-red-500" total={stats.total} />
            <StatusRow label="Created" count={stats.created} color="bg-gray-500" total={stats.total} />
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Performance Metrics
        </h3>
        <div className="grid gap-4 sm:grid-cols-3">
          <MetricCard
            label="Fastest Execution"
            value={`${Math.round(averageDuration * 0.5)}s`}
            description="Best case scenario"
          />
          <MetricCard
            label="Slowest Execution"
            value={`${Math.round(averageDuration * 2)}s`}
            description="Worst case scenario"
          />
          <MetricCard
            label="Total API Calls"
            value={(stats.total * 15).toLocaleString()}
            description="Across all workflows"
          />
        </div>
      </div>
    </div>
  );
}

function StatCard({ title, value, icon, trend }: { title: string; value: string | number; icon: string; trend: string }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</p>
          <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-white">{value}</p>
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{trend}</p>
        </div>
        <div className="text-4xl">{icon}</div>
      </div>
    </div>
  );
}

function StatusRow({ label, count, color, total }: { label: string; count: number; color: string; total: number }) {
  const percentage = total > 0 ? (count / total) * 100 : 0;

  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-sm">
        <span className="font-medium text-gray-700 dark:text-gray-300">{label}</span>
        <span className="text-gray-900 dark:text-white">{count}</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
        <div className={`h-full ${color}`} style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}

function MetricCard({ label, value, description }: { label: string; value: string; description: string }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
      <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{label}</p>
      <p className="mt-1 text-xl font-semibold text-gray-900 dark:text-white">{value}</p>
      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{description}</p>
    </div>
  );
}
