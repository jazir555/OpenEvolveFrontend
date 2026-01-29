/**
 * Analytics Dashboard Page
 * Display execution statistics and metrics
 */

import { createFileRoute } from '@tanstack/react-router';
import { useMemo } from 'react';
import { useWorkflows } from '../hooks/use-workflows-api';
import { useWorkflowStats } from '../stores/workflowStore';
import { BarChart, LineChart, PieChart } from '../components/analytics/Charts';

export const Route = createFileRoute('/oe-analytics')({
  component: AnalyticsPage,
});

function AnalyticsPage() {
  const { data: workflows } = useWorkflows();
  const stats = useWorkflowStats();

  const durations = useMemo(() => {
    if (!workflows) return [];
    return workflows
      .map((workflow: any) => {
        const start =
          workflow.started_at || workflow.created_at || workflow.updated_at;
        const end = workflow.completed_at || workflow.updated_at;
        if (!start || !end) return null;
        const startTime = new Date(start).getTime();
        const endTime = new Date(end).getTime();
        if (Number.isNaN(startTime) || Number.isNaN(endTime)) return null;
        return Math.max((endTime - startTime) / 1000, 0);
      })
      .filter((duration: number | null): duration is number => duration !== null);
  }, [workflows]);

  const averageDuration =
    durations.length > 0
      ? durations.reduce((sum, duration) => sum + duration, 0) /
        durations.length
      : 0;

  const fastestDuration =
    durations.length > 0 ? Math.min(...durations) : 0;
  const slowestDuration =
    durations.length > 0 ? Math.max(...durations) : 0;

  const successRate = stats.total > 0 ? stats.completed / stats.total : 0;

  const statusChartData = useMemo(
    () => [
      { label: 'Completed', value: stats.completed, color: '#10B981' },
      { label: 'Running', value: stats.running, color: '#F59E0B' },
      { label: 'Failed', value: stats.failed, color: '#EF4444' },
      { label: 'Created', value: stats.created, color: '#9CA3AF' },
    ],
    [stats]
  );

  const timelineData = useMemo(() => {
    const now = new Date();
    const formatter = new Intl.DateTimeFormat('en-US', {
      weekday: 'short',
    });
    const buckets = Array.from({ length: 7 }).map((_, idx) => {
      const date = new Date(now);
      date.setDate(now.getDate() - (6 - idx));
      const key = date.toISOString().slice(0, 10);
      return {
        key,
        label: formatter.format(date),
        count: 0,
      };
    });
    const bucketMap = new Map(buckets.map((b) => [b.key, b]));

    workflows?.forEach((workflow: any) => {
      const timestamp =
        workflow.completed_at || workflow.created_at || workflow.updated_at;
      if (!timestamp) return;
      const dateKey = new Date(timestamp).toISOString().slice(0, 10);
      const bucket = bucketMap.get(dateKey);
      if (bucket) {
        bucket.count += 1;
      }
    });

    return buckets.map((bucket) => ({
      x: bucket.label,
      y: bucket.count,
    }));
  }, [workflows]);

  const durationChartData = useMemo(
    () => [
      {
        label: 'Fastest',
        value: Math.round(fastestDuration),
        color: 'bg-emerald-500',
      },
      {
        label: 'Average',
        value: Math.round(averageDuration),
        color: 'bg-blue-500',
      },
      {
        label: 'Slowest',
        value: Math.round(slowestDuration),
        color: 'bg-rose-500',
      },
    ],
    [averageDuration, fastestDuration, slowestDuration]
  );

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
        <LineChart
          data={timelineData}
          title="Execution Timeline (Last 7 Days)"
          height={220}
          width={520}
        />
        <PieChart data={statusChartData} title="Status Distribution" />
      </div>

      {/* Performance Metrics */}
      <div className="grid gap-6 lg:grid-cols-2">
        <BarChart
          data={durationChartData}
          title="Execution Duration (s)"
          horizontal
          height={220}
        />
        <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Performance Metrics
          </h3>
          <div className="grid gap-4 sm:grid-cols-3">
            <MetricCard
              label="Fastest Execution"
              value={`${Math.round(fastestDuration)}s`}
              description="Best case scenario"
            />
            <MetricCard
              label="Slowest Execution"
              value={`${Math.round(slowestDuration)}s`}
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

function MetricCard({ label, value, description }: { label: string; value: string; description: string }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
      <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{label}</p>
      <p className="mt-1 text-xl font-semibold text-gray-900 dark:text-white">{value}</p>
      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{description}</p>
    </div>
  );
}
