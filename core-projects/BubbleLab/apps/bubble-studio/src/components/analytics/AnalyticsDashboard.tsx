/**
 * AnalyticsDashboard Component
 * Displays OpenEvolve analytics: statistics summary, performance metrics,
 * workflow metrics, and knowledge stats. Uses the pre-built openevolveApi
 * analytics client methods. No charting library is present in the project,
 * so distributions/metrics are rendered with lightweight CSS bars and tables.
 */

import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  ArrowPathIcon,
  BarChart3,
  BookOpen,
  CheckCircle2,
  Cpu,
  Database,
  PlayCircle,
  Workflow,
  XCircle,
} from 'lucide-react';
import { openevolveApi } from '@/services/openevolveApi';
import type {
  AnalyticsKnowledgeStats,
  AnalyticsWorkflowMetric,
  PerformanceMetric,
  StatisticsSummary,
} from '@/types/openevolve';
import { Card } from '../common/Card';
import { Alert } from '../common/Alert';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { StatCard, type StatAccent } from './StatCard';
import { DistributionBars } from './DistributionBars';
import { MetricTable, type TableColumn } from './MetricTable';

function formatNum(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return `${value.toFixed(1)}%`;
}

function formatTimestamp(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === '') return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

/**
 * Render a loading or error placeholder for a single query-driven section.
 * Returns the provided children only when data is available.
 */
function useQueryState<T>(
  query: { isLoading: boolean; isError: boolean; error: unknown; data?: T }
) {
  return {
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    data: query.data,
  };
}

export function AnalyticsDashboard() {
  const statisticsQuery = useQuery({
    queryKey: ['analytics', 'statistics'],
    queryFn: () => openevolveApi.getStatistics(),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });

  const performanceQuery = useQuery({
    queryKey: ['analytics', 'performance-metrics'],
    queryFn: () => openevolveApi.getPerformanceMetrics(undefined, 200),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });

  const workflowQuery = useQuery({
    queryKey: ['analytics', 'workflow-metrics'],
    queryFn: () => openevolveApi.getWorkflowMetrics(),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });

  const knowledgeQuery = useQuery({
    queryKey: ['analytics', 'knowledge-stats'],
    queryFn: () => openevolveApi.getAnalyticsKnowledgeStats(),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });

  const isRefreshing =
    statisticsQuery.isFetching ||
    performanceQuery.isFetching ||
    workflowQuery.isFetching ||
    knowledgeQuery.isFetching;

  const handleRefresh = () => {
    void statisticsQuery.refetch();
    void performanceQuery.refetch();
    void workflowQuery.refetch();
    void knowledgeQuery.refetch();
  };

  const errorMessages = [
    statisticsQuery.isError && 'statistics',
    performanceQuery.isError && 'performance metrics',
    workflowQuery.isError && 'workflow metrics',
    knowledgeQuery.isError && 'knowledge stats',
  ].filter(Boolean) as string[];

  const performanceColumns: TableColumn<PerformanceMetric>[] = [
    {
      key: 'entity_type',
      header: 'Entity Type',
      render: (row) => (
        <span className="font-medium text-gray-900 dark:text-white">{row.entity_type}</span>
      ),
    },
    {
      key: 'entity_id',
      header: 'Entity ID',
      render: (row) => <code className="text-xs text-gray-500 dark:text-gray-400">{row.entity_id}</code>,
    },
    { key: 'domain', header: 'Domain', render: (row) => row.domain ?? '-' },
    { key: 'problem_type', header: 'Problem Type', render: (row) => row.problem_type ?? '-' },
    {
      key: 'metrics',
      header: 'Metrics',
      render: (row) => (
        <div className="flex flex-wrap gap-1">
          {Object.entries(row.metrics ?? {}).map(([k, v]) => (
            <span
              key={k}
              className="inline-flex items-center rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-700 dark:bg-gray-700 dark:text-gray-200"
              title={`${k}: ${v}`}
            >
              {k}: {formatNum(v)}
            </span>
          ))}
          {(!row.metrics || Object.keys(row.metrics).length === 0) && (
            <span className="text-gray-400">-</span>
          )}
        </div>
      ),
    },
    {
      key: 'timestamp',
      header: 'Timestamp',
      render: (row) => (
        <span className="whitespace-nowrap text-xs text-gray-500 dark:text-gray-400">
          {formatTimestamp(row.timestamp)}
        </span>
      ),
    },
  ];

  const workflowColumns: TableColumn<AnalyticsWorkflowMetric>[] = [
    {
      key: 'timestamp',
      header: 'Timestamp',
      render: (row) => (
        <span className="whitespace-nowrap text-xs text-gray-500 dark:text-gray-400">
          {formatTimestamp(row.timestamp)}
        </span>
      ),
    },
    {
      key: 'workflow_id',
      header: 'Workflow',
      render: (row) => <code className="text-xs text-gray-500 dark:text-gray-400">{row.workflow_id}</code>,
    },
    {
      key: 'status',
      header: 'Status',
      render: (row) => {
        const status = row.status ?? '-';
        const tone =
          status === 'completed' || status === 'success'
            ? 'text-green-600 dark:text-green-400'
            : status === 'failed' || status === 'error'
            ? 'text-red-600 dark:text-red-400'
            : 'text-gray-600 dark:text-gray-400';
        return <span className={`font-medium ${tone}`}>{status}</span>;
      },
    },
    { key: 'progress', header: 'Progress', align: 'right', render: (row) => formatPct(row.progress) },
    { key: 'best_fitness', header: 'Best Fit', align: 'right', render: (row) => formatNum(row.best_fitness) },
    { key: 'avg_fitness', header: 'Avg Fit', align: 'right', render: (row) => formatNum(row.avg_fitness) },
    { key: 'diversity', header: 'Diversity', align: 'right', render: (row) => formatNum(row.diversity) },
    {
      key: 'tokens_used',
      header: 'Tokens',
      align: 'right',
      render: (row) => formatNum(row.tokens_used),
    },
    {
      key: 'execution_time',
      header: 'Exec (s)',
      align: 'right',
      render: (row) => formatNum(row.execution_time),
    },
    {
      key: 'generation',
      header: 'Gen',
      align: 'right',
      render: (row) => formatNum(row.generation),
    },
  ];

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Analytics</h1>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            OpenEvolve performance, workflow, and knowledge statistics.
          </p>
        </div>
        <button
          type="button"
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        >
          <ArrowPathIcon className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          {isRefreshing ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {errorMessages.length > 0 && (
        <Alert type="error" title="Failed to load analytics">
          Could not load: {errorMessages.join(', ')}.
        </Alert>
      )}

      {/* ===================== Statistics Summary ===================== */}
      <Section
        title="Overview"
        subtitle="Aggregate statistics"
        icon={<BarChart3 className="h-5 w-5" />}
        query={useQueryState(statisticsQuery)}
        loadingText="Loading statistics…"
      >
        {(stats: StatisticsSummary) => (
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
            <StatCard
              label="Workflows"
              value={stats.total_workflows}
              accent="blue"
              icon={<Workflow className="h-5 w-5" />}
            />
            <StatCard
              label="Completed"
              value={stats.completed}
              accent="green"
              icon={<CheckCircle2 className="h-5 w-5" />}
            />
            <StatCard
              label="Failed"
              value={stats.failed}
              accent="red"
              icon={<XCircle className="h-5 w-5" />}
            />
            <StatCard
              label="Running"
              value={stats.running}
              accent="yellow"
              icon={<PlayCircle className="h-5 w-5" />}
            />
            <StatCard
              label="Teams"
              value={stats.total_teams}
              accent="purple"
              icon={<Cpu className="h-5 w-5" />}
            />
            <StatCard
              label="Gauntlets"
              value={stats.total_gauntlets}
              accent="gray"
              icon={<Activity className="h-5 w-5" />}
            />
          </div>
        )}
      </Section>

      {/* ===================== Performance Metrics ===================== */}
      <Section
        title="Performance Metrics"
        subtitle={`${performanceQuery.data?.total ?? 0} recorded entities`}
        icon={<Activity className="h-5 w-5" />}
        query={useQueryState(performanceQuery)}
        loadingText="Loading performance metrics…"
      >
        {(result: { metrics: PerformanceMetric[]; total: number }) => (
          <MetricTable
            columns={performanceColumns}
            data={result.metrics}
            rowKey={(row, index) => `${row.entity_type}-${row.entity_id}-${index}`}
            maxHeight={480}
            emptyMessage="No performance metrics recorded yet."
          />
        )}
      </Section>

      {/* ===================== Workflow Metrics ===================== */}
      <Section
        title="Workflow Metrics"
        subtitle={`${workflowQuery.data?.total ?? 0} workflow records`}
        icon={<Workflow className="h-5 w-5" />}
        query={useQueryState(workflowQuery)}
        loadingText="Loading workflow metrics…"
      >
        {(result: { metrics: AnalyticsWorkflowMetric[]; total: number }) => (
          <MetricTable
            columns={workflowColumns}
            data={result.metrics}
            rowKey={(row, index) => `${row.workflow_id}-${index}`}
            maxHeight={480}
            emptyMessage="No workflow metrics recorded yet."
          />
        )}
      </Section>

      {/* ===================== Knowledge Stats ===================== */}
      <Section
        title="Knowledge Stats"
        subtitle="Knowledge artifact usage and effectiveness"
        icon={<Database className="h-5 w-5" />}
        query={useQueryState(knowledgeQuery)}
        loadingText="Loading knowledge stats…"
      >
        {(stats: AnalyticsKnowledgeStats) => (
          <div className="space-y-6">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <StatCard
                label="Total Artifacts"
                value={stats.total_artifacts}
                accent="blue"
                icon={<BookOpen className="h-5 w-5" />}
              />
              <StatCard
                label="Total Usage"
                value={stats.total_usage}
                accent="green"
                icon={<Activity className="h-5 w-5" />}
              />
              <StatCard
                label="Avg Effectiveness"
                value={formatPct(stats.avg_effectiveness)}
                accent="purple"
                icon={<BarChart3 className="h-5 w-5" />}
              />
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <DistributionBars
                data={stats.artifact_type_distribution}
                title="Artifacts by Type"
                subtitle="Distribution across artifact types"
              />
              <DistributionBars
                data={stats.domain_distribution}
                title="Artifacts by Domain"
                subtitle="Distribution across domains"
              />
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <ArtifactList
                title="Top Used Artifacts"
                records={stats.top_used_artifacts}
                metricKey="usage_count"
                metricLabel="Usage"
              />
              <ArtifactList
                title="Top Effective Artifacts"
                records={stats.top_effective_artifacts}
                metricKey="effectiveness_score"
                metricLabel="Effectiveness"
              />
            </div>
          </div>
        )}
      </Section>
    </div>
  );
}

interface SectionProps<T> {
  title: string;
  subtitle?: string;
  icon: React.ReactNode;
  query: { isLoading: boolean; isError: boolean; error: unknown; data?: T };
  loadingText: string;
  children: (data: T) => React.ReactNode;
}

function Section<T>({ title, subtitle, icon, query, loadingText, children }: SectionProps<T>) {
  return (
    <Card
      title={title}
      subtitle={subtitle}
      actions={<span className="text-gray-400">{icon}</span>}
    >
      {query.isLoading && !query.data ? (
        <div className="flex items-center justify-center py-12">
          <LoadingSpinner text={loadingText} />
        </div>
      ) : query.isError && !query.data ? (
        <Alert type="error" title="Error">
          {(query.error as Error)?.message ?? 'Failed to load data.'}
        </Alert>
      ) : query.data ? (
        <>{children(query.data)}</>
      ) : (
        <p className="text-sm text-gray-500 dark:text-gray-400">No data available.</p>
      )}
    </Card>
  );
}

interface ArtifactListProps {
  title: string;
  records: Array<Record<string, unknown>>;
  metricKey: string;
  metricLabel: string;
}

function ArtifactList({ title, records, metricKey, metricLabel }: ArtifactListProps) {
  const describe = (record: Record<string, unknown>) => {
    const titleValue =
      (record.artifact_id as string) ??
      (record.id as string) ??
      (record.name as string) ??
      'Unknown artifact';
    const metric = record[metricKey];
    const metricText =
      typeof metric === 'number' ? metric.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '-';
    return { titleValue, metricText };
  };

  return (
    <Card title={title}>
      {records.length === 0 ? (
        <p className="text-sm text-gray-500 dark:text-gray-400">No artifacts available.</p>
      ) : (
        <ul className="divide-y divide-gray-200 dark:divide-gray-700">
          {records.map((record, index) => {
            const { titleValue, metricText } = describe(record);
            return (
              <li
                key={index}
                className="flex items-center justify-between gap-3 py-3"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-gray-900 dark:text-white">
                    {titleValue}
                  </p>
                  <p className="truncate text-xs text-gray-500 dark:text-gray-400">
                    {Object.entries(record)
                      .filter(([k]) => k !== 'artifact_id' && k !== 'id' && k !== 'name' && k !== metricKey)
                      .slice(0, 2)
                      .map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`)
                      .join(' · ') || '—'}
                  </p>
                </div>
                <div className="flex-shrink-0 text-right">
                  <p className="text-sm font-semibold text-gray-900 dark:text-white">{metricText}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">{metricLabel}</p>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}
