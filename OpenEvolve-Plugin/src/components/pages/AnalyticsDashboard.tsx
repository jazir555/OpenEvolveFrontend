import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useAnalytics, useMonitoring } from '@/services/hooks/useApi';
import { contentApi, monitoringApi } from '@/services/api';
import { StatGrid } from '@/components/analytics/StatGrid';
import { PerformanceChart } from '@/components/analytics/PerformanceChart';
import { ArtifactTable } from '@/components/analytics/ArtifactTable';
import { BubbleButton, BubbleCard } from '@/components/bubblelab';
import { VizErrorBoundary } from '@/components/shared/VizErrorBoundary';
import { PageErrorBoundary } from '@/components/shared/PageErrorBoundary';
import { withComponentBoundary } from '@/components/shared/ComponentErrorBoundary';
import {
  ArtifactViz,
  GlobalAnalyticsViz,
  LineageViz,
  PyGraphistryViz,
} from 'openevolve-pygraphistry-plugin';
import type { GraphNode, GraphEdge } from 'openevolve-pygraphistry-plugin';

function AnalyticsDashboardBase() {
  const [dateRange, setDateRange] = useState<'7d' | '30d' | '90d' | 'all'>('30d');
  const range = useMemo(() => {
    const now = new Date();
    let start = new Date(now);
    if (dateRange === '7d') {
      start.setDate(now.getDate() - 7);
    } else if (dateRange === '30d') {
      start.setDate(now.getDate() - 30);
    } else if (dateRange === '90d') {
      start.setDate(now.getDate() - 90);
    } else {
      start = new Date('2000-01-01T00:00:00Z');
    }
    return {
      start: start.toISOString(),
      end: now.toISOString(),
      granularity: dateRange === '90d' ? 'week' : 'day',
    } as const;
  }, [dateRange]);

  const { metrics, performance, isLoading, error } = useAnalytics(range);
  const { health } = useMonitoring(15000);

  const artifactsQuery = useQuery({
    queryKey: ['analytics', 'artifacts', 'recent'],
    queryFn: async () => {
      const response = await contentApi.list({ limit: 10, offset: 0 });
      return response?.content || [];
    },
    staleTime: 60000,
  });

  const activityQuery = useQuery({
    queryKey: ['monitoring', 'logs', 'recent'],
    queryFn: async () => {
      const response = await monitoringApi.getLogs({ limit: 6, offset: 0 });
      return response?.logs || [];
    },
    refetchInterval: 30000,
  });

  const aggregated = metrics?.metrics;
  const timeSeries = metrics?.time_series || [];

  const stats = [
    {
      title: 'Total Evolutions',
      value: aggregated?.total_evolutions || 0,
      change: 0,
      unit: '',
      trend: 'up' as const,
    },
    {
      title: 'Success Rate',
      value: Math.round((aggregated?.success_rate ?? 0) * 100),
      change: 0,
      unit: '%',
      trend: 'up' as const,
    },
    {
      title: 'Adversarial Tests',
      value: aggregated?.total_adversarial_tests || 0,
      change: 0,
      unit: '',
      trend: 'up' as const,
    },
    {
      title: 'Avg. Fitness Gain',
      value: aggregated?.average_fitness_improvement || 0,
      change: 0,
      unit: '',
      trend: 'up' as const,
    },
  ];

  const performanceData =
    timeSeries.length > 0
      ? timeSeries.map((point) => ({
          timestamp: point.timestamp,
          value: point.average_fitness ?? 0,
        }))
      : [];

  const evolutionsOverTime =
    timeSeries.length > 0
      ? timeSeries.map((point) => ({
          timestamp: point.timestamp,
          value: point.evolutions ?? 0,
        }))
      : [];

  const topModels = performance?.model_performance || [];
  const artifacts = (artifactsQuery.data || []).map((artifact) => ({
    id: artifact.artifact_id || artifact.id || artifact.title,
    name: artifact.title || 'Untitled',
    type: artifact.language || 'artifact',
    created: artifact.updated_at || artifact.created_at || new Date().toISOString(),
    status: 'success',
  }));
  const activityLogs = activityQuery.data || [];
  const formatLogDate = (value: string) => {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'Unknown' : date.toLocaleString();
  };

  const graphData = useMemo(() => {
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];

    if (!aggregated) {
      return { nodes, edges };
    }

    nodes.push(
      { id: 'analytics', label: 'Analytics', type: 'summary' },
      { id: 'total_evolutions', label: `Evolutions ${aggregated.total_evolutions}`, type: 'metric' },
      { id: 'success_rate', label: `Success ${((aggregated.success_rate ?? 0) * 100).toFixed(0)}%`, type: 'metric' },
      { id: 'avg_fitness', label: `Fitness ${aggregated.average_fitness_improvement ?? 0}`, type: 'metric' },
    );

    edges.push(
      { source: 'analytics', target: 'total_evolutions', type: 'metric' },
      { source: 'analytics', target: 'success_rate', type: 'metric' },
      { source: 'analytics', target: 'avg_fitness', type: 'metric' },
    );

    topModels.slice(0, 5).forEach((model) => {
      const nodeId = `model-${model.provider}-${model.model}`;
      nodes.push({
        id: nodeId,
        label: model.model,
        type: model.provider,
      });
      edges.push({
        source: 'analytics',
        target: nodeId,
        type: 'model',
        weight: model.success_rate ?? 0,
      });
    });

    return { nodes, edges };
  }, [aggregated, topModels]);

  const pageError =
    (error instanceof Error ? error.message : error ? String(error) : null) ||
    (artifactsQuery.error instanceof Error ? artifactsQuery.error.message : null) ||
    (activityQuery.error instanceof Error ? activityQuery.error.message : null);

  return (
    <PageErrorBoundary label="Analytics dashboard">
      <div className="analytics-dashboard min-h-screen bg-slate-50 p-6">
      <header className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Analytics Dashboard</h1>
            <p className="text-slate-600 mt-1">Track your evolution performance and metrics</p>
          </div>

          <div className="flex gap-2">
            {(['7d', '30d', '90d', 'all'] as const).map((range) => (
              <BubbleButton
                key={range}
                onClick={() => setDateRange(range)}
                variant={dateRange === range ? 'primary' : 'secondary'}
              >
                {range === '7d' && 'Last 7 Days'}
                {range === '30d' && 'Last 30 Days'}
                {range === '90d' && 'Last 90 Days'}
                {range === 'all' && 'All Time'}
              </BubbleButton>
            ))}
          </div>
        </div>
      </header>

      {pageError ? (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          Failed to load analytics data: {pageError}
        </div>
      ) : isLoading ? (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
          <p className="mt-4 text-gray-600">Loading analytics...</p>
        </div>
      ) : (
        <div className="space-y-6">
          <StatGrid stats={stats} columns={4} />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PerformanceChart
              data={performanceData}
              type="line"
              title="Fitness Over Time"
              color="#3b82f6"
            />
            <PerformanceChart
              data={evolutionsOverTime}
              type="bar"
              title="Evolutions Completed"
              color="#10b981"
            />
          </div>

          <BubbleCard title="Recent Artifacts" description="Artifacts created during this period.">
            <ArtifactTable
              artifacts={artifacts}
              onRowClick={(artifact) => console.log('Selected:', artifact)}
            />
          </BubbleCard>

          <BubbleCard
            title="Graphistry Global Analytics"
            description="PyGraphistry-backed global performance and cost breakdowns."
          >
            <VizErrorBoundary label="Global analytics visualization">
            <GlobalAnalyticsViz />
          </VizErrorBoundary>
          </BubbleCard>

          <BubbleCard
            title="Analytics Graph"
            description="Graphistry view of core metrics and top model performance."
          >
            <VizErrorBoundary label="Analytics graph">
              <PyGraphistryViz
                nodes={graphData.nodes}
                edges={graphData.edges}
                height={360}
              />
            </VizErrorBoundary>
          </BubbleCard>

          <BubbleCard
            title="Artifact Graph"
            description="PyGraphistry artifact relationship map generated from OpenEvolve outputs."
          >
            <VizErrorBoundary label="Artifact graph">
            <ArtifactViz />
          </VizErrorBoundary>
          </BubbleCard>

          <BubbleCard
            title="Evolution Lineage"
            description="Ancestral traces for the latest evolutionary runs."
          >
            <VizErrorBoundary label="Evolution lineage">
            <LineageViz />
          </VizErrorBoundary>
          </BubbleCard>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <BubbleCard title="Top Performing Models" description="Quality and latency leaders.">
              <div className="space-y-2">
                {topModels.length === 0 && (
                  <p className="text-sm text-slate-400">No model performance data.</p>
                )}
                {topModels.slice(0, 5).map((model) => (
                  <div key={`${model.provider}-${model.model}`} className="flex items-center justify-between text-sm">
                    <span className="text-slate-800">{model.model}</span>
                    <span className="text-slate-600">{Math.round((model.success_rate ?? 0) * 100)}%</span>
                  </div>
                ))}
              </div>
            </BubbleCard>

            <BubbleCard title="Resource Usage" description="Live utilization from monitoring.">
              <div className="space-y-3 text-sm text-slate-600">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">CPU</span>
                  <span className="font-medium">{health?.resource_usage?.cpu_percent ?? 0}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Memory</span>
                  <span className="font-medium">{health?.resource_usage?.memory_percent ?? 0}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Disk</span>
                  <span className="font-medium">{health?.resource_usage?.disk_percent ?? 0}%</span>
                </div>
              </div>
            </BubbleCard>

            <BubbleCard title="Recent Activity" description="Audit events and workflow updates.">
              {activityQuery.isLoading && (
                <p className="text-sm text-slate-500">Loading activity...</p>
              )}
              {!activityQuery.isLoading && activityLogs.length === 0 && (
                <p className="text-sm text-slate-500">No recent activity.</p>
              )}
              {activityLogs.length > 0 && (
                <div className="space-y-3 text-sm text-slate-600">
                  {activityLogs.map((log) => (
                    <div key={`${log.timestamp}-${log.message}`} className="rounded-md bg-slate-50 px-3 py-2">
                      <div className="flex items-center justify-between text-xs text-slate-400">
                        <span>{log.level}</span>
                        <span>{formatLogDate(log.timestamp)}</span>
                      </div>
                      <p className="text-sm text-slate-700">{log.message}</p>
                    </div>
                  ))}
                </div>
              )}
            </BubbleCard>
          </div>
        </div>
      )}
      </div>
    </PageErrorBoundary>
  );
}

export const AnalyticsDashboard = withComponentBoundary(
  AnalyticsDashboardBase,
  'AnalyticsDashboard'
);
