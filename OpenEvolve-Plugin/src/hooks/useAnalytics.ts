/**
 * useAnalytics Hook
 *
 * Provides a lightweight analytics summary derived from the analytics APIs.
 */

import { useQuery } from '@tanstack/react-query';
import { analyticsApi } from '@/services/api';
import type { AnalyticsData as AnalyticsResponse } from '@/stores/analyticsStore';

export interface AnalyticsData {
  totalEvolutions: number;
  evolutionsChange: number;
  successRate: number;
  avgExecutionTime: number;
  successRateChange?: number;
  avgDuration?: number;
  durationChange?: number;
  activeWorkflows?: number;
  workflowsChange?: number;
  performanceOverTime?: Array<{ timestamp: string; value: number }>;
  artifacts?: Array<any>;
}

type DateRangeInput = string | { start: string; end: string; granularity?: 'hour' | 'day' | 'week' | 'month' };

const parseDateRange = (range: DateRangeInput) => {
  if (typeof range !== 'string') {
    return {
      start: range.start,
      end: range.end,
      granularity: range.granularity || 'day',
    };
  }

  const now = new Date();
  const end = now.toISOString();

  if (range.includes(':')) {
    const [startRaw, endRaw] = range.split(':');
    const startDate = new Date(startRaw.trim());
    const endDate = new Date(endRaw.trim());
    return {
      start: isNaN(startDate.getTime()) ? end : startDate.toISOString(),
      end: isNaN(endDate.getTime()) ? end : endDate.toISOString(),
      granularity: 'day' as const,
    };
  }

  const days = range.endsWith('d') ? parseInt(range, 10) : undefined;
  if (days && !Number.isNaN(days)) {
    const startDate = new Date(now);
    startDate.setDate(now.getDate() - days);
    return { start: startDate.toISOString(), end, granularity: 'day' as const };
  }

  return {
    start: new Date('2000-01-01T00:00:00Z').toISOString(),
    end,
    granularity: 'month' as const,
  };
};

const buildSummary = (metrics: AnalyticsResponse, performance: any): AnalyticsData => {
  const aggregated = metrics?.metrics;
  const timeSeries = Array.isArray(metrics?.time_series) ? metrics.time_series : [];
  const modelPerformance = Array.isArray(performance?.model_performance)
    ? performance.model_performance
    : [];

  const avgExecutionTime =
    modelPerformance.length
      ? Math.round(
          modelPerformance.reduce(
            (sum: number, model: { average_latency: number }) =>
              sum + (model.average_latency || 0),
            0
          ) / modelPerformance.length
        )
      : 0;

  return {
    totalEvolutions: aggregated?.total_evolutions || 0,
    evolutionsChange: 0,
    successRate: aggregated?.success_rate || 0,
    avgExecutionTime,
    performanceOverTime: timeSeries.map((point) => ({
      timestamp: point.timestamp,
      value: point.average_fitness ?? 0,
    })),
  };
};

export function useAnalytics(dateRange: DateRangeInput) {
  return useQuery({
    queryKey: ['analytics', dateRange],
    queryFn: async (): Promise<AnalyticsData> => {
      const range = parseDateRange(dateRange);
      const [metrics, performance] = await Promise.all([
        analyticsApi.getMetrics({
          start_date: range.start,
          end_date: range.end,
          granularity: range.granularity,
        }),
        analyticsApi.getPerformance(),
      ]);

      return buildSummary(metrics as AnalyticsResponse, performance);
    },
  });
}
