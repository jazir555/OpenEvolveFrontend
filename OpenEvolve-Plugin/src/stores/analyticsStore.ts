import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { errorLogger } from '@/utils';

/**
 * Time series data point
 */
export interface TimeSeriesDataPoint {
  timestamp: string;
  evolutions: number;
  adversarial_tests: number;
  average_fitness: number;
}

/**
 * Aggregated metrics
 */
export interface AggregatedMetrics {
  total_evolutions: number;
  total_adversarial_tests: number;
  average_fitness_improvement: number;
  average_vulnerabilities_found: number;
  success_rate: number;
}

/**
 * Model performance data
 */
export interface ModelPerformance {
  model: string;
  provider: string;
  total_calls: number;
  average_latency: number;
  success_rate: number;
  average_quality_score: number;
}

/**
 * Cost analysis data
 */
export interface CostAnalysis {
  total_cost: number;
  cost_by_model: Record<string, number>;
}

/**
 * Performance analytics
 */
export interface PerformanceAnalytics {
  model_performance: ModelPerformance[];
  cost_analysis: CostAnalysis;
}

/**
 * Analytics data response
 */
export interface AnalyticsData {
  period: {
    start: string;
    end: string;
    granularity: 'hour' | 'day' | 'week' | 'month';
  };
  metrics: AggregatedMetrics;
  time_series: TimeSeriesDataPoint[];
}

/**
 * Analytics state interface
 */
interface AnalyticsState {
  // Data
  metrics: AggregatedMetrics | null;
  timeSeries: TimeSeriesDataPoint[];
  performance: PerformanceAnalytics | null;

  // Filters
  dateRange: {
    start: Date | null;
    end: Date | null;
  };
  granularity: 'hour' | 'day' | 'week' | 'month';

  // UI state
  isLoading: boolean;
  error: string | null;
  selectedMetrics: string[];
  autoRefreshEnabled: boolean;
  refreshInterval: number; // seconds

  // Actions
  setMetrics: (metrics: AggregatedMetrics) => void;
  setTimeSeries: (data: TimeSeriesDataPoint[]) => void;
  setPerformance: (performance: PerformanceAnalytics) => void;
  setAnalyticsData: (data: AnalyticsData) => void;

  setDateRange: (start: Date | null, end: Date | null) => void;
  setGranularity: (granularity: 'hour' | 'day' | 'week' | 'month') => void;

  toggleMetric: (metric: string) => void;
  setSelectedMetrics: (metrics: string[]) => void;

  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  toggleAutoRefresh: () => void;
  setRefreshInterval: (interval: number) => void;

  reset: () => void;
}

/**
 * Default selected metrics
 */
const defaultSelectedMetrics = [
  'total_evolutions',
  'average_fitness_improvement',
  'success_rate',
];

/**
 * Analytics store
 */
export const useAnalyticsStore = create<AnalyticsState>()(
  devtools(
    (set, get) => ({
      metrics: null,
      timeSeries: [],
      performance: null,
      dateRange: {
        start: null,
        end: null,
      },
      granularity: 'day',
      isLoading: false,
      error: null,
      selectedMetrics: defaultSelectedMetrics,
      autoRefreshEnabled: true,
      refreshInterval: 30, // 30 seconds

      setMetrics: (metrics) => {
        try {
          set({ metrics });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AnalyticsStore', function: 'setMetrics', additionalData: { metrics } }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to set metrics' });
        }
      },

      setTimeSeries: (data) => {
        try {
          set({ timeSeries: data });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AnalyticsStore', function: 'setTimeSeries', additionalData: { data } }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to set time series data' });
        }
      },

      setPerformance: (performance) => {
        try {
          set({ performance });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AnalyticsStore', function: 'setPerformance', additionalData: { performance } }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to set performance data' });
        }
      },

      setAnalyticsData: (data) => {
        try {
          set({
            metrics: data.metrics,
            timeSeries: data.time_series,
          });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AnalyticsStore', function: 'setAnalyticsData', additionalData: { data } }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to set analytics data' });
        }
      },

      setDateRange: (start, end) => {
        try {
          set((state) => ({
            dateRange: { start: start || state.dateRange.start, end: end || state.dateRange.end },
          }));
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AnalyticsStore', function: 'setDateRange', additionalData: { start, end } }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to set date range' });
        }
      },

      setGranularity: (granularity) => {
        try {
          set({ granularity });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AnalyticsStore', function: 'setGranularity', additionalData: { granularity } }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to set granularity' });
        }
      },

      toggleMetric: (metric) => {
        try {
          set((state) => ({
            selectedMetrics: state.selectedMetrics.includes(metric)
              ? state.selectedMetrics.filter((m) => m !== metric)
              : [...state.selectedMetrics, metric],
          }));
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AnalyticsStore', function: 'toggleMetric', additionalData: { metric } }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to toggle metric' });
        }
      },

      setSelectedMetrics: (metrics) => {
        try {
          set({ selectedMetrics: metrics });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AnalyticsStore', function: 'setSelectedMetrics', additionalData: { metrics } }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to set selected metrics' });
        }
      },

      setLoading: (loading) => {
        try {
          set({ isLoading: loading });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AnalyticsStore', function: 'setLoading', additionalData: { loading } }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to set loading state' });
        }
      },

      setError: (error) => {
        try {
          set({ error: error });
        } catch (setErrorError) {
          errorLogger.logError(
            setErrorError instanceof Error ? setErrorError : new Error(String(setErrorError)),
            'error',
            { component: 'AnalyticsStore', function: 'setError', additionalData: { error } }
          );
        }
      },

      toggleAutoRefresh: () => {
        try {
          set((state) => ({
            autoRefreshEnabled: !state.autoRefreshEnabled,
          }));
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AnalyticsStore', function: 'toggleAutoRefresh' }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to toggle auto-refresh' });
        }
      },

      setRefreshInterval: (interval) => {
        try {
          set({ refreshInterval: interval });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AnalyticsStore', function: 'setRefreshInterval', additionalData: { interval } }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to set refresh interval' });
        }
      },

      reset: () => {
        try {
          set({
            metrics: null,
            timeSeries: [],
            performance: null,
            error: null,
          });
        } catch (error) {
          errorLogger.logError(
            error instanceof Error ? error : new Error(String(error)),
            'error',
            { component: 'AnalyticsStore', function: 'reset' }
          );
          set({ error: error instanceof Error ? error.message : 'Failed to reset analytics store' });
        }
      },
    }),
    { name: 'AnalyticsStore' }
  )
);
