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
    metrics: AggregatedMetrics | null;
    timeSeries: TimeSeriesDataPoint[];
    performance: PerformanceAnalytics | null;
    dateRange: {
        start: Date | null;
        end: Date | null;
    };
    granularity: 'hour' | 'day' | 'week' | 'month';
    isLoading: boolean;
    error: string | null;
    selectedMetrics: string[];
    autoRefreshEnabled: boolean;
    refreshInterval: number;
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
 * Analytics store
 */
export declare const useAnalyticsStore: import('zustand').UseBoundStore<Omit<import('zustand').StoreApi<AnalyticsState>, "setState"> & {
    setState<A extends string | {
        type: string;
    }>(partial: AnalyticsState | Partial<AnalyticsState> | ((state: AnalyticsState) => AnalyticsState | Partial<AnalyticsState>), replace?: boolean, action?: A): void;
}>;
export {};
