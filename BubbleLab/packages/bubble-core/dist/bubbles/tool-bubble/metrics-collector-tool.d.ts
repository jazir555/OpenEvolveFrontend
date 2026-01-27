/**
 * METRICS COLLECTOR TOOL
 *
 * A comprehensive tool for collecting, aggregating, and analyzing metrics from various sources.
 * Supports application performance monitoring, business metrics, and custom metric collection.
 *
 * Features:
 * - Multi-source metric collection (APIs, databases, logs, files)
 * - Real-time and batch collection
 * - Metric aggregation and rollup
 * - Threshold-based alerting
 * - Metric visualization data generation
 * - Export to various formats (Prometheus, Graphite, JSON)
 * - Metric retention and archival
 */
import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Parameters schema
 */
declare const MetricsCollectorToolParamsSchema: z.ZodObject<{
    operation: z.ZodEnum<["collect", "aggregate", "query", "export", "alert", "compare", "forecast"]>;
    sources: z.ZodOptional<z.ZodArray<z.ZodObject<{
        type: z.ZodEnum<["api", "database", "file", "prometheus", "cloudwatch"]>;
        endpoint: z.ZodOptional<z.ZodString>;
        query: z.ZodOptional<z.ZodString>;
        interval: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        type: "file" | "database" | "api" | "prometheus" | "cloudwatch";
        query?: string | undefined;
        endpoint?: string | undefined;
        interval?: number | undefined;
    }, {
        type: "file" | "database" | "api" | "prometheus" | "cloudwatch";
        query?: string | undefined;
        endpoint?: string | undefined;
        interval?: number | undefined;
    }>, "many">>;
    metrics: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        value: z.ZodNumber;
        timestamp: z.ZodString;
        labels: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        unit: z.ZodOptional<z.ZodString>;
        type: z.ZodEnum<["gauge", "counter", "histogram", "summary"]>;
    }, "strip", z.ZodTypeAny, {
        value: number;
        type: "summary" | "gauge" | "counter" | "histogram";
        name: string;
        timestamp: string;
        labels?: Record<string, string> | undefined;
        unit?: string | undefined;
    }, {
        value: number;
        type: "summary" | "gauge" | "counter" | "histogram";
        name: string;
        timestamp: string;
        labels?: Record<string, string> | undefined;
        unit?: string | undefined;
    }>, "many">>;
    query: z.ZodOptional<z.ZodObject<{
        name: z.ZodOptional<z.ZodString>;
        labels: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        startTime: z.ZodOptional<z.ZodString>;
        endTime: z.ZodOptional<z.ZodString>;
        aggregation: z.ZodOptional<z.ZodEnum<["sum", "avg", "min", "max", "count"]>>;
        step: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name?: string | undefined;
        labels?: Record<string, string> | undefined;
        startTime?: string | undefined;
        endTime?: string | undefined;
        step?: string | undefined;
        aggregation?: "min" | "max" | "count" | "sum" | "avg" | undefined;
    }, {
        name?: string | undefined;
        labels?: Record<string, string> | undefined;
        startTime?: string | undefined;
        endTime?: string | undefined;
        step?: string | undefined;
        aggregation?: "min" | "max" | "count" | "sum" | "avg" | undefined;
    }>>;
    aggregation: z.ZodOptional<z.ZodObject<{
        window: z.ZodString;
        functions: z.ZodOptional<z.ZodDefault<z.ZodArray<z.ZodEnum<["sum", "avg", "min", "max", "count", "p50", "p95", "p99"]>, "many">>>;
        groupBy: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        window: string;
        functions?: ("min" | "max" | "count" | "sum" | "avg" | "p50" | "p95" | "p99")[] | undefined;
        groupBy?: string[] | undefined;
    }, {
        window: string;
        functions?: ("min" | "max" | "count" | "sum" | "avg" | "p50" | "p95" | "p99")[] | undefined;
        groupBy?: string[] | undefined;
    }>>;
    alerts: z.ZodOptional<z.ZodArray<z.ZodObject<{
        metricName: z.ZodString;
        condition: z.ZodEnum<["gt", "lt", "eq", "gte", "lte"]>;
        threshold: z.ZodNumber;
        duration: z.ZodOptional<z.ZodNumber>;
        severity: z.ZodEnum<["info", "warning", "critical"]>;
    }, "strip", z.ZodTypeAny, {
        severity: "info" | "warning" | "critical";
        condition: "lt" | "eq" | "gt" | "gte" | "lte";
        metricName: string;
        threshold: number;
        duration?: number | undefined;
    }, {
        severity: "info" | "warning" | "critical";
        condition: "lt" | "eq" | "gt" | "gte" | "lte";
        metricName: string;
        threshold: number;
        duration?: number | undefined;
    }>, "many">>;
    exportFormat: z.ZodOptional<z.ZodEnum<["json", "prometheus", "graphite", "csv", "influxdb"]>>;
    compareWith: z.ZodOptional<z.ZodObject<{
        period: z.ZodString;
        startTime: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        period: string;
        startTime?: string | undefined;
    }, {
        period: string;
        startTime?: string | undefined;
    }>>;
    forecast: z.ZodOptional<z.ZodObject<{
        horizon: z.ZodString;
        method: z.ZodOptional<z.ZodDefault<z.ZodEnum<["linear", "moving_average", "exponential"]>>>;
    }, "strip", z.ZodTypeAny, {
        horizon: string;
        method?: "exponential" | "linear" | "moving_average" | undefined;
    }, {
        horizon: string;
        method?: "exponential" | "linear" | "moving_average" | undefined;
    }>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    operation: "query" | "aggregate" | "export" | "collect" | "alert" | "compare" | "forecast";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    config?: Record<string, unknown> | undefined;
    query?: {
        name?: string | undefined;
        labels?: Record<string, string> | undefined;
        startTime?: string | undefined;
        endTime?: string | undefined;
        step?: string | undefined;
        aggregation?: "min" | "max" | "count" | "sum" | "avg" | undefined;
    } | undefined;
    sources?: {
        type: "file" | "database" | "api" | "prometheus" | "cloudwatch";
        query?: string | undefined;
        endpoint?: string | undefined;
        interval?: number | undefined;
    }[] | undefined;
    metrics?: {
        value: number;
        type: "summary" | "gauge" | "counter" | "histogram";
        name: string;
        timestamp: string;
        labels?: Record<string, string> | undefined;
        unit?: string | undefined;
    }[] | undefined;
    forecast?: {
        horizon: string;
        method?: "exponential" | "linear" | "moving_average" | undefined;
    } | undefined;
    aggregation?: {
        window: string;
        functions?: ("min" | "max" | "count" | "sum" | "avg" | "p50" | "p95" | "p99")[] | undefined;
        groupBy?: string[] | undefined;
    } | undefined;
    alerts?: {
        severity: "info" | "warning" | "critical";
        condition: "lt" | "eq" | "gt" | "gte" | "lte";
        metricName: string;
        threshold: number;
        duration?: number | undefined;
    }[] | undefined;
    exportFormat?: "json" | "csv" | "prometheus" | "graphite" | "influxdb" | undefined;
    compareWith?: {
        period: string;
        startTime?: string | undefined;
    } | undefined;
}, {
    operation: "query" | "aggregate" | "export" | "collect" | "alert" | "compare" | "forecast";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    config?: Record<string, unknown> | undefined;
    query?: {
        name?: string | undefined;
        labels?: Record<string, string> | undefined;
        startTime?: string | undefined;
        endTime?: string | undefined;
        step?: string | undefined;
        aggregation?: "min" | "max" | "count" | "sum" | "avg" | undefined;
    } | undefined;
    sources?: {
        type: "file" | "database" | "api" | "prometheus" | "cloudwatch";
        query?: string | undefined;
        endpoint?: string | undefined;
        interval?: number | undefined;
    }[] | undefined;
    metrics?: {
        value: number;
        type: "summary" | "gauge" | "counter" | "histogram";
        name: string;
        timestamp: string;
        labels?: Record<string, string> | undefined;
        unit?: string | undefined;
    }[] | undefined;
    forecast?: {
        horizon: string;
        method?: "exponential" | "linear" | "moving_average" | undefined;
    } | undefined;
    aggregation?: {
        window: string;
        functions?: ("min" | "max" | "count" | "sum" | "avg" | "p50" | "p95" | "p99")[] | undefined;
        groupBy?: string[] | undefined;
    } | undefined;
    alerts?: {
        severity: "info" | "warning" | "critical";
        condition: "lt" | "eq" | "gt" | "gte" | "lte";
        metricName: string;
        threshold: number;
        duration?: number | undefined;
    }[] | undefined;
    exportFormat?: "json" | "csv" | "prometheus" | "graphite" | "influxdb" | undefined;
    compareWith?: {
        period: string;
        startTime?: string | undefined;
    } | undefined;
}>;
/**
 * Result schema
 */
declare const MetricsCollectorToolResultSchema: z.ZodObject<{
    operation: z.ZodString;
    metrics: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        value: z.ZodNumber;
        timestamp: z.ZodString;
        labels: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        unit: z.ZodOptional<z.ZodString>;
        type: z.ZodEnum<["gauge", "counter", "histogram", "summary"]>;
    }, "strip", z.ZodTypeAny, {
        value: number;
        type: "summary" | "gauge" | "counter" | "histogram";
        name: string;
        timestamp: string;
        labels?: Record<string, string> | undefined;
        unit?: string | undefined;
    }, {
        value: number;
        type: "summary" | "gauge" | "counter" | "histogram";
        name: string;
        timestamp: string;
        labels?: Record<string, string> | undefined;
        unit?: string | undefined;
    }>, "many">>;
    aggregations: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        count: z.ZodNumber;
        min: z.ZodNumber;
        max: z.ZodNumber;
        avg: z.ZodNumber;
        sum: z.ZodNumber;
        p50: z.ZodOptional<z.ZodNumber>;
        p95: z.ZodOptional<z.ZodNumber>;
        p99: z.ZodOptional<z.ZodNumber>;
        timestamp: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        name: string;
        min: number;
        max: number;
        timestamp: string;
        count: number;
        sum: number;
        avg: number;
        p50?: number | undefined;
        p95?: number | undefined;
        p99?: number | undefined;
    }, {
        name: string;
        min: number;
        max: number;
        timestamp: string;
        count: number;
        sum: number;
        avg: number;
        p50?: number | undefined;
        p95?: number | undefined;
        p99?: number | undefined;
    }>, "many">>;
    alerts: z.ZodOptional<z.ZodArray<z.ZodObject<{
        condition: z.ZodObject<{
            metricName: z.ZodString;
            condition: z.ZodEnum<["gt", "lt", "eq", "gte", "lte"]>;
            threshold: z.ZodNumber;
            duration: z.ZodOptional<z.ZodNumber>;
            severity: z.ZodEnum<["info", "warning", "critical"]>;
        }, "strip", z.ZodTypeAny, {
            severity: "info" | "warning" | "critical";
            condition: "lt" | "eq" | "gt" | "gte" | "lte";
            metricName: string;
            threshold: number;
            duration?: number | undefined;
        }, {
            severity: "info" | "warning" | "critical";
            condition: "lt" | "eq" | "gt" | "gte" | "lte";
            metricName: string;
            threshold: number;
            duration?: number | undefined;
        }>;
        triggered: z.ZodBoolean;
        value: z.ZodNumber;
        message: z.ZodString;
        timestamp: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        value: number;
        message: string;
        timestamp: string;
        condition: {
            severity: "info" | "warning" | "critical";
            condition: "lt" | "eq" | "gt" | "gte" | "lte";
            metricName: string;
            threshold: number;
            duration?: number | undefined;
        };
        triggered: boolean;
    }, {
        value: number;
        message: string;
        timestamp: string;
        condition: {
            severity: "info" | "warning" | "critical";
            condition: "lt" | "eq" | "gt" | "gte" | "lte";
            metricName: string;
            threshold: number;
            duration?: number | undefined;
        };
        triggered: boolean;
    }>, "many">>;
    comparison: z.ZodOptional<z.ZodObject<{
        current: z.ZodRecord<z.ZodString, z.ZodNumber>;
        previous: z.ZodRecord<z.ZodString, z.ZodNumber>;
        change: z.ZodRecord<z.ZodString, z.ZodNumber>;
        changePercent: z.ZodRecord<z.ZodString, z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        change: Record<string, number>;
        current: Record<string, number>;
        previous: Record<string, number>;
        changePercent: Record<string, number>;
    }, {
        change: Record<string, number>;
        current: Record<string, number>;
        previous: Record<string, number>;
        changePercent: Record<string, number>;
    }>>;
    forecast: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        timestamp: z.ZodString;
        value: z.ZodNumber;
        confidence: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        value: number;
        name: string;
        timestamp: string;
        confidence?: number | undefined;
    }, {
        value: number;
        name: string;
        timestamp: string;
        confidence?: number | undefined;
    }>, "many">>;
    exportedData: z.ZodOptional<z.ZodString>;
    metadata: z.ZodObject<{
        metricsCollected: z.ZodNumber;
        sourcesQueried: z.ZodNumber;
        collectionTime: z.ZodNumber;
        timestamp: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        metricsCollected: number;
        sourcesQueried: number;
        collectionTime: number;
    }, {
        timestamp: string;
        metricsCollected: number;
        sourcesQueried: number;
        collectionTime: number;
    }>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: string;
    metadata: {
        timestamp: string;
        metricsCollected: number;
        sourcesQueried: number;
        collectionTime: number;
    };
    metrics?: {
        value: number;
        type: "summary" | "gauge" | "counter" | "histogram";
        name: string;
        timestamp: string;
        labels?: Record<string, string> | undefined;
        unit?: string | undefined;
    }[] | undefined;
    aggregations?: {
        name: string;
        min: number;
        max: number;
        timestamp: string;
        count: number;
        sum: number;
        avg: number;
        p50?: number | undefined;
        p95?: number | undefined;
        p99?: number | undefined;
    }[] | undefined;
    forecast?: {
        value: number;
        name: string;
        timestamp: string;
        confidence?: number | undefined;
    }[] | undefined;
    alerts?: {
        value: number;
        message: string;
        timestamp: string;
        condition: {
            severity: "info" | "warning" | "critical";
            condition: "lt" | "eq" | "gt" | "gte" | "lte";
            metricName: string;
            threshold: number;
            duration?: number | undefined;
        };
        triggered: boolean;
    }[] | undefined;
    comparison?: {
        change: Record<string, number>;
        current: Record<string, number>;
        previous: Record<string, number>;
        changePercent: Record<string, number>;
    } | undefined;
    exportedData?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: string;
    metadata: {
        timestamp: string;
        metricsCollected: number;
        sourcesQueried: number;
        collectionTime: number;
    };
    metrics?: {
        value: number;
        type: "summary" | "gauge" | "counter" | "histogram";
        name: string;
        timestamp: string;
        labels?: Record<string, string> | undefined;
        unit?: string | undefined;
    }[] | undefined;
    aggregations?: {
        name: string;
        min: number;
        max: number;
        timestamp: string;
        count: number;
        sum: number;
        avg: number;
        p50?: number | undefined;
        p95?: number | undefined;
        p99?: number | undefined;
    }[] | undefined;
    forecast?: {
        value: number;
        name: string;
        timestamp: string;
        confidence?: number | undefined;
    }[] | undefined;
    alerts?: {
        value: number;
        message: string;
        timestamp: string;
        condition: {
            severity: "info" | "warning" | "critical";
            condition: "lt" | "eq" | "gt" | "gte" | "lte";
            metricName: string;
            threshold: number;
            duration?: number | undefined;
        };
        triggered: boolean;
    }[] | undefined;
    comparison?: {
        change: Record<string, number>;
        current: Record<string, number>;
        previous: Record<string, number>;
        changePercent: Record<string, number>;
    } | undefined;
    exportedData?: string | undefined;
}>;
type MetricsCollectorToolParams = z.output<typeof MetricsCollectorToolParamsSchema>;
type MetricsCollectorToolResult = z.output<typeof MetricsCollectorToolResultSchema>;
type MetricsCollectorToolParamsInput = z.input<typeof MetricsCollectorToolParamsSchema>;
/**
 * Metrics Collector Tool
 * Collects and analyzes metrics from various sources
 */
export declare class MetricsCollectorTool extends ToolBubble<MetricsCollectorToolParams, MetricsCollectorToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        operation: z.ZodEnum<["collect", "aggregate", "query", "export", "alert", "compare", "forecast"]>;
        sources: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["api", "database", "file", "prometheus", "cloudwatch"]>;
            endpoint: z.ZodOptional<z.ZodString>;
            query: z.ZodOptional<z.ZodString>;
            interval: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            type: "file" | "database" | "api" | "prometheus" | "cloudwatch";
            query?: string | undefined;
            endpoint?: string | undefined;
            interval?: number | undefined;
        }, {
            type: "file" | "database" | "api" | "prometheus" | "cloudwatch";
            query?: string | undefined;
            endpoint?: string | undefined;
            interval?: number | undefined;
        }>, "many">>;
        metrics: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            value: z.ZodNumber;
            timestamp: z.ZodString;
            labels: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
            unit: z.ZodOptional<z.ZodString>;
            type: z.ZodEnum<["gauge", "counter", "histogram", "summary"]>;
        }, "strip", z.ZodTypeAny, {
            value: number;
            type: "summary" | "gauge" | "counter" | "histogram";
            name: string;
            timestamp: string;
            labels?: Record<string, string> | undefined;
            unit?: string | undefined;
        }, {
            value: number;
            type: "summary" | "gauge" | "counter" | "histogram";
            name: string;
            timestamp: string;
            labels?: Record<string, string> | undefined;
            unit?: string | undefined;
        }>, "many">>;
        query: z.ZodOptional<z.ZodObject<{
            name: z.ZodOptional<z.ZodString>;
            labels: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
            startTime: z.ZodOptional<z.ZodString>;
            endTime: z.ZodOptional<z.ZodString>;
            aggregation: z.ZodOptional<z.ZodEnum<["sum", "avg", "min", "max", "count"]>>;
            step: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name?: string | undefined;
            labels?: Record<string, string> | undefined;
            startTime?: string | undefined;
            endTime?: string | undefined;
            step?: string | undefined;
            aggregation?: "min" | "max" | "count" | "sum" | "avg" | undefined;
        }, {
            name?: string | undefined;
            labels?: Record<string, string> | undefined;
            startTime?: string | undefined;
            endTime?: string | undefined;
            step?: string | undefined;
            aggregation?: "min" | "max" | "count" | "sum" | "avg" | undefined;
        }>>;
        aggregation: z.ZodOptional<z.ZodObject<{
            window: z.ZodString;
            functions: z.ZodOptional<z.ZodDefault<z.ZodArray<z.ZodEnum<["sum", "avg", "min", "max", "count", "p50", "p95", "p99"]>, "many">>>;
            groupBy: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            window: string;
            functions?: ("min" | "max" | "count" | "sum" | "avg" | "p50" | "p95" | "p99")[] | undefined;
            groupBy?: string[] | undefined;
        }, {
            window: string;
            functions?: ("min" | "max" | "count" | "sum" | "avg" | "p50" | "p95" | "p99")[] | undefined;
            groupBy?: string[] | undefined;
        }>>;
        alerts: z.ZodOptional<z.ZodArray<z.ZodObject<{
            metricName: z.ZodString;
            condition: z.ZodEnum<["gt", "lt", "eq", "gte", "lte"]>;
            threshold: z.ZodNumber;
            duration: z.ZodOptional<z.ZodNumber>;
            severity: z.ZodEnum<["info", "warning", "critical"]>;
        }, "strip", z.ZodTypeAny, {
            severity: "info" | "warning" | "critical";
            condition: "lt" | "eq" | "gt" | "gte" | "lte";
            metricName: string;
            threshold: number;
            duration?: number | undefined;
        }, {
            severity: "info" | "warning" | "critical";
            condition: "lt" | "eq" | "gt" | "gte" | "lte";
            metricName: string;
            threshold: number;
            duration?: number | undefined;
        }>, "many">>;
        exportFormat: z.ZodOptional<z.ZodEnum<["json", "prometheus", "graphite", "csv", "influxdb"]>>;
        compareWith: z.ZodOptional<z.ZodObject<{
            period: z.ZodString;
            startTime: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            period: string;
            startTime?: string | undefined;
        }, {
            period: string;
            startTime?: string | undefined;
        }>>;
        forecast: z.ZodOptional<z.ZodObject<{
            horizon: z.ZodString;
            method: z.ZodOptional<z.ZodDefault<z.ZodEnum<["linear", "moving_average", "exponential"]>>>;
        }, "strip", z.ZodTypeAny, {
            horizon: string;
            method?: "exponential" | "linear" | "moving_average" | undefined;
        }, {
            horizon: string;
            method?: "exponential" | "linear" | "moving_average" | undefined;
        }>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
        config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        operation: "query" | "aggregate" | "export" | "collect" | "alert" | "compare" | "forecast";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
        query?: {
            name?: string | undefined;
            labels?: Record<string, string> | undefined;
            startTime?: string | undefined;
            endTime?: string | undefined;
            step?: string | undefined;
            aggregation?: "min" | "max" | "count" | "sum" | "avg" | undefined;
        } | undefined;
        sources?: {
            type: "file" | "database" | "api" | "prometheus" | "cloudwatch";
            query?: string | undefined;
            endpoint?: string | undefined;
            interval?: number | undefined;
        }[] | undefined;
        metrics?: {
            value: number;
            type: "summary" | "gauge" | "counter" | "histogram";
            name: string;
            timestamp: string;
            labels?: Record<string, string> | undefined;
            unit?: string | undefined;
        }[] | undefined;
        forecast?: {
            horizon: string;
            method?: "exponential" | "linear" | "moving_average" | undefined;
        } | undefined;
        aggregation?: {
            window: string;
            functions?: ("min" | "max" | "count" | "sum" | "avg" | "p50" | "p95" | "p99")[] | undefined;
            groupBy?: string[] | undefined;
        } | undefined;
        alerts?: {
            severity: "info" | "warning" | "critical";
            condition: "lt" | "eq" | "gt" | "gte" | "lte";
            metricName: string;
            threshold: number;
            duration?: number | undefined;
        }[] | undefined;
        exportFormat?: "json" | "csv" | "prometheus" | "graphite" | "influxdb" | undefined;
        compareWith?: {
            period: string;
            startTime?: string | undefined;
        } | undefined;
    }, {
        operation: "query" | "aggregate" | "export" | "collect" | "alert" | "compare" | "forecast";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
        query?: {
            name?: string | undefined;
            labels?: Record<string, string> | undefined;
            startTime?: string | undefined;
            endTime?: string | undefined;
            step?: string | undefined;
            aggregation?: "min" | "max" | "count" | "sum" | "avg" | undefined;
        } | undefined;
        sources?: {
            type: "file" | "database" | "api" | "prometheus" | "cloudwatch";
            query?: string | undefined;
            endpoint?: string | undefined;
            interval?: number | undefined;
        }[] | undefined;
        metrics?: {
            value: number;
            type: "summary" | "gauge" | "counter" | "histogram";
            name: string;
            timestamp: string;
            labels?: Record<string, string> | undefined;
            unit?: string | undefined;
        }[] | undefined;
        forecast?: {
            horizon: string;
            method?: "exponential" | "linear" | "moving_average" | undefined;
        } | undefined;
        aggregation?: {
            window: string;
            functions?: ("min" | "max" | "count" | "sum" | "avg" | "p50" | "p95" | "p99")[] | undefined;
            groupBy?: string[] | undefined;
        } | undefined;
        alerts?: {
            severity: "info" | "warning" | "critical";
            condition: "lt" | "eq" | "gt" | "gte" | "lte";
            metricName: string;
            threshold: number;
            duration?: number | undefined;
        }[] | undefined;
        exportFormat?: "json" | "csv" | "prometheus" | "graphite" | "influxdb" | undefined;
        compareWith?: {
            period: string;
            startTime?: string | undefined;
        } | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        operation: z.ZodString;
        metrics: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            value: z.ZodNumber;
            timestamp: z.ZodString;
            labels: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
            unit: z.ZodOptional<z.ZodString>;
            type: z.ZodEnum<["gauge", "counter", "histogram", "summary"]>;
        }, "strip", z.ZodTypeAny, {
            value: number;
            type: "summary" | "gauge" | "counter" | "histogram";
            name: string;
            timestamp: string;
            labels?: Record<string, string> | undefined;
            unit?: string | undefined;
        }, {
            value: number;
            type: "summary" | "gauge" | "counter" | "histogram";
            name: string;
            timestamp: string;
            labels?: Record<string, string> | undefined;
            unit?: string | undefined;
        }>, "many">>;
        aggregations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            count: z.ZodNumber;
            min: z.ZodNumber;
            max: z.ZodNumber;
            avg: z.ZodNumber;
            sum: z.ZodNumber;
            p50: z.ZodOptional<z.ZodNumber>;
            p95: z.ZodOptional<z.ZodNumber>;
            p99: z.ZodOptional<z.ZodNumber>;
            timestamp: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            name: string;
            min: number;
            max: number;
            timestamp: string;
            count: number;
            sum: number;
            avg: number;
            p50?: number | undefined;
            p95?: number | undefined;
            p99?: number | undefined;
        }, {
            name: string;
            min: number;
            max: number;
            timestamp: string;
            count: number;
            sum: number;
            avg: number;
            p50?: number | undefined;
            p95?: number | undefined;
            p99?: number | undefined;
        }>, "many">>;
        alerts: z.ZodOptional<z.ZodArray<z.ZodObject<{
            condition: z.ZodObject<{
                metricName: z.ZodString;
                condition: z.ZodEnum<["gt", "lt", "eq", "gte", "lte"]>;
                threshold: z.ZodNumber;
                duration: z.ZodOptional<z.ZodNumber>;
                severity: z.ZodEnum<["info", "warning", "critical"]>;
            }, "strip", z.ZodTypeAny, {
                severity: "info" | "warning" | "critical";
                condition: "lt" | "eq" | "gt" | "gte" | "lte";
                metricName: string;
                threshold: number;
                duration?: number | undefined;
            }, {
                severity: "info" | "warning" | "critical";
                condition: "lt" | "eq" | "gt" | "gte" | "lte";
                metricName: string;
                threshold: number;
                duration?: number | undefined;
            }>;
            triggered: z.ZodBoolean;
            value: z.ZodNumber;
            message: z.ZodString;
            timestamp: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            value: number;
            message: string;
            timestamp: string;
            condition: {
                severity: "info" | "warning" | "critical";
                condition: "lt" | "eq" | "gt" | "gte" | "lte";
                metricName: string;
                threshold: number;
                duration?: number | undefined;
            };
            triggered: boolean;
        }, {
            value: number;
            message: string;
            timestamp: string;
            condition: {
                severity: "info" | "warning" | "critical";
                condition: "lt" | "eq" | "gt" | "gte" | "lte";
                metricName: string;
                threshold: number;
                duration?: number | undefined;
            };
            triggered: boolean;
        }>, "many">>;
        comparison: z.ZodOptional<z.ZodObject<{
            current: z.ZodRecord<z.ZodString, z.ZodNumber>;
            previous: z.ZodRecord<z.ZodString, z.ZodNumber>;
            change: z.ZodRecord<z.ZodString, z.ZodNumber>;
            changePercent: z.ZodRecord<z.ZodString, z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            change: Record<string, number>;
            current: Record<string, number>;
            previous: Record<string, number>;
            changePercent: Record<string, number>;
        }, {
            change: Record<string, number>;
            current: Record<string, number>;
            previous: Record<string, number>;
            changePercent: Record<string, number>;
        }>>;
        forecast: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            timestamp: z.ZodString;
            value: z.ZodNumber;
            confidence: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            value: number;
            name: string;
            timestamp: string;
            confidence?: number | undefined;
        }, {
            value: number;
            name: string;
            timestamp: string;
            confidence?: number | undefined;
        }>, "many">>;
        exportedData: z.ZodOptional<z.ZodString>;
        metadata: z.ZodObject<{
            metricsCollected: z.ZodNumber;
            sourcesQueried: z.ZodNumber;
            collectionTime: z.ZodNumber;
            timestamp: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            timestamp: string;
            metricsCollected: number;
            sourcesQueried: number;
            collectionTime: number;
        }, {
            timestamp: string;
            metricsCollected: number;
            sourcesQueried: number;
            collectionTime: number;
        }>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: string;
        metadata: {
            timestamp: string;
            metricsCollected: number;
            sourcesQueried: number;
            collectionTime: number;
        };
        metrics?: {
            value: number;
            type: "summary" | "gauge" | "counter" | "histogram";
            name: string;
            timestamp: string;
            labels?: Record<string, string> | undefined;
            unit?: string | undefined;
        }[] | undefined;
        aggregations?: {
            name: string;
            min: number;
            max: number;
            timestamp: string;
            count: number;
            sum: number;
            avg: number;
            p50?: number | undefined;
            p95?: number | undefined;
            p99?: number | undefined;
        }[] | undefined;
        forecast?: {
            value: number;
            name: string;
            timestamp: string;
            confidence?: number | undefined;
        }[] | undefined;
        alerts?: {
            value: number;
            message: string;
            timestamp: string;
            condition: {
                severity: "info" | "warning" | "critical";
                condition: "lt" | "eq" | "gt" | "gte" | "lte";
                metricName: string;
                threshold: number;
                duration?: number | undefined;
            };
            triggered: boolean;
        }[] | undefined;
        comparison?: {
            change: Record<string, number>;
            current: Record<string, number>;
            previous: Record<string, number>;
            changePercent: Record<string, number>;
        } | undefined;
        exportedData?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: string;
        metadata: {
            timestamp: string;
            metricsCollected: number;
            sourcesQueried: number;
            collectionTime: number;
        };
        metrics?: {
            value: number;
            type: "summary" | "gauge" | "counter" | "histogram";
            name: string;
            timestamp: string;
            labels?: Record<string, string> | undefined;
            unit?: string | undefined;
        }[] | undefined;
        aggregations?: {
            name: string;
            min: number;
            max: number;
            timestamp: string;
            count: number;
            sum: number;
            avg: number;
            p50?: number | undefined;
            p95?: number | undefined;
            p99?: number | undefined;
        }[] | undefined;
        forecast?: {
            value: number;
            name: string;
            timestamp: string;
            confidence?: number | undefined;
        }[] | undefined;
        alerts?: {
            value: number;
            message: string;
            timestamp: string;
            condition: {
                severity: "info" | "warning" | "critical";
                condition: "lt" | "eq" | "gt" | "gte" | "lte";
                metricName: string;
                threshold: number;
                duration?: number | undefined;
            };
            triggered: boolean;
        }[] | undefined;
        comparison?: {
            change: Record<string, number>;
            current: Record<string, number>;
            previous: Record<string, number>;
            changePercent: Record<string, number>;
        } | undefined;
        exportedData?: string | undefined;
    }>;
    static readonly shortDescription = "Collect and analyze metrics";
    static readonly longDescription = "\n    Comprehensive metrics collection and analysis tool.\n\n    Operations:\n    - collect: Collect metrics from various sources\n    - aggregate: Aggregate metrics over time windows\n    - query: Query stored metrics\n    - export: Export metrics to different formats\n    - alert: Check metric values against thresholds\n    - compare: Compare metrics with previous periods\n    - forecast: Forecast future metric values\n\n    Supported sources:\n    - REST APIs\n    - Databases (via queries)\n    - Files (JSON, CSV)\n    - Prometheus\n    - CloudWatch\n\n    Features:\n    - Real-time collection\n    - Time-series aggregation\n    - Threshold alerting\n    - Period comparison\n    - Trend forecasting\n    - Multi-format export\n  ";
    static readonly alias = "metrics";
    static readonly type = "tool";
    private static metricStore;
    private static readonly MAX_METRICS_PER_NAME;
    private static readonly METRIC_TTL;
    private static readonly CLEANUP_INTERVAL;
    private static lastCleanup;
    private static aggregator;
    constructor(params?: MetricsCollectorToolParamsInput, context?: BubbleContext);
    performAction(): Promise<MetricsCollectorToolResult>;
    private collectMetrics;
    private collectFromSource;
    private collectFromAPI;
    private collectFromPrometheus;
    private collectFromFile;
    /**
     * Parse CSV line handling quoted fields
     */
    private parseCSVLine;
    private aggregateMetrics;
    private calculatePercentile;
    private queryMetrics;
    private exportMetrics;
    private exportToPrometheusFormat;
    private exportToGraphiteFormat;
    private exportToCsvFormat;
    private checkAlerts;
    private compareMetrics;
    private calculateAggregateValues;
    private parsePeriod;
    private forecastMetrics;
    private applyForecastMethod;
    private createErrorResult;
    /**
     * Cleanup old metrics based on TTL
     * Runs periodically based on CLEANUP_INTERVAL
     */
    private cleanupOldMetrics;
}
export {};
//# sourceMappingURL=metrics-collector-tool.d.ts.map