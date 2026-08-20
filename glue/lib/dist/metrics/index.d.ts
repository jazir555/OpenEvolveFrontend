/**
 * Metrics and Monitoring Library
 *
 * Central exports and initialization for the monitoring system
 *
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: All config via environment variables
 * - Observability: Comprehensive monitoring with Prometheus, OpenTelemetry
 */
import { MetricsCollector, getMetricsCollector } from './metrics-collector';
import { HealthChecker, HealthEndpointHandler } from './health-checker';
import { Tracer, getTracer } from './tracer';
import { AlertManager, getAlertManager } from './alert-manager';
export interface MonitoringConfig {
    serviceName: string;
    prometheus?: {
        prefix?: string;
        port?: number;
    };
    otel?: {
        endpoint?: string;
        headers?: Record<string, string>;
    };
    health?: {
        enabled?: boolean;
        endpoint?: string;
    };
    alerts?: {
        enabled?: boolean;
        webhookUrl?: string;
        slackWebhookUrl?: string;
    };
}
/**
 * Initialize monitoring system
 */
export declare function initializeMonitoring(config: MonitoringConfig): Promise<{
    metrics: MetricsCollector;
    health: HealthChecker;
    tracer: Tracer;
    alerts: AlertManager;
}>;
/**
 * Create Express middleware for metrics endpoint
 */
export declare function createMetricsMiddleware(): (req: any, res: any, next: any) => Promise<void>;
/**
 * Create Express middleware for health endpoints
 */
export declare function createHealthMiddleware(health: HealthChecker): (req: any, res: any, next: any) => Promise<void>;
/**
 * Create HTTP request tracking middleware
 */
export declare function createRequestTrackingMiddleware(serviceName?: string): (req: any, res: any, next: any) => Promise<void>;
export { MetricsCollector, getMetricsCollector, HealthChecker, HealthEndpointHandler, Tracer, getTracer, AlertManager, getAlertManager, };
export type { MetricsLabels, KnowledgeExtractionLabels } from './metrics-collector';
export type { HealthStatus, HealthCheckResult, HealthCheckOptions, HealthCheckFunction, } from './health-checker';
export type { TraceOptions, SpanMetadata } from './tracer';
export type { AlertSeverity, AlertRule, AlertCondition, NotificationChannel, Alert, AlertHistory, } from './alert-manager';
//# sourceMappingURL=index.d.ts.map