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
export declare function initializeMonitoring(config: MonitoringConfig): Promise<{
    metrics: MetricsCollector;
    health: HealthChecker;
    tracer: Tracer;
    alerts: AlertManager;
}>;
export declare function createMetricsMiddleware(): (req: any, res: any, next: any) => Promise<void>;
export declare function createHealthMiddleware(health: HealthChecker): (req: any, res: any, next: any) => Promise<void>;
export declare function createRequestTrackingMiddleware(serviceName?: string): (req: any, res: any, next: any) => Promise<void>;
export { MetricsCollector, getMetricsCollector, HealthChecker, HealthEndpointHandler, Tracer, getTracer, AlertManager, getAlertManager, };
export type { MetricsLabels, KnowledgeExtractionLabels, HealthStatus, HealthCheckResult, HealthCheckOptions, HealthCheckFunction, TraceOptions, SpanMetadata, AlertSeverity, AlertRule, AlertCondition, NotificationChannel, Alert, AlertHistory, };
//# sourceMappingURL=index.d.ts.map