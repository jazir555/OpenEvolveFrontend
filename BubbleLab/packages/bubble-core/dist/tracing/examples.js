/**
 * Example usage of distributed tracing with BubbleLab
 *
 * This file demonstrates common patterns for using OpenTelemetry
 * distributed tracing in BubbleLab applications.
 */
import { TracingManager } from './tracing-manager.js';
import { traceAsync, recordException } from './tracer.js';
import { BubbleTracer } from './bubble-tracer.js';
import { injectContext, extractContext } from './context-propagator.js';
import { TraceMetricsAnalyzer } from './trace-metrics.js';
import { TraceAlertManager, CommonAlertRules } from './trace-alerts.js';
import { ExporterType } from './types.js';
// ============================================
// Example 1: Basic Tracing Setup
// ============================================
async function basicSetupExample() {
    const manager = TracingManager.getInstance();
    // Initialize tracing with Jaeger (local development)
    await manager.initialize({
        serviceName: 'bubble-lab-api',
        enabled: true,
        sampleRate: 1.0, // 100% sampling for development
        exporter: {
            type: ExporterType.COLLECTOR,
            options: {
                endpoint: 'localhost:4317',
            },
        },
        resourceAttributes: {
            'environment': 'development',
            'version': '1.0.0',
        },
    });
    console.log('Tracing initialized:', manager.getStats());
}
// ============================================
// Example 2: Tracing a Bubble Operation
// ============================================
async function traceBubbleOperation() {
    return traceAsync({
        name: 'bubble.execution',
        attributes: {
            'bubble.name': 'ai-agent',
            'bubble.type': 'service',
            'bubble.operation': 'generate-text',
        },
    }, async (span) => {
        // Your bubble operation here
        const result = await performAICompletion();
        // Add custom attributes
        if (span) {
            span.setAttribute('result.tokens', result.tokenCount);
            span.setAttribute('result.duration', result.duration);
        }
        return result;
    });
}
// ============================================
// Example 3: Bubble-Specific Tracing
// ============================================
async function bubbleSpecificTracing() {
    const tracer = new BubbleTracer();
    // Trace bubble instantiation
    const instantiationSpan = tracer.createInstantiationSpan({
        bubbleName: 'postgresql',
        bubbleType: 'service',
        variableName: 'db',
        className: 'PostgreSQLBubble',
    });
    try {
        // Instantiate bubble
        const bubble = await createPostgreSQLBubble();
        instantiationSpan?.end();
        // Trace bubble execution
        const result = await tracer.traceBubbleAction({
            bubbleName: 'postgresql',
            bubbleType: 'service',
            operation: 'query',
            correlationId: 'req-123',
            executionId: 'exec-456',
        }, async (span) => {
            // Execute query
            const result = await bubble.query();
            // Add query-specific attributes
            if (span) {
                span.setAttribute('db.rows', result.rows.length);
                span.setAttribute('db.duration', result.duration || 0);
            }
            return result;
        });
        return result;
    }
    catch (error) {
        if (instantiationSpan) {
            instantiationSpan.recordException(error);
            instantiationSpan.end();
        }
        throw error;
    }
}
// ============================================
// Example 4: Context Propagation
// ============================================
async function contextPropagationExample() {
    // Service A - Outgoing request
    async function callServiceB() {
        const headers = {
            'Content-Type': 'application/json',
        };
        // Inject trace context into headers
        injectContext(headers);
        // Make HTTP request with trace context
        const response = await fetch('http://service-b/api/data', {
            method: 'POST',
            headers,
            body: JSON.stringify({ data: 'test' }),
        });
        return response.json();
    }
    // Service B - Incoming request
    async function handleServiceBRequest(request) {
        // Extract trace context from incoming headers
        const headers = Object.fromEntries(request.headers.entries());
        const ctx = extractContext(headers);
        // Execute within the extracted context
        return traceAsync({
            name: 'service-b.process-data',
            attributes: {
                'service.name': 'service-b',
            },
        }, async () => {
            // Process data
            return { result: 'processed' };
        });
    }
}
// ============================================
// Example 5: Tracing HTTP Requests
// ============================================
async function traceHTTPRequest() {
    return traceAsync({
        name: 'http.request',
        attributes: {
            'http.method': 'POST',
            'http.url': 'https://api.example.com/data',
            'bubble.name': 'http-bubble',
        },
    }, async (span) => {
        const startTime = Date.now();
        try {
            const response = await fetch('https://api.example.com/data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: 'test' }),
            });
            const data = await response.json();
            // Add HTTP response attributes
            if (span) {
                span.setAttribute('http.status_code', response.status);
                span.setAttribute('http.response_size', JSON.stringify(data).length);
                span.setAttribute('duration.ms', Date.now() - startTime);
            }
            return data;
        }
        catch (error) {
            if (error instanceof Error) {
                recordException(error);
            }
            throw error;
        }
    });
}
// ============================================
// Example 6: Tracing Database Queries
// ============================================
async function traceDatabaseQuery() {
    return traceAsync({
        name: 'database.query',
        attributes: {
            'db.system': 'postgresql',
            'db.name': 'bubblelab',
            'db.operation': 'SELECT',
            'db.statement': 'SELECT * FROM users WHERE id = $1',
        },
    }, async (span) => {
        const startTime = Date.now();
        try {
            const result = await db.query('SELECT * FROM users WHERE id = $1', [123]);
            if (span) {
                span.setAttribute('db.rows_affected', result.rowCount);
                span.setAttribute('duration.ms', Date.now() - startTime);
            }
            return result;
        }
        catch (error) {
            if (error instanceof Error) {
                recordException(error);
                // Add database-specific error attributes
                if (span) {
                    span.setAttribute('db.error.code', error.code || 'UNKNOWN');
                    span.setAttribute('db.error.message', error.message);
                }
            }
            throw error;
        }
    });
}
// ============================================
// Example 7: Performance Metrics
// ============================================
async function performanceMetricsExample() {
    const metrics = new TraceMetricsAnalyzer();
    // Record operation performance
    const startTime = Date.now();
    try {
        await performOperation();
        const duration = Date.now() - startTime;
        metrics.recordOperation('my-operation', duration);
    }
    catch (error) {
        metrics.recordError('my-operation');
        throw error;
    }
    // Get overall metrics
    const overallMetrics = metrics.getMetrics();
    console.log('Overall metrics:', overallMetrics);
    // Get operation-specific metrics
    const operationMetrics = metrics.getOperationMetrics('my-operation');
    console.log('Operation metrics:', operationMetrics);
    // Analyze performance
    const analysis = metrics.analyzePerformance();
    console.log('Bottlenecks:', analysis.bottlenecks);
    console.log('Recommendations:', analysis.recommendations);
}
// ============================================
// Example 8: Alerting
// ============================================
async function alertingExample() {
    const alertManager = new TraceAlertManager();
    // Add predefined alert rules
    alertManager.addRule(CommonAlertRules.highP95Latency(30000));
    alertManager.addRule(CommonAlertRules.highErrorRate(5));
    alertManager.addRule(CommonAlertRules.missingSpans(300));
    // Register notification callback
    alertManager.registerNotificationCallback('default', (alert) => {
        console.warn('Alert triggered:', {
            rule: alert.rule.name,
            severity: alert.rule.severity,
            actualValue: alert.actualValue,
            threshold: alert.threshold,
        });
        // Send to your notification system
        sendNotification(alert);
    });
    // Evaluate rules (call this periodically)
    setInterval(() => {
        const triggers = alertManager.evaluateRules();
        console.log(`Evaluated rules, ${triggers.length} alerts triggered`);
    }, 60000); // Every minute
}
// ============================================
// Example 9: Workflow Tracing
// ============================================
async function workflowTracingExample() {
    const tracer = new BubbleTracer();
    return traceAsync({
        name: 'workflow.execute',
        attributes: {
            'workflow.name': 'data-pipeline',
            'workflow.id': 'wf-123',
        },
    }, async (workflowSpan) => {
        // Step 1: Fetch data
        const data = await tracer.traceBubbleAction({
            bubbleName: 'http',
            bubbleType: 'service',
            operation: 'fetch-data',
        }, async () => {
            return fetchDataFromAPI();
        });
        // Step 2: Process data
        const processed = await tracer.traceBubbleAction({
            bubbleName: 'ai-agent',
            bubbleType: 'service',
            operation: 'process-data',
        }, async () => {
            return processDataWithAI(data);
        });
        // Step 3: Store results
        const stored = await tracer.traceBubbleAction({
            bubbleName: 'postgresql',
            bubbleType: 'service',
            operation: 'store-results',
        }, async () => {
            return storeResults(processed);
        });
        return stored;
    });
}
// ============================================
// Example 10: Production Configuration
// ============================================
async function productionConfigExample() {
    const manager = TracingManager.getInstance();
    await manager.initialize({
        serviceName: process.env.OTEL_SERVICE_NAME || 'bubble-lab-api',
        enabled: process.env.OTEL_ENABLED === 'true',
        sampleRate: parseFloat(process.env.OTEL_SAMPLE_RATE || '0.1'),
        exporter: {
            type: ExporterType.COLLECTOR,
            options: {
                endpoint: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://otel-collector:4317',
                headers: {
                    'Authorization': `Bearer ${process.env.OTEL_AUTH_TOKEN}`,
                },
            },
        },
        batchExport: {
            exportIntervalMillis: 5000,
            maxQueueSize: 2048,
            maxExportBatchSize: 512,
            exportTimeoutMillis: 30000,
        },
        resourceAttributes: {
            'environment': process.env.NODE_ENV || 'production',
            'version': process.env.npm_package_version || '1.0.0',
            'region': process.env.AWS_REGION || 'us-east-1',
        },
    });
}
// ============================================
// Example 11: Error Handling
// ============================================
async function errorHandlingExample() {
    return traceAsync({
        name: 'operation-with-error-handling',
    }, async (span) => {
        try {
            const result = await riskyOperation();
            if (span) {
                span.setStatus({ code: 1 }); // OK
                span.setAttribute('success', true);
            }
            return result;
        }
        catch (error) {
            if (error instanceof Error) {
                // Record exception in span
                recordException(error);
                // Add custom error attributes
                if (span) {
                    span.setAttribute('error.type', error.constructor.name);
                    span.setAttribute('error.category', categorizeError(error));
                    span.setAttribute('success', false);
                }
            }
            throw error;
        }
    });
}
// ============================================
// Example 12: Custom Span Attributes
// ============================================
async function customAttributesExample() {
    return traceAsync({
        name: 'operation-with-custom-attributes',
        attributes: {
            'user.id': 'user-123',
            'operation.type': 'data-processing',
            'feature.name': 'advanced-analytics',
            'a/b.test.variant': 'experiment-b',
        },
    }, async (span) => {
        // Add dynamic attributes during execution
        if (span) {
            const memoryUsage = process.memoryUsage();
            span.setAttribute('memory.used.mb', memoryUsage.heapUsed / 1024 / 1024);
            span.setAttribute('cpu.usage.percent', getCpuUsage());
            span.setAttribute('cache.hit', true);
            span.setAttribute('cache.key', 'data:user-123');
        }
        return performOperation();
    });
}
// ============================================
// Helper Functions
// ============================================
async function performAICompletion() {
    return { tokenCount: 100, duration: 1234 };
}
async function createPostgreSQLBubble() {
    return { query: async () => ({ rows: [], rowCount: 0 }) };
}
async function performOperation() {
    return { success: true };
}
async function riskyOperation() {
    throw new Error('Something went wrong');
}
function categorizeError(error) {
    if (error.message.includes('timeout'))
        return 'timeout';
    if (error.message.includes('connection'))
        return 'network';
    return 'unknown';
}
function getCpuUsage() {
    return 50.5;
}
function sendNotification(alert) {
    // Implementation depends on notification system
    console.log('Sending notification:', alert);
}
// Helper functions for examples
const db = {
    query: async (sql, params) => ({
        rows: [{ id: 1, name: 'test' }],
        rowCount: 1,
    }),
};
async function fetchDataFromAPI() {
    return { data: 'example' };
}
async function processDataWithAI(data) {
    return { processed: data };
}
async function storeResults(data) {
    return { stored: true };
}
// Export examples
export { basicSetupExample, traceBubbleOperation, bubbleSpecificTracing, contextPropagationExample, traceHTTPRequest, traceDatabaseQuery, performanceMetricsExample, alertingExample, workflowTracingExample, productionConfigExample, errorHandlingExample, customAttributesExample, };
//# sourceMappingURL=examples.js.map