/**
 * Service Dependency Scanner
 * Purpose: Scan and map service dependencies across the infrastructure
 * Category: Infrastructure Automation
 * Event Type: schedule/cron
 * Schedule: 0 4 * * * (Daily at 4 AM)
 *
 * Required Credentials:
 * - KUBERNETES_CONFIG: Kubernetes cluster configuration
 * - PROMETHEUS_URL: Prometheus for service metrics
 * - POSTGRES_CONNECTION_STRING: To store dependency maps
 * - API_KEY: API key for authentication (required)
 *
 * Security Fixes Applied (Wave 5):
 * - Environment variable validation at startup
 * - API key authentication
 * - Rate limiting
 * - Input validation for all user inputs
 * - SQL injection prevention with parameterized queries
 * - Error message sanitization
 * - Structured logging with correlation IDs
 * - URL validation for all endpoints
 */

import {
  BubbleFlow,
  HttpBubble,
  AIAgentBubble,
  PostgreSQLBubble,
  type CronEvent
} from '@bubblelab/bubble-core';
import {
  validateEnvironment,
  authenticateRequest,
  requireAuthentication,
  RateLimiter,
  InputValidator,
  sanitizeError,
  StructuredLogger,
  generateCorrelationId,
  buildParameterizedQuery,
  SecuritySchemas,
} from '../security-utils';

interface ServiceDependency {
  source: string;
  target: string;
  type: 'http' | 'grpc' | 'database' | 'cache' | 'message_queue';
  calls: number;
  errors: number;
  latency: number;
}

interface DependencyGraph {
  timestamp: string;
  services: string[];
  dependencies: ServiceDependency[];
  criticalPaths: string[][];
  correlationId: string;
}

// Security: Environment variable validation
validateEnvironment({
  required: ['KUBERNETES_API', 'KUBERNETES_TOKEN', 'PROMETHEUS_URL', 'POSTGRES_CONNECTION_STRING', 'API_KEY'],
  schemas: {
    API_KEY: SecuritySchemas.apiKey,
    KUBERNETES_API: SecuritySchemas.url,
    PROMETHEUS_URL: SecuritySchemas.url,
  },
});

export class ServiceDependencyScanner extends BubbleFlow<'schedule/cron'> {
  readonly cronSchedule = '0 4 * * *';
  readonly name = 'Service Dependency Scanner';
  readonly description = 'Scan and map service dependencies across infrastructure';

  private logger = new StructuredLogger('service-dependency-scanner');
  private rateLimiter = new RateLimiter({
    maxRequests: 1, // 1 scan per minute
    windowMs: 60000, // 1 minute
  });

  async handle(payload: CronEvent): Promise<DependencyGraph> {
    // Security: Generate correlation ID for tracing
    const correlationId = generateCorrelationId();
    this.logger = this.logger.child({ correlationId });

    const timestamp = new Date().toISOString();

    // Security: Rate limiting check
    if (!this.rateLimiter.checkLimit(correlationId)) {
      throw new Error('Rate limit exceeded. Please try again later.');
    }

    // Security: API key authentication
    const authContext = authenticateRequest(
      payload.headers?.['x-api-key'],
      process.env.API_KEY,
      { correlationId, ip: payload.headers?.['x-forwarded-for'] }
    );
    requireAuthentication(authContext);

    this.logger.info({
      msg: 'Starting service dependency scan',
      timestamp,
    });

    // Step 1: Get all services from Kubernetes
    const validatedKubernetesApi = InputValidator.validateUrl(process.env.KUBERNETES_API);

    const getServices = new HttpBubble({
      url: InputValidator.validateUrl(`${validatedKubernetesApi}/api/v1/namespaces/default/services`),
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${process.env.KUBERNETES_TOKEN}`,
      },
      timeout: 10000,
    });

    const servicesResponse = await getServices.action();
    const services = servicesResponse.data.items
      .filter((item: any) => item.spec.type !== 'ClusterIP' || item.metadata.name !== 'kubernetes')
      .map((item: any) => InputValidator.validateServiceName(item.metadata.name));

    const dependencies: ServiceDependency[] = [];

    // Step 2: Query Prometheus for service-to-service communication
    const serviceMeshQuery = InputValidator.sanitizeString(
      'sum(rate(http_requests_total[5m])) by (source_service, destination_service)',
      500
    );

    const validatedPrometheusUrl = InputValidator.validateUrl(process.env.PROMETHEUS_URL);

    const meshData = new HttpBubble({
      url: InputValidator.validateUrl(`${validatedPrometheusUrl}/api/v1/query`),
      method: 'GET',
      params: {
        query: serviceMeshQuery,
      },
      timeout: 15000,
    });

    const meshResponse = await meshData.action();
    const metrics = meshResponse.data.data.result || [];

    // Step 3: Build dependency list from metrics
    for (const metric of metrics) {
      const source = InputValidator.validateServiceName(metric.metric.source_service);
      const target = InputValidator.validateServiceName(metric.metric.destination_service);
      const calls = InputValidator.sanitizeNumber(parseFloat(metric.value[1]), 0);

      if (source && target && services.includes(source) && services.includes(target)) {
        // Get error rate
        const errorQuery = InputValidator.sanitizeString(
          `sum(rate(http_requests_total{status=~"5..",source_service="${source}",destination_service="${target}"}[5m]))`,
          500
        );

        const errorData = new HttpBubble({
          url: InputValidator.validateUrl(`${validatedPrometheusUrl}/api/v1/query`),
          method: 'GET',
          params: { query: errorQuery },
          timeout: 10000,
        });

        const errorResponse = await errorData.action();
        const errors = InputValidator.sanitizeNumber(parseFloat(errorResponse.data.data.result[0]?.value[1]) || 0);

        // Get latency
        const latencyQuery = InputValidator.sanitizeString(
          `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{source_service="${source}",destination_service="${target}"}[5m]))`,
          500
        );

        const latencyData = new HttpBubble({
          url: InputValidator.validateUrl(`${validatedPrometheusUrl}/api/v1/query`),
          method: 'GET',
          params: { query: latencyQuery },
          timeout: 10000,
        });

        const latencyResponse = await latencyData.action();
        const latency = InputValidator.sanitizeNumber(parseFloat(latencyResponse.data.data.result[0]?.value[1]) || 0);

        dependencies.push({
          source,
          target,
          type: 'http',
          calls,
          errors,
          latency: latency * 1000, // Convert to ms
        });
      }
    }

    // Step 4: Identify critical paths using AI
    const agent = new AIAgentBubble({
      model: { model: 'openai/gpt-4' },
      systemPrompt: 'Analyze service dependencies and identify critical paths through the system',
      message: InputValidator.sanitizeString(`
Services: ${services.join(', ')}

Dependencies:
${dependencies.map(d => `${d.source} -> ${d.target} (${d.calls} calls, ${d.errors} errors)`).join('\n')}

Identify the most critical paths (chains of dependencies that are heavily used or error-prone).
Return as JSON array of arrays:
[["service1", "service2", "service3"], ["service4", "service5"]]
      `.trim(), 10000),
    });

    const analysis = await agent.action();

    let criticalPaths: string[][] = [];
    try {
      criticalPaths = JSON.parse(analysis.data.response);
    } catch (parseError) {
      // Fallback: simple critical path detection
      const highTrafficDeps = dependencies.filter(d => d.calls > 1000);
      criticalPaths = highTrafficDeps.map(d => [d.source, d.target]);
    }

    const graph: DependencyGraph = {
      timestamp,
      services,
      dependencies,
      criticalPaths,
      correlationId,
    };

    // Step 5: Store dependency graph
    // Security: SQL injection prevention - use parameterized query
    const storeGraphQuery = buildParameterizedQuery(
      `
        INSERT INTO dependency_graphs (timestamp, services, dependencies, critical_paths)
        VALUES ($1, $2, $3, $4)
      `,
      [
        timestamp,
        JSON.stringify(services),
        JSON.stringify(dependencies),
        JSON.stringify(criticalPaths),
      ]
    );

    try {
      const storeGraph = new PostgreSQLBubble({
        connectionString: process.env.POSTGRES_CONNECTION_STRING,
        query: storeGraphQuery.query,
        params: storeGraphQuery.params,
      });

      await storeGraph.action();
    } catch (error) {
      this.logger.error({
        msg: 'Failed to store dependency graph',
      }, error);
      throw new Error('Failed to store dependency graph in database');
    }

    this.logger.info({
      msg: 'Dependency scan completed',
      serviceCount: services.length,
      dependencyCount: dependencies.length,
      criticalPathCount: criticalPaths.length,
    });

    return graph;
  }
}

export default ServiceDependencyScanner;
