/**
 * Integration of distributed tracing with BubbleRunner
 *
 * This file demonstrates how to integrate OpenTelemetry tracing
 * into the BubbleRunner for end-to-end workflow tracing.
 */

import { TracingManager } from '@bubblelab/bubble-core/tracing';
import { BubbleTracer } from '@bubblelab/bubble-core/tracing';
import { traceAsync } from '@bubblelab/bubble-core/tracing';
import { injectContext } from '@bubblelab/bubble-core/tracing';

/**
 * Initialize tracing for BubbleRunner
 */
export async function initializeBubbleRunnerTracing(serviceName: string = 'bubble-lab-runner') {
  const manager = TracingManager.getInstance();

  // Check if already initialized
  if (manager.getStats().initialized) {
    return manager;
  }

  await manager.initialize({
    serviceName,
    enabled: process.env.OTEL_ENABLED === 'true' || true,
    sampleRate: parseFloat(process.env.OTEL_SAMPLE_RATE || '1.0'),
    exporter: {
      type: (process.env.OTEL_EXPORTER_TYPE as any) || 'jaeger',
      options: {
        host: process.env.JAEGER_HOST || 'localhost',
        port: parseInt(process.env.JAEGER_PORT || '6832'),
      },
    },
    batchExport: {
      exportIntervalMillis: 5000,
      maxQueueSize: 2048,
      maxExportBatchSize: 512,
      exportTimeoutMillis: 30000,
    },
    resourceAttributes: {
      'environment': process.env.NODE_ENV || 'development',
      'version': process.env.npm_package_version || '1.0.0',
    },
  });

  return manager;
}

/**
 * Trace bubble flow execution
 */
export async function traceBubbleFlowExecution(
  flowName: string,
  flowId: string,
  executeFn: () => Promise<any>
) {
  const tracer = new BubbleTracer();

  return tracer.traceBubbleAction(
    {
      bubbleName: flowName,
      bubbleType: 'workflow',
      operation: 'execute-flow',
      executionId: flowId,
    },
    async (span) => {
      if (span) {
        span.setAttribute('flow.name', flowName);
        span.setAttribute('flow.id', flowId);
        span.setAttribute('flow.type', 'bubbleflow');
      }

      return executeFn();
    }
  );
}

/**
 * Trace bubble step execution
 */
export async function traceBubbleStep(
  stepId: string,
  bubbleName: string,
  bubbleType: 'service' | 'tool' | 'workflow',
  executeFn: () => Promise<any>
) {
  const tracer = new BubbleTracer();

  return tracer.traceBubbleAction(
    {
      bubbleName,
      bubbleType,
      operation: 'execute-step',
    },
    async (span) => {
      if (span) {
        span.setAttribute('step.id', stepId);
        span.setAttribute('bubble.variable_name', bubbleName);
      }

      return executeFn();
    }
  );
}

/**
 * Trace HTTP requests in bubbles
 */
export async function traceBubbleHTTPRequest(
  bubbleName: string,
  url: string,
  method: string,
  executeFn: () => Promise<any>
) {
  return traceAsync(
    {
      name: 'bubble.http.request',
      attributes: {
        'http.method': method,
        'http.url': url,
        'bubble.name': bubbleName,
      },
    },
    async (span) => {
      const startTime = Date.now();

      try {
        const result = await executeFn();

        if (span) {
          span.setAttribute('http.status_code', result.status || 200);
          span.setAttribute('duration.ms', Date.now() - startTime);
        }

        return result;
      } catch (error) {
        if (span) {
          span.setAttribute('http.status_code', 500);
          span.setAttribute('duration.ms', Date.now() - startTime);
        }
        throw error;
      }
    }
  );
}

/**
 * Trace database queries in bubbles
 */
export async function traceBubbleDatabaseQuery(
  bubbleName: string,
  dbSystem: string,
  dbName: string,
  query: string,
  executeFn: () => Promise<any>
) {
  return traceAsync(
    {
      name: 'bubble.database.query',
      attributes: {
        'db.system': dbSystem,
        'db.name': dbName,
        'db.statement': query.substring(0, 1000), // Limit query length
        'bubble.name': bubbleName,
      },
    },
    async (span) => {
      const startTime = Date.now();

      try {
        const result = await executeFn();

        if (span) {
          span.setAttribute('db.rows_affected', result?.rowCount || 0);
          span.setAttribute('duration.ms', Date.now() - startTime);
        }

        return result;
      } catch (error) {
        if (span && error instanceof Error) {
          span.setAttribute('db.error.code', (error as any).code);
          span.setAttribute('db.error.message', error.message);
        }
        throw error;
      }
    }
  );
}

/**
 * Inject trace context into webhook payload
 */
export function injectTraceContextIntoWebhook(payload: any): any {
  const headers = {};
  injectContext(headers);

  return {
    ...payload,
    _traceContext: headers,
  };
}

/**
 * Extract trace context from webhook payload
 */
export function extractTraceContextFromWebhook(payload: any) {
  if (payload._traceContext) {
    return payload._traceContext;
  }
  return {};
}

/**
 * Wrap a bubble class method with tracing
 */
export function traceBubbleMethod(
  bubbleName: string,
  bubbleType: 'service' | 'tool' | 'workflow',
  methodName: string
) {
  return function (
    _target: any,
    _propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;

    descriptor.value = async function (...args: any[]) {
      return traceAsync(
        {
          name: `bubble.${methodName}`,
          attributes: {
            'bubble.name': bubbleName,
            'bubble.type': bubbleType,
            'bubble.method': methodName,
          },
        },
        async (span) => {
          try {
            const result = await originalMethod.apply(this, args);

            if (span) {
              span.setAttribute('success', true);
            }

            return result;
          } catch (error) {
            if (span && error instanceof Error) {
              span.recordException(error);
              span.setAttribute('success', false);
              span.setAttribute('error.type', error.constructor.name);
            }
            throw error;
          }
        }
      );
    };

    return descriptor;
  };
}

/**
 * Create a traced version of a bubble class
 */
export function createTracedBubbleClass<T extends { new (...args: any[]): any }>(
  bubbleName: string,
  bubbleType: 'service' | 'tool' | 'workflow',
  BubbleClass: T
): T {
  return class extends BubbleClass {
    constructor(...args: any[]) {
      super(...args);
    }

    async action(...args: any[]) {
      return traceAsync(
        {
          name: 'bubble.action',
          attributes: {
            'bubble.name': bubbleName,
            'bubble.type': bubbleType,
            'bubble.method': 'action',
          },
        },
        async (span) => {
          try {
            const result = await super.action(...args);

            if (span) {
              span.setAttribute('success', true);
            }

            return result;
          } catch (error) {
            if (span && error instanceof Error) {
              span.recordException(error);
              span.setAttribute('success', false);
            }
            throw error;
          }
        }
      );
    }
  } as T;
}
