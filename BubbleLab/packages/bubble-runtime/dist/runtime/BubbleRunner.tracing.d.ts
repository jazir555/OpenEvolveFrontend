/**
 * Integration of distributed tracing with BubbleRunner
 *
 * This file demonstrates how to integrate OpenTelemetry tracing
 * into the BubbleRunner for end-to-end workflow tracing.
 */
import { TracingManager } from '@bubblelab/bubble-core/tracing';
/**
 * Initialize tracing for BubbleRunner
 */
export declare function initializeBubbleRunnerTracing(serviceName?: string): Promise<TracingManager>;
/**
 * Trace bubble flow execution
 */
export declare function traceBubbleFlowExecution(flowName: string, flowId: string, executeFn: () => Promise<any>): Promise<any>;
/**
 * Trace bubble step execution
 */
export declare function traceBubbleStep(stepId: string, bubbleName: string, bubbleType: 'service' | 'tool' | 'workflow', executeFn: () => Promise<any>): Promise<any>;
/**
 * Trace HTTP requests in bubbles
 */
export declare function traceBubbleHTTPRequest(bubbleName: string, url: string, method: string, executeFn: () => Promise<any>): Promise<any>;
/**
 * Trace database queries in bubbles
 */
export declare function traceBubbleDatabaseQuery(bubbleName: string, dbSystem: string, dbName: string, query: string, executeFn: () => Promise<any>): Promise<any>;
/**
 * Inject trace context into webhook payload
 */
export declare function injectTraceContextIntoWebhook(payload: any): any;
/**
 * Extract trace context from webhook payload
 */
export declare function extractTraceContextFromWebhook(payload: any): any;
/**
 * Wrap a bubble class method with tracing
 */
export declare function traceBubbleMethod(bubbleName: string, bubbleType: 'service' | 'tool' | 'workflow', methodName: string): (_target: any, _propertyKey: string, descriptor: PropertyDescriptor) => PropertyDescriptor;
/**
 * Create a traced version of a bubble class
 */
export declare function createTracedBubbleClass<T extends {
    new (...args: any[]): any;
}>(bubbleName: string, bubbleType: 'service' | 'tool' | 'workflow', BubbleClass: T): T;
//# sourceMappingURL=BubbleRunner.tracing.d.ts.map