/**
 * Example usage of distributed tracing with BubbleLab
 *
 * This file demonstrates common patterns for using OpenTelemetry
 * distributed tracing in BubbleLab applications.
 */
declare function basicSetupExample(): Promise<void>;
declare function traceBubbleOperation(): Promise<{
    tokenCount: number;
    duration: number;
}>;
declare function bubbleSpecificTracing(): Promise<{
    rows: never[];
    rowCount: number;
}>;
declare function contextPropagationExample(): Promise<void>;
declare function traceHTTPRequest(): Promise<any>;
declare function traceDatabaseQuery(): Promise<{
    rows: {
        id: number;
        name: string;
    }[];
    rowCount: number;
}>;
declare function performanceMetricsExample(): Promise<void>;
declare function alertingExample(): Promise<void>;
declare function workflowTracingExample(): Promise<unknown>;
declare function productionConfigExample(): Promise<void>;
declare function errorHandlingExample(): Promise<void>;
declare function customAttributesExample(): Promise<{
    success: boolean;
}>;
export { basicSetupExample, traceBubbleOperation, bubbleSpecificTracing, contextPropagationExample, traceHTTPRequest, traceDatabaseQuery, performanceMetricsExample, alertingExample, workflowTracingExample, productionConfigExample, errorHandlingExample, customAttributesExample, };
//# sourceMappingURL=examples.d.ts.map