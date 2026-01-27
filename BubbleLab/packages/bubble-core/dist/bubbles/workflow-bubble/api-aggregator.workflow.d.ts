import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const APIAggregatorParamsSchema: z.ZodObject<{
    apis: z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        url: z.ZodString;
        method: z.ZodDefault<z.ZodEnum<["GET", "POST", "PUT", "PATCH"]>>;
        headers: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        body: z.ZodOptional<z.ZodUnknown>;
        timeout: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        name: string;
        url: string;
        method: "GET" | "POST" | "PUT" | "PATCH";
        headers?: Record<string, string> | undefined;
        body?: unknown;
    }, {
        name: string;
        url: string;
        timeout?: number | undefined;
        headers?: Record<string, string> | undefined;
        method?: "GET" | "POST" | "PUT" | "PATCH" | undefined;
        body?: unknown;
    }>, "many">;
    aggregationStrategy: z.ZodDefault<z.ZodEnum<["parallel", "sequential", "batch"]>>;
    mergeStrategy: z.ZodDefault<z.ZodEnum<["concat", "merge", "zip"]>>;
    errorHandling: z.ZodDefault<z.ZodEnum<["fail", "continue", "partial"]>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    apis: {
        timeout: number;
        name: string;
        url: string;
        method: "GET" | "POST" | "PUT" | "PATCH";
        headers?: Record<string, string> | undefined;
        body?: unknown;
    }[];
    aggregationStrategy: "batch" | "parallel" | "sequential";
    mergeStrategy: "concat" | "merge" | "zip";
    errorHandling: "partial" | "continue" | "fail";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    apis: {
        name: string;
        url: string;
        timeout?: number | undefined;
        headers?: Record<string, string> | undefined;
        method?: "GET" | "POST" | "PUT" | "PATCH" | undefined;
        body?: unknown;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    aggregationStrategy?: "batch" | "parallel" | "sequential" | undefined;
    mergeStrategy?: "concat" | "merge" | "zip" | undefined;
    errorHandling?: "partial" | "continue" | "fail" | undefined;
}>;
type APIAggregatorParams = z.input<typeof APIAggregatorParamsSchema>;
declare const APIAggregatorResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    error: z.ZodString;
    results: z.ZodArray<z.ZodObject<{
        api: z.ZodString;
        success: z.ZodBoolean;
        data: z.ZodOptional<z.ZodUnknown>;
        error: z.ZodOptional<z.ZodString>;
        responseTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        responseTime: number;
        api: string;
        error?: string | undefined;
        data?: unknown;
    }, {
        success: boolean;
        responseTime: number;
        api: string;
        error?: string | undefined;
        data?: unknown;
    }>, "many">;
    mergedData: z.ZodOptional<z.ZodUnknown>;
    totalResponseTime: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    results: {
        success: boolean;
        responseTime: number;
        api: string;
        error?: string | undefined;
        data?: unknown;
    }[];
    totalResponseTime: number;
    mergedData?: unknown;
}, {
    error: string;
    success: boolean;
    results: {
        success: boolean;
        responseTime: number;
        api: string;
        error?: string | undefined;
        data?: unknown;
    }[];
    totalResponseTime: number;
    mergedData?: unknown;
}>;
export declare class APIAggregatorWorkflow extends WorkflowBubble<APIAggregatorParams, z.infer<typeof APIAggregatorResultSchema>> {
    static readonly type: "workflow";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        apis: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            url: z.ZodString;
            method: z.ZodDefault<z.ZodEnum<["GET", "POST", "PUT", "PATCH"]>>;
            headers: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
            body: z.ZodOptional<z.ZodUnknown>;
            timeout: z.ZodDefault<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            timeout: number;
            name: string;
            url: string;
            method: "GET" | "POST" | "PUT" | "PATCH";
            headers?: Record<string, string> | undefined;
            body?: unknown;
        }, {
            name: string;
            url: string;
            timeout?: number | undefined;
            headers?: Record<string, string> | undefined;
            method?: "GET" | "POST" | "PUT" | "PATCH" | undefined;
            body?: unknown;
        }>, "many">;
        aggregationStrategy: z.ZodDefault<z.ZodEnum<["parallel", "sequential", "batch"]>>;
        mergeStrategy: z.ZodDefault<z.ZodEnum<["concat", "merge", "zip"]>>;
        errorHandling: z.ZodDefault<z.ZodEnum<["fail", "continue", "partial"]>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        apis: {
            timeout: number;
            name: string;
            url: string;
            method: "GET" | "POST" | "PUT" | "PATCH";
            headers?: Record<string, string> | undefined;
            body?: unknown;
        }[];
        aggregationStrategy: "batch" | "parallel" | "sequential";
        mergeStrategy: "concat" | "merge" | "zip";
        errorHandling: "partial" | "continue" | "fail";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        apis: {
            name: string;
            url: string;
            timeout?: number | undefined;
            headers?: Record<string, string> | undefined;
            method?: "GET" | "POST" | "PUT" | "PATCH" | undefined;
            body?: unknown;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        aggregationStrategy?: "batch" | "parallel" | "sequential" | undefined;
        mergeStrategy?: "concat" | "merge" | "zip" | undefined;
        errorHandling?: "partial" | "continue" | "fail" | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        error: z.ZodString;
        results: z.ZodArray<z.ZodObject<{
            api: z.ZodString;
            success: z.ZodBoolean;
            data: z.ZodOptional<z.ZodUnknown>;
            error: z.ZodOptional<z.ZodString>;
            responseTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            responseTime: number;
            api: string;
            error?: string | undefined;
            data?: unknown;
        }, {
            success: boolean;
            responseTime: number;
            api: string;
            error?: string | undefined;
            data?: unknown;
        }>, "many">;
        mergedData: z.ZodOptional<z.ZodUnknown>;
        totalResponseTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        results: {
            success: boolean;
            responseTime: number;
            api: string;
            error?: string | undefined;
            data?: unknown;
        }[];
        totalResponseTime: number;
        mergedData?: unknown;
    }, {
        error: string;
        success: boolean;
        results: {
            success: boolean;
            responseTime: number;
            api: string;
            error?: string | undefined;
            data?: unknown;
        }[];
        totalResponseTime: number;
        mergedData?: unknown;
    }>;
    static readonly shortDescription = "Aggregate multiple API calls into unified response";
    static readonly longDescription = "Calls multiple APIs in parallel or sequence and merges results into unified response.";
    static readonly alias = "aggregate-apis";
    constructor(params: APIAggregatorParams, context?: BubbleContext);
    protected performAction(): Promise<{
        success: boolean;
        error: string;
        results: {
            success: boolean;
            responseTime: number;
            api: string;
            error?: string | undefined;
            data?: unknown;
        }[];
        mergedData: Record<string, unknown> | {}[];
        totalResponseTime: number;
    } | {
        success: boolean;
        error: string;
        results: never[];
        totalResponseTime: number;
        mergedData?: undefined;
    }>;
    private callAPI;
    private mergeResults;
}
export {};
//# sourceMappingURL=api-aggregator.workflow.d.ts.map