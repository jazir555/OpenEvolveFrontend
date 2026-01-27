import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const ETLPipelineParamsSchema: z.ZodObject<{
    phase: z.ZodEnum<["extract", "transform", "load"]>;
    source: z.ZodObject<{
        type: z.ZodEnum<["database", "api", "file", "csv", "json"]>;
        config: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    }, "strip", z.ZodTypeAny, {
        type: "file" | "json" | "database" | "csv" | "api";
        config: Record<string, unknown>;
    }, {
        type: "file" | "json" | "database" | "csv" | "api";
        config: Record<string, unknown>;
    }>;
    destination: z.ZodObject<{
        type: z.ZodEnum<["database", "api", "file"]>;
        config: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    }, "strip", z.ZodTypeAny, {
        type: "file" | "database" | "api";
        config: Record<string, unknown>;
    }, {
        type: "file" | "database" | "api";
        config: Record<string, unknown>;
    }>;
    transform: z.ZodOptional<z.ZodObject<{
        rules: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
        function: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        function?: string | undefined;
        rules?: Record<string, unknown>[] | undefined;
    }, {
        function?: string | undefined;
        rules?: Record<string, unknown>[] | undefined;
    }>>;
    batchSize: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    source: {
        type: "file" | "json" | "database" | "csv" | "api";
        config: Record<string, unknown>;
    };
    phase: "extract" | "transform" | "load";
    destination: {
        type: "file" | "database" | "api";
        config: Record<string, unknown>;
    };
    batchSize: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    transform?: {
        function?: string | undefined;
        rules?: Record<string, unknown>[] | undefined;
    } | undefined;
}, {
    source: {
        type: "file" | "json" | "database" | "csv" | "api";
        config: Record<string, unknown>;
    };
    phase: "extract" | "transform" | "load";
    destination: {
        type: "file" | "database" | "api";
        config: Record<string, unknown>;
    };
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    transform?: {
        function?: string | undefined;
        rules?: Record<string, unknown>[] | undefined;
    } | undefined;
    batchSize?: number | undefined;
}>;
type ETLPipelineParams = z.input<typeof ETLPipelineParamsSchema>;
declare const ETLPipelineResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    error: z.ZodString;
    phase: z.ZodString;
    recordsProcessed: z.ZodOptional<z.ZodNumber>;
    recordsSucceeded: z.ZodOptional<z.ZodNumber>;
    recordsFailed: z.ZodOptional<z.ZodNumber>;
    duration: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    phase: string;
    duration?: number | undefined;
    recordsProcessed?: number | undefined;
    recordsSucceeded?: number | undefined;
    recordsFailed?: number | undefined;
}, {
    error: string;
    success: boolean;
    phase: string;
    duration?: number | undefined;
    recordsProcessed?: number | undefined;
    recordsSucceeded?: number | undefined;
    recordsFailed?: number | undefined;
}>;
export declare class ETLPipelineWorkflow extends WorkflowBubble<ETLPipelineParams, z.infer<typeof ETLPipelineResultSchema>> {
    static readonly type: "workflow";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        phase: z.ZodEnum<["extract", "transform", "load"]>;
        source: z.ZodObject<{
            type: z.ZodEnum<["database", "api", "file", "csv", "json"]>;
            config: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        }, "strip", z.ZodTypeAny, {
            type: "file" | "json" | "database" | "csv" | "api";
            config: Record<string, unknown>;
        }, {
            type: "file" | "json" | "database" | "csv" | "api";
            config: Record<string, unknown>;
        }>;
        destination: z.ZodObject<{
            type: z.ZodEnum<["database", "api", "file"]>;
            config: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        }, "strip", z.ZodTypeAny, {
            type: "file" | "database" | "api";
            config: Record<string, unknown>;
        }, {
            type: "file" | "database" | "api";
            config: Record<string, unknown>;
        }>;
        transform: z.ZodOptional<z.ZodObject<{
            rules: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
            function: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            function?: string | undefined;
            rules?: Record<string, unknown>[] | undefined;
        }, {
            function?: string | undefined;
            rules?: Record<string, unknown>[] | undefined;
        }>>;
        batchSize: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        source: {
            type: "file" | "json" | "database" | "csv" | "api";
            config: Record<string, unknown>;
        };
        phase: "extract" | "transform" | "load";
        destination: {
            type: "file" | "database" | "api";
            config: Record<string, unknown>;
        };
        batchSize: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        transform?: {
            function?: string | undefined;
            rules?: Record<string, unknown>[] | undefined;
        } | undefined;
    }, {
        source: {
            type: "file" | "json" | "database" | "csv" | "api";
            config: Record<string, unknown>;
        };
        phase: "extract" | "transform" | "load";
        destination: {
            type: "file" | "database" | "api";
            config: Record<string, unknown>;
        };
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        transform?: {
            function?: string | undefined;
            rules?: Record<string, unknown>[] | undefined;
        } | undefined;
        batchSize?: number | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        error: z.ZodString;
        phase: z.ZodString;
        recordsProcessed: z.ZodOptional<z.ZodNumber>;
        recordsSucceeded: z.ZodOptional<z.ZodNumber>;
        recordsFailed: z.ZodOptional<z.ZodNumber>;
        duration: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        phase: string;
        duration?: number | undefined;
        recordsProcessed?: number | undefined;
        recordsSucceeded?: number | undefined;
        recordsFailed?: number | undefined;
    }, {
        error: string;
        success: boolean;
        phase: string;
        duration?: number | undefined;
        recordsProcessed?: number | undefined;
        recordsSucceeded?: number | undefined;
        recordsFailed?: number | undefined;
    }>;
    static readonly shortDescription = "Extract, Transform, Load data pipeline";
    static readonly longDescription = "Comprehensive ETL pipeline for data movement and transformation between multiple sources.";
    static readonly alias = "etl";
    constructor(params: ETLPipelineParams, context?: BubbleContext);
    protected performAction(): Promise<{
        success: boolean;
        error: string;
        phase: string;
        recordsProcessed: number;
        recordsSucceeded: number;
        recordsFailed: number;
        duration: number;
    } | {
        success: boolean;
        error: string;
        phase: never;
        duration?: undefined;
    } | {
        success: boolean;
        error: string;
        phase: "extract" | "transform" | "load";
        duration: number;
    }>;
    private extract;
    private transform;
    private load;
}
export {};
//# sourceMappingURL=etl-pipeline.workflow.d.ts.map