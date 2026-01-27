import { z } from '@hono/zod-openapi';
import { CredentialType } from './types';
export declare const ServiceUsageSchema: z.ZodObject<{
    service: z.ZodNativeEnum<typeof CredentialType>;
    subService: z.ZodOptional<z.ZodString>;
    unit: z.ZodString;
    usage: z.ZodNumber;
    unitCost: z.ZodNumber;
    totalCost: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    service: CredentialType;
    unit: string;
    usage: number;
    unitCost: number;
    totalCost: number;
    subService?: string | undefined;
}, {
    service: CredentialType;
    unit: string;
    usage: number;
    unitCost: number;
    totalCost: number;
    subService?: string | undefined;
}>;
export type ServiceUsage = z.infer<typeof ServiceUsageSchema>;
export declare const ExecutionSummarySchema: z.ZodObject<{
    result: z.ZodOptional<z.ZodAny>;
    totalDuration: z.ZodNumber;
    lineExecutionCount: z.ZodOptional<z.ZodNumber>;
    bubbleExecutionCount: z.ZodOptional<z.ZodNumber>;
    errorCount: z.ZodOptional<z.ZodNumber>;
    totalCost: z.ZodNumber;
    warningCount: z.ZodOptional<z.ZodNumber>;
    errors: z.ZodOptional<z.ZodArray<z.ZodObject<{
        message: z.ZodString;
        timestamp: z.ZodNumber;
        bubbleName: z.ZodOptional<z.ZodString>;
        variableId: z.ZodOptional<z.ZodNumber>;
        lineNumber: z.ZodOptional<z.ZodNumber>;
        additionalData: z.ZodOptional<z.ZodAny>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        timestamp: number;
        variableId?: number | undefined;
        bubbleName?: string | undefined;
        lineNumber?: number | undefined;
        additionalData?: any;
    }, {
        message: string;
        timestamp: number;
        variableId?: number | undefined;
        bubbleName?: string | undefined;
        lineNumber?: number | undefined;
        additionalData?: any;
    }>, "many">>;
    warnings: z.ZodOptional<z.ZodArray<z.ZodObject<{
        message: z.ZodString;
        timestamp: z.ZodNumber;
        bubbleName: z.ZodOptional<z.ZodString>;
        variableId: z.ZodOptional<z.ZodNumber>;
        lineNumber: z.ZodOptional<z.ZodNumber>;
        additionalData: z.ZodOptional<z.ZodAny>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        timestamp: number;
        variableId?: number | undefined;
        bubbleName?: string | undefined;
        lineNumber?: number | undefined;
        additionalData?: any;
    }, {
        message: string;
        timestamp: number;
        variableId?: number | undefined;
        bubbleName?: string | undefined;
        lineNumber?: number | undefined;
        additionalData?: any;
    }>, "many">>;
    averageLineExecutionTime: z.ZodOptional<z.ZodNumber>;
    slowestLines: z.ZodOptional<z.ZodArray<z.ZodObject<{
        lineNumber: z.ZodNumber;
        duration: z.ZodNumber;
        message: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        message: string;
        duration: number;
        lineNumber: number;
    }, {
        message: string;
        duration: number;
        lineNumber: number;
    }>, "many">>;
    memoryPeakUsage: z.ZodOptional<z.ZodAny>;
    startTime: z.ZodOptional<z.ZodNumber>;
    endTime: z.ZodOptional<z.ZodNumber>;
    serviceUsage: z.ZodOptional<z.ZodArray<z.ZodObject<{
        service: z.ZodNativeEnum<typeof CredentialType>;
        subService: z.ZodOptional<z.ZodString>;
        unit: z.ZodString;
        usage: z.ZodNumber;
        unitCost: z.ZodNumber;
        totalCost: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        service: CredentialType;
        unit: string;
        usage: number;
        unitCost: number;
        totalCost: number;
        subService?: string | undefined;
    }, {
        service: CredentialType;
        unit: string;
        usage: number;
        unitCost: number;
        totalCost: number;
        subService?: string | undefined;
    }>, "many">>;
    serviceUsageByService: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodObject<{
        service: z.ZodNativeEnum<typeof CredentialType>;
        subService: z.ZodOptional<z.ZodString>;
        unit: z.ZodString;
        usage: z.ZodNumber;
        unitCost: z.ZodNumber;
        totalCost: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        service: CredentialType;
        unit: string;
        usage: number;
        unitCost: number;
        totalCost: number;
        subService?: string | undefined;
    }, {
        service: CredentialType;
        unit: string;
        usage: number;
        unitCost: number;
        totalCost: number;
        subService?: string | undefined;
    }>>>;
}, "strip", z.ZodTypeAny, {
    totalCost: number;
    totalDuration: number;
    result?: any;
    lineExecutionCount?: number | undefined;
    bubbleExecutionCount?: number | undefined;
    errorCount?: number | undefined;
    warningCount?: number | undefined;
    errors?: {
        message: string;
        timestamp: number;
        variableId?: number | undefined;
        bubbleName?: string | undefined;
        lineNumber?: number | undefined;
        additionalData?: any;
    }[] | undefined;
    warnings?: {
        message: string;
        timestamp: number;
        variableId?: number | undefined;
        bubbleName?: string | undefined;
        lineNumber?: number | undefined;
        additionalData?: any;
    }[] | undefined;
    averageLineExecutionTime?: number | undefined;
    slowestLines?: {
        message: string;
        duration: number;
        lineNumber: number;
    }[] | undefined;
    memoryPeakUsage?: any;
    startTime?: number | undefined;
    endTime?: number | undefined;
    serviceUsage?: {
        service: CredentialType;
        unit: string;
        usage: number;
        unitCost: number;
        totalCost: number;
        subService?: string | undefined;
    }[] | undefined;
    serviceUsageByService?: Record<string, {
        service: CredentialType;
        unit: string;
        usage: number;
        unitCost: number;
        totalCost: number;
        subService?: string | undefined;
    }> | undefined;
}, {
    totalCost: number;
    totalDuration: number;
    result?: any;
    lineExecutionCount?: number | undefined;
    bubbleExecutionCount?: number | undefined;
    errorCount?: number | undefined;
    warningCount?: number | undefined;
    errors?: {
        message: string;
        timestamp: number;
        variableId?: number | undefined;
        bubbleName?: string | undefined;
        lineNumber?: number | undefined;
        additionalData?: any;
    }[] | undefined;
    warnings?: {
        message: string;
        timestamp: number;
        variableId?: number | undefined;
        bubbleName?: string | undefined;
        lineNumber?: number | undefined;
        additionalData?: any;
    }[] | undefined;
    averageLineExecutionTime?: number | undefined;
    slowestLines?: {
        message: string;
        duration: number;
        lineNumber: number;
    }[] | undefined;
    memoryPeakUsage?: any;
    startTime?: number | undefined;
    endTime?: number | undefined;
    serviceUsage?: {
        service: CredentialType;
        unit: string;
        usage: number;
        unitCost: number;
        totalCost: number;
        subService?: string | undefined;
    }[] | undefined;
    serviceUsageByService?: Record<string, {
        service: CredentialType;
        unit: string;
        usage: number;
        unitCost: number;
        totalCost: number;
        subService?: string | undefined;
    }> | undefined;
}>;
export type ExecutionSummary = z.infer<typeof ExecutionSummarySchema>;
export declare const bubbleFlowExecutionSchema: z.ZodObject<{
    id: z.ZodNumber;
    status: z.ZodEnum<["running", "success", "error"]>;
    payload: z.ZodRecord<z.ZodString, z.ZodAny>;
    result: z.ZodOptional<z.ZodAny>;
    error: z.ZodOptional<z.ZodString>;
    startedAt: z.ZodString;
    webhook_url: z.ZodString;
    completedAt: z.ZodOptional<z.ZodString>;
    code: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    status: "running" | "success" | "error";
    id: number;
    payload: Record<string, any>;
    startedAt: string;
    webhook_url: string;
    code?: string | undefined;
    result?: any;
    error?: string | undefined;
    completedAt?: string | undefined;
}, {
    status: "running" | "success" | "error";
    id: number;
    payload: Record<string, any>;
    startedAt: string;
    webhook_url: string;
    code?: string | undefined;
    result?: any;
    error?: string | undefined;
    completedAt?: string | undefined;
}>;
export declare const listBubbleFlowExecutionsResponseSchema: z.ZodArray<z.ZodObject<{
    id: z.ZodNumber;
    status: z.ZodEnum<["running", "success", "error"]>;
    payload: z.ZodRecord<z.ZodString, z.ZodAny>;
    result: z.ZodOptional<z.ZodAny>;
    error: z.ZodOptional<z.ZodString>;
    startedAt: z.ZodString;
    webhook_url: z.ZodString;
    completedAt: z.ZodOptional<z.ZodString>;
    code: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    status: "running" | "success" | "error";
    id: number;
    payload: Record<string, any>;
    startedAt: string;
    webhook_url: string;
    code?: string | undefined;
    result?: any;
    error?: string | undefined;
    completedAt?: string | undefined;
}, {
    status: "running" | "success" | "error";
    id: number;
    payload: Record<string, any>;
    startedAt: string;
    webhook_url: string;
    code?: string | undefined;
    result?: any;
    error?: string | undefined;
    completedAt?: string | undefined;
}>, "many">;
export type ListBubbleFlowExecutionsResponse = z.infer<typeof listBubbleFlowExecutionsResponseSchema>;
export declare const executeBubbleFlowResponseSchema: z.ZodObject<{
    executionId: z.ZodNumber;
    success: z.ZodBoolean;
    data: z.ZodOptional<z.ZodAny>;
    summary: z.ZodOptional<z.ZodObject<{
        result: z.ZodOptional<z.ZodAny>;
        totalDuration: z.ZodNumber;
        lineExecutionCount: z.ZodOptional<z.ZodNumber>;
        bubbleExecutionCount: z.ZodOptional<z.ZodNumber>;
        errorCount: z.ZodOptional<z.ZodNumber>;
        totalCost: z.ZodNumber;
        warningCount: z.ZodOptional<z.ZodNumber>;
        errors: z.ZodOptional<z.ZodArray<z.ZodObject<{
            message: z.ZodString;
            timestamp: z.ZodNumber;
            bubbleName: z.ZodOptional<z.ZodString>;
            variableId: z.ZodOptional<z.ZodNumber>;
            lineNumber: z.ZodOptional<z.ZodNumber>;
            additionalData: z.ZodOptional<z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            message: string;
            timestamp: number;
            variableId?: number | undefined;
            bubbleName?: string | undefined;
            lineNumber?: number | undefined;
            additionalData?: any;
        }, {
            message: string;
            timestamp: number;
            variableId?: number | undefined;
            bubbleName?: string | undefined;
            lineNumber?: number | undefined;
            additionalData?: any;
        }>, "many">>;
        warnings: z.ZodOptional<z.ZodArray<z.ZodObject<{
            message: z.ZodString;
            timestamp: z.ZodNumber;
            bubbleName: z.ZodOptional<z.ZodString>;
            variableId: z.ZodOptional<z.ZodNumber>;
            lineNumber: z.ZodOptional<z.ZodNumber>;
            additionalData: z.ZodOptional<z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            message: string;
            timestamp: number;
            variableId?: number | undefined;
            bubbleName?: string | undefined;
            lineNumber?: number | undefined;
            additionalData?: any;
        }, {
            message: string;
            timestamp: number;
            variableId?: number | undefined;
            bubbleName?: string | undefined;
            lineNumber?: number | undefined;
            additionalData?: any;
        }>, "many">>;
        averageLineExecutionTime: z.ZodOptional<z.ZodNumber>;
        slowestLines: z.ZodOptional<z.ZodArray<z.ZodObject<{
            lineNumber: z.ZodNumber;
            duration: z.ZodNumber;
            message: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            message: string;
            duration: number;
            lineNumber: number;
        }, {
            message: string;
            duration: number;
            lineNumber: number;
        }>, "many">>;
        memoryPeakUsage: z.ZodOptional<z.ZodAny>;
        startTime: z.ZodOptional<z.ZodNumber>;
        endTime: z.ZodOptional<z.ZodNumber>;
        serviceUsage: z.ZodOptional<z.ZodArray<z.ZodObject<{
            service: z.ZodNativeEnum<typeof CredentialType>;
            subService: z.ZodOptional<z.ZodString>;
            unit: z.ZodString;
            usage: z.ZodNumber;
            unitCost: z.ZodNumber;
            totalCost: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            service: CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }, {
            service: CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }>, "many">>;
        serviceUsageByService: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodObject<{
            service: z.ZodNativeEnum<typeof CredentialType>;
            subService: z.ZodOptional<z.ZodString>;
            unit: z.ZodString;
            usage: z.ZodNumber;
            unitCost: z.ZodNumber;
            totalCost: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            service: CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }, {
            service: CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }>>>;
    }, "strip", z.ZodTypeAny, {
        totalCost: number;
        totalDuration: number;
        result?: any;
        lineExecutionCount?: number | undefined;
        bubbleExecutionCount?: number | undefined;
        errorCount?: number | undefined;
        warningCount?: number | undefined;
        errors?: {
            message: string;
            timestamp: number;
            variableId?: number | undefined;
            bubbleName?: string | undefined;
            lineNumber?: number | undefined;
            additionalData?: any;
        }[] | undefined;
        warnings?: {
            message: string;
            timestamp: number;
            variableId?: number | undefined;
            bubbleName?: string | undefined;
            lineNumber?: number | undefined;
            additionalData?: any;
        }[] | undefined;
        averageLineExecutionTime?: number | undefined;
        slowestLines?: {
            message: string;
            duration: number;
            lineNumber: number;
        }[] | undefined;
        memoryPeakUsage?: any;
        startTime?: number | undefined;
        endTime?: number | undefined;
        serviceUsage?: {
            service: CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }[] | undefined;
        serviceUsageByService?: Record<string, {
            service: CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }> | undefined;
    }, {
        totalCost: number;
        totalDuration: number;
        result?: any;
        lineExecutionCount?: number | undefined;
        bubbleExecutionCount?: number | undefined;
        errorCount?: number | undefined;
        warningCount?: number | undefined;
        errors?: {
            message: string;
            timestamp: number;
            variableId?: number | undefined;
            bubbleName?: string | undefined;
            lineNumber?: number | undefined;
            additionalData?: any;
        }[] | undefined;
        warnings?: {
            message: string;
            timestamp: number;
            variableId?: number | undefined;
            bubbleName?: string | undefined;
            lineNumber?: number | undefined;
            additionalData?: any;
        }[] | undefined;
        averageLineExecutionTime?: number | undefined;
        slowestLines?: {
            message: string;
            duration: number;
            lineNumber: number;
        }[] | undefined;
        memoryPeakUsage?: any;
        startTime?: number | undefined;
        endTime?: number | undefined;
        serviceUsage?: {
            service: CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }[] | undefined;
        serviceUsageByService?: Record<string, {
            service: CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }> | undefined;
    }>>;
    error: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    success: boolean;
    executionId: number;
    error?: string | undefined;
    data?: any;
    summary?: {
        totalCost: number;
        totalDuration: number;
        result?: any;
        lineExecutionCount?: number | undefined;
        bubbleExecutionCount?: number | undefined;
        errorCount?: number | undefined;
        warningCount?: number | undefined;
        errors?: {
            message: string;
            timestamp: number;
            variableId?: number | undefined;
            bubbleName?: string | undefined;
            lineNumber?: number | undefined;
            additionalData?: any;
        }[] | undefined;
        warnings?: {
            message: string;
            timestamp: number;
            variableId?: number | undefined;
            bubbleName?: string | undefined;
            lineNumber?: number | undefined;
            additionalData?: any;
        }[] | undefined;
        averageLineExecutionTime?: number | undefined;
        slowestLines?: {
            message: string;
            duration: number;
            lineNumber: number;
        }[] | undefined;
        memoryPeakUsage?: any;
        startTime?: number | undefined;
        endTime?: number | undefined;
        serviceUsage?: {
            service: CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }[] | undefined;
        serviceUsageByService?: Record<string, {
            service: CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }> | undefined;
    } | undefined;
}, {
    success: boolean;
    executionId: number;
    error?: string | undefined;
    data?: any;
    summary?: {
        totalCost: number;
        totalDuration: number;
        result?: any;
        lineExecutionCount?: number | undefined;
        bubbleExecutionCount?: number | undefined;
        errorCount?: number | undefined;
        warningCount?: number | undefined;
        errors?: {
            message: string;
            timestamp: number;
            variableId?: number | undefined;
            bubbleName?: string | undefined;
            lineNumber?: number | undefined;
            additionalData?: any;
        }[] | undefined;
        warnings?: {
            message: string;
            timestamp: number;
            variableId?: number | undefined;
            bubbleName?: string | undefined;
            lineNumber?: number | undefined;
            additionalData?: any;
        }[] | undefined;
        averageLineExecutionTime?: number | undefined;
        slowestLines?: {
            message: string;
            duration: number;
            lineNumber: number;
        }[] | undefined;
        memoryPeakUsage?: any;
        startTime?: number | undefined;
        endTime?: number | undefined;
        serviceUsage?: {
            service: CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }[] | undefined;
        serviceUsageByService?: Record<string, {
            service: CredentialType;
            unit: string;
            usage: number;
            unitCost: number;
            totalCost: number;
            subService?: string | undefined;
        }> | undefined;
    } | undefined;
}>;
export type ExecuteBubbleFlowResponse = z.infer<typeof executeBubbleFlowResponseSchema>;
export type ExecutionResult = ExecuteBubbleFlowResponse;
export declare const validateBubbleFlowCodeSchema: z.ZodObject<{
    code: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        includeDetails: z.ZodDefault<z.ZodBoolean>;
        strictMode: z.ZodDefault<z.ZodBoolean>;
        syncInputsWithFlow: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        includeDetails: boolean;
        strictMode: boolean;
        syncInputsWithFlow: boolean;
    }, {
        includeDetails?: boolean | undefined;
        strictMode?: boolean | undefined;
        syncInputsWithFlow?: boolean | undefined;
    }>>;
    flowId: z.ZodOptional<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodRecord<z.ZodString, z.ZodNumber>>>;
    defaultInputs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    activateCron: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    code: string;
    options?: {
        includeDetails: boolean;
        strictMode: boolean;
        syncInputsWithFlow: boolean;
    } | undefined;
    flowId?: number | undefined;
    credentials?: Record<string, Record<string, number>> | undefined;
    defaultInputs?: Record<string, unknown> | undefined;
    activateCron?: boolean | undefined;
}, {
    code: string;
    options?: {
        includeDetails?: boolean | undefined;
        strictMode?: boolean | undefined;
        syncInputsWithFlow?: boolean | undefined;
    } | undefined;
    flowId?: number | undefined;
    credentials?: Record<string, Record<string, number>> | undefined;
    defaultInputs?: Record<string, unknown> | undefined;
    activateCron?: boolean | undefined;
}>;
export declare const validateBubbleFlowCodeResponseSchema: z.ZodObject<{
    eventType: z.ZodString;
    webhookPath: z.ZodString;
    valid: z.ZodBoolean;
    errors: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    bubbleCount: z.ZodOptional<z.ZodNumber>;
    inputSchema: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    bubbles: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodObject<{
        variableName: z.ZodString;
        bubbleName: z.ZodType<import("./types").BubbleName>;
        className: z.ZodString;
        parameters: z.ZodArray<z.ZodObject<{
            location: z.ZodOptional<z.ZodObject<{
                startLine: z.ZodNumber;
                startCol: z.ZodNumber;
                endLine: z.ZodNumber;
                endCol: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }>>;
            variableId: z.ZodOptional<z.ZodNumber>;
            name: z.ZodString;
            value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodArray<z.ZodUnknown, "many">]>;
            type: z.ZodNativeEnum<typeof import("./bubble-definition-schema").BubbleParameterType>;
            source: z.ZodOptional<z.ZodEnum<["object-property", "first-arg", "spread"]>>;
        }, "strip", z.ZodTypeAny, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: import("./bubble-definition-schema").BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: import("./bubble-definition-schema").BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }>, "many">;
        hasAwait: z.ZodBoolean;
        hasActionCall: z.ZodBoolean;
        dependencies: z.ZodOptional<z.ZodArray<z.ZodType<import("./types").BubbleName, z.ZodTypeDef, import("./types").BubbleName>, "many">>;
        dependencyGraph: z.ZodOptional<z.ZodType<import("./bubble-definition-schema").DependencyGraphNode, z.ZodTypeDef, import("./bubble-definition-schema").DependencyGraphNode>>;
        variableId: z.ZodNumber;
        nodeType: z.ZodEnum<["service", "tool", "workflow", "unknown"]>;
        location: z.ZodObject<{
            startLine: z.ZodNumber;
            startCol: z.ZodNumber;
            endLine: z.ZodNumber;
            endCol: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        }, {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        }>;
        description: z.ZodOptional<z.ZodString>;
        invocationCallSiteKey: z.ZodOptional<z.ZodString>;
        clonedFromVariableId: z.ZodOptional<z.ZodNumber>;
        isInsideCustomTool: z.ZodOptional<z.ZodBoolean>;
        containingCustomToolId: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: import("./bubble-definition-schema").BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: import("./types").BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: import("./types").BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: import("./bubble-definition-schema").BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: import("./types").BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: import("./types").BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>>>;
    workflow: z.ZodOptional<z.ZodObject<{
        root: z.ZodArray<z.ZodType<import("./bubble-definition-schema").WorkflowNode, z.ZodTypeDef, import("./bubble-definition-schema").WorkflowNode>, "many">;
        bubbles: z.ZodRecord<z.ZodNumber, z.ZodObject<{
            variableName: z.ZodString;
            bubbleName: z.ZodType<import("./types").BubbleName>;
            className: z.ZodString;
            parameters: z.ZodArray<z.ZodObject<{
                location: z.ZodOptional<z.ZodObject<{
                    startLine: z.ZodNumber;
                    startCol: z.ZodNumber;
                    endLine: z.ZodNumber;
                    endCol: z.ZodNumber;
                }, "strip", z.ZodTypeAny, {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                }, {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                }>>;
                variableId: z.ZodOptional<z.ZodNumber>;
                name: z.ZodString;
                value: z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodArray<z.ZodUnknown, "many">]>;
                type: z.ZodNativeEnum<typeof import("./bubble-definition-schema").BubbleParameterType>;
                source: z.ZodOptional<z.ZodEnum<["object-property", "first-arg", "spread"]>>;
            }, "strip", z.ZodTypeAny, {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: import("./bubble-definition-schema").BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }, {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: import("./bubble-definition-schema").BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }>, "many">;
            hasAwait: z.ZodBoolean;
            hasActionCall: z.ZodBoolean;
            dependencies: z.ZodOptional<z.ZodArray<z.ZodType<import("./types").BubbleName, z.ZodTypeDef, import("./types").BubbleName>, "many">>;
            dependencyGraph: z.ZodOptional<z.ZodType<import("./bubble-definition-schema").DependencyGraphNode, z.ZodTypeDef, import("./bubble-definition-schema").DependencyGraphNode>>;
            variableId: z.ZodNumber;
            nodeType: z.ZodEnum<["service", "tool", "workflow", "unknown"]>;
            location: z.ZodObject<{
                startLine: z.ZodNumber;
                startCol: z.ZodNumber;
                endLine: z.ZodNumber;
                endCol: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }, {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            }>;
            description: z.ZodOptional<z.ZodString>;
            invocationCallSiteKey: z.ZodOptional<z.ZodString>;
            clonedFromVariableId: z.ZodOptional<z.ZodNumber>;
            isInsideCustomTool: z.ZodOptional<z.ZodBoolean>;
            containingCustomToolId: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: import("./bubble-definition-schema").BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: import("./types").BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: import("./types").BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: import("./bubble-definition-schema").BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: import("./types").BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: import("./types").BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        root: import("./bubble-definition-schema").WorkflowNode[];
        bubbles: Record<number, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: import("./bubble-definition-schema").BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: import("./types").BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: import("./types").BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>;
    }, {
        root: import("./bubble-definition-schema").WorkflowNode[];
        bubbles: Record<number, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: import("./bubble-definition-schema").BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: import("./types").BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: import("./types").BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>;
    }>>;
    requiredCredentials: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodString, "many">>>;
    metadata: z.ZodObject<{
        validatedAt: z.ZodString;
        codeLength: z.ZodNumber;
        strictMode: z.ZodBoolean;
        flowUpdated: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        strictMode: boolean;
        validatedAt: string;
        codeLength: number;
        flowUpdated?: boolean | undefined;
    }, {
        strictMode: boolean;
        validatedAt: string;
        codeLength: number;
        flowUpdated?: boolean | undefined;
    }>;
    cron: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    cronActive: z.ZodOptional<z.ZodBoolean>;
    defaultInputs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    valid: boolean;
    metadata: {
        strictMode: boolean;
        validatedAt: string;
        codeLength: number;
        flowUpdated?: boolean | undefined;
    };
    success: boolean;
    error: string;
    eventType: string;
    webhookPath: string;
    inputSchema: Record<string, unknown>;
    workflow?: {
        root: import("./bubble-definition-schema").WorkflowNode[];
        bubbles: Record<number, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: import("./bubble-definition-schema").BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: import("./types").BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: import("./types").BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>;
    } | undefined;
    bubbles?: Record<string, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: import("./bubble-definition-schema").BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: import("./types").BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: import("./types").BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }> | undefined;
    errors?: string[] | undefined;
    defaultInputs?: Record<string, unknown> | undefined;
    bubbleCount?: number | undefined;
    requiredCredentials?: Record<string, string[]> | undefined;
    cron?: string | null | undefined;
    cronActive?: boolean | undefined;
}, {
    valid: boolean;
    metadata: {
        strictMode: boolean;
        validatedAt: string;
        codeLength: number;
        flowUpdated?: boolean | undefined;
    };
    success: boolean;
    error: string;
    eventType: string;
    webhookPath: string;
    inputSchema: Record<string, unknown>;
    workflow?: {
        root: import("./bubble-definition-schema").WorkflowNode[];
        bubbles: Record<number, {
            location: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            };
            variableId: number;
            variableName: string;
            className: string;
            parameters: {
                value: string | number | boolean | unknown[] | Record<string, unknown>;
                type: import("./bubble-definition-schema").BubbleParameterType;
                name: string;
                location?: {
                    startLine: number;
                    startCol: number;
                    endLine: number;
                    endCol: number;
                } | undefined;
                variableId?: number | undefined;
                source?: "object-property" | "first-arg" | "spread" | undefined;
            }[];
            hasAwait: boolean;
            hasActionCall: boolean;
            bubbleName: import("./types").BubbleName;
            nodeType: "unknown" | "service" | "tool" | "workflow";
            description?: string | undefined;
            dependencies?: import("./types").BubbleName[] | undefined;
            dependencyGraph?: import("./bubble-definition-schema").DependencyGraphNode | undefined;
            invocationCallSiteKey?: string | undefined;
            clonedFromVariableId?: number | undefined;
            isInsideCustomTool?: boolean | undefined;
            containingCustomToolId?: string | undefined;
        }>;
    } | undefined;
    bubbles?: Record<string, {
        location: {
            startLine: number;
            startCol: number;
            endLine: number;
            endCol: number;
        };
        variableId: number;
        variableName: string;
        className: string;
        parameters: {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: import("./bubble-definition-schema").BubbleParameterType;
            name: string;
            location?: {
                startLine: number;
                startCol: number;
                endLine: number;
                endCol: number;
            } | undefined;
            variableId?: number | undefined;
            source?: "object-property" | "first-arg" | "spread" | undefined;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: import("./types").BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: import("./types").BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }> | undefined;
    errors?: string[] | undefined;
    defaultInputs?: Record<string, unknown> | undefined;
    bubbleCount?: number | undefined;
    requiredCredentials?: Record<string, string[]> | undefined;
    cron?: string | null | undefined;
    cronActive?: boolean | undefined;
}>;
export type ValidateBubbleFlowResponse = z.infer<typeof validateBubbleFlowCodeResponseSchema>;
export type BubbleFlowExecution = z.infer<typeof bubbleFlowExecutionSchema>;
//# sourceMappingURL=bubbleflow-execution-schema.d.ts.map