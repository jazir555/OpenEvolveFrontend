import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const WorkflowOrchestratorBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"createWorkflow">;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    steps: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        type: z.ZodEnum<["task", "condition", "loop", "parallel", "delay", "bubble"]>;
        config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        nextStepId: z.ZodOptional<z.ZodString>;
        condition: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
        name: string;
        id: string;
        config?: Record<string, unknown> | undefined;
        condition?: Record<string, unknown> | undefined;
        nextStepId?: string | undefined;
    }, {
        type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
        name: string;
        id: string;
        config?: Record<string, unknown> | undefined;
        condition?: Record<string, unknown> | undefined;
        nextStepId?: string | undefined;
    }>, "many">;
    inputSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    outputSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    timeout: z.ZodOptional<z.ZodNumber>;
    retryPolicy: z.ZodOptional<z.ZodObject<{
        maxAttempts: z.ZodOptional<z.ZodNumber>;
        backoff: z.ZodOptional<z.ZodEnum<["linear", "exponential"]>>;
    }, "strip", z.ZodTypeAny, {
        maxAttempts?: number | undefined;
        backoff?: "exponential" | "linear" | undefined;
    }, {
        maxAttempts?: number | undefined;
        backoff?: "exponential" | "linear" | undefined;
    }>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    operation: "createWorkflow";
    steps: {
        type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
        name: string;
        id: string;
        config?: Record<string, unknown> | undefined;
        condition?: Record<string, unknown> | undefined;
        nextStepId?: string | undefined;
    }[];
    timeout?: number | undefined;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    inputSchema?: Record<string, unknown> | undefined;
    outputSchema?: Record<string, unknown> | undefined;
    retryPolicy?: {
        maxAttempts?: number | undefined;
        backoff?: "exponential" | "linear" | undefined;
    } | undefined;
}, {
    name: string;
    operation: "createWorkflow";
    steps: {
        type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
        name: string;
        id: string;
        config?: Record<string, unknown> | undefined;
        condition?: Record<string, unknown> | undefined;
        nextStepId?: string | undefined;
    }[];
    timeout?: number | undefined;
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    inputSchema?: Record<string, unknown> | undefined;
    outputSchema?: Record<string, unknown> | undefined;
    retryPolicy?: {
        maxAttempts?: number | undefined;
        backoff?: "exponential" | "linear" | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"executeWorkflow">;
    workflowId: z.ZodString;
    inputs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    context: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    async: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "executeWorkflow";
    workflowId: string;
    async: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    context?: Record<string, unknown> | undefined;
    inputs?: Record<string, unknown> | undefined;
}, {
    operation: "executeWorkflow";
    workflowId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    context?: Record<string, unknown> | undefined;
    inputs?: Record<string, unknown> | undefined;
    async?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"scheduleWorkflow">;
    workflowId: z.ZodString;
    scheduledTime: z.ZodString;
    inputs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    timezone: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "scheduleWorkflow";
    workflowId: string;
    scheduledTime: string;
    timezone: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    inputs?: Record<string, unknown> | undefined;
}, {
    operation: "scheduleWorkflow";
    workflowId: string;
    scheduledTime: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    inputs?: Record<string, unknown> | undefined;
    timezone?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"pauseWorkflow">;
    executionId: z.ZodString;
    reason: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "pauseWorkflow";
    executionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    reason?: string | undefined;
}, {
    operation: "pauseWorkflow";
    executionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    reason?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"resumeWorkflow">;
    executionId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "resumeWorkflow";
    executionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "resumeWorkflow";
    executionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"cancelWorkflow">;
    executionId: z.ZodString;
    reason: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "cancelWorkflow";
    executionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    reason?: string | undefined;
}, {
    operation: "cancelWorkflow";
    executionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    reason?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getWorkflowStatus">;
    executionId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getWorkflowStatus";
    executionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getWorkflowStatus";
    executionId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listWorkflows">;
    status: z.ZodDefault<z.ZodOptional<z.ZodEnum<["all", "running", "completed", "failed", "paused"]>>>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    status: "all" | "completed" | "running" | "paused" | "failed";
    operation: "listWorkflows";
    limit: number;
    offset: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "listWorkflows";
    status?: "all" | "completed" | "running" | "paused" | "failed" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"updateWorkflow">;
    workflowId: z.ZodString;
    name: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    steps: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        type: z.ZodEnum<["task", "condition", "loop", "parallel", "delay", "bubble"]>;
        config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        nextStepId: z.ZodOptional<z.ZodString>;
        condition: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
        name: string;
        id: string;
        config?: Record<string, unknown> | undefined;
        condition?: Record<string, unknown> | undefined;
        nextStepId?: string | undefined;
    }, {
        type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
        name: string;
        id: string;
        config?: Record<string, unknown> | undefined;
        condition?: Record<string, unknown> | undefined;
        nextStepId?: string | undefined;
    }>, "many">>;
    inputSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    outputSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    timeout: z.ZodOptional<z.ZodNumber>;
    retryPolicy: z.ZodOptional<z.ZodObject<{
        maxAttempts: z.ZodOptional<z.ZodNumber>;
        backoff: z.ZodOptional<z.ZodEnum<["linear", "exponential"]>>;
    }, "strip", z.ZodTypeAny, {
        maxAttempts?: number | undefined;
        backoff?: "exponential" | "linear" | undefined;
    }, {
        maxAttempts?: number | undefined;
        backoff?: "exponential" | "linear" | undefined;
    }>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "updateWorkflow";
    workflowId: string;
    timeout?: number | undefined;
    description?: string | undefined;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    inputSchema?: Record<string, unknown> | undefined;
    outputSchema?: Record<string, unknown> | undefined;
    steps?: {
        type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
        name: string;
        id: string;
        config?: Record<string, unknown> | undefined;
        condition?: Record<string, unknown> | undefined;
        nextStepId?: string | undefined;
    }[] | undefined;
    retryPolicy?: {
        maxAttempts?: number | undefined;
        backoff?: "exponential" | "linear" | undefined;
    } | undefined;
}, {
    operation: "updateWorkflow";
    workflowId: string;
    timeout?: number | undefined;
    description?: string | undefined;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    inputSchema?: Record<string, unknown> | undefined;
    outputSchema?: Record<string, unknown> | undefined;
    steps?: {
        type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
        name: string;
        id: string;
        config?: Record<string, unknown> | undefined;
        condition?: Record<string, unknown> | undefined;
        nextStepId?: string | undefined;
    }[] | undefined;
    retryPolicy?: {
        maxAttempts?: number | undefined;
        backoff?: "exponential" | "linear" | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteWorkflow">;
    workflowId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteWorkflow";
    workflowId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteWorkflow";
    workflowId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
type WorkflowOrchestratorBubbleParams = z.input<typeof WorkflowOrchestratorBubbleParamsSchema>;
declare const WorkflowOrchestratorBubbleResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodUnknown;
    error: z.ZodString;
    meta: z.ZodObject<{
        operation: z.ZodString;
        workflowId: z.ZodOptional<z.ZodString>;
        executionId: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        executionId?: string | undefined;
        workflowId?: string | undefined;
    }, {
        operation: string;
        executionId?: string | undefined;
        workflowId?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        executionId?: string | undefined;
        workflowId?: string | undefined;
    };
    data?: unknown;
}, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        executionId?: string | undefined;
        workflowId?: string | undefined;
    };
    data?: unknown;
}>;
type WorkflowOrchestratorBubbleResult = z.output<typeof WorkflowOrchestratorBubbleResultSchema>;
export declare class WorkflowOrchestratorBubble extends ServiceBubble<WorkflowOrchestratorBubbleParams, WorkflowOrchestratorBubbleResult> {
    static readonly service = "workflow-orchestrator";
    static readonly authType: "apikey";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"createWorkflow">;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        steps: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            type: z.ZodEnum<["task", "condition", "loop", "parallel", "delay", "bubble"]>;
            config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            nextStepId: z.ZodOptional<z.ZodString>;
            condition: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
            name: string;
            id: string;
            config?: Record<string, unknown> | undefined;
            condition?: Record<string, unknown> | undefined;
            nextStepId?: string | undefined;
        }, {
            type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
            name: string;
            id: string;
            config?: Record<string, unknown> | undefined;
            condition?: Record<string, unknown> | undefined;
            nextStepId?: string | undefined;
        }>, "many">;
        inputSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        outputSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        timeout: z.ZodOptional<z.ZodNumber>;
        retryPolicy: z.ZodOptional<z.ZodObject<{
            maxAttempts: z.ZodOptional<z.ZodNumber>;
            backoff: z.ZodOptional<z.ZodEnum<["linear", "exponential"]>>;
        }, "strip", z.ZodTypeAny, {
            maxAttempts?: number | undefined;
            backoff?: "exponential" | "linear" | undefined;
        }, {
            maxAttempts?: number | undefined;
            backoff?: "exponential" | "linear" | undefined;
        }>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        operation: "createWorkflow";
        steps: {
            type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
            name: string;
            id: string;
            config?: Record<string, unknown> | undefined;
            condition?: Record<string, unknown> | undefined;
            nextStepId?: string | undefined;
        }[];
        timeout?: number | undefined;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        inputSchema?: Record<string, unknown> | undefined;
        outputSchema?: Record<string, unknown> | undefined;
        retryPolicy?: {
            maxAttempts?: number | undefined;
            backoff?: "exponential" | "linear" | undefined;
        } | undefined;
    }, {
        name: string;
        operation: "createWorkflow";
        steps: {
            type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
            name: string;
            id: string;
            config?: Record<string, unknown> | undefined;
            condition?: Record<string, unknown> | undefined;
            nextStepId?: string | undefined;
        }[];
        timeout?: number | undefined;
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        inputSchema?: Record<string, unknown> | undefined;
        outputSchema?: Record<string, unknown> | undefined;
        retryPolicy?: {
            maxAttempts?: number | undefined;
            backoff?: "exponential" | "linear" | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"executeWorkflow">;
        workflowId: z.ZodString;
        inputs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        context: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        async: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "executeWorkflow";
        workflowId: string;
        async: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        context?: Record<string, unknown> | undefined;
        inputs?: Record<string, unknown> | undefined;
    }, {
        operation: "executeWorkflow";
        workflowId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        context?: Record<string, unknown> | undefined;
        inputs?: Record<string, unknown> | undefined;
        async?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"scheduleWorkflow">;
        workflowId: z.ZodString;
        scheduledTime: z.ZodString;
        inputs: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        timezone: z.ZodDefault<z.ZodOptional<z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "scheduleWorkflow";
        workflowId: string;
        scheduledTime: string;
        timezone: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        inputs?: Record<string, unknown> | undefined;
    }, {
        operation: "scheduleWorkflow";
        workflowId: string;
        scheduledTime: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        inputs?: Record<string, unknown> | undefined;
        timezone?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"pauseWorkflow">;
        executionId: z.ZodString;
        reason: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "pauseWorkflow";
        executionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        reason?: string | undefined;
    }, {
        operation: "pauseWorkflow";
        executionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        reason?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"resumeWorkflow">;
        executionId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "resumeWorkflow";
        executionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "resumeWorkflow";
        executionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"cancelWorkflow">;
        executionId: z.ZodString;
        reason: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "cancelWorkflow";
        executionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        reason?: string | undefined;
    }, {
        operation: "cancelWorkflow";
        executionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        reason?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getWorkflowStatus">;
        executionId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getWorkflowStatus";
        executionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getWorkflowStatus";
        executionId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listWorkflows">;
        status: z.ZodDefault<z.ZodOptional<z.ZodEnum<["all", "running", "completed", "failed", "paused"]>>>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        offset: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        status: "all" | "completed" | "running" | "paused" | "failed";
        operation: "listWorkflows";
        limit: number;
        offset: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "listWorkflows";
        status?: "all" | "completed" | "running" | "paused" | "failed" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"updateWorkflow">;
        workflowId: z.ZodString;
        name: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        steps: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            type: z.ZodEnum<["task", "condition", "loop", "parallel", "delay", "bubble"]>;
            config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            nextStepId: z.ZodOptional<z.ZodString>;
            condition: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
            name: string;
            id: string;
            config?: Record<string, unknown> | undefined;
            condition?: Record<string, unknown> | undefined;
            nextStepId?: string | undefined;
        }, {
            type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
            name: string;
            id: string;
            config?: Record<string, unknown> | undefined;
            condition?: Record<string, unknown> | undefined;
            nextStepId?: string | undefined;
        }>, "many">>;
        inputSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        outputSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        timeout: z.ZodOptional<z.ZodNumber>;
        retryPolicy: z.ZodOptional<z.ZodObject<{
            maxAttempts: z.ZodOptional<z.ZodNumber>;
            backoff: z.ZodOptional<z.ZodEnum<["linear", "exponential"]>>;
        }, "strip", z.ZodTypeAny, {
            maxAttempts?: number | undefined;
            backoff?: "exponential" | "linear" | undefined;
        }, {
            maxAttempts?: number | undefined;
            backoff?: "exponential" | "linear" | undefined;
        }>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "updateWorkflow";
        workflowId: string;
        timeout?: number | undefined;
        description?: string | undefined;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        inputSchema?: Record<string, unknown> | undefined;
        outputSchema?: Record<string, unknown> | undefined;
        steps?: {
            type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
            name: string;
            id: string;
            config?: Record<string, unknown> | undefined;
            condition?: Record<string, unknown> | undefined;
            nextStepId?: string | undefined;
        }[] | undefined;
        retryPolicy?: {
            maxAttempts?: number | undefined;
            backoff?: "exponential" | "linear" | undefined;
        } | undefined;
    }, {
        operation: "updateWorkflow";
        workflowId: string;
        timeout?: number | undefined;
        description?: string | undefined;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        inputSchema?: Record<string, unknown> | undefined;
        outputSchema?: Record<string, unknown> | undefined;
        steps?: {
            type: "task" | "bubble" | "delay" | "condition" | "loop" | "parallel";
            name: string;
            id: string;
            config?: Record<string, unknown> | undefined;
            condition?: Record<string, unknown> | undefined;
            nextStepId?: string | undefined;
        }[] | undefined;
        retryPolicy?: {
            maxAttempts?: number | undefined;
            backoff?: "exponential" | "linear" | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteWorkflow">;
        workflowId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteWorkflow";
        workflowId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteWorkflow";
        workflowId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodUnknown;
        error: z.ZodString;
        meta: z.ZodObject<{
            operation: z.ZodString;
            workflowId: z.ZodOptional<z.ZodString>;
            executionId: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            executionId?: string | undefined;
            workflowId?: string | undefined;
        }, {
            operation: string;
            executionId?: string | undefined;
            workflowId?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            executionId?: string | undefined;
            workflowId?: string | undefined;
        };
        data?: unknown;
    }, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            executionId?: string | undefined;
            workflowId?: string | undefined;
        };
        data?: unknown;
    }>;
    static readonly shortDescription = "Workflow orchestration and automation engine";
    static readonly longDescription = "\n    Workflow Orchestrator Bubble for complex multi-step processes.\n\n    Features:\n    - Create and manage workflow definitions\n    - Execute workflows with input validation\n    - Schedule workflows for future execution\n    - Pause, resume, and cancel running workflows\n    - Track execution status and history\n    - Retry policies and error handling\n    - Support for conditions, loops, and parallel tasks\n\n    Use cases:\n    - Multi-step business processes\n    - Data pipeline orchestration\n    - Automated approval workflows\n    - Batch job processing\n    - CI/CD pipeline automation\n  ";
    static readonly alias = "workflow";
    constructor(params: WorkflowOrchestratorBubbleParams, context?: BubbleContext, instanceId?: string);
    protected getCredentialType(): CredentialType;
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    protected performAction(context?: BubbleContext): Promise<WorkflowOrchestratorBubbleResult>;
    private createWorkflow;
    private executeWorkflow;
    private simulateExecution;
    private scheduleWorkflow;
    private pauseWorkflow;
    private resumeWorkflow;
    private cancelWorkflow;
    private getWorkflowStatus;
    private listWorkflows;
    private updateWorkflow;
    private deleteWorkflow;
    private extractWorkflowId;
    private extractExecutionId;
}
export {};
//# sourceMappingURL=workflow-orchestrator-bubble.d.ts.map