import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const ScheduledTaskParamsSchema: z.ZodObject<{
    taskName: z.ZodString;
    schedule: z.ZodObject<{
        type: z.ZodEnum<["cron", "interval", "once"]>;
        expression: z.ZodOptional<z.ZodString>;
        runAt: z.ZodOptional<z.ZodDate>;
    }, "strip", z.ZodTypeAny, {
        type: "once" | "interval" | "cron";
        expression?: string | undefined;
        runAt?: Date | undefined;
    }, {
        type: "once" | "interval" | "cron";
        expression?: string | undefined;
        runAt?: Date | undefined;
    }>;
    action: z.ZodObject<{
        type: z.ZodEnum<["http", "workflow", "function"]>;
        config: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    }, "strip", z.ZodTypeAny, {
        type: "function" | "http" | "workflow";
        config: Record<string, unknown>;
    }, {
        type: "function" | "http" | "workflow";
        config: Record<string, unknown>;
    }>;
    timeout: z.ZodDefault<z.ZodNumber>;
    retryOnFailure: z.ZodDefault<z.ZodBoolean>;
    maxRetries: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    maxRetries: number;
    action: {
        type: "function" | "http" | "workflow";
        config: Record<string, unknown>;
    };
    schedule: {
        type: "once" | "interval" | "cron";
        expression?: string | undefined;
        runAt?: Date | undefined;
    };
    taskName: string;
    retryOnFailure: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    action: {
        type: "function" | "http" | "workflow";
        config: Record<string, unknown>;
    };
    schedule: {
        type: "once" | "interval" | "cron";
        expression?: string | undefined;
        runAt?: Date | undefined;
    };
    taskName: string;
    timeout?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    maxRetries?: number | undefined;
    retryOnFailure?: boolean | undefined;
}>;
type ScheduledTaskParams = z.input<typeof ScheduledTaskParamsSchema>;
declare const ScheduledTaskResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    error: z.ZodString;
    taskId: z.ZodString;
    status: z.ZodEnum<["scheduled", "running", "completed", "failed", "cancelled"]>;
    nextRun: z.ZodOptional<z.ZodDate>;
    result: z.ZodOptional<z.ZodUnknown>;
}, "strip", z.ZodTypeAny, {
    error: string;
    status: "completed" | "running" | "failed" | "cancelled" | "scheduled";
    success: boolean;
    taskId: string;
    result?: unknown;
    nextRun?: Date | undefined;
}, {
    error: string;
    status: "completed" | "running" | "failed" | "cancelled" | "scheduled";
    success: boolean;
    taskId: string;
    result?: unknown;
    nextRun?: Date | undefined;
}>;
export declare class ScheduledTaskWorkflow extends WorkflowBubble<ScheduledTaskParams, z.infer<typeof ScheduledTaskResultSchema>> {
    static readonly type: "workflow";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        taskName: z.ZodString;
        schedule: z.ZodObject<{
            type: z.ZodEnum<["cron", "interval", "once"]>;
            expression: z.ZodOptional<z.ZodString>;
            runAt: z.ZodOptional<z.ZodDate>;
        }, "strip", z.ZodTypeAny, {
            type: "once" | "interval" | "cron";
            expression?: string | undefined;
            runAt?: Date | undefined;
        }, {
            type: "once" | "interval" | "cron";
            expression?: string | undefined;
            runAt?: Date | undefined;
        }>;
        action: z.ZodObject<{
            type: z.ZodEnum<["http", "workflow", "function"]>;
            config: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        }, "strip", z.ZodTypeAny, {
            type: "function" | "http" | "workflow";
            config: Record<string, unknown>;
        }, {
            type: "function" | "http" | "workflow";
            config: Record<string, unknown>;
        }>;
        timeout: z.ZodDefault<z.ZodNumber>;
        retryOnFailure: z.ZodDefault<z.ZodBoolean>;
        maxRetries: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        maxRetries: number;
        action: {
            type: "function" | "http" | "workflow";
            config: Record<string, unknown>;
        };
        schedule: {
            type: "once" | "interval" | "cron";
            expression?: string | undefined;
            runAt?: Date | undefined;
        };
        taskName: string;
        retryOnFailure: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        action: {
            type: "function" | "http" | "workflow";
            config: Record<string, unknown>;
        };
        schedule: {
            type: "once" | "interval" | "cron";
            expression?: string | undefined;
            runAt?: Date | undefined;
        };
        taskName: string;
        timeout?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        maxRetries?: number | undefined;
        retryOnFailure?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        error: z.ZodString;
        taskId: z.ZodString;
        status: z.ZodEnum<["scheduled", "running", "completed", "failed", "cancelled"]>;
        nextRun: z.ZodOptional<z.ZodDate>;
        result: z.ZodOptional<z.ZodUnknown>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: "completed" | "running" | "failed" | "cancelled" | "scheduled";
        success: boolean;
        taskId: string;
        result?: unknown;
        nextRun?: Date | undefined;
    }, {
        error: string;
        status: "completed" | "running" | "failed" | "cancelled" | "scheduled";
        success: boolean;
        taskId: string;
        result?: unknown;
        nextRun?: Date | undefined;
    }>;
    static readonly shortDescription = "Run tasks on schedule with cron/interval support";
    static readonly longDescription = "Schedule and execute tasks using cron expressions, intervals, or specific times with retry support.";
    static readonly alias = "scheduled-task";
    private static scheduledTasks;
    constructor(params: ScheduledTaskParams, context?: BubbleContext);
    protected performAction(): Promise<{
        success: boolean;
        error: string;
        taskId: string;
        status: "scheduled";
        nextRun: Date | undefined;
    } | {
        success: boolean;
        error: string;
        taskId: string;
        status: "failed";
        nextRun: undefined;
    }>;
    private executeTask;
    private generateTaskId;
    static cancelTask(taskId: string): boolean;
}
export {};
//# sourceMappingURL=scheduled-task.workflow.d.ts.map