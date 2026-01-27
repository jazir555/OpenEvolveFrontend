/**
 * EVENT HANDLER WORKFLOW
 *
 * Route and handle events with pattern matching and middleware support.
 */
import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const EventHandlerParamsSchema: z.ZodObject<{
    eventType: z.ZodEnum<["webhook", "message", "schedule", "custom"]>;
    eventPayload: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    routingRules: z.ZodArray<z.ZodObject<{
        condition: z.ZodString;
        handler: z.ZodObject<{
            type: z.ZodEnum<["http", "workflow", "function", "slack", "email"]>;
            config: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        }, "strip", z.ZodTypeAny, {
            type: "function" | "slack" | "http" | "workflow" | "email";
            config: Record<string, unknown>;
        }, {
            type: "function" | "slack" | "http" | "workflow" | "email";
            config: Record<string, unknown>;
        }>;
        priority: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        priority: number;
        condition: string;
        handler: {
            type: "function" | "slack" | "http" | "workflow" | "email";
            config: Record<string, unknown>;
        };
    }, {
        condition: string;
        handler: {
            type: "function" | "slack" | "http" | "workflow" | "email";
            config: Record<string, unknown>;
        };
        priority?: number | undefined;
    }>, "many">;
    middleware: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        config?: Record<string, unknown> | undefined;
    }, {
        name: string;
        config?: Record<string, unknown> | undefined;
    }>, "many">>;
    errorHandling: z.ZodDefault<z.ZodEnum<["continue", "stop", "retry"]>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    eventType: "webhook" | "message" | "custom" | "schedule";
    errorHandling: "stop" | "retry" | "continue";
    eventPayload: Record<string, unknown>;
    routingRules: {
        priority: number;
        condition: string;
        handler: {
            type: "function" | "slack" | "http" | "workflow" | "email";
            config: Record<string, unknown>;
        };
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    middleware?: {
        name: string;
        config?: Record<string, unknown> | undefined;
    }[] | undefined;
}, {
    eventType: "webhook" | "message" | "custom" | "schedule";
    eventPayload: Record<string, unknown>;
    routingRules: {
        condition: string;
        handler: {
            type: "function" | "slack" | "http" | "workflow" | "email";
            config: Record<string, unknown>;
        };
        priority?: number | undefined;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    errorHandling?: "stop" | "retry" | "continue" | undefined;
    middleware?: {
        name: string;
        config?: Record<string, unknown> | undefined;
    }[] | undefined;
}>;
type EventHandlerParams = z.input<typeof EventHandlerParamsSchema>;
declare const EventHandlerResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    error: z.ZodString;
    matchedHandlers: z.ZodArray<z.ZodObject<{
        condition: z.ZodString;
        handlerType: z.ZodString;
        result: z.ZodOptional<z.ZodUnknown>;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        condition: string;
        handlerType: string;
        result?: unknown;
    }, {
        success: boolean;
        condition: string;
        handlerType: string;
        result?: unknown;
    }>, "many">;
    middlewareResults: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        success: boolean;
        error?: string | undefined;
    }, {
        name: string;
        success: boolean;
        error?: string | undefined;
    }>, "many">>;
    executionTime: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    executionTime: number;
    matchedHandlers: {
        success: boolean;
        condition: string;
        handlerType: string;
        result?: unknown;
    }[];
    middlewareResults?: {
        name: string;
        success: boolean;
        error?: string | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    executionTime: number;
    matchedHandlers: {
        success: boolean;
        condition: string;
        handlerType: string;
        result?: unknown;
    }[];
    middlewareResults?: {
        name: string;
        success: boolean;
        error?: string | undefined;
    }[] | undefined;
}>;
export declare class EventHandlerWorkflow extends WorkflowBubble<EventHandlerParams, z.infer<typeof EventHandlerResultSchema>> {
    static readonly type: "workflow";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        eventType: z.ZodEnum<["webhook", "message", "schedule", "custom"]>;
        eventPayload: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        routingRules: z.ZodArray<z.ZodObject<{
            condition: z.ZodString;
            handler: z.ZodObject<{
                type: z.ZodEnum<["http", "workflow", "function", "slack", "email"]>;
                config: z.ZodRecord<z.ZodString, z.ZodUnknown>;
            }, "strip", z.ZodTypeAny, {
                type: "function" | "slack" | "http" | "workflow" | "email";
                config: Record<string, unknown>;
            }, {
                type: "function" | "slack" | "http" | "workflow" | "email";
                config: Record<string, unknown>;
            }>;
            priority: z.ZodDefault<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            priority: number;
            condition: string;
            handler: {
                type: "function" | "slack" | "http" | "workflow" | "email";
                config: Record<string, unknown>;
            };
        }, {
            condition: string;
            handler: {
                type: "function" | "slack" | "http" | "workflow" | "email";
                config: Record<string, unknown>;
            };
            priority?: number | undefined;
        }>, "many">;
        middleware: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            config?: Record<string, unknown> | undefined;
        }, {
            name: string;
            config?: Record<string, unknown> | undefined;
        }>, "many">>;
        errorHandling: z.ZodDefault<z.ZodEnum<["continue", "stop", "retry"]>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        eventType: "webhook" | "message" | "custom" | "schedule";
        errorHandling: "stop" | "retry" | "continue";
        eventPayload: Record<string, unknown>;
        routingRules: {
            priority: number;
            condition: string;
            handler: {
                type: "function" | "slack" | "http" | "workflow" | "email";
                config: Record<string, unknown>;
            };
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        middleware?: {
            name: string;
            config?: Record<string, unknown> | undefined;
        }[] | undefined;
    }, {
        eventType: "webhook" | "message" | "custom" | "schedule";
        eventPayload: Record<string, unknown>;
        routingRules: {
            condition: string;
            handler: {
                type: "function" | "slack" | "http" | "workflow" | "email";
                config: Record<string, unknown>;
            };
            priority?: number | undefined;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        errorHandling?: "stop" | "retry" | "continue" | undefined;
        middleware?: {
            name: string;
            config?: Record<string, unknown> | undefined;
        }[] | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        error: z.ZodString;
        matchedHandlers: z.ZodArray<z.ZodObject<{
            condition: z.ZodString;
            handlerType: z.ZodString;
            result: z.ZodOptional<z.ZodUnknown>;
            success: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            condition: string;
            handlerType: string;
            result?: unknown;
        }, {
            success: boolean;
            condition: string;
            handlerType: string;
            result?: unknown;
        }>, "many">;
        middlewareResults: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            success: z.ZodBoolean;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            success: boolean;
            error?: string | undefined;
        }, {
            name: string;
            success: boolean;
            error?: string | undefined;
        }>, "many">>;
        executionTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        executionTime: number;
        matchedHandlers: {
            success: boolean;
            condition: string;
            handlerType: string;
            result?: unknown;
        }[];
        middlewareResults?: {
            name: string;
            success: boolean;
            error?: string | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        executionTime: number;
        matchedHandlers: {
            success: boolean;
            condition: string;
            handlerType: string;
            result?: unknown;
        }[];
        middlewareResults?: {
            name: string;
            success: boolean;
            error?: string | undefined;
        }[] | undefined;
    }>;
    static readonly shortDescription = "Route and handle events with pattern matching";
    static readonly longDescription = "\n    Event routing and handling system with pattern matching, middleware support, and flexible handlers.\n\n    Features:\n    - Pattern-based event routing with JavaScript expressions\n    - Multiple handler types (HTTP, workflow, function, Slack, email)\n    - Middleware pipeline for pre/post processing\n    - Priority-based handler execution\n    - Comprehensive error handling strategies\n\n    Use cases:\n    - Webhook event processing\n    - Message queue handling\n    - Event-driven architecture\n    - Custom event routing\n    - Integration event processing\n  ";
    static readonly alias = "handle-event";
    constructor(params: EventHandlerParams, context?: BubbleContext);
    protected performAction(): Promise<z.infer<typeof EventHandlerResultSchema>>;
    /**
     * Execute middleware
     */
    private executeMiddleware;
    /**
     * Evaluate condition against event payload
     */
    private evaluateCondition;
    /**
     * Execute handler
     */
    private executeHandler;
}
export {};
//# sourceMappingURL=event-handler.workflow.d.ts.map