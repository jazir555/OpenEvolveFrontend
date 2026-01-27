import { z } from '@hono/zod-openapi';
import { BubbleParameterType } from './bubble-definition-schema';
import { CredentialType } from './types';
export declare const generateBubbleFlowCodeSchema: z.ZodObject<{
    prompt: z.ZodString;
    flowId: z.ZodOptional<z.ZodNumber>;
    messages: z.ZodOptional<z.ZodArray<z.ZodDiscriminatedUnion<"type", [z.ZodObject<{
        id: z.ZodString;
        timestamp: z.ZodString;
    } & {
        type: z.ZodLiteral<"user">;
        content: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        type: "user";
        content: string;
        id: string;
        timestamp: string;
    }, {
        type: "user";
        content: string;
        id: string;
        timestamp: string;
    }>, z.ZodObject<{
        id: z.ZodString;
        timestamp: z.ZodString;
    } & {
        type: z.ZodLiteral<"assistant">;
        content: z.ZodString;
        code: z.ZodOptional<z.ZodString>;
        resultType: z.ZodOptional<z.ZodEnum<["code", "question", "answer", "reject"]>>;
        bubbleParameters: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        type: "assistant";
        content: string;
        id: string;
        timestamp: string;
        code?: string | undefined;
        bubbleParameters?: Record<string, unknown> | undefined;
        resultType?: "code" | "question" | "answer" | "reject" | undefined;
    }, {
        type: "assistant";
        content: string;
        id: string;
        timestamp: string;
        code?: string | undefined;
        bubbleParameters?: Record<string, unknown> | undefined;
        resultType?: "code" | "question" | "answer" | "reject" | undefined;
    }>, z.ZodObject<{
        id: z.ZodString;
        timestamp: z.ZodString;
    } & {
        type: z.ZodLiteral<"clarification_request">;
        questions: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            question: z.ZodString;
            choices: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                label: z.ZodString;
                description: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                id: string;
                label: string;
                description?: string | undefined;
            }, {
                id: string;
                label: string;
                description?: string | undefined;
            }>, "many">;
            context: z.ZodOptional<z.ZodString>;
            allowMultiple: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            question: string;
            choices: {
                id: string;
                label: string;
                description?: string | undefined;
            }[];
            allowMultiple: boolean;
            context?: string | undefined;
        }, {
            id: string;
            question: string;
            choices: {
                id: string;
                label: string;
                description?: string | undefined;
            }[];
            context?: string | undefined;
            allowMultiple?: boolean | undefined;
        }>, "many">;
    }, "strip", z.ZodTypeAny, {
        type: "clarification_request";
        id: string;
        timestamp: string;
        questions: {
            id: string;
            question: string;
            choices: {
                id: string;
                label: string;
                description?: string | undefined;
            }[];
            allowMultiple: boolean;
            context?: string | undefined;
        }[];
    }, {
        type: "clarification_request";
        id: string;
        timestamp: string;
        questions: {
            id: string;
            question: string;
            choices: {
                id: string;
                label: string;
                description?: string | undefined;
            }[];
            context?: string | undefined;
            allowMultiple?: boolean | undefined;
        }[];
    }>, z.ZodObject<{
        id: z.ZodString;
        timestamp: z.ZodString;
    } & {
        type: z.ZodLiteral<"clarification_response">;
        answers: z.ZodRecord<z.ZodString, z.ZodArray<z.ZodString, "many">>;
        originalQuestions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            question: z.ZodString;
            choices: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                label: z.ZodString;
                description: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                id: string;
                label: string;
                description?: string | undefined;
            }, {
                id: string;
                label: string;
                description?: string | undefined;
            }>, "many">;
            context: z.ZodOptional<z.ZodString>;
            allowMultiple: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            question: string;
            choices: {
                id: string;
                label: string;
                description?: string | undefined;
            }[];
            allowMultiple: boolean;
            context?: string | undefined;
        }, {
            id: string;
            question: string;
            choices: {
                id: string;
                label: string;
                description?: string | undefined;
            }[];
            context?: string | undefined;
            allowMultiple?: boolean | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        type: "clarification_response";
        id: string;
        timestamp: string;
        answers: Record<string, string[]>;
        originalQuestions?: {
            id: string;
            question: string;
            choices: {
                id: string;
                label: string;
                description?: string | undefined;
            }[];
            allowMultiple: boolean;
            context?: string | undefined;
        }[] | undefined;
    }, {
        type: "clarification_response";
        id: string;
        timestamp: string;
        answers: Record<string, string[]>;
        originalQuestions?: {
            id: string;
            question: string;
            choices: {
                id: string;
                label: string;
                description?: string | undefined;
            }[];
            context?: string | undefined;
            allowMultiple?: boolean | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        id: z.ZodString;
        timestamp: z.ZodString;
    } & {
        type: z.ZodLiteral<"context_request">;
        request: z.ZodObject<{
            flowId: z.ZodString;
            flowCode: z.ZodString;
            requiredCredentials: z.ZodArray<z.ZodNativeEnum<typeof CredentialType>, "many">;
            description: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            description: string;
            flowId: string;
            requiredCredentials: CredentialType[];
            flowCode: string;
        }, {
            description: string;
            flowId: string;
            requiredCredentials: CredentialType[];
            flowCode: string;
        }>;
    }, "strip", z.ZodTypeAny, {
        type: "context_request";
        id: string;
        timestamp: string;
        request: {
            description: string;
            flowId: string;
            requiredCredentials: CredentialType[];
            flowCode: string;
        };
    }, {
        type: "context_request";
        id: string;
        timestamp: string;
        request: {
            description: string;
            flowId: string;
            requiredCredentials: CredentialType[];
            flowCode: string;
        };
    }>, z.ZodObject<{
        id: z.ZodString;
        timestamp: z.ZodString;
    } & {
        type: z.ZodLiteral<"context_response">;
        answer: z.ZodObject<{
            flowId: z.ZodString;
            status: z.ZodEnum<["success", "rejected", "error"]>;
            result: z.ZodOptional<z.ZodUnknown>;
            error: z.ZodOptional<z.ZodString>;
            originalRequest: z.ZodOptional<z.ZodObject<{
                flowId: z.ZodString;
                flowCode: z.ZodString;
                requiredCredentials: z.ZodArray<z.ZodNativeEnum<typeof CredentialType>, "many">;
                description: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                description: string;
                flowId: string;
                requiredCredentials: CredentialType[];
                flowCode: string;
            }, {
                description: string;
                flowId: string;
                requiredCredentials: CredentialType[];
                flowCode: string;
            }>>;
        }, "strip", z.ZodTypeAny, {
            status: "success" | "error" | "rejected";
            flowId: string;
            result?: unknown;
            error?: string | undefined;
            originalRequest?: {
                description: string;
                flowId: string;
                requiredCredentials: CredentialType[];
                flowCode: string;
            } | undefined;
        }, {
            status: "success" | "error" | "rejected";
            flowId: string;
            result?: unknown;
            error?: string | undefined;
            originalRequest?: {
                description: string;
                flowId: string;
                requiredCredentials: CredentialType[];
                flowCode: string;
            } | undefined;
        }>;
        credentialTypes: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        type: "context_response";
        id: string;
        timestamp: string;
        answer: {
            status: "success" | "error" | "rejected";
            flowId: string;
            result?: unknown;
            error?: string | undefined;
            originalRequest?: {
                description: string;
                flowId: string;
                requiredCredentials: CredentialType[];
                flowCode: string;
            } | undefined;
        };
        credentialTypes?: string[] | undefined;
    }, {
        type: "context_response";
        id: string;
        timestamp: string;
        answer: {
            status: "success" | "error" | "rejected";
            flowId: string;
            result?: unknown;
            error?: string | undefined;
            originalRequest?: {
                description: string;
                flowId: string;
                requiredCredentials: CredentialType[];
                flowCode: string;
            } | undefined;
        };
        credentialTypes?: string[] | undefined;
    }>, z.ZodObject<{
        id: z.ZodString;
        timestamp: z.ZodString;
    } & {
        type: z.ZodLiteral<"plan">;
        plan: z.ZodObject<{
            summary: z.ZodString;
            steps: z.ZodArray<z.ZodObject<{
                title: z.ZodString;
                description: z.ZodString;
                bubblesUsed: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                description: string;
                title: string;
                bubblesUsed?: string[] | undefined;
            }, {
                description: string;
                title: string;
                bubblesUsed?: string[] | undefined;
            }>, "many">;
            estimatedBubbles: z.ZodArray<z.ZodString, "many">;
        }, "strip", z.ZodTypeAny, {
            summary: string;
            steps: {
                description: string;
                title: string;
                bubblesUsed?: string[] | undefined;
            }[];
            estimatedBubbles: string[];
        }, {
            summary: string;
            steps: {
                description: string;
                title: string;
                bubblesUsed?: string[] | undefined;
            }[];
            estimatedBubbles: string[];
        }>;
    }, "strip", z.ZodTypeAny, {
        type: "plan";
        id: string;
        timestamp: string;
        plan: {
            summary: string;
            steps: {
                description: string;
                title: string;
                bubblesUsed?: string[] | undefined;
            }[];
            estimatedBubbles: string[];
        };
    }, {
        type: "plan";
        id: string;
        timestamp: string;
        plan: {
            summary: string;
            steps: {
                description: string;
                title: string;
                bubblesUsed?: string[] | undefined;
            }[];
            estimatedBubbles: string[];
        };
    }>, z.ZodObject<{
        id: z.ZodString;
        timestamp: z.ZodString;
    } & {
        type: z.ZodLiteral<"plan_approval">;
        approved: z.ZodBoolean;
        comment: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        type: "plan_approval";
        id: string;
        timestamp: string;
        approved: boolean;
        comment?: string | undefined;
    }, {
        type: "plan_approval";
        id: string;
        timestamp: string;
        approved: boolean;
        comment?: string | undefined;
    }>, z.ZodObject<{
        id: z.ZodString;
        timestamp: z.ZodString;
    } & {
        type: z.ZodLiteral<"system">;
        content: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        type: "system";
        content: string;
        id: string;
        timestamp: string;
    }, {
        type: "system";
        content: string;
        id: string;
        timestamp: string;
    }>, z.ZodObject<{
        id: z.ZodString;
        timestamp: z.ZodString;
    } & {
        type: z.ZodLiteral<"tool_result">;
        toolName: z.ZodString;
        toolCallId: z.ZodString;
        input: z.ZodUnknown;
        output: z.ZodUnknown;
        duration: z.ZodNumber;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        type: "tool_result";
        duration: number;
        id: string;
        timestamp: string;
        success: boolean;
        toolName: string;
        toolCallId: string;
        input?: unknown;
        output?: unknown;
    }, {
        type: "tool_result";
        duration: number;
        id: string;
        timestamp: string;
        success: boolean;
        toolName: string;
        toolCallId: string;
        input?: unknown;
        output?: unknown;
    }>]>, "many">>;
    planContext: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    prompt: string;
    flowId?: number | undefined;
    messages?: ({
        type: "user";
        content: string;
        id: string;
        timestamp: string;
    } | {
        type: "assistant";
        content: string;
        id: string;
        timestamp: string;
        code?: string | undefined;
        bubbleParameters?: Record<string, unknown> | undefined;
        resultType?: "code" | "question" | "answer" | "reject" | undefined;
    } | {
        type: "clarification_request";
        id: string;
        timestamp: string;
        questions: {
            id: string;
            question: string;
            choices: {
                id: string;
                label: string;
                description?: string | undefined;
            }[];
            allowMultiple: boolean;
            context?: string | undefined;
        }[];
    } | {
        type: "clarification_response";
        id: string;
        timestamp: string;
        answers: Record<string, string[]>;
        originalQuestions?: {
            id: string;
            question: string;
            choices: {
                id: string;
                label: string;
                description?: string | undefined;
            }[];
            allowMultiple: boolean;
            context?: string | undefined;
        }[] | undefined;
    } | {
        type: "context_request";
        id: string;
        timestamp: string;
        request: {
            description: string;
            flowId: string;
            requiredCredentials: CredentialType[];
            flowCode: string;
        };
    } | {
        type: "context_response";
        id: string;
        timestamp: string;
        answer: {
            status: "success" | "error" | "rejected";
            flowId: string;
            result?: unknown;
            error?: string | undefined;
            originalRequest?: {
                description: string;
                flowId: string;
                requiredCredentials: CredentialType[];
                flowCode: string;
            } | undefined;
        };
        credentialTypes?: string[] | undefined;
    } | {
        type: "plan";
        id: string;
        timestamp: string;
        plan: {
            summary: string;
            steps: {
                description: string;
                title: string;
                bubblesUsed?: string[] | undefined;
            }[];
            estimatedBubbles: string[];
        };
    } | {
        type: "plan_approval";
        id: string;
        timestamp: string;
        approved: boolean;
        comment?: string | undefined;
    } | {
        type: "system";
        content: string;
        id: string;
        timestamp: string;
    } | {
        type: "tool_result";
        duration: number;
        id: string;
        timestamp: string;
        success: boolean;
        toolName: string;
        toolCallId: string;
        input?: unknown;
        output?: unknown;
    })[] | undefined;
    planContext?: string | undefined;
}, {
    prompt: string;
    flowId?: number | undefined;
    messages?: ({
        type: "user";
        content: string;
        id: string;
        timestamp: string;
    } | {
        type: "assistant";
        content: string;
        id: string;
        timestamp: string;
        code?: string | undefined;
        bubbleParameters?: Record<string, unknown> | undefined;
        resultType?: "code" | "question" | "answer" | "reject" | undefined;
    } | {
        type: "clarification_request";
        id: string;
        timestamp: string;
        questions: {
            id: string;
            question: string;
            choices: {
                id: string;
                label: string;
                description?: string | undefined;
            }[];
            context?: string | undefined;
            allowMultiple?: boolean | undefined;
        }[];
    } | {
        type: "clarification_response";
        id: string;
        timestamp: string;
        answers: Record<string, string[]>;
        originalQuestions?: {
            id: string;
            question: string;
            choices: {
                id: string;
                label: string;
                description?: string | undefined;
            }[];
            context?: string | undefined;
            allowMultiple?: boolean | undefined;
        }[] | undefined;
    } | {
        type: "context_request";
        id: string;
        timestamp: string;
        request: {
            description: string;
            flowId: string;
            requiredCredentials: CredentialType[];
            flowCode: string;
        };
    } | {
        type: "context_response";
        id: string;
        timestamp: string;
        answer: {
            status: "success" | "error" | "rejected";
            flowId: string;
            result?: unknown;
            error?: string | undefined;
            originalRequest?: {
                description: string;
                flowId: string;
                requiredCredentials: CredentialType[];
                flowCode: string;
            } | undefined;
        };
        credentialTypes?: string[] | undefined;
    } | {
        type: "plan";
        id: string;
        timestamp: string;
        plan: {
            summary: string;
            steps: {
                description: string;
                title: string;
                bubblesUsed?: string[] | undefined;
            }[];
            estimatedBubbles: string[];
        };
    } | {
        type: "plan_approval";
        id: string;
        timestamp: string;
        approved: boolean;
        comment?: string | undefined;
    } | {
        type: "system";
        content: string;
        id: string;
        timestamp: string;
    } | {
        type: "tool_result";
        duration: number;
        id: string;
        timestamp: string;
        success: boolean;
        toolName: string;
        toolCallId: string;
        input?: unknown;
        output?: unknown;
    })[] | undefined;
    planContext?: string | undefined;
}>;
export declare const generateBubbleFlowCodeResponseSchema: z.ZodObject<{
    generatedCode: z.ZodString;
    isValid: z.ZodBoolean;
    success: z.ZodBoolean;
    error: z.ZodString;
    bubbleParameters: z.ZodRecord<z.ZodString, z.ZodObject<{
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
            type: z.ZodNativeEnum<typeof BubbleParameterType>;
            source: z.ZodOptional<z.ZodEnum<["object-property", "first-arg", "spread"]>>;
        }, "strip", z.ZodTypeAny, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
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
            type: BubbleParameterType;
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
            type: BubbleParameterType;
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
            type: BubbleParameterType;
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
    requiredCredentials: z.ZodRecord<z.ZodString, z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    success: boolean;
    error: string;
    requiredCredentials: Record<string, string[]>;
    bubbleParameters: Record<string, {
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
            type: BubbleParameterType;
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
    generatedCode: string;
    isValid: boolean;
}, {
    success: boolean;
    error: string;
    requiredCredentials: Record<string, string[]>;
    bubbleParameters: Record<string, {
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
            type: BubbleParameterType;
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
    generatedCode: string;
    isValid: boolean;
}>;
/**
 * Schema for the result of BubbleFlow generation
 * Used by the BubbleFlowGeneratorWorkflow
 */
export declare const GenerationResultSchema: z.ZodObject<{
    generatedCode: z.ZodString;
    isValid: z.ZodBoolean;
    success: z.ZodBoolean;
    error: z.ZodString;
    flowId: z.ZodOptional<z.ZodNumber>;
    toolCalls: z.ZodArray<z.ZodUnknown, "many">;
    summary: z.ZodDefault<z.ZodString>;
    inputsSchema: z.ZodDefault<z.ZodString>;
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
    bubbleCount: z.ZodOptional<z.ZodNumber>;
    codeLength: z.ZodOptional<z.ZodNumber>;
    bubbleParameters: z.ZodOptional<z.ZodRecord<z.ZodUnion<[z.ZodString, z.ZodNumber]>, z.ZodObject<{
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
            type: z.ZodNativeEnum<typeof BubbleParameterType>;
            source: z.ZodOptional<z.ZodEnum<["object-property", "first-arg", "spread"]>>;
        }, "strip", z.ZodTypeAny, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: BubbleParameterType;
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
            type: BubbleParameterType;
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
            type: BubbleParameterType;
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
            type: BubbleParameterType;
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
}, "strip", z.ZodTypeAny, {
    success: boolean;
    error: string;
    summary: string;
    generatedCode: string;
    isValid: boolean;
    toolCalls: unknown[];
    inputsSchema: string;
    serviceUsage?: {
        service: CredentialType;
        unit: string;
        usage: number;
        unitCost: number;
        totalCost: number;
        subService?: string | undefined;
    }[] | undefined;
    flowId?: number | undefined;
    bubbleCount?: number | undefined;
    codeLength?: number | undefined;
    bubbleParameters?: Record<string | number, {
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
            type: BubbleParameterType;
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
}, {
    success: boolean;
    error: string;
    generatedCode: string;
    isValid: boolean;
    toolCalls: unknown[];
    serviceUsage?: {
        service: CredentialType;
        unit: string;
        usage: number;
        unitCost: number;
        totalCost: number;
        subService?: string | undefined;
    }[] | undefined;
    summary?: string | undefined;
    flowId?: number | undefined;
    bubbleCount?: number | undefined;
    codeLength?: number | undefined;
    bubbleParameters?: Record<string | number, {
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
            type: BubbleParameterType;
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
    inputsSchema?: string | undefined;
}>;
export declare const generateBubbleFlowTemplateSchema: z.ZodObject<{
    name: z.ZodString;
    description: z.ZodString;
    roles: z.ZodString;
    useCase: z.ZodLiteral<"slack-data-scientist">;
    verbosity: z.ZodOptional<z.ZodEnum<["1", "2", "3", "4", "5"]>>;
    technicality: z.ZodOptional<z.ZodEnum<["1", "2", "3", "4", "5"]>>;
    includeQuery: z.ZodOptional<z.ZodBoolean>;
    includeExplanation: z.ZodOptional<z.ZodBoolean>;
    maxQueries: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    description: string;
    name: string;
    roles: string;
    useCase: "slack-data-scientist";
    verbosity?: "1" | "2" | "3" | "4" | "5" | undefined;
    technicality?: "1" | "2" | "3" | "4" | "5" | undefined;
    includeQuery?: boolean | undefined;
    includeExplanation?: boolean | undefined;
    maxQueries?: number | undefined;
}, {
    description: string;
    name: string;
    roles: string;
    useCase: "slack-data-scientist";
    verbosity?: "1" | "2" | "3" | "4" | "5" | undefined;
    technicality?: "1" | "2" | "3" | "4" | "5" | undefined;
    includeQuery?: boolean | undefined;
    includeExplanation?: boolean | undefined;
    maxQueries?: number | undefined;
}>;
export declare const generateDocumentGenerationTemplateSchema: z.ZodObject<{
    name: z.ZodString;
    description: z.ZodDefault<z.ZodString>;
    outputDescription: z.ZodString;
    outputFormat: z.ZodOptional<z.ZodEnum<["html", "csv", "json"]>>;
    conversionOptions: z.ZodOptional<z.ZodObject<{
        preserveStructure: z.ZodOptional<z.ZodBoolean>;
        includeVisualDescriptions: z.ZodOptional<z.ZodBoolean>;
        extractNumericalData: z.ZodOptional<z.ZodBoolean>;
        combinePages: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        preserveStructure?: boolean | undefined;
        includeVisualDescriptions?: boolean | undefined;
        extractNumericalData?: boolean | undefined;
        combinePages?: boolean | undefined;
    }, {
        preserveStructure?: boolean | undefined;
        includeVisualDescriptions?: boolean | undefined;
        extractNumericalData?: boolean | undefined;
        combinePages?: boolean | undefined;
    }>>;
    imageOptions: z.ZodOptional<z.ZodObject<{
        format: z.ZodOptional<z.ZodEnum<["png", "jpg", "jpeg"]>>;
        quality: z.ZodOptional<z.ZodNumber>;
        dpi: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        format?: "png" | "jpg" | "jpeg" | undefined;
        quality?: number | undefined;
        dpi?: number | undefined;
    }, {
        format?: "png" | "jpg" | "jpeg" | undefined;
        quality?: number | undefined;
        dpi?: number | undefined;
    }>>;
    aiOptions: z.ZodOptional<z.ZodObject<{
        model: z.ZodOptional<z.ZodString>;
        temperature: z.ZodOptional<z.ZodNumber>;
        maxTokens: z.ZodOptional<z.ZodNumber>;
        jsonMode: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        model?: string | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
        jsonMode?: boolean | undefined;
    }, {
        model?: string | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
        jsonMode?: boolean | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    description: string;
    name: string;
    outputDescription: string;
    metadata?: Record<string, unknown> | undefined;
    outputFormat?: "html" | "csv" | "json" | undefined;
    conversionOptions?: {
        preserveStructure?: boolean | undefined;
        includeVisualDescriptions?: boolean | undefined;
        extractNumericalData?: boolean | undefined;
        combinePages?: boolean | undefined;
    } | undefined;
    imageOptions?: {
        format?: "png" | "jpg" | "jpeg" | undefined;
        quality?: number | undefined;
        dpi?: number | undefined;
    } | undefined;
    aiOptions?: {
        model?: string | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
        jsonMode?: boolean | undefined;
    } | undefined;
}, {
    name: string;
    outputDescription: string;
    description?: string | undefined;
    metadata?: Record<string, unknown> | undefined;
    outputFormat?: "html" | "csv" | "json" | undefined;
    conversionOptions?: {
        preserveStructure?: boolean | undefined;
        includeVisualDescriptions?: boolean | undefined;
        extractNumericalData?: boolean | undefined;
        combinePages?: boolean | undefined;
    } | undefined;
    imageOptions?: {
        format?: "png" | "jpg" | "jpeg" | undefined;
        quality?: number | undefined;
        dpi?: number | undefined;
    } | undefined;
    aiOptions?: {
        model?: string | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
        jsonMode?: boolean | undefined;
    } | undefined;
}>;
export declare const bubbleFlowTemplateResponseSchema: z.ZodObject<{
    id: z.ZodNumber;
    name: z.ZodString;
    description: z.ZodString;
    eventType: z.ZodString;
    displayedBubbleParameters: z.ZodRecord<z.ZodString, z.ZodObject<{
        variableName: z.ZodString;
        bubbleName: z.ZodString;
        className: z.ZodString;
        parameters: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            value: z.ZodUnknown;
            type: z.ZodNativeEnum<typeof BubbleParameterType>;
        }, "strip", z.ZodTypeAny, {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }, {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }>, "many">;
        hasAwait: z.ZodBoolean;
        hasActionCall: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        variableName: string;
        className: string;
        parameters: {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: string;
    }, {
        variableName: string;
        className: string;
        parameters: {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: string;
    }>>;
    flowDecomposition: z.ZodOptional<z.ZodObject<{
        displayedParameters: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            displayName: z.ZodString;
            value: z.ZodUnknown;
            type: z.ZodNativeEnum<typeof BubbleParameterType>;
            isRequired: z.ZodBoolean;
            isConfigurable: z.ZodBoolean;
            description: z.ZodOptional<z.ZodString>;
            group: z.ZodOptional<z.ZodString>;
            source: z.ZodEnum<["literal", "reference", "environment", "computed"]>;
        }, "strip", z.ZodTypeAny, {
            type: BubbleParameterType;
            name: string;
            source: "literal" | "reference" | "environment" | "computed";
            displayName: string;
            isRequired: boolean;
            isConfigurable: boolean;
            value?: unknown;
            description?: string | undefined;
            group?: string | undefined;
        }, {
            type: BubbleParameterType;
            name: string;
            source: "literal" | "reference" | "environment" | "computed";
            displayName: string;
            isRequired: boolean;
            isConfigurable: boolean;
            value?: unknown;
            description?: string | undefined;
            group?: string | undefined;
        }>, "many">;
        dependencies: z.ZodObject<{
            nodes: z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                type: z.ZodEnum<["bubble", "parameter", "trigger"]>;
                label: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                type: "bubble" | "parameter" | "trigger";
                id: string;
                label: string;
            }, {
                type: "bubble" | "parameter" | "trigger";
                id: string;
                label: string;
            }>, "many">;
            edges: z.ZodArray<z.ZodObject<{
                from: z.ZodString;
                to: z.ZodString;
                type: z.ZodEnum<["data", "control", "resource"]>;
                description: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                type: "data" | "control" | "resource";
                from: string;
                to: string;
                description?: string | undefined;
            }, {
                type: "data" | "control" | "resource";
                from: string;
                to: string;
                description?: string | undefined;
            }>, "many">;
        }, "strip", z.ZodTypeAny, {
            nodes: {
                type: "bubble" | "parameter" | "trigger";
                id: string;
                label: string;
            }[];
            edges: {
                type: "data" | "control" | "resource";
                from: string;
                to: string;
                description?: string | undefined;
            }[];
        }, {
            nodes: {
                type: "bubble" | "parameter" | "trigger";
                id: string;
                label: string;
            }[];
            edges: {
                type: "data" | "control" | "resource";
                from: string;
                to: string;
                description?: string | undefined;
            }[];
        }>;
        validationRules: z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["required", "format", "range", "custom"]>;
            message: z.ZodString;
            severity: z.ZodOptional<z.ZodEnum<["error", "warning", "info"]>>;
        }, "strip", z.ZodTypeAny, {
            message: string;
            type: "format" | "required" | "custom" | "range";
            severity?: "error" | "warning" | "info" | undefined;
        }, {
            message: string;
            type: "format" | "required" | "custom" | "range";
            severity?: "error" | "warning" | "info" | undefined;
        }>, "many">;
        metadata: z.ZodObject<{
            totalParameters: z.ZodNumber;
            requiredParameters: z.ZodNumber;
            configurableParameters: z.ZodNumber;
            environmentParameters: z.ZodNumber;
            nestedParameterCount: z.ZodNumber;
            conditionalParameterCount: z.ZodNumber;
            hasCircularDependencies: z.ZodBoolean;
            estimatedComplexity: z.ZodEnum<["simple", "medium", "complex"]>;
        }, "strip", z.ZodTypeAny, {
            totalParameters: number;
            requiredParameters: number;
            configurableParameters: number;
            environmentParameters: number;
            nestedParameterCount: number;
            conditionalParameterCount: number;
            hasCircularDependencies: boolean;
            estimatedComplexity: "simple" | "medium" | "complex";
        }, {
            totalParameters: number;
            requiredParameters: number;
            configurableParameters: number;
            environmentParameters: number;
            nestedParameterCount: number;
            conditionalParameterCount: number;
            hasCircularDependencies: boolean;
            estimatedComplexity: "simple" | "medium" | "complex";
        }>;
    }, "strip", z.ZodTypeAny, {
        dependencies: {
            nodes: {
                type: "bubble" | "parameter" | "trigger";
                id: string;
                label: string;
            }[];
            edges: {
                type: "data" | "control" | "resource";
                from: string;
                to: string;
                description?: string | undefined;
            }[];
        };
        metadata: {
            totalParameters: number;
            requiredParameters: number;
            configurableParameters: number;
            environmentParameters: number;
            nestedParameterCount: number;
            conditionalParameterCount: number;
            hasCircularDependencies: boolean;
            estimatedComplexity: "simple" | "medium" | "complex";
        };
        displayedParameters: {
            type: BubbleParameterType;
            name: string;
            source: "literal" | "reference" | "environment" | "computed";
            displayName: string;
            isRequired: boolean;
            isConfigurable: boolean;
            value?: unknown;
            description?: string | undefined;
            group?: string | undefined;
        }[];
        validationRules: {
            message: string;
            type: "format" | "required" | "custom" | "range";
            severity?: "error" | "warning" | "info" | undefined;
        }[];
    }, {
        dependencies: {
            nodes: {
                type: "bubble" | "parameter" | "trigger";
                id: string;
                label: string;
            }[];
            edges: {
                type: "data" | "control" | "resource";
                from: string;
                to: string;
                description?: string | undefined;
            }[];
        };
        metadata: {
            totalParameters: number;
            requiredParameters: number;
            configurableParameters: number;
            environmentParameters: number;
            nestedParameterCount: number;
            conditionalParameterCount: number;
            hasCircularDependencies: boolean;
            estimatedComplexity: "simple" | "medium" | "complex";
        };
        displayedParameters: {
            type: BubbleParameterType;
            name: string;
            source: "literal" | "reference" | "environment" | "computed";
            displayName: string;
            isRequired: boolean;
            isConfigurable: boolean;
            value?: unknown;
            description?: string | undefined;
            group?: string | undefined;
        }[];
        validationRules: {
            message: string;
            type: "format" | "required" | "custom" | "range";
            severity?: "error" | "warning" | "info" | undefined;
        }[];
    }>>;
    bubbleParameters: z.ZodRecord<z.ZodString, z.ZodObject<{
        variableName: z.ZodString;
        bubbleName: z.ZodString;
        className: z.ZodString;
        parameters: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            value: z.ZodUnknown;
            type: z.ZodNativeEnum<typeof BubbleParameterType>;
        }, "strip", z.ZodTypeAny, {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }, {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }>, "many">;
        hasAwait: z.ZodBoolean;
        hasActionCall: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        variableName: string;
        className: string;
        parameters: {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: string;
    }, {
        variableName: string;
        className: string;
        parameters: {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: string;
    }>>;
    requiredCredentials: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodNativeEnum<typeof CredentialType>, "many">>>;
    createdAt: z.ZodString;
    updatedAt: z.ZodString;
    webhook: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        url: z.ZodString;
        path: z.ZodString;
        active: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        path: string;
        url: string;
        id: number;
        active: boolean;
    }, {
        path: string;
        url: string;
        id: number;
        active: boolean;
    }>>;
}, "strip", z.ZodTypeAny, {
    description: string;
    name: string;
    id: number;
    createdAt: string;
    updatedAt: string;
    eventType: string;
    bubbleParameters: Record<string, {
        variableName: string;
        className: string;
        parameters: {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: string;
    }>;
    displayedBubbleParameters: Record<string, {
        variableName: string;
        className: string;
        parameters: {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: string;
    }>;
    webhook?: {
        path: string;
        url: string;
        id: number;
        active: boolean;
    } | undefined;
    requiredCredentials?: Record<string, CredentialType[]> | undefined;
    flowDecomposition?: {
        dependencies: {
            nodes: {
                type: "bubble" | "parameter" | "trigger";
                id: string;
                label: string;
            }[];
            edges: {
                type: "data" | "control" | "resource";
                from: string;
                to: string;
                description?: string | undefined;
            }[];
        };
        metadata: {
            totalParameters: number;
            requiredParameters: number;
            configurableParameters: number;
            environmentParameters: number;
            nestedParameterCount: number;
            conditionalParameterCount: number;
            hasCircularDependencies: boolean;
            estimatedComplexity: "simple" | "medium" | "complex";
        };
        displayedParameters: {
            type: BubbleParameterType;
            name: string;
            source: "literal" | "reference" | "environment" | "computed";
            displayName: string;
            isRequired: boolean;
            isConfigurable: boolean;
            value?: unknown;
            description?: string | undefined;
            group?: string | undefined;
        }[];
        validationRules: {
            message: string;
            type: "format" | "required" | "custom" | "range";
            severity?: "error" | "warning" | "info" | undefined;
        }[];
    } | undefined;
}, {
    description: string;
    name: string;
    id: number;
    createdAt: string;
    updatedAt: string;
    eventType: string;
    bubbleParameters: Record<string, {
        variableName: string;
        className: string;
        parameters: {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: string;
    }>;
    displayedBubbleParameters: Record<string, {
        variableName: string;
        className: string;
        parameters: {
            type: BubbleParameterType;
            name: string;
            value?: unknown;
        }[];
        hasAwait: boolean;
        hasActionCall: boolean;
        bubbleName: string;
    }>;
    webhook?: {
        path: string;
        url: string;
        id: number;
        active: boolean;
    } | undefined;
    requiredCredentials?: Record<string, CredentialType[]> | undefined;
    flowDecomposition?: {
        dependencies: {
            nodes: {
                type: "bubble" | "parameter" | "trigger";
                id: string;
                label: string;
            }[];
            edges: {
                type: "data" | "control" | "resource";
                from: string;
                to: string;
                description?: string | undefined;
            }[];
        };
        metadata: {
            totalParameters: number;
            requiredParameters: number;
            configurableParameters: number;
            environmentParameters: number;
            nestedParameterCount: number;
            conditionalParameterCount: number;
            hasCircularDependencies: boolean;
            estimatedComplexity: "simple" | "medium" | "complex";
        };
        displayedParameters: {
            type: BubbleParameterType;
            name: string;
            source: "literal" | "reference" | "environment" | "computed";
            displayName: string;
            isRequired: boolean;
            isConfigurable: boolean;
            value?: unknown;
            description?: string | undefined;
            group?: string | undefined;
        }[];
        validationRules: {
            message: string;
            type: "format" | "required" | "custom" | "range";
            severity?: "error" | "warning" | "info" | undefined;
        }[];
    } | undefined;
}>;
export type GenerateBubbleFlowCodeResponse = z.infer<typeof generateBubbleFlowCodeResponseSchema>;
export type GenerateBubbleFlowTemplateRequest = z.infer<typeof generateBubbleFlowTemplateSchema>;
export type GenerateDocumentGenerationTemplateRequest = z.infer<typeof generateDocumentGenerationTemplateSchema>;
export type BubbleFlowTemplateResponse = z.infer<typeof bubbleFlowTemplateResponseSchema>;
export type GenerationResult = z.infer<typeof GenerationResultSchema>;
//# sourceMappingURL=generate-bubbleflow-schema.d.ts.map