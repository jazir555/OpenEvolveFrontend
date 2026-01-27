import { z } from 'zod';
import { CredentialType } from './types.js';
export declare const COFFEE_MAX_ITERATIONS = 30;
export declare const COFFEE_MAX_QUESTIONS = 3;
export declare const COFFEE_DEFAULT_MODEL: "google/gemini-3-flash-preview";
/** A single choice option for a clarification question */
export declare const ClarificationChoiceSchema: z.ZodObject<{
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
}>;
/** A clarification question with multiple choices */
export declare const ClarificationQuestionSchema: z.ZodObject<{
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
}>;
/** Event sent to frontend containing clarification questions */
export declare const CoffeeClarificationEventSchema: z.ZodObject<{
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
}>;
/**
 * Event sent when Coffee requests external context via running a BubbleFlow.
 * This pauses the planning process until the user provides credentials and approves execution.
 */
export declare const CoffeeRequestExternalContextEventSchema: z.ZodObject<{
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
/**
 * Answer sent back to Coffee after user provides credentials and flow executes.
 * This is used to resume the planning process with enriched context.
 */
export declare const CoffeeContextAnswerSchema: z.ZodObject<{
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
/**
 * Context request info that the agent generates when it wants to run a flow.
 */
export declare const CoffeeContextRequestInfoSchema: z.ZodObject<{
    purpose: z.ZodString;
    flowDescription: z.ZodString;
}, "strip", z.ZodTypeAny, {
    purpose: string;
    flowDescription: string;
}, {
    purpose: string;
    flowDescription: string;
}>;
/** Legacy context gathering status (used in streaming events) */
export declare const CoffeeContextEventSchema: z.ZodObject<{
    status: z.ZodEnum<["gathering", "complete"]>;
    miniFlowDescription: z.ZodOptional<z.ZodString>;
    result: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    status: "gathering" | "complete";
    result?: string | undefined;
    miniFlowDescription?: string | undefined;
}, {
    status: "gathering" | "complete";
    result?: string | undefined;
    miniFlowDescription?: string | undefined;
}>;
/** A single step in the implementation plan */
export declare const PlanStepSchema: z.ZodObject<{
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
}>;
/** The complete implementation plan generated by Coffee */
export declare const CoffeePlanEventSchema: z.ZodObject<{
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
/** Regular user text message */
export declare const UserMessageSchema: z.ZodObject<{
    id: z.ZodString;
    timestamp: z.ZodString;
} & {
    type: z.ZodLiteral<"user">;
    content: z.ZodString;
}, "strip", z.ZodTypeAny, {
    content: string;
    type: "user";
    timestamp: string;
    id: string;
}, {
    content: string;
    type: "user";
    timestamp: string;
    id: string;
}>;
/** Regular assistant text message */
export declare const AssistantMessageSchema: z.ZodObject<{
    id: z.ZodString;
    timestamp: z.ZodString;
} & {
    type: z.ZodLiteral<"assistant">;
    content: z.ZodString;
    code: z.ZodOptional<z.ZodString>;
    resultType: z.ZodOptional<z.ZodEnum<["code", "question", "answer", "reject"]>>;
    bubbleParameters: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    content: string;
    type: "assistant";
    timestamp: string;
    id: string;
    code?: string | undefined;
    bubbleParameters?: Record<string, unknown> | undefined;
    resultType?: "code" | "question" | "answer" | "reject" | undefined;
}, {
    content: string;
    type: "assistant";
    timestamp: string;
    id: string;
    code?: string | undefined;
    bubbleParameters?: Record<string, unknown> | undefined;
    resultType?: "code" | "question" | "answer" | "reject" | undefined;
}>;
/** Coffee asking clarification questions */
export declare const ClarificationRequestMessageSchema: z.ZodObject<{
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
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
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
}>;
/** User's answers to clarification questions */
export declare const ClarificationResponseMessageSchema: z.ZodObject<{
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
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
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
}>;
/** Coffee requesting external context */
export declare const ContextRequestMessageSchema: z.ZodObject<{
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
    timestamp: string;
    id: string;
    request: {
        description: string;
        flowId: string;
        requiredCredentials: CredentialType[];
        flowCode: string;
    };
}, {
    type: "context_request";
    timestamp: string;
    id: string;
    request: {
        description: string;
        flowId: string;
        requiredCredentials: CredentialType[];
        flowCode: string;
    };
}>;
/** User's response to context request */
export declare const ContextResponseMessageSchema: z.ZodObject<{
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
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
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
}>;
/** Coffee's generated plan */
export declare const PlanMessageSchema: z.ZodObject<{
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
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
    plan: {
        summary: string;
        steps: {
            description: string;
            title: string;
            bubblesUsed?: string[] | undefined;
        }[];
        estimatedBubbles: string[];
    };
}>;
/** User's plan approval with optional comment */
export declare const PlanApprovalMessageSchema: z.ZodObject<{
    id: z.ZodString;
    timestamp: z.ZodString;
} & {
    type: z.ZodLiteral<"plan_approval">;
    approved: z.ZodBoolean;
    comment: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    type: "plan_approval";
    timestamp: string;
    id: string;
    approved: boolean;
    comment?: string | undefined;
}, {
    type: "plan_approval";
    timestamp: string;
    id: string;
    approved: boolean;
    comment?: string | undefined;
}>;
/** System message (for retries, errors, etc.) */
export declare const SystemMessageSchema: z.ZodObject<{
    id: z.ZodString;
    timestamp: z.ZodString;
} & {
    type: z.ZodLiteral<"system">;
    content: z.ZodString;
}, "strip", z.ZodTypeAny, {
    content: string;
    type: "system";
    timestamp: string;
    id: string;
}, {
    content: string;
    type: "system";
    timestamp: string;
    id: string;
}>;
/** Tool result message - persists successful tool call results */
export declare const ToolResultMessageSchema: z.ZodObject<{
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
    toolCallId: string;
    type: "tool_result";
    timestamp: string;
    duration: number;
    id: string;
    success: boolean;
    toolName: string;
    input?: unknown;
    output?: unknown;
}, {
    toolCallId: string;
    type: "tool_result";
    timestamp: string;
    duration: number;
    id: string;
    success: boolean;
    toolName: string;
    input?: unknown;
    output?: unknown;
}>;
/** Union of all Coffee message types */
export declare const CoffeeMessageSchema: z.ZodDiscriminatedUnion<"type", [z.ZodObject<{
    id: z.ZodString;
    timestamp: z.ZodString;
} & {
    type: z.ZodLiteral<"user">;
    content: z.ZodString;
}, "strip", z.ZodTypeAny, {
    content: string;
    type: "user";
    timestamp: string;
    id: string;
}, {
    content: string;
    type: "user";
    timestamp: string;
    id: string;
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
    content: string;
    type: "assistant";
    timestamp: string;
    id: string;
    code?: string | undefined;
    bubbleParameters?: Record<string, unknown> | undefined;
    resultType?: "code" | "question" | "answer" | "reject" | undefined;
}, {
    content: string;
    type: "assistant";
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
    request: {
        description: string;
        flowId: string;
        requiredCredentials: CredentialType[];
        flowCode: string;
    };
}, {
    type: "context_request";
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
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
    timestamp: string;
    id: string;
    approved: boolean;
    comment?: string | undefined;
}, {
    type: "plan_approval";
    timestamp: string;
    id: string;
    approved: boolean;
    comment?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    timestamp: z.ZodString;
} & {
    type: z.ZodLiteral<"system">;
    content: z.ZodString;
}, "strip", z.ZodTypeAny, {
    content: string;
    type: "system";
    timestamp: string;
    id: string;
}, {
    content: string;
    type: "system";
    timestamp: string;
    id: string;
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
    toolCallId: string;
    type: "tool_result";
    timestamp: string;
    duration: number;
    id: string;
    success: boolean;
    toolName: string;
    input?: unknown;
    output?: unknown;
}, {
    toolCallId: string;
    type: "tool_result";
    timestamp: string;
    duration: number;
    id: string;
    success: boolean;
    toolName: string;
    input?: unknown;
    output?: unknown;
}>]>;
/** Request to the Generate BubbleFlow */
export declare const CoffeeRequestSchema: z.ZodObject<{
    prompt: z.ZodString;
    flowId: z.ZodOptional<z.ZodNumber>;
    messages: z.ZodOptional<z.ZodArray<z.ZodDiscriminatedUnion<"type", [z.ZodObject<{
        id: z.ZodString;
        timestamp: z.ZodString;
    } & {
        type: z.ZodLiteral<"user">;
        content: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        content: string;
        type: "user";
        timestamp: string;
        id: string;
    }, {
        content: string;
        type: "user";
        timestamp: string;
        id: string;
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
        content: string;
        type: "assistant";
        timestamp: string;
        id: string;
        code?: string | undefined;
        bubbleParameters?: Record<string, unknown> | undefined;
        resultType?: "code" | "question" | "answer" | "reject" | undefined;
    }, {
        content: string;
        type: "assistant";
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
        request: {
            description: string;
            flowId: string;
            requiredCredentials: CredentialType[];
            flowCode: string;
        };
    }, {
        type: "context_request";
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
        approved: boolean;
        comment?: string | undefined;
    }, {
        type: "plan_approval";
        timestamp: string;
        id: string;
        approved: boolean;
        comment?: string | undefined;
    }>, z.ZodObject<{
        id: z.ZodString;
        timestamp: z.ZodString;
    } & {
        type: z.ZodLiteral<"system">;
        content: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        content: string;
        type: "system";
        timestamp: string;
        id: string;
    }, {
        content: string;
        type: "system";
        timestamp: string;
        id: string;
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
        toolCallId: string;
        type: "tool_result";
        timestamp: string;
        duration: number;
        id: string;
        success: boolean;
        toolName: string;
        input?: unknown;
        output?: unknown;
    }, {
        toolCallId: string;
        type: "tool_result";
        timestamp: string;
        duration: number;
        id: string;
        success: boolean;
        toolName: string;
        input?: unknown;
        output?: unknown;
    }>]>, "many">>;
}, "strip", z.ZodTypeAny, {
    prompt: string;
    flowId?: number | undefined;
    messages?: ({
        content: string;
        type: "user";
        timestamp: string;
        id: string;
    } | {
        content: string;
        type: "assistant";
        timestamp: string;
        id: string;
        code?: string | undefined;
        bubbleParameters?: Record<string, unknown> | undefined;
        resultType?: "code" | "question" | "answer" | "reject" | undefined;
    } | {
        type: "clarification_request";
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
        request: {
            description: string;
            flowId: string;
            requiredCredentials: CredentialType[];
            flowCode: string;
        };
    } | {
        type: "context_response";
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
        approved: boolean;
        comment?: string | undefined;
    } | {
        content: string;
        type: "system";
        timestamp: string;
        id: string;
    } | {
        toolCallId: string;
        type: "tool_result";
        timestamp: string;
        duration: number;
        id: string;
        success: boolean;
        toolName: string;
        input?: unknown;
        output?: unknown;
    })[] | undefined;
}, {
    prompt: string;
    flowId?: number | undefined;
    messages?: ({
        content: string;
        type: "user";
        timestamp: string;
        id: string;
    } | {
        content: string;
        type: "assistant";
        timestamp: string;
        id: string;
        code?: string | undefined;
        bubbleParameters?: Record<string, unknown> | undefined;
        resultType?: "code" | "question" | "answer" | "reject" | undefined;
    } | {
        type: "clarification_request";
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
        request: {
            description: string;
            flowId: string;
            requiredCredentials: CredentialType[];
            flowCode: string;
        };
    } | {
        type: "context_response";
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
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
        timestamp: string;
        id: string;
        approved: boolean;
        comment?: string | undefined;
    } | {
        content: string;
        type: "system";
        timestamp: string;
        id: string;
    } | {
        toolCallId: string;
        type: "tool_result";
        timestamp: string;
        duration: number;
        id: string;
        success: boolean;
        toolName: string;
        input?: unknown;
        output?: unknown;
    })[] | undefined;
}>;
/** Response from the Coffee agent */
export declare const CoffeeResponseSchema: z.ZodObject<{
    type: z.ZodEnum<["clarification", "plan", "context_request", "error"]>;
    clarification: z.ZodOptional<z.ZodObject<{
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
    }>>;
    plan: z.ZodOptional<z.ZodObject<{
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
    }>>;
    contextRequest: z.ZodOptional<z.ZodObject<{
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
    error: z.ZodOptional<z.ZodString>;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    type: "error" | "context_request" | "plan" | "clarification";
    success: boolean;
    error?: string | undefined;
    plan?: {
        summary: string;
        steps: {
            description: string;
            title: string;
            bubblesUsed?: string[] | undefined;
        }[];
        estimatedBubbles: string[];
    } | undefined;
    clarification?: {
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
    } | undefined;
    contextRequest?: {
        description: string;
        flowId: string;
        requiredCredentials: CredentialType[];
        flowCode: string;
    } | undefined;
}, {
    type: "error" | "context_request" | "plan" | "clarification";
    success: boolean;
    error?: string | undefined;
    plan?: {
        summary: string;
        steps: {
            description: string;
            title: string;
            bubblesUsed?: string[] | undefined;
        }[];
        estimatedBubbles: string[];
    } | undefined;
    clarification?: {
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
    } | undefined;
    contextRequest?: {
        description: string;
        flowId: string;
        requiredCredentials: CredentialType[];
        flowCode: string;
    } | undefined;
}>;
/** Internal output format from the Coffee AI agent */
export declare const CoffeeAgentOutputSchema: z.ZodObject<{
    action: z.ZodEnum<["askClarification", "generatePlan", "requestContext"]>;
    questions: z.ZodOptional<z.ZodArray<z.ZodObject<{
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
    plan: z.ZodOptional<z.ZodObject<{
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
    }>>;
    contextRequest: z.ZodOptional<z.ZodObject<{
        purpose: z.ZodString;
        flowDescription: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        purpose: string;
        flowDescription: string;
    }, {
        purpose: string;
        flowDescription: string;
    }>>;
}, "strip", z.ZodTypeAny, {
    action: "askClarification" | "generatePlan" | "requestContext";
    questions?: {
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
    plan?: {
        summary: string;
        steps: {
            description: string;
            title: string;
            bubblesUsed?: string[] | undefined;
        }[];
        estimatedBubbles: string[];
    } | undefined;
    contextRequest?: {
        purpose: string;
        flowDescription: string;
    } | undefined;
}, {
    action: "askClarification" | "generatePlan" | "requestContext";
    questions?: {
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
    plan?: {
        summary: string;
        steps: {
            description: string;
            title: string;
            bubblesUsed?: string[] | undefined;
        }[];
        estimatedBubbles: string[];
    } | undefined;
    contextRequest?: {
        purpose: string;
        flowDescription: string;
    } | undefined;
}>;
export type ClarificationChoice = z.infer<typeof ClarificationChoiceSchema>;
export type ClarificationQuestion = z.infer<typeof ClarificationQuestionSchema>;
export type CoffeeClarificationEvent = z.infer<typeof CoffeeClarificationEventSchema>;
export type CoffeeRequestExternalContextEvent = z.infer<typeof CoffeeRequestExternalContextEventSchema>;
export type CoffeeContextAnswer = z.infer<typeof CoffeeContextAnswerSchema>;
export type CoffeeContextEvent = z.infer<typeof CoffeeContextEventSchema>;
export type CoffeeContextRequestInfo = z.infer<typeof CoffeeContextRequestInfoSchema>;
export type PlanStep = z.infer<typeof PlanStepSchema>;
export type CoffeePlanEvent = z.infer<typeof CoffeePlanEventSchema>;
export type CoffeeRequest = z.infer<typeof CoffeeRequestSchema>;
export type CoffeeResponse = z.infer<typeof CoffeeResponseSchema>;
export type CoffeeAgentOutput = z.infer<typeof CoffeeAgentOutputSchema>;
export type UserMessage = z.infer<typeof UserMessageSchema>;
export type AssistantMessage = z.infer<typeof AssistantMessageSchema>;
export type ClarificationRequestMessage = z.infer<typeof ClarificationRequestMessageSchema>;
export type ClarificationResponseMessage = z.infer<typeof ClarificationResponseMessageSchema>;
export type ContextRequestMessage = z.infer<typeof ContextRequestMessageSchema>;
export type ContextResponseMessage = z.infer<typeof ContextResponseMessageSchema>;
export type PlanMessage = z.infer<typeof PlanMessageSchema>;
export type PlanApprovalMessage = z.infer<typeof PlanApprovalMessageSchema>;
export type SystemMessage = z.infer<typeof SystemMessageSchema>;
export type ToolResultMessage = z.infer<typeof ToolResultMessageSchema>;
export type CoffeeMessage = z.infer<typeof CoffeeMessageSchema>;
//# sourceMappingURL=coffee.d.ts.map