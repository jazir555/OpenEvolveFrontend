import { z } from 'zod';
import { type AvailableModel } from './ai-models.js';
import { CredentialType } from './types.js';
export declare const PEARL_DEFAULT_MODEL: AvailableModel;
/**
 * Request schema for Pearl agent
 * Pearl helps users build complete workflows without requiring specific bubbles
 */
export declare const PearlRequestSchema: z.ZodObject<{
    userRequest: z.ZodString;
    currentCode: z.ZodOptional<z.ZodString>;
    userName: z.ZodString;
    availableVariables: z.ZodArray<z.ZodAny, "many">;
    conversationHistory: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodObject<{
        role: z.ZodEnum<["user", "assistant", "tool"]>;
        content: z.ZodString;
        toolCallId: z.ZodOptional<z.ZodString>;
        name: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        content: string;
        role: "tool" | "user" | "assistant";
        name?: string | undefined;
        toolCallId?: string | undefined;
    }, {
        content: string;
        role: "tool" | "user" | "assistant";
        name?: string | undefined;
        toolCallId?: string | undefined;
    }>, "many">>>;
    model: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
    additionalContext: z.ZodOptional<z.ZodString>;
    uploadedFiles: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        content: z.ZodString;
        fileType: z.ZodEnum<["image", "text"]>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        content: string;
        fileType: "text" | "image";
    }, {
        name: string;
        content: string;
        fileType: "text" | "image";
    }>, "many">>>;
}, "strip", z.ZodTypeAny, {
    model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
    userRequest: string;
    userName: string;
    availableVariables: any[];
    conversationHistory: {
        content: string;
        role: "tool" | "user" | "assistant";
        name?: string | undefined;
        toolCallId?: string | undefined;
    }[];
    uploadedFiles: {
        name: string;
        content: string;
        fileType: "text" | "image";
    }[];
    currentCode?: string | undefined;
    additionalContext?: string | undefined;
}, {
    userRequest: string;
    userName: string;
    availableVariables: any[];
    model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
    currentCode?: string | undefined;
    conversationHistory?: {
        content: string;
        role: "tool" | "user" | "assistant";
        name?: string | undefined;
        toolCallId?: string | undefined;
    }[] | undefined;
    additionalContext?: string | undefined;
    uploadedFiles?: {
        name: string;
        content: string;
        fileType: "text" | "image";
    }[] | undefined;
}>;
/**
 * Response schema for Pearl agent
 */
export declare const PearlResponseSchema: z.ZodObject<{
    type: z.ZodEnum<["code", "question", "answer", "reject"]>;
    message: z.ZodString;
    snippet: z.ZodOptional<z.ZodString>;
    bubbleParameters: z.ZodOptional<z.ZodRecord<z.ZodNumber, z.ZodObject<{
        variableName: z.ZodString;
        bubbleName: z.ZodType<import("./types.js").BubbleName>;
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
            type: z.ZodNativeEnum<typeof import("./bubble-definition-schema.js").BubbleParameterType>;
            source: z.ZodOptional<z.ZodEnum<["object-property", "first-arg", "spread"]>>;
        }, "strip", z.ZodTypeAny, {
            value: string | number | boolean | unknown[] | Record<string, unknown>;
            type: import("./bubble-definition-schema.js").BubbleParameterType;
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
            type: import("./bubble-definition-schema.js").BubbleParameterType;
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
        dependencies: z.ZodOptional<z.ZodArray<z.ZodType<import("./types.js").BubbleName, z.ZodTypeDef, import("./types.js").BubbleName>, "many">>;
        dependencyGraph: z.ZodOptional<z.ZodType<import("./bubble-definition-schema.js").DependencyGraphNode, z.ZodTypeDef, import("./bubble-definition-schema.js").DependencyGraphNode>>;
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
            type: import("./bubble-definition-schema.js").BubbleParameterType;
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
        bubbleName: import("./types.js").BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: import("./types.js").BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
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
            type: import("./bubble-definition-schema.js").BubbleParameterType;
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
        bubbleName: import("./types.js").BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: import("./types.js").BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }>>>;
    inputSchema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    requiredCredentials: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodNativeEnum<typeof CredentialType>, "many">>>;
    success: z.ZodBoolean;
    error: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    message: string;
    type: "code" | "question" | "answer" | "reject";
    success: boolean;
    error?: string | undefined;
    inputSchema?: Record<string, unknown> | undefined;
    requiredCredentials?: Record<string, CredentialType[]> | undefined;
    bubbleParameters?: Record<number, {
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
            type: import("./bubble-definition-schema.js").BubbleParameterType;
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
        bubbleName: import("./types.js").BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: import("./types.js").BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }> | undefined;
    snippet?: string | undefined;
}, {
    message: string;
    type: "code" | "question" | "answer" | "reject";
    success: boolean;
    error?: string | undefined;
    inputSchema?: Record<string, unknown> | undefined;
    requiredCredentials?: Record<string, CredentialType[]> | undefined;
    bubbleParameters?: Record<number, {
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
            type: import("./bubble-definition-schema.js").BubbleParameterType;
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
        bubbleName: import("./types.js").BubbleName;
        nodeType: "unknown" | "service" | "tool" | "workflow";
        description?: string | undefined;
        dependencies?: import("./types.js").BubbleName[] | undefined;
        dependencyGraph?: import("./bubble-definition-schema.js").DependencyGraphNode | undefined;
        invocationCallSiteKey?: string | undefined;
        clonedFromVariableId?: number | undefined;
        isInsideCustomTool?: boolean | undefined;
        containingCustomToolId?: string | undefined;
    }> | undefined;
    snippet?: string | undefined;
}>;
/**
 * Internal agent response format (JSON mode output from AI)
 */
export declare const PearlAgentOutputSchema: z.ZodObject<{
    type: z.ZodEnum<["code", "question", "answer", "reject", "text"]>;
    message: z.ZodString;
    snippet: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    message: string;
    type: "code" | "text" | "question" | "answer" | "reject";
    snippet?: string | undefined;
}, {
    message: string;
    type: "code" | "text" | "question" | "answer" | "reject";
    snippet?: string | undefined;
}>;
export type PearlRequest = z.infer<typeof PearlRequestSchema>;
export type PearlResponse = z.infer<typeof PearlResponseSchema>;
export type PearlAgentOutput = z.infer<typeof PearlAgentOutputSchema>;
//# sourceMappingURL=pearl.d.ts.map