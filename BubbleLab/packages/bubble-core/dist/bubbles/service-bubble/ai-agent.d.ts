import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { type AvailableTool } from '../../types/available-tools.js';
import type { BubbleName, BubbleResult } from '@bubblelab/shared-schemas';
import type { StreamingEvent } from '@bubblelab/shared-schemas';
import { ConversationMessageSchema } from '@bubblelab/shared-schemas';
export type ToolHookContext = {
    toolName: AvailableTool;
    toolInput: unknown;
    toolOutput?: BubbleResult<unknown>;
    messages: BaseMessage[];
};
export type ToolHookAfter = (context: ToolHookContext) => Promise<{
    messages: BaseMessage[];
    shouldStop?: boolean;
}>;
export type ToolHookBefore = (context: ToolHookContext) => Promise<{
    messages: BaseMessage[];
    toolInput: Record<string, any>;
}>;
export type AfterLLMCallContext = {
    messages: BaseMessage[];
    lastAIMessage: AIMessage | AIMessageChunk;
    hasToolCalls: boolean;
};
export type AfterLLMCallHook = (context: AfterLLMCallContext) => Promise<{
    messages: BaseMessage[];
    continueToAgent?: boolean;
}>;
export type StreamingCallback = (event: StreamingEvent) => Promise<void> | void;
/** Type for conversation history messages - enables KV cache optimization */
export type ConversationMessage = z.infer<typeof ConversationMessageSchema>;
declare const AIAgentParamsSchema: z.ZodObject<{
    message: z.ZodString;
    images: z.ZodDefault<z.ZodArray<z.ZodDiscriminatedUnion<"type", [z.ZodObject<{
        type: z.ZodDefault<z.ZodLiteral<"base64">>;
        data: z.ZodString;
        mimeType: z.ZodDefault<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        type: "base64";
        data: string;
        mimeType: string;
        description?: string | undefined;
    }, {
        data: string;
        type?: "base64" | undefined;
        description?: string | undefined;
        mimeType?: string | undefined;
    }>, z.ZodObject<{
        type: z.ZodLiteral<"url">;
        url: z.ZodEffects<z.ZodString, string, string>;
        description: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        type: "url";
        url: string;
        description?: string | undefined;
    }, {
        type: "url";
        url: string;
        description?: string | undefined;
    }>]>, "many">>;
    conversationHistory: z.ZodOptional<z.ZodArray<z.ZodObject<{
        role: z.ZodEnum<["user", "assistant", "tool"]>;
        content: z.ZodString;
        toolCallId: z.ZodOptional<z.ZodString>;
        name: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        role: "user" | "assistant" | "tool";
        content: string;
        toolCallId?: string | undefined;
        name?: string | undefined;
    }, {
        role: "user" | "assistant" | "tool";
        content: string;
        toolCallId?: string | undefined;
        name?: string | undefined;
    }>, "many">>;
    systemPrompt: z.ZodDefault<z.ZodString>;
    name: z.ZodOptional<z.ZodDefault<z.ZodString>>;
    model: z.ZodDefault<z.ZodObject<{
        model: z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>;
        temperature: z.ZodDefault<z.ZodNumber>;
        maxTokens: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        reasoningEffort: z.ZodOptional<z.ZodEnum<["low", "medium", "high"]>>;
        maxRetries: z.ZodDefault<z.ZodNumber>;
        provider: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        jsonMode: z.ZodDefault<z.ZodBoolean>;
        backupModel: z.ZodOptional<z.ZodDefault<z.ZodObject<{
            model: z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>;
            temperature: z.ZodOptional<z.ZodNumber>;
            maxTokens: z.ZodOptional<z.ZodNumber>;
            reasoningEffort: z.ZodOptional<z.ZodEnum<["low", "medium", "high"]>>;
            maxRetries: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            reasoningEffort?: "low" | "medium" | "high" | undefined;
            maxRetries?: number | undefined;
        }, {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            reasoningEffort?: "low" | "medium" | "high" | undefined;
            maxRetries?: number | undefined;
        }>>>;
    }, "strip", z.ZodTypeAny, {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature: number;
        maxTokens: number;
        maxRetries: number;
        jsonMode: boolean;
        reasoningEffort?: "low" | "medium" | "high" | undefined;
        provider?: string[] | undefined;
        backupModel?: {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            reasoningEffort?: "low" | "medium" | "high" | undefined;
            maxRetries?: number | undefined;
        } | undefined;
    }, {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature?: number | undefined;
        maxTokens?: number | undefined;
        reasoningEffort?: "low" | "medium" | "high" | undefined;
        maxRetries?: number | undefined;
        provider?: string[] | undefined;
        jsonMode?: boolean | undefined;
        backupModel?: {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            reasoningEffort?: "low" | "medium" | "high" | undefined;
            maxRetries?: number | undefined;
        } | undefined;
    }>>;
    tools: z.ZodDefault<z.ZodArray<z.ZodObject<{
        name: z.ZodEnum<["web-search-tool", "web-scrape-tool", "web-crawl-tool", "web-extract-tool", "research-agent-tool", "reddit-scrape-tool", "instagram-tool", "list-bubbles-tool", "get-bubble-details-tool", "bubbleflow-validation-tool", "code-edit-tool", "chart-js-tool", "sql-query-tool"]>;
        credentials: z.ZodOptional<z.ZodDefault<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>>;
        config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        name: "get-bubble-details-tool" | "list-bubbles-tool" | "sql-query-tool" | "chart-js-tool" | "web-search-tool" | "web-scrape-tool" | "web-crawl-tool" | "web-extract-tool" | "research-agent-tool" | "reddit-scrape-tool" | "bubbleflow-validation-tool" | "code-edit-tool" | "instagram-tool";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
    }, {
        name: "get-bubble-details-tool" | "list-bubbles-tool" | "sql-query-tool" | "chart-js-tool" | "web-search-tool" | "web-scrape-tool" | "web-crawl-tool" | "web-extract-tool" | "research-agent-tool" | "reddit-scrape-tool" | "bubbleflow-validation-tool" | "code-edit-tool" | "instagram-tool";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
    }>, "many">>;
    customTools: z.ZodOptional<z.ZodDefault<z.ZodArray<z.ZodEffects<z.ZodObject<{
        name: z.ZodString;
        description: z.ZodString;
        schema: z.ZodUnion<[z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodType<z.ZodTypeAny, z.ZodTypeDef, z.ZodTypeAny>]>;
        func: z.ZodFunction<z.ZodTuple<[z.ZodRecord<z.ZodString, z.ZodUnknown>], z.ZodUnknown>, z.ZodPromise<z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        description: string;
        name: string;
        schema: z.ZodTypeAny | Record<string, unknown>;
        func: (args_0: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
    }, {
        description: string;
        name: string;
        schema: z.ZodTypeAny | Record<string, unknown>;
        func: (args_0: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
    }>, {
        description: string;
        name: string;
        schema: z.ZodTypeAny | Record<string, unknown>;
        func: (args_0: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
    }, {
        description: string;
        name: string;
        schema: z.ZodTypeAny | Record<string, unknown>;
        func: (args_0: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
    }>, "many">>>;
    maxIterations: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    streaming: z.ZodDefault<z.ZodBoolean>;
    expectedOutputSchema: z.ZodOptional<z.ZodUnion<[z.ZodType<z.ZodTypeAny, z.ZodTypeDef, z.ZodTypeAny>, z.ZodString]>>;
}, "strip", z.ZodTypeAny, {
    message: string;
    images: ({
        type: "base64";
        data: string;
        mimeType: string;
        description?: string | undefined;
    } | {
        type: "url";
        url: string;
        description?: string | undefined;
    })[];
    systemPrompt: string;
    model: {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature: number;
        maxTokens: number;
        maxRetries: number;
        jsonMode: boolean;
        reasoningEffort?: "low" | "medium" | "high" | undefined;
        provider?: string[] | undefined;
        backupModel?: {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            reasoningEffort?: "low" | "medium" | "high" | undefined;
            maxRetries?: number | undefined;
        } | undefined;
    };
    tools: {
        name: "get-bubble-details-tool" | "list-bubbles-tool" | "sql-query-tool" | "chart-js-tool" | "web-search-tool" | "web-scrape-tool" | "web-crawl-tool" | "web-extract-tool" | "research-agent-tool" | "reddit-scrape-tool" | "bubbleflow-validation-tool" | "code-edit-tool" | "instagram-tool";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
    }[];
    maxIterations: number;
    streaming: boolean;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    conversationHistory?: {
        role: "user" | "assistant" | "tool";
        content: string;
        toolCallId?: string | undefined;
        name?: string | undefined;
    }[] | undefined;
    customTools?: {
        description: string;
        name: string;
        schema: z.ZodTypeAny | Record<string, unknown>;
        func: (args_0: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
    }[] | undefined;
    expectedOutputSchema?: string | z.ZodTypeAny | undefined;
}, {
    message: string;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    images?: ({
        data: string;
        type?: "base64" | undefined;
        description?: string | undefined;
        mimeType?: string | undefined;
    } | {
        type: "url";
        url: string;
        description?: string | undefined;
    })[] | undefined;
    conversationHistory?: {
        role: "user" | "assistant" | "tool";
        content: string;
        toolCallId?: string | undefined;
        name?: string | undefined;
    }[] | undefined;
    systemPrompt?: string | undefined;
    model?: {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature?: number | undefined;
        maxTokens?: number | undefined;
        reasoningEffort?: "low" | "medium" | "high" | undefined;
        maxRetries?: number | undefined;
        provider?: string[] | undefined;
        jsonMode?: boolean | undefined;
        backupModel?: {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            reasoningEffort?: "low" | "medium" | "high" | undefined;
            maxRetries?: number | undefined;
        } | undefined;
    } | undefined;
    tools?: {
        name: "get-bubble-details-tool" | "list-bubbles-tool" | "sql-query-tool" | "chart-js-tool" | "web-search-tool" | "web-scrape-tool" | "web-crawl-tool" | "web-extract-tool" | "research-agent-tool" | "reddit-scrape-tool" | "bubbleflow-validation-tool" | "code-edit-tool" | "instagram-tool";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
    }[] | undefined;
    customTools?: {
        description: string;
        name: string;
        schema: z.ZodTypeAny | Record<string, unknown>;
        func: (args_0: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
    }[] | undefined;
    maxIterations?: number | undefined;
    streaming?: boolean | undefined;
    expectedOutputSchema?: string | z.ZodTypeAny | undefined;
}>;
declare const AIAgentResultSchema: z.ZodObject<{
    response: z.ZodString;
    toolCalls: z.ZodArray<z.ZodObject<{
        tool: z.ZodString;
        input: z.ZodUnknown;
        output: z.ZodUnknown;
    }, "strip", z.ZodTypeAny, {
        tool: string;
        input?: unknown;
        output?: unknown;
    }, {
        tool: string;
        input?: unknown;
        output?: unknown;
    }>, "many">;
    iterations: z.ZodNumber;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    response: string;
    toolCalls: {
        tool: string;
        input?: unknown;
        output?: unknown;
    }[];
    iterations: number;
}, {
    error: string;
    success: boolean;
    response: string;
    toolCalls: {
        tool: string;
        input?: unknown;
        output?: unknown;
    }[];
    iterations: number;
}>;
type AIAgentParams = z.input<typeof AIAgentParamsSchema> & {
    beforeToolCall?: ToolHookBefore;
    afterToolCall?: ToolHookAfter;
    afterLLMCall?: AfterLLMCallHook;
    streamingCallback?: StreamingCallback;
};
type AIAgentParamsParsed = z.output<typeof AIAgentParamsSchema> & {
    beforeToolCall?: ToolHookBefore;
    afterToolCall?: ToolHookAfter;
    afterLLMCall?: AfterLLMCallHook;
    streamingCallback?: StreamingCallback;
};
type AIAgentResult = z.output<typeof AIAgentResultSchema>;
export declare class AIAgentBubble extends ServiceBubble<AIAgentParamsParsed, AIAgentResult> {
    static readonly type: "service";
    static readonly service = "ai-agent";
    static readonly authType: "apikey";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        message: z.ZodString;
        images: z.ZodDefault<z.ZodArray<z.ZodDiscriminatedUnion<"type", [z.ZodObject<{
            type: z.ZodDefault<z.ZodLiteral<"base64">>;
            data: z.ZodString;
            mimeType: z.ZodDefault<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: "base64";
            data: string;
            mimeType: string;
            description?: string | undefined;
        }, {
            data: string;
            type?: "base64" | undefined;
            description?: string | undefined;
            mimeType?: string | undefined;
        }>, z.ZodObject<{
            type: z.ZodLiteral<"url">;
            url: z.ZodEffects<z.ZodString, string, string>;
            description: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: "url";
            url: string;
            description?: string | undefined;
        }, {
            type: "url";
            url: string;
            description?: string | undefined;
        }>]>, "many">>;
        conversationHistory: z.ZodOptional<z.ZodArray<z.ZodObject<{
            role: z.ZodEnum<["user", "assistant", "tool"]>;
            content: z.ZodString;
            toolCallId: z.ZodOptional<z.ZodString>;
            name: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            role: "user" | "assistant" | "tool";
            content: string;
            toolCallId?: string | undefined;
            name?: string | undefined;
        }, {
            role: "user" | "assistant" | "tool";
            content: string;
            toolCallId?: string | undefined;
            name?: string | undefined;
        }>, "many">>;
        systemPrompt: z.ZodDefault<z.ZodString>;
        name: z.ZodOptional<z.ZodDefault<z.ZodString>>;
        model: z.ZodDefault<z.ZodObject<{
            model: z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>;
            temperature: z.ZodDefault<z.ZodNumber>;
            maxTokens: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
            reasoningEffort: z.ZodOptional<z.ZodEnum<["low", "medium", "high"]>>;
            maxRetries: z.ZodDefault<z.ZodNumber>;
            provider: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            jsonMode: z.ZodDefault<z.ZodBoolean>;
            backupModel: z.ZodOptional<z.ZodDefault<z.ZodObject<{
                model: z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>;
                temperature: z.ZodOptional<z.ZodNumber>;
                maxTokens: z.ZodOptional<z.ZodNumber>;
                reasoningEffort: z.ZodOptional<z.ZodEnum<["low", "medium", "high"]>>;
                maxRetries: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
                temperature?: number | undefined;
                maxTokens?: number | undefined;
                reasoningEffort?: "low" | "medium" | "high" | undefined;
                maxRetries?: number | undefined;
            }, {
                model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
                temperature?: number | undefined;
                maxTokens?: number | undefined;
                reasoningEffort?: "low" | "medium" | "high" | undefined;
                maxRetries?: number | undefined;
            }>>>;
        }, "strip", z.ZodTypeAny, {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature: number;
            maxTokens: number;
            maxRetries: number;
            jsonMode: boolean;
            reasoningEffort?: "low" | "medium" | "high" | undefined;
            provider?: string[] | undefined;
            backupModel?: {
                model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
                temperature?: number | undefined;
                maxTokens?: number | undefined;
                reasoningEffort?: "low" | "medium" | "high" | undefined;
                maxRetries?: number | undefined;
            } | undefined;
        }, {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            reasoningEffort?: "low" | "medium" | "high" | undefined;
            maxRetries?: number | undefined;
            provider?: string[] | undefined;
            jsonMode?: boolean | undefined;
            backupModel?: {
                model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
                temperature?: number | undefined;
                maxTokens?: number | undefined;
                reasoningEffort?: "low" | "medium" | "high" | undefined;
                maxRetries?: number | undefined;
            } | undefined;
        }>>;
        tools: z.ZodDefault<z.ZodArray<z.ZodObject<{
            name: z.ZodEnum<["web-search-tool", "web-scrape-tool", "web-crawl-tool", "web-extract-tool", "research-agent-tool", "reddit-scrape-tool", "instagram-tool", "list-bubbles-tool", "get-bubble-details-tool", "bubbleflow-validation-tool", "code-edit-tool", "chart-js-tool", "sql-query-tool"]>;
            credentials: z.ZodOptional<z.ZodDefault<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>>;
            config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            name: "get-bubble-details-tool" | "list-bubbles-tool" | "sql-query-tool" | "chart-js-tool" | "web-search-tool" | "web-scrape-tool" | "web-crawl-tool" | "web-extract-tool" | "research-agent-tool" | "reddit-scrape-tool" | "bubbleflow-validation-tool" | "code-edit-tool" | "instagram-tool";
            credentials?: Partial<Record<CredentialType, string>> | undefined;
            config?: Record<string, unknown> | undefined;
        }, {
            name: "get-bubble-details-tool" | "list-bubbles-tool" | "sql-query-tool" | "chart-js-tool" | "web-search-tool" | "web-scrape-tool" | "web-crawl-tool" | "web-extract-tool" | "research-agent-tool" | "reddit-scrape-tool" | "bubbleflow-validation-tool" | "code-edit-tool" | "instagram-tool";
            credentials?: Partial<Record<CredentialType, string>> | undefined;
            config?: Record<string, unknown> | undefined;
        }>, "many">>;
        customTools: z.ZodOptional<z.ZodDefault<z.ZodArray<z.ZodEffects<z.ZodObject<{
            name: z.ZodString;
            description: z.ZodString;
            schema: z.ZodUnion<[z.ZodRecord<z.ZodString, z.ZodUnknown>, z.ZodType<z.ZodTypeAny, z.ZodTypeDef, z.ZodTypeAny>]>;
            func: z.ZodFunction<z.ZodTuple<[z.ZodRecord<z.ZodString, z.ZodUnknown>], z.ZodUnknown>, z.ZodPromise<z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            description: string;
            name: string;
            schema: z.ZodTypeAny | Record<string, unknown>;
            func: (args_0: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
        }, {
            description: string;
            name: string;
            schema: z.ZodTypeAny | Record<string, unknown>;
            func: (args_0: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
        }>, {
            description: string;
            name: string;
            schema: z.ZodTypeAny | Record<string, unknown>;
            func: (args_0: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
        }, {
            description: string;
            name: string;
            schema: z.ZodTypeAny | Record<string, unknown>;
            func: (args_0: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
        }>, "many">>>;
        maxIterations: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
        streaming: z.ZodDefault<z.ZodBoolean>;
        expectedOutputSchema: z.ZodOptional<z.ZodUnion<[z.ZodType<z.ZodTypeAny, z.ZodTypeDef, z.ZodTypeAny>, z.ZodString]>>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        images: ({
            type: "base64";
            data: string;
            mimeType: string;
            description?: string | undefined;
        } | {
            type: "url";
            url: string;
            description?: string | undefined;
        })[];
        systemPrompt: string;
        model: {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature: number;
            maxTokens: number;
            maxRetries: number;
            jsonMode: boolean;
            reasoningEffort?: "low" | "medium" | "high" | undefined;
            provider?: string[] | undefined;
            backupModel?: {
                model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
                temperature?: number | undefined;
                maxTokens?: number | undefined;
                reasoningEffort?: "low" | "medium" | "high" | undefined;
                maxRetries?: number | undefined;
            } | undefined;
        };
        tools: {
            name: "get-bubble-details-tool" | "list-bubbles-tool" | "sql-query-tool" | "chart-js-tool" | "web-search-tool" | "web-scrape-tool" | "web-crawl-tool" | "web-extract-tool" | "research-agent-tool" | "reddit-scrape-tool" | "bubbleflow-validation-tool" | "code-edit-tool" | "instagram-tool";
            credentials?: Partial<Record<CredentialType, string>> | undefined;
            config?: Record<string, unknown> | undefined;
        }[];
        maxIterations: number;
        streaming: boolean;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        conversationHistory?: {
            role: "user" | "assistant" | "tool";
            content: string;
            toolCallId?: string | undefined;
            name?: string | undefined;
        }[] | undefined;
        customTools?: {
            description: string;
            name: string;
            schema: z.ZodTypeAny | Record<string, unknown>;
            func: (args_0: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
        }[] | undefined;
        expectedOutputSchema?: string | z.ZodTypeAny | undefined;
    }, {
        message: string;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        images?: ({
            data: string;
            type?: "base64" | undefined;
            description?: string | undefined;
            mimeType?: string | undefined;
        } | {
            type: "url";
            url: string;
            description?: string | undefined;
        })[] | undefined;
        conversationHistory?: {
            role: "user" | "assistant" | "tool";
            content: string;
            toolCallId?: string | undefined;
            name?: string | undefined;
        }[] | undefined;
        systemPrompt?: string | undefined;
        model?: {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            reasoningEffort?: "low" | "medium" | "high" | undefined;
            maxRetries?: number | undefined;
            provider?: string[] | undefined;
            jsonMode?: boolean | undefined;
            backupModel?: {
                model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
                temperature?: number | undefined;
                maxTokens?: number | undefined;
                reasoningEffort?: "low" | "medium" | "high" | undefined;
                maxRetries?: number | undefined;
            } | undefined;
        } | undefined;
        tools?: {
            name: "get-bubble-details-tool" | "list-bubbles-tool" | "sql-query-tool" | "chart-js-tool" | "web-search-tool" | "web-scrape-tool" | "web-crawl-tool" | "web-extract-tool" | "research-agent-tool" | "reddit-scrape-tool" | "bubbleflow-validation-tool" | "code-edit-tool" | "instagram-tool";
            credentials?: Partial<Record<CredentialType, string>> | undefined;
            config?: Record<string, unknown> | undefined;
        }[] | undefined;
        customTools?: {
            description: string;
            name: string;
            schema: z.ZodTypeAny | Record<string, unknown>;
            func: (args_0: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
        }[] | undefined;
        maxIterations?: number | undefined;
        streaming?: boolean | undefined;
        expectedOutputSchema?: string | z.ZodTypeAny | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        response: z.ZodString;
        toolCalls: z.ZodArray<z.ZodObject<{
            tool: z.ZodString;
            input: z.ZodUnknown;
            output: z.ZodUnknown;
        }, "strip", z.ZodTypeAny, {
            tool: string;
            input?: unknown;
            output?: unknown;
        }, {
            tool: string;
            input?: unknown;
            output?: unknown;
        }>, "many">;
        iterations: z.ZodNumber;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        response: string;
        toolCalls: {
            tool: string;
            input?: unknown;
            output?: unknown;
        }[];
        iterations: number;
    }, {
        error: string;
        success: boolean;
        response: string;
        toolCalls: {
            tool: string;
            input?: unknown;
            output?: unknown;
        }[];
        iterations: number;
    }>;
    static readonly shortDescription = "AI agent with LangGraph for tool-enabled conversations, multimodal support, and JSON mode";
    static readonly longDescription = "\n    An AI agent powered by LangGraph that can use any tool bubble to answer questions.\n    Use cases:\n    - Add tools to enhance the AI agent's capabilities (web-search-tool, web-scrape-tool)\n    - Multi-step reasoning with tool assistance\n    - Tool-augmented conversations with any registered tool\n    - JSON mode for structured output (strips markdown formatting)\n  ";
    static readonly alias = "agent";
    private static conversationCache;
    private static toolResultCache;
    private static readonly CLEANUP_INTERVAL;
    private static lastCleanup;
    private factory;
    private beforeToolCallHook;
    private afterToolCallHook;
    private afterLLMCallHook;
    private streamingCallback;
    private shouldStopAfterTools;
    private shouldContinueToAgent;
    private logger;
    constructor(params?: AIAgentParams, context?: BubbleContext, instanceId?: string);
    /**
     * Perform periodic cleanup of cache entries
     */
    private performCacheCleanup;
    testCredential(): Promise<boolean>;
    /**
     * Build effective model config from primary and optional backup settings
     */
    private buildModelConfig;
    /**
     * Core execution logic for running the agent with a given model config
     */
    private executeWithModel;
    /**
     * Modify params before execution - centralizes all param transformations
     */
    private beforeAction;
    protected performAction(context?: BubbleContext): Promise<AIAgentResult>;
    protected getCredentialType(): CredentialType;
    /**
     * Get credential type for a specific model string
     */
    private getCredentialTypeForModel;
    protected chooseCredential(): string | undefined;
    private initializeModel;
    private initializeTools;
    /**
     * Custom tool execution node that supports hooks
     */
    private executeToolsWithHooks;
    private createAgentGraph;
    private executeAgent;
}
export {};
//# sourceMappingURL=ai-agent.d.ts.map