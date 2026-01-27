import { z } from 'zod';
/**
 * Conversation message schema for milk tea multi-turn conversations
 */
/**
 * Request schema for Milk Tea agent
 * Milk Tea helps users configure bubble parameters through conversation
 */
export declare const MilkTeaRequestSchema: z.ZodObject<{
    userRequest: z.ZodString;
    bubbleName: z.ZodString;
    bubbleSchema: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    currentCode: z.ZodOptional<z.ZodString>;
    availableCredentials: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    userName: z.ZodString;
    insertLocation: z.ZodOptional<z.ZodString>;
    conversationHistory: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodObject<{
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
    }>, "many">>>;
    model: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
}, "strip", z.ZodTypeAny, {
    bubbleName: string;
    model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
    userRequest: string;
    bubbleSchema: Record<string, unknown>;
    availableCredentials: string[];
    userName: string;
    conversationHistory: {
        role: "user" | "assistant" | "tool";
        content: string;
        toolCallId?: string | undefined;
        name?: string | undefined;
    }[];
    currentCode?: string | undefined;
    insertLocation?: string | undefined;
}, {
    bubbleName: string;
    userRequest: string;
    bubbleSchema: Record<string, unknown>;
    userName: string;
    model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
    currentCode?: string | undefined;
    availableCredentials?: string[] | undefined;
    insertLocation?: string | undefined;
    conversationHistory?: {
        role: "user" | "assistant" | "tool";
        content: string;
        toolCallId?: string | undefined;
        name?: string | undefined;
    }[] | undefined;
}>;
/**
 * Response schema for Milk Tea agent
 */
export declare const MilkTeaResponseSchema: z.ZodObject<{
    type: z.ZodEnum<["code", "question", "reject"]>;
    message: z.ZodString;
    snippet: z.ZodOptional<z.ZodString>;
    success: z.ZodBoolean;
    error: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    message: string;
    type: "code" | "question" | "reject";
    success: boolean;
    error?: string | undefined;
    snippet?: string | undefined;
}, {
    message: string;
    type: "code" | "question" | "reject";
    success: boolean;
    error?: string | undefined;
    snippet?: string | undefined;
}>;
/**
 * Internal agent response format (JSON mode output from AI)
 */
export declare const MilkTeaAgentOutputSchema: z.ZodObject<{
    type: z.ZodEnum<["code", "question", "reject"]>;
    message: z.ZodString;
    snippet: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    message: string;
    type: "code" | "question" | "reject";
    snippet?: string | undefined;
}, {
    message: string;
    type: "code" | "question" | "reject";
    snippet?: string | undefined;
}>;
export type MilkTeaRequest = z.infer<typeof MilkTeaRequestSchema>;
export type MilkTeaResponse = z.infer<typeof MilkTeaResponseSchema>;
export type MilkTeaAgentOutput = z.infer<typeof MilkTeaAgentOutputSchema>;
//# sourceMappingURL=milk-tea.d.ts.map