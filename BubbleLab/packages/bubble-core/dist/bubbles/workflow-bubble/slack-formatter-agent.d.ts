import { z } from 'zod';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import type { BubbleName } from '@bubblelab/shared-schemas';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
declare const SlackFormatterAgentParamsSchema: z.ZodObject<{
    message: z.ZodString;
    verbosity: z.ZodDefault<z.ZodEnum<["1", "2", "3", "4", "5"]>>;
    technicality: z.ZodDefault<z.ZodEnum<["1", "2", "3", "4", "5"]>>;
    includeBlockKit: z.ZodDefault<z.ZodBoolean>;
    includeQuery: z.ZodDefault<z.ZodBoolean>;
    includeExplanation: z.ZodDefault<z.ZodBoolean>;
    model: z.ZodDefault<z.ZodObject<{
        model: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
        temperature: z.ZodDefault<z.ZodNumber>;
        maxTokens: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    }, "strip", z.ZodTypeAny, {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature: number;
        maxTokens: number;
    }, {
        model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
    }>>;
    tools: z.ZodDefault<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        credentials: z.ZodOptional<z.ZodDefault<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>>;
        config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
    }, {
        name: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
    }>, "many">>;
    maxIterations: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    additionalContext: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    message: string;
    model: {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature: number;
        maxTokens: number;
    };
    tools: {
        name: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
    }[];
    maxIterations: number;
    verbosity: "1" | "2" | "3" | "4" | "5";
    technicality: "1" | "2" | "3" | "4" | "5";
    includeQuery: boolean;
    includeExplanation: boolean;
    includeBlockKit: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    additionalContext?: string | undefined;
}, {
    message: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    model?: {
        model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
    } | undefined;
    tools?: {
        name: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        config?: Record<string, unknown> | undefined;
    }[] | undefined;
    maxIterations?: number | undefined;
    verbosity?: "1" | "2" | "3" | "4" | "5" | undefined;
    technicality?: "1" | "2" | "3" | "4" | "5" | undefined;
    includeQuery?: boolean | undefined;
    includeExplanation?: boolean | undefined;
    additionalContext?: string | undefined;
    includeBlockKit?: boolean | undefined;
}>;
declare const SlackFormatterAgentResultSchema: z.ZodObject<{
    response: z.ZodString;
    blocks: z.ZodOptional<z.ZodArray<z.ZodObject<{
        type: z.ZodEnum<["section", "header", "divider", "context", "actions", "input", "file", "image"]>;
        text: z.ZodOptional<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>>;
        block_id: z.ZodOptional<z.ZodString>;
        accessory: z.ZodOptional<z.ZodUnknown>;
        fields: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>, "many">>;
        element: z.ZodOptional<z.ZodUnknown>;
        label: z.ZodOptional<z.ZodUnknown>;
        hint: z.ZodOptional<z.ZodUnknown>;
        optional: z.ZodOptional<z.ZodBoolean>;
        alt_text: z.ZodOptional<z.ZodString>;
        image_url: z.ZodOptional<z.ZodString>;
        title: z.ZodOptional<z.ZodObject<{
            type: z.ZodEnum<["plain_text"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text";
            text: string;
            emoji?: boolean | undefined;
        }, {
            type: "plain_text";
            text: string;
            emoji?: boolean | undefined;
        }>>;
        elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        type: "input" | "image" | "file" | "section" | "header" | "divider" | "context" | "actions";
        title?: {
            type: "plain_text";
            text: string;
            emoji?: boolean | undefined;
        } | undefined;
        fields?: {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }[] | undefined;
        text?: {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        } | undefined;
        image_url?: string | undefined;
        alt_text?: string | undefined;
        elements?: {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }[] | undefined;
        label?: unknown;
        optional?: boolean | undefined;
        element?: unknown;
        block_id?: string | undefined;
        accessory?: unknown;
        hint?: unknown;
    }, {
        type: "input" | "image" | "file" | "section" | "header" | "divider" | "context" | "actions";
        title?: {
            type: "plain_text";
            text: string;
            emoji?: boolean | undefined;
        } | undefined;
        fields?: {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }[] | undefined;
        text?: {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        } | undefined;
        image_url?: string | undefined;
        alt_text?: string | undefined;
        elements?: {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }[] | undefined;
        label?: unknown;
        optional?: boolean | undefined;
        element?: unknown;
        block_id?: string | undefined;
        accessory?: unknown;
        hint?: unknown;
    }>, "many">>;
    metadata: z.ZodObject<{
        verbosityLevel: z.ZodString;
        technicalityLevel: z.ZodString;
        wordCount: z.ZodNumber;
        blockCount: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        wordCount: number;
        verbosityLevel: string;
        technicalityLevel: string;
        blockCount?: number | undefined;
    }, {
        wordCount: number;
        verbosityLevel: string;
        technicalityLevel: string;
        blockCount?: number | undefined;
    }>;
    toolCalls: z.ZodOptional<z.ZodArray<z.ZodObject<{
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
    }>, "many">>;
    iterations: z.ZodNumber;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    response: string;
    iterations: number;
    metadata: {
        wordCount: number;
        verbosityLevel: string;
        technicalityLevel: string;
        blockCount?: number | undefined;
    };
    toolCalls?: {
        tool: string;
        input?: unknown;
        output?: unknown;
    }[] | undefined;
    blocks?: {
        type: "input" | "image" | "file" | "section" | "header" | "divider" | "context" | "actions";
        title?: {
            type: "plain_text";
            text: string;
            emoji?: boolean | undefined;
        } | undefined;
        fields?: {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }[] | undefined;
        text?: {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        } | undefined;
        image_url?: string | undefined;
        alt_text?: string | undefined;
        elements?: {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }[] | undefined;
        label?: unknown;
        optional?: boolean | undefined;
        element?: unknown;
        block_id?: string | undefined;
        accessory?: unknown;
        hint?: unknown;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    response: string;
    iterations: number;
    metadata: {
        wordCount: number;
        verbosityLevel: string;
        technicalityLevel: string;
        blockCount?: number | undefined;
    };
    toolCalls?: {
        tool: string;
        input?: unknown;
        output?: unknown;
    }[] | undefined;
    blocks?: {
        type: "input" | "image" | "file" | "section" | "header" | "divider" | "context" | "actions";
        title?: {
            type: "plain_text";
            text: string;
            emoji?: boolean | undefined;
        } | undefined;
        fields?: {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }[] | undefined;
        text?: {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        } | undefined;
        image_url?: string | undefined;
        alt_text?: string | undefined;
        elements?: {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }[] | undefined;
        label?: unknown;
        optional?: boolean | undefined;
        element?: unknown;
        block_id?: string | undefined;
        accessory?: unknown;
        hint?: unknown;
    }[] | undefined;
}>;
type SlackFormatterAgentParams = z.input<typeof SlackFormatterAgentParamsSchema>;
type SlackFormatterAgentParamsParsed = z.output<typeof SlackFormatterAgentParamsSchema>;
type SlackFormatterAgentResult = z.output<typeof SlackFormatterAgentResultSchema>;
export declare class SlackFormatterAgentBubble extends WorkflowBubble<SlackFormatterAgentParamsParsed, SlackFormatterAgentResult> {
    static readonly type: "service";
    static readonly service = "slack-formatter-agent";
    static readonly authType: "apikey";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        message: z.ZodString;
        verbosity: z.ZodDefault<z.ZodEnum<["1", "2", "3", "4", "5"]>>;
        technicality: z.ZodDefault<z.ZodEnum<["1", "2", "3", "4", "5"]>>;
        includeBlockKit: z.ZodDefault<z.ZodBoolean>;
        includeQuery: z.ZodDefault<z.ZodBoolean>;
        includeExplanation: z.ZodDefault<z.ZodBoolean>;
        model: z.ZodDefault<z.ZodObject<{
            model: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
            temperature: z.ZodDefault<z.ZodNumber>;
            maxTokens: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        }, "strip", z.ZodTypeAny, {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature: number;
            maxTokens: number;
        }, {
            model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
            temperature?: number | undefined;
            maxTokens?: number | undefined;
        }>>;
        tools: z.ZodDefault<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            credentials: z.ZodOptional<z.ZodDefault<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>>;
            config: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            credentials?: Partial<Record<CredentialType, string>> | undefined;
            config?: Record<string, unknown> | undefined;
        }, {
            name: string;
            credentials?: Partial<Record<CredentialType, string>> | undefined;
            config?: Record<string, unknown> | undefined;
        }>, "many">>;
        maxIterations: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
        additionalContext: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        model: {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature: number;
            maxTokens: number;
        };
        tools: {
            name: string;
            credentials?: Partial<Record<CredentialType, string>> | undefined;
            config?: Record<string, unknown> | undefined;
        }[];
        maxIterations: number;
        verbosity: "1" | "2" | "3" | "4" | "5";
        technicality: "1" | "2" | "3" | "4" | "5";
        includeQuery: boolean;
        includeExplanation: boolean;
        includeBlockKit: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        additionalContext?: string | undefined;
    }, {
        message: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        model?: {
            model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
            temperature?: number | undefined;
            maxTokens?: number | undefined;
        } | undefined;
        tools?: {
            name: string;
            credentials?: Partial<Record<CredentialType, string>> | undefined;
            config?: Record<string, unknown> | undefined;
        }[] | undefined;
        maxIterations?: number | undefined;
        verbosity?: "1" | "2" | "3" | "4" | "5" | undefined;
        technicality?: "1" | "2" | "3" | "4" | "5" | undefined;
        includeQuery?: boolean | undefined;
        includeExplanation?: boolean | undefined;
        additionalContext?: string | undefined;
        includeBlockKit?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        response: z.ZodString;
        blocks: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["section", "header", "divider", "context", "actions", "input", "file", "image"]>;
            text: z.ZodOptional<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>>;
            block_id: z.ZodOptional<z.ZodString>;
            accessory: z.ZodOptional<z.ZodUnknown>;
            fields: z.ZodOptional<z.ZodArray<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>, "many">>;
            element: z.ZodOptional<z.ZodUnknown>;
            label: z.ZodOptional<z.ZodUnknown>;
            hint: z.ZodOptional<z.ZodUnknown>;
            optional: z.ZodOptional<z.ZodBoolean>;
            alt_text: z.ZodOptional<z.ZodString>;
            image_url: z.ZodOptional<z.ZodString>;
            title: z.ZodOptional<z.ZodObject<{
                type: z.ZodEnum<["plain_text"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text";
                text: string;
                emoji?: boolean | undefined;
            }, {
                type: "plain_text";
                text: string;
                emoji?: boolean | undefined;
            }>>;
            elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            type: "input" | "image" | "file" | "section" | "header" | "divider" | "context" | "actions";
            title?: {
                type: "plain_text";
                text: string;
                emoji?: boolean | undefined;
            } | undefined;
            fields?: {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }[] | undefined;
            text?: {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            } | undefined;
            image_url?: string | undefined;
            alt_text?: string | undefined;
            elements?: {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }[] | undefined;
            label?: unknown;
            optional?: boolean | undefined;
            element?: unknown;
            block_id?: string | undefined;
            accessory?: unknown;
            hint?: unknown;
        }, {
            type: "input" | "image" | "file" | "section" | "header" | "divider" | "context" | "actions";
            title?: {
                type: "plain_text";
                text: string;
                emoji?: boolean | undefined;
            } | undefined;
            fields?: {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }[] | undefined;
            text?: {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            } | undefined;
            image_url?: string | undefined;
            alt_text?: string | undefined;
            elements?: {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }[] | undefined;
            label?: unknown;
            optional?: boolean | undefined;
            element?: unknown;
            block_id?: string | undefined;
            accessory?: unknown;
            hint?: unknown;
        }>, "many">>;
        metadata: z.ZodObject<{
            verbosityLevel: z.ZodString;
            technicalityLevel: z.ZodString;
            wordCount: z.ZodNumber;
            blockCount: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            wordCount: number;
            verbosityLevel: string;
            technicalityLevel: string;
            blockCount?: number | undefined;
        }, {
            wordCount: number;
            verbosityLevel: string;
            technicalityLevel: string;
            blockCount?: number | undefined;
        }>;
        toolCalls: z.ZodOptional<z.ZodArray<z.ZodObject<{
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
        }>, "many">>;
        iterations: z.ZodNumber;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        response: string;
        iterations: number;
        metadata: {
            wordCount: number;
            verbosityLevel: string;
            technicalityLevel: string;
            blockCount?: number | undefined;
        };
        toolCalls?: {
            tool: string;
            input?: unknown;
            output?: unknown;
        }[] | undefined;
        blocks?: {
            type: "input" | "image" | "file" | "section" | "header" | "divider" | "context" | "actions";
            title?: {
                type: "plain_text";
                text: string;
                emoji?: boolean | undefined;
            } | undefined;
            fields?: {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }[] | undefined;
            text?: {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            } | undefined;
            image_url?: string | undefined;
            alt_text?: string | undefined;
            elements?: {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }[] | undefined;
            label?: unknown;
            optional?: boolean | undefined;
            element?: unknown;
            block_id?: string | undefined;
            accessory?: unknown;
            hint?: unknown;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        response: string;
        iterations: number;
        metadata: {
            wordCount: number;
            verbosityLevel: string;
            technicalityLevel: string;
            blockCount?: number | undefined;
        };
        toolCalls?: {
            tool: string;
            input?: unknown;
            output?: unknown;
        }[] | undefined;
        blocks?: {
            type: "input" | "image" | "file" | "section" | "header" | "divider" | "context" | "actions";
            title?: {
                type: "plain_text";
                text: string;
                emoji?: boolean | undefined;
            } | undefined;
            fields?: {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }[] | undefined;
            text?: {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            } | undefined;
            image_url?: string | undefined;
            alt_text?: string | undefined;
            elements?: {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }[] | undefined;
            label?: unknown;
            optional?: boolean | undefined;
            element?: unknown;
            block_id?: string | undefined;
            accessory?: unknown;
            hint?: unknown;
        }[] | undefined;
    }>;
    static readonly shortDescription = "AI agent for creating well-formatted Slack messages with adjustable verbosity and technicality";
    static readonly longDescription = "\n    An AI agent that specializes in generating properly formatted Slack messages with:\n    - Adjustable verbosity levels (1-5): from concise bullet points to comprehensive explanations\n    - Adjustable technicality levels (1-5): from plain English to expert terminology\n    - Native Slack markdown formatting (bold, italic, code blocks, lists)\n    - Optional Slack Block Kit JSON for rich interactive messages\n    - Tool integration for dynamic content generation\n    \n    Perfect for:\n    - Creating consistent Slack notifications with appropriate detail level\n    - Adapting any content for different audiences\n    - Generating interactive Slack messages with Block Kit\n    - Formatting summaries, reports, and updates for Slack channels\n    - Building engaging team communications with proper structure\n    - Converting any information into Slack-friendly format\n  ";
    static readonly alias = "slack-format";
    private factory;
    constructor(params?: SlackFormatterAgentParams, context?: BubbleContext);
    testCredential(): Promise<boolean>;
    protected performAction(context?: BubbleContext): Promise<SlackFormatterAgentResult>;
    protected chooseCredential(): string | undefined;
    private createSlackFormatterPrompt;
    private initializeModel;
    private initializeTools;
    private createAgentGraph;
    private executeAgent;
    private extractSlackBlocks;
    private validateAndFixSlackBlocks;
}
export {};
//# sourceMappingURL=slack-formatter-agent.d.ts.map