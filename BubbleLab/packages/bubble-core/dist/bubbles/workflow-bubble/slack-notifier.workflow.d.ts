import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const SlackNotifierParamsSchema: z.ZodObject<{
    /**
     * The raw data or content to be formatted for Slack notification
     */
    contentToFormat: z.ZodString;
    /**
     * The original user query or context that generated this content
     */
    originalUserQuery: z.ZodOptional<z.ZodString>;
    /**
     * Target Slack channel name (without #) or channel ID
     */
    targetChannel: z.ZodString;
    /**
     * Optional custom message title/header for the notification
     */
    messageTitle: z.ZodOptional<z.ZodString>;
    /**
     * Tone and style for the AI formatting (professional, casual, technical, etc.)
     */
    messageStyle: z.ZodDefault<z.ZodEnum<["professional", "casual", "technical", "concise", "detailed"]>>;
    /**
     * Whether to include emojis and rich formatting in the message
     */
    includeFormatting: z.ZodDefault<z.ZodBoolean>;
    /**
     * Maximum message length (Slack has limits)
     */
    maxMessageLength: z.ZodDefault<z.ZodNumber>;
    /**
     * AI model configuration for content formatting
     */
    aiModel: z.ZodOptional<z.ZodObject<{
        model: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
        temperature: z.ZodDefault<z.ZodNumber>;
        maxTokens: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature: number;
        maxTokens: number;
    }, {
        model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
    }>>;
    /**
     * Injected credentials from the system
     */
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    contentToFormat: string;
    targetChannel: string;
    messageStyle: "professional" | "casual" | "technical" | "concise" | "detailed";
    includeFormatting: boolean;
    maxMessageLength: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    originalUserQuery?: string | undefined;
    messageTitle?: string | undefined;
    aiModel?: {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature: number;
        maxTokens: number;
    } | undefined;
}, {
    contentToFormat: string;
    targetChannel: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    originalUserQuery?: string | undefined;
    messageTitle?: string | undefined;
    messageStyle?: "professional" | "casual" | "technical" | "concise" | "detailed" | undefined;
    includeFormatting?: boolean | undefined;
    maxMessageLength?: number | undefined;
    aiModel?: {
        model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
    } | undefined;
}>;
type SlackNotifierParamsInput = z.input<typeof SlackNotifierParamsSchema>;
type SlackNotifierParams = z.output<typeof SlackNotifierParamsSchema>;
declare const SlackNotifierResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    error: z.ZodString;
    /**
     * Information about the sent Slack message
     */
    messageInfo: z.ZodOptional<z.ZodObject<{
        /**
         * Slack message timestamp (unique identifier)
         */
        messageTimestamp: z.ZodOptional<z.ZodString>;
        /**
         * Channel ID where message was sent
         */
        channelId: z.ZodOptional<z.ZodString>;
        /**
         * Channel name where message was sent
         */
        channelName: z.ZodOptional<z.ZodString>;
        /**
         * The formatted message that was sent
         */
        formattedMessage: z.ZodOptional<z.ZodString>;
        /**
         * Message length in characters
         */
        messageLength: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        channelName?: string | undefined;
        messageTimestamp?: string | undefined;
        channelId?: string | undefined;
        formattedMessage?: string | undefined;
        messageLength?: number | undefined;
    }, {
        channelName?: string | undefined;
        messageTimestamp?: string | undefined;
        channelId?: string | undefined;
        formattedMessage?: string | undefined;
        messageLength?: number | undefined;
    }>>;
    /**
     * AI formatting process information
     */
    formattingInfo: z.ZodOptional<z.ZodObject<{
        /**
         * AI model used for formatting
         */
        modelUsed: z.ZodOptional<z.ZodString>;
        /**
         * Whether content was truncated due to length limits
         */
        wasTruncated: z.ZodDefault<z.ZodBoolean>;
        /**
         * Original content length before formatting
         */
        originalLength: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        wasTruncated: boolean;
        modelUsed?: string | undefined;
        originalLength?: number | undefined;
    }, {
        modelUsed?: string | undefined;
        wasTruncated?: boolean | undefined;
        originalLength?: number | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    messageInfo?: {
        channelName?: string | undefined;
        messageTimestamp?: string | undefined;
        channelId?: string | undefined;
        formattedMessage?: string | undefined;
        messageLength?: number | undefined;
    } | undefined;
    formattingInfo?: {
        wasTruncated: boolean;
        modelUsed?: string | undefined;
        originalLength?: number | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    messageInfo?: {
        channelName?: string | undefined;
        messageTimestamp?: string | undefined;
        channelId?: string | undefined;
        formattedMessage?: string | undefined;
        messageLength?: number | undefined;
    } | undefined;
    formattingInfo?: {
        modelUsed?: string | undefined;
        wasTruncated?: boolean | undefined;
        originalLength?: number | undefined;
    } | undefined;
}>;
type SlackNotifierResult = z.infer<typeof SlackNotifierResultSchema>;
/**
 * SlackNotifierWorkflowBubble - Your personal data analyst for Slack communications
 *
 * This workflow bubble acts like a seasoned data analyst who transforms raw information
 * into compelling, actionable Slack messages that your team will actually want to read:
 *
 * 1. **Analyzes content** - Spots patterns, key insights, and business implications
 * 2. **Tells the story** - Converts dry data into engaging narratives with context
 * 3. **Makes it actionable** - Provides recommendations and next steps
 * 4. **Delivers naturally** - Uses conversational, human-like communication style
 * 5. **Handles logistics** - Finds channels and manages message delivery
 *
 * Perfect for:
 * - Sharing analysis results that drive decisions
 * - Automated insights that feel personally crafted
 * - Business intelligence updates with personality
 * - Data discoveries that need immediate attention
 * - Reports that people actually read and act on
 */
export declare class SlackNotifierWorkflowBubble extends WorkflowBubble<SlackNotifierParams, SlackNotifierResult> {
    static readonly bubbleName = "slack-notifier";
    static readonly schema: z.ZodObject<{
        /**
         * The raw data or content to be formatted for Slack notification
         */
        contentToFormat: z.ZodString;
        /**
         * The original user query or context that generated this content
         */
        originalUserQuery: z.ZodOptional<z.ZodString>;
        /**
         * Target Slack channel name (without #) or channel ID
         */
        targetChannel: z.ZodString;
        /**
         * Optional custom message title/header for the notification
         */
        messageTitle: z.ZodOptional<z.ZodString>;
        /**
         * Tone and style for the AI formatting (professional, casual, technical, etc.)
         */
        messageStyle: z.ZodDefault<z.ZodEnum<["professional", "casual", "technical", "concise", "detailed"]>>;
        /**
         * Whether to include emojis and rich formatting in the message
         */
        includeFormatting: z.ZodDefault<z.ZodBoolean>;
        /**
         * Maximum message length (Slack has limits)
         */
        maxMessageLength: z.ZodDefault<z.ZodNumber>;
        /**
         * AI model configuration for content formatting
         */
        aiModel: z.ZodOptional<z.ZodObject<{
            model: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
            temperature: z.ZodDefault<z.ZodNumber>;
            maxTokens: z.ZodDefault<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature: number;
            maxTokens: number;
        }, {
            model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
            temperature?: number | undefined;
            maxTokens?: number | undefined;
        }>>;
        /**
         * Injected credentials from the system
         */
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        contentToFormat: string;
        targetChannel: string;
        messageStyle: "professional" | "casual" | "technical" | "concise" | "detailed";
        includeFormatting: boolean;
        maxMessageLength: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        originalUserQuery?: string | undefined;
        messageTitle?: string | undefined;
        aiModel?: {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature: number;
            maxTokens: number;
        } | undefined;
    }, {
        contentToFormat: string;
        targetChannel: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        originalUserQuery?: string | undefined;
        messageTitle?: string | undefined;
        messageStyle?: "professional" | "casual" | "technical" | "concise" | "detailed" | undefined;
        includeFormatting?: boolean | undefined;
        maxMessageLength?: number | undefined;
        aiModel?: {
            model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
            temperature?: number | undefined;
            maxTokens?: number | undefined;
        } | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        error: z.ZodString;
        /**
         * Information about the sent Slack message
         */
        messageInfo: z.ZodOptional<z.ZodObject<{
            /**
             * Slack message timestamp (unique identifier)
             */
            messageTimestamp: z.ZodOptional<z.ZodString>;
            /**
             * Channel ID where message was sent
             */
            channelId: z.ZodOptional<z.ZodString>;
            /**
             * Channel name where message was sent
             */
            channelName: z.ZodOptional<z.ZodString>;
            /**
             * The formatted message that was sent
             */
            formattedMessage: z.ZodOptional<z.ZodString>;
            /**
             * Message length in characters
             */
            messageLength: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            channelName?: string | undefined;
            messageTimestamp?: string | undefined;
            channelId?: string | undefined;
            formattedMessage?: string | undefined;
            messageLength?: number | undefined;
        }, {
            channelName?: string | undefined;
            messageTimestamp?: string | undefined;
            channelId?: string | undefined;
            formattedMessage?: string | undefined;
            messageLength?: number | undefined;
        }>>;
        /**
         * AI formatting process information
         */
        formattingInfo: z.ZodOptional<z.ZodObject<{
            /**
             * AI model used for formatting
             */
            modelUsed: z.ZodOptional<z.ZodString>;
            /**
             * Whether content was truncated due to length limits
             */
            wasTruncated: z.ZodDefault<z.ZodBoolean>;
            /**
             * Original content length before formatting
             */
            originalLength: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            wasTruncated: boolean;
            modelUsed?: string | undefined;
            originalLength?: number | undefined;
        }, {
            modelUsed?: string | undefined;
            wasTruncated?: boolean | undefined;
            originalLength?: number | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        messageInfo?: {
            channelName?: string | undefined;
            messageTimestamp?: string | undefined;
            channelId?: string | undefined;
            formattedMessage?: string | undefined;
            messageLength?: number | undefined;
        } | undefined;
        formattingInfo?: {
            wasTruncated: boolean;
            modelUsed?: string | undefined;
            originalLength?: number | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        messageInfo?: {
            channelName?: string | undefined;
            messageTimestamp?: string | undefined;
            channelId?: string | undefined;
            formattedMessage?: string | undefined;
            messageLength?: number | undefined;
        } | undefined;
        formattingInfo?: {
            modelUsed?: string | undefined;
            wasTruncated?: boolean | undefined;
            originalLength?: number | undefined;
        } | undefined;
    }>;
    static readonly shortDescription = "Data analyst-powered Slack notifications that tell compelling stories";
    static readonly longDescription = "Transforms raw data and insights into engaging, conversational Slack messages that colleagues actually want to read. Uses AI with a data analyst personality to spot patterns, provide context, and make information actionable. Perfect for sharing analysis results, automated reports, and business intelligence updates with natural, human-like communication.";
    static readonly alias = "notify-slack";
    static readonly type: "workflow";
    constructor(params: SlackNotifierParamsInput, context?: BubbleContext, instanceId?: string);
    protected performAction(): Promise<SlackNotifierResult>;
    /**
     * Find the target Slack channel by name or ID
     */
    private findSlackChannel;
    /**
     * Format content using AI for better Slack presentation
     */
    private formatContentWithAI;
    /**
     * Send the formatted message to Slack
     */
    private sendToSlack;
    /**
     * Build the AI formatting prompt based on parameters
     */
    private buildFormattingPrompt;
}
export {};
//# sourceMappingURL=slack-notifier.workflow.d.ts.map