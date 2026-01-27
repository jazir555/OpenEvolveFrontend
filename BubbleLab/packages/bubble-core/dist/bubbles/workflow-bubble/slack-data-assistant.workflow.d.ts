import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const SlackDataAssistantParamsSchema: z.ZodObject<{
    slackChannel: z.ZodString;
    slackThreadTs: z.ZodOptional<z.ZodString>;
    userQuestion: z.ZodString;
    userName: z.ZodOptional<z.ZodString>;
    name: z.ZodDefault<z.ZodString>;
    dataSourceType: z.ZodDefault<z.ZodEnum<["postgresql", "mysql", "sqlite", "mariadb", "mssql"]>>;
    databaseUrl: z.ZodOptional<z.ZodString>;
    ignoreSSLErrors: z.ZodDefault<z.ZodBoolean>;
    aiModel: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
    temperature: z.ZodDefault<z.ZodNumber>;
    verbosity: z.ZodDefault<z.ZodEnum<["1", "2", "3", "4", "5"]>>;
    technicality: z.ZodDefault<z.ZodEnum<["1", "2", "3", "4", "5"]>>;
    includeQuery: z.ZodDefault<z.ZodBoolean>;
    includeExplanation: z.ZodDefault<z.ZodBoolean>;
    injectedMetadata: z.ZodOptional<z.ZodObject<{
        tables: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodRecord<z.ZodString, z.ZodString>>>;
        tableNotes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        rules: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        tables?: Record<string, Record<string, string>> | undefined;
        tableNotes?: Record<string, string> | undefined;
        rules?: string[] | undefined;
    }, {
        tables?: Record<string, Record<string, string>> | undefined;
        tableNotes?: Record<string, string> | undefined;
        rules?: string[] | undefined;
    }>>;
    additionalContext: z.ZodOptional<z.ZodString>;
    maxQueries: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    temperature: number;
    dataSourceType: "postgresql" | "mysql" | "sqlite" | "mariadb" | "mssql";
    ignoreSSLErrors: boolean;
    aiModel: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
    slackChannel: string;
    userQuestion: string;
    verbosity: "1" | "2" | "3" | "4" | "5";
    technicality: "1" | "2" | "3" | "4" | "5";
    includeQuery: boolean;
    includeExplanation: boolean;
    maxQueries: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    userName?: string | undefined;
    injectedMetadata?: {
        tables?: Record<string, Record<string, string>> | undefined;
        tableNotes?: Record<string, string> | undefined;
        rules?: string[] | undefined;
    } | undefined;
    slackThreadTs?: string | undefined;
    databaseUrl?: string | undefined;
    additionalContext?: string | undefined;
}, {
    slackChannel: string;
    userQuestion: string;
    name?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    temperature?: number | undefined;
    userName?: string | undefined;
    dataSourceType?: "postgresql" | "mysql" | "sqlite" | "mariadb" | "mssql" | undefined;
    ignoreSSLErrors?: boolean | undefined;
    injectedMetadata?: {
        tables?: Record<string, Record<string, string>> | undefined;
        tableNotes?: Record<string, string> | undefined;
        rules?: string[] | undefined;
    } | undefined;
    aiModel?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
    slackThreadTs?: string | undefined;
    databaseUrl?: string | undefined;
    verbosity?: "1" | "2" | "3" | "4" | "5" | undefined;
    technicality?: "1" | "2" | "3" | "4" | "5" | undefined;
    includeQuery?: boolean | undefined;
    includeExplanation?: boolean | undefined;
    additionalContext?: string | undefined;
    maxQueries?: number | undefined;
}>;
declare const SlackDataAssistantResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    error: z.ZodString;
    query: z.ZodOptional<z.ZodString>;
    queryExplanation: z.ZodOptional<z.ZodString>;
    queryResults: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
    formattedResponse: z.ZodOptional<z.ZodString>;
    slackBlocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    slackMessageTs: z.ZodOptional<z.ZodString>;
    isDataQuestion: z.ZodOptional<z.ZodBoolean>;
    metadata: z.ZodOptional<z.ZodObject<{
        executionTime: z.ZodNumber;
        rowCount: z.ZodOptional<z.ZodNumber>;
        wordCount: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        executionTime: number;
        rowCount?: number | undefined;
        wordCount?: number | undefined;
    }, {
        executionTime: number;
        rowCount?: number | undefined;
        wordCount?: number | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    query?: string | undefined;
    metadata?: {
        executionTime: number;
        rowCount?: number | undefined;
        wordCount?: number | undefined;
    } | undefined;
    queryExplanation?: string | undefined;
    queryResults?: Record<string, unknown>[] | undefined;
    formattedResponse?: string | undefined;
    slackBlocks?: unknown[] | undefined;
    slackMessageTs?: string | undefined;
    isDataQuestion?: boolean | undefined;
}, {
    error: string;
    success: boolean;
    query?: string | undefined;
    metadata?: {
        executionTime: number;
        rowCount?: number | undefined;
        wordCount?: number | undefined;
    } | undefined;
    queryExplanation?: string | undefined;
    queryResults?: Record<string, unknown>[] | undefined;
    formattedResponse?: string | undefined;
    slackBlocks?: unknown[] | undefined;
    slackMessageTs?: string | undefined;
    isDataQuestion?: boolean | undefined;
}>;
type SlackDataAssistantParams = z.input<typeof SlackDataAssistantParamsSchema>;
type SlackDataAssistantResult = z.output<typeof SlackDataAssistantResultSchema>;
export declare class SlackDataAssistantWorkflow extends WorkflowBubble<SlackDataAssistantParams, SlackDataAssistantResult> {
    static readonly type: "workflow";
    static readonly service = "slack-data-assistant";
    static readonly bubbleName = "slack-data-assistant";
    static readonly schema: z.ZodObject<{
        slackChannel: z.ZodString;
        slackThreadTs: z.ZodOptional<z.ZodString>;
        userQuestion: z.ZodString;
        userName: z.ZodOptional<z.ZodString>;
        name: z.ZodDefault<z.ZodString>;
        dataSourceType: z.ZodDefault<z.ZodEnum<["postgresql", "mysql", "sqlite", "mariadb", "mssql"]>>;
        databaseUrl: z.ZodOptional<z.ZodString>;
        ignoreSSLErrors: z.ZodDefault<z.ZodBoolean>;
        aiModel: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
        temperature: z.ZodDefault<z.ZodNumber>;
        verbosity: z.ZodDefault<z.ZodEnum<["1", "2", "3", "4", "5"]>>;
        technicality: z.ZodDefault<z.ZodEnum<["1", "2", "3", "4", "5"]>>;
        includeQuery: z.ZodDefault<z.ZodBoolean>;
        includeExplanation: z.ZodDefault<z.ZodBoolean>;
        injectedMetadata: z.ZodOptional<z.ZodObject<{
            tables: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodRecord<z.ZodString, z.ZodString>>>;
            tableNotes: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
            rules: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            tables?: Record<string, Record<string, string>> | undefined;
            tableNotes?: Record<string, string> | undefined;
            rules?: string[] | undefined;
        }, {
            tables?: Record<string, Record<string, string>> | undefined;
            tableNotes?: Record<string, string> | undefined;
            rules?: string[] | undefined;
        }>>;
        additionalContext: z.ZodOptional<z.ZodString>;
        maxQueries: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        temperature: number;
        dataSourceType: "postgresql" | "mysql" | "sqlite" | "mariadb" | "mssql";
        ignoreSSLErrors: boolean;
        aiModel: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        slackChannel: string;
        userQuestion: string;
        verbosity: "1" | "2" | "3" | "4" | "5";
        technicality: "1" | "2" | "3" | "4" | "5";
        includeQuery: boolean;
        includeExplanation: boolean;
        maxQueries: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        userName?: string | undefined;
        injectedMetadata?: {
            tables?: Record<string, Record<string, string>> | undefined;
            tableNotes?: Record<string, string> | undefined;
            rules?: string[] | undefined;
        } | undefined;
        slackThreadTs?: string | undefined;
        databaseUrl?: string | undefined;
        additionalContext?: string | undefined;
    }, {
        slackChannel: string;
        userQuestion: string;
        name?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        temperature?: number | undefined;
        userName?: string | undefined;
        dataSourceType?: "postgresql" | "mysql" | "sqlite" | "mariadb" | "mssql" | undefined;
        ignoreSSLErrors?: boolean | undefined;
        injectedMetadata?: {
            tables?: Record<string, Record<string, string>> | undefined;
            tableNotes?: Record<string, string> | undefined;
            rules?: string[] | undefined;
        } | undefined;
        aiModel?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
        slackThreadTs?: string | undefined;
        databaseUrl?: string | undefined;
        verbosity?: "1" | "2" | "3" | "4" | "5" | undefined;
        technicality?: "1" | "2" | "3" | "4" | "5" | undefined;
        includeQuery?: boolean | undefined;
        includeExplanation?: boolean | undefined;
        additionalContext?: string | undefined;
        maxQueries?: number | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        error: z.ZodString;
        query: z.ZodOptional<z.ZodString>;
        queryExplanation: z.ZodOptional<z.ZodString>;
        queryResults: z.ZodOptional<z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnknown>, "many">>;
        formattedResponse: z.ZodOptional<z.ZodString>;
        slackBlocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        slackMessageTs: z.ZodOptional<z.ZodString>;
        isDataQuestion: z.ZodOptional<z.ZodBoolean>;
        metadata: z.ZodOptional<z.ZodObject<{
            executionTime: z.ZodNumber;
            rowCount: z.ZodOptional<z.ZodNumber>;
            wordCount: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            executionTime: number;
            rowCount?: number | undefined;
            wordCount?: number | undefined;
        }, {
            executionTime: number;
            rowCount?: number | undefined;
            wordCount?: number | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        query?: string | undefined;
        metadata?: {
            executionTime: number;
            rowCount?: number | undefined;
            wordCount?: number | undefined;
        } | undefined;
        queryExplanation?: string | undefined;
        queryResults?: Record<string, unknown>[] | undefined;
        formattedResponse?: string | undefined;
        slackBlocks?: unknown[] | undefined;
        slackMessageTs?: string | undefined;
        isDataQuestion?: boolean | undefined;
    }, {
        error: string;
        success: boolean;
        query?: string | undefined;
        metadata?: {
            executionTime: number;
            rowCount?: number | undefined;
            wordCount?: number | undefined;
        } | undefined;
        queryExplanation?: string | undefined;
        queryResults?: Record<string, unknown>[] | undefined;
        formattedResponse?: string | undefined;
        slackBlocks?: unknown[] | undefined;
        slackMessageTs?: string | undefined;
        isDataQuestion?: boolean | undefined;
    }>;
    static readonly shortDescription = "AI-powered Slack bot that answers data questions by querying databases";
    static readonly longDescription = "\n    A comprehensive workflow that creates an intelligent Slack bot capable of:\n    - Receiving questions from Slack mentions\n    - Analyzing database schema\n    - Generating appropriate SQL queries using AI\n    - Executing queries safely (read-only)\n    - Formatting results in a user-friendly way\n    - Responding in Slack with rich block formatting\n    \n    Perfect for:\n    - Business intelligence chat-bots\n    - Data analytics assistants\n    - Database query automation\n    - Self-service data access\n  ";
    static readonly alias = "slack-data-bot";
    constructor(params: SlackDataAssistantParams, context?: BubbleContext);
    /**
     * Extract first name from a full name string
     */
    private extractFirstName;
    /**
     * Clean bot name by removing common suffixes and formatting
     */
    private cleanBotName;
    /**
     * Clean username by converting formats like "john.doe" to "John"
     */
    private cleanUsername;
    /**
     * Generate a readable name from a Slack user ID when API calls fail
     * Converts "U07UTL8MA9Y" to "User07" etc.
     */
    private generateReadableNameFromUserId;
    protected performAction(): Promise<SlackDataAssistantResult>;
    private aggregateQueryResults;
}
export {};
//# sourceMappingURL=slack-data-assistant.workflow.d.ts.map