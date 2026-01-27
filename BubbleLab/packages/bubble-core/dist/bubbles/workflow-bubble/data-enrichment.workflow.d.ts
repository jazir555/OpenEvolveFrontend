/**
 * DATA ENRICHMENT WORKFLOW
 *
 * A comprehensive workflow for enriching data records from multiple sources
 * including web search, vector search, and AI analysis.
 *
 * This workflow combines:
 * 1. Web search tool for external data enrichment
 * 2. Vector search for similar record discovery
 * 3. AI agent for intelligent analysis and synthesis
 * 4. Multi-source data merging and validation
 */
import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Parameters schema for data enrichment workflow
 */
declare const DataEnrichmentParamsSchema: z.ZodObject<{
    /**
     * Input record to enrich
     */
    record: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    /**
     * Enrichment sources to use
     */
    sources: z.ZodOptional<z.ZodObject<{
        webSearch: z.ZodDefault<z.ZodBoolean>;
        vectorSearch: z.ZodDefault<z.ZodBoolean>;
        aiAnalysis: z.ZodDefault<z.ZodBoolean>;
        databaseLookup: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        aiAnalysis: boolean;
        webSearch: boolean;
        vectorSearch: boolean;
        databaseLookup: boolean;
    }, {
        aiAnalysis?: boolean | undefined;
        webSearch?: boolean | undefined;
        vectorSearch?: boolean | undefined;
        databaseLookup?: boolean | undefined;
    }>>;
    /**
     * Web search configuration
     */
    webSearchConfig: z.ZodOptional<z.ZodObject<{
        searchEngine: z.ZodDefault<z.ZodEnum<["google", "bing", "duckduckgo"]>>;
        maxResults: z.ZodDefault<z.ZodNumber>;
        searchQuery: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        maxResults: number;
        searchEngine: "google" | "bing" | "duckduckgo";
        searchQuery?: string | undefined;
    }, {
        maxResults?: number | undefined;
        searchEngine?: "google" | "bing" | "duckduckgo" | undefined;
        searchQuery?: string | undefined;
    }>>;
    /**
     * Vector search configuration
     */
    vectorSearchConfig: z.ZodOptional<z.ZodObject<{
        vectorEndpoint: z.ZodString;
        topK: z.ZodDefault<z.ZodNumber>;
        threshold: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        topK: number;
        threshold: number;
        vectorEndpoint: string;
    }, {
        vectorEndpoint: string;
        topK?: number | undefined;
        threshold?: number | undefined;
    }>>;
    /**
     * Database lookup configuration
     */
    databaseConfig: z.ZodOptional<z.ZodObject<{
        connectionString: z.ZodString;
        query: z.ZodString;
        queryKey: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        query: string;
        connectionString: string;
        queryKey: string;
    }, {
        query: string;
        connectionString: string;
        queryKey: string;
    }>>;
    /**
     * AI analysis configuration
     */
    aiConfig: z.ZodOptional<z.ZodObject<{
        model: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
        temperature: z.ZodDefault<z.ZodNumber>;
        maxTokens: z.ZodDefault<z.ZodNumber>;
        analysisPrompt: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature: number;
        maxTokens: number;
        analysisPrompt?: string | undefined;
    }, {
        model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
        analysisPrompt?: string | undefined;
    }>>;
    /**
     * Output format
     */
    outputFormat: z.ZodDefault<z.ZodEnum<["merged", "append", "replace"]>>;
    /**
     * Credentials
     */
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    outputFormat: "replace" | "merged" | "append";
    record: Record<string, unknown>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    sources?: {
        aiAnalysis: boolean;
        webSearch: boolean;
        vectorSearch: boolean;
        databaseLookup: boolean;
    } | undefined;
    webSearchConfig?: {
        maxResults: number;
        searchEngine: "google" | "bing" | "duckduckgo";
        searchQuery?: string | undefined;
    } | undefined;
    vectorSearchConfig?: {
        topK: number;
        threshold: number;
        vectorEndpoint: string;
    } | undefined;
    databaseConfig?: {
        query: string;
        connectionString: string;
        queryKey: string;
    } | undefined;
    aiConfig?: {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature: number;
        maxTokens: number;
        analysisPrompt?: string | undefined;
    } | undefined;
}, {
    record: Record<string, unknown>;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    outputFormat?: "replace" | "merged" | "append" | undefined;
    sources?: {
        aiAnalysis?: boolean | undefined;
        webSearch?: boolean | undefined;
        vectorSearch?: boolean | undefined;
        databaseLookup?: boolean | undefined;
    } | undefined;
    webSearchConfig?: {
        maxResults?: number | undefined;
        searchEngine?: "google" | "bing" | "duckduckgo" | undefined;
        searchQuery?: string | undefined;
    } | undefined;
    vectorSearchConfig?: {
        vectorEndpoint: string;
        topK?: number | undefined;
        threshold?: number | undefined;
    } | undefined;
    databaseConfig?: {
        query: string;
        connectionString: string;
        queryKey: string;
    } | undefined;
    aiConfig?: {
        model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
        analysisPrompt?: string | undefined;
    } | undefined;
}>;
type DataEnrichmentParams = z.input<typeof DataEnrichmentParamsSchema>;
/**
 * Result schema for data enrichment workflow
 */
declare const DataEnrichmentResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    error: z.ZodString;
    /**
     * Enriched record
     */
    enrichedRecord: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    /**
     * Enrichment sources results
     */
    enrichmentResults: z.ZodOptional<z.ZodObject<{
        webSearch: z.ZodOptional<z.ZodObject<{
            success: z.ZodBoolean;
            results: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            count: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            count?: number | undefined;
            results?: unknown[] | undefined;
        }, {
            success: boolean;
            count?: number | undefined;
            results?: unknown[] | undefined;
        }>>;
        vectorSearch: z.ZodOptional<z.ZodObject<{
            success: z.ZodBoolean;
            similarRecords: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            count: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            count?: number | undefined;
            similarRecords?: unknown[] | undefined;
        }, {
            success: boolean;
            count?: number | undefined;
            similarRecords?: unknown[] | undefined;
        }>>;
        aiAnalysis: z.ZodOptional<z.ZodObject<{
            success: z.ZodBoolean;
            insights: z.ZodOptional<z.ZodString>;
            confidence: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            confidence?: number | undefined;
            insights?: string | undefined;
        }, {
            success: boolean;
            confidence?: number | undefined;
            insights?: string | undefined;
        }>>;
        databaseLookup: z.ZodOptional<z.ZodObject<{
            success: z.ZodBoolean;
            data: z.ZodOptional<z.ZodUnknown>;
            rowsAffected: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            success: boolean;
            data?: unknown;
            rowsAffected?: number | undefined;
        }, {
            success: boolean;
            data?: unknown;
            rowsAffected?: number | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        aiAnalysis?: {
            success: boolean;
            confidence?: number | undefined;
            insights?: string | undefined;
        } | undefined;
        webSearch?: {
            success: boolean;
            count?: number | undefined;
            results?: unknown[] | undefined;
        } | undefined;
        vectorSearch?: {
            success: boolean;
            count?: number | undefined;
            similarRecords?: unknown[] | undefined;
        } | undefined;
        databaseLookup?: {
            success: boolean;
            data?: unknown;
            rowsAffected?: number | undefined;
        } | undefined;
    }, {
        aiAnalysis?: {
            success: boolean;
            confidence?: number | undefined;
            insights?: string | undefined;
        } | undefined;
        webSearch?: {
            success: boolean;
            count?: number | undefined;
            results?: unknown[] | undefined;
        } | undefined;
        vectorSearch?: {
            success: boolean;
            count?: number | undefined;
            similarRecords?: unknown[] | undefined;
        } | undefined;
        databaseLookup?: {
            success: boolean;
            data?: unknown;
            rowsAffected?: number | undefined;
        } | undefined;
    }>>;
    /**
     * Enrichment metadata
     */
    metadata: z.ZodOptional<z.ZodObject<{
        sourcesUsed: z.ZodArray<z.ZodString, "many">;
        enrichmentTimestamp: z.ZodDate;
        processingTime: z.ZodNumber;
        fieldsAdded: z.ZodNumber;
        dataQualityScore: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        processingTime: number;
        sourcesUsed: string[];
        fieldsAdded: number;
        enrichmentTimestamp: Date;
        dataQualityScore: number;
    }, {
        processingTime: number;
        sourcesUsed: string[];
        fieldsAdded: number;
        enrichmentTimestamp: Date;
        dataQualityScore: number;
    }>>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    metadata?: {
        processingTime: number;
        sourcesUsed: string[];
        fieldsAdded: number;
        enrichmentTimestamp: Date;
        dataQualityScore: number;
    } | undefined;
    enrichedRecord?: Record<string, unknown> | undefined;
    enrichmentResults?: {
        aiAnalysis?: {
            success: boolean;
            confidence?: number | undefined;
            insights?: string | undefined;
        } | undefined;
        webSearch?: {
            success: boolean;
            count?: number | undefined;
            results?: unknown[] | undefined;
        } | undefined;
        vectorSearch?: {
            success: boolean;
            count?: number | undefined;
            similarRecords?: unknown[] | undefined;
        } | undefined;
        databaseLookup?: {
            success: boolean;
            data?: unknown;
            rowsAffected?: number | undefined;
        } | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    metadata?: {
        processingTime: number;
        sourcesUsed: string[];
        fieldsAdded: number;
        enrichmentTimestamp: Date;
        dataQualityScore: number;
    } | undefined;
    enrichedRecord?: Record<string, unknown> | undefined;
    enrichmentResults?: {
        aiAnalysis?: {
            success: boolean;
            confidence?: number | undefined;
            insights?: string | undefined;
        } | undefined;
        webSearch?: {
            success: boolean;
            count?: number | undefined;
            results?: unknown[] | undefined;
        } | undefined;
        vectorSearch?: {
            success: boolean;
            count?: number | undefined;
            similarRecords?: unknown[] | undefined;
        } | undefined;
        databaseLookup?: {
            success: boolean;
            data?: unknown;
            rowsAffected?: number | undefined;
        } | undefined;
    } | undefined;
}>;
type DataEnrichmentResult = z.infer<typeof DataEnrichmentResultSchema>;
/**
 * Data Enrichment Workflow
 *
 * Enriches data records from multiple sources with intelligent merging and validation.
 */
export declare class DataEnrichmentWorkflow extends WorkflowBubble<DataEnrichmentParams, DataEnrichmentResult> {
    static readonly type: "workflow";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        /**
         * Input record to enrich
         */
        record: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        /**
         * Enrichment sources to use
         */
        sources: z.ZodOptional<z.ZodObject<{
            webSearch: z.ZodDefault<z.ZodBoolean>;
            vectorSearch: z.ZodDefault<z.ZodBoolean>;
            aiAnalysis: z.ZodDefault<z.ZodBoolean>;
            databaseLookup: z.ZodDefault<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            aiAnalysis: boolean;
            webSearch: boolean;
            vectorSearch: boolean;
            databaseLookup: boolean;
        }, {
            aiAnalysis?: boolean | undefined;
            webSearch?: boolean | undefined;
            vectorSearch?: boolean | undefined;
            databaseLookup?: boolean | undefined;
        }>>;
        /**
         * Web search configuration
         */
        webSearchConfig: z.ZodOptional<z.ZodObject<{
            searchEngine: z.ZodDefault<z.ZodEnum<["google", "bing", "duckduckgo"]>>;
            maxResults: z.ZodDefault<z.ZodNumber>;
            searchQuery: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            maxResults: number;
            searchEngine: "google" | "bing" | "duckduckgo";
            searchQuery?: string | undefined;
        }, {
            maxResults?: number | undefined;
            searchEngine?: "google" | "bing" | "duckduckgo" | undefined;
            searchQuery?: string | undefined;
        }>>;
        /**
         * Vector search configuration
         */
        vectorSearchConfig: z.ZodOptional<z.ZodObject<{
            vectorEndpoint: z.ZodString;
            topK: z.ZodDefault<z.ZodNumber>;
            threshold: z.ZodDefault<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            topK: number;
            threshold: number;
            vectorEndpoint: string;
        }, {
            vectorEndpoint: string;
            topK?: number | undefined;
            threshold?: number | undefined;
        }>>;
        /**
         * Database lookup configuration
         */
        databaseConfig: z.ZodOptional<z.ZodObject<{
            connectionString: z.ZodString;
            query: z.ZodString;
            queryKey: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            query: string;
            connectionString: string;
            queryKey: string;
        }, {
            query: string;
            connectionString: string;
            queryKey: string;
        }>>;
        /**
         * AI analysis configuration
         */
        aiConfig: z.ZodOptional<z.ZodObject<{
            model: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
            temperature: z.ZodDefault<z.ZodNumber>;
            maxTokens: z.ZodDefault<z.ZodNumber>;
            analysisPrompt: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature: number;
            maxTokens: number;
            analysisPrompt?: string | undefined;
        }, {
            model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            analysisPrompt?: string | undefined;
        }>>;
        /**
         * Output format
         */
        outputFormat: z.ZodDefault<z.ZodEnum<["merged", "append", "replace"]>>;
        /**
         * Credentials
         */
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        outputFormat: "replace" | "merged" | "append";
        record: Record<string, unknown>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        sources?: {
            aiAnalysis: boolean;
            webSearch: boolean;
            vectorSearch: boolean;
            databaseLookup: boolean;
        } | undefined;
        webSearchConfig?: {
            maxResults: number;
            searchEngine: "google" | "bing" | "duckduckgo";
            searchQuery?: string | undefined;
        } | undefined;
        vectorSearchConfig?: {
            topK: number;
            threshold: number;
            vectorEndpoint: string;
        } | undefined;
        databaseConfig?: {
            query: string;
            connectionString: string;
            queryKey: string;
        } | undefined;
        aiConfig?: {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature: number;
            maxTokens: number;
            analysisPrompt?: string | undefined;
        } | undefined;
    }, {
        record: Record<string, unknown>;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        outputFormat?: "replace" | "merged" | "append" | undefined;
        sources?: {
            aiAnalysis?: boolean | undefined;
            webSearch?: boolean | undefined;
            vectorSearch?: boolean | undefined;
            databaseLookup?: boolean | undefined;
        } | undefined;
        webSearchConfig?: {
            maxResults?: number | undefined;
            searchEngine?: "google" | "bing" | "duckduckgo" | undefined;
            searchQuery?: string | undefined;
        } | undefined;
        vectorSearchConfig?: {
            vectorEndpoint: string;
            topK?: number | undefined;
            threshold?: number | undefined;
        } | undefined;
        databaseConfig?: {
            query: string;
            connectionString: string;
            queryKey: string;
        } | undefined;
        aiConfig?: {
            model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            analysisPrompt?: string | undefined;
        } | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        error: z.ZodString;
        /**
         * Enriched record
         */
        enrichedRecord: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        /**
         * Enrichment sources results
         */
        enrichmentResults: z.ZodOptional<z.ZodObject<{
            webSearch: z.ZodOptional<z.ZodObject<{
                success: z.ZodBoolean;
                results: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
                count: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                success: boolean;
                count?: number | undefined;
                results?: unknown[] | undefined;
            }, {
                success: boolean;
                count?: number | undefined;
                results?: unknown[] | undefined;
            }>>;
            vectorSearch: z.ZodOptional<z.ZodObject<{
                success: z.ZodBoolean;
                similarRecords: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
                count: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                success: boolean;
                count?: number | undefined;
                similarRecords?: unknown[] | undefined;
            }, {
                success: boolean;
                count?: number | undefined;
                similarRecords?: unknown[] | undefined;
            }>>;
            aiAnalysis: z.ZodOptional<z.ZodObject<{
                success: z.ZodBoolean;
                insights: z.ZodOptional<z.ZodString>;
                confidence: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                success: boolean;
                confidence?: number | undefined;
                insights?: string | undefined;
            }, {
                success: boolean;
                confidence?: number | undefined;
                insights?: string | undefined;
            }>>;
            databaseLookup: z.ZodOptional<z.ZodObject<{
                success: z.ZodBoolean;
                data: z.ZodOptional<z.ZodUnknown>;
                rowsAffected: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                success: boolean;
                data?: unknown;
                rowsAffected?: number | undefined;
            }, {
                success: boolean;
                data?: unknown;
                rowsAffected?: number | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            aiAnalysis?: {
                success: boolean;
                confidence?: number | undefined;
                insights?: string | undefined;
            } | undefined;
            webSearch?: {
                success: boolean;
                count?: number | undefined;
                results?: unknown[] | undefined;
            } | undefined;
            vectorSearch?: {
                success: boolean;
                count?: number | undefined;
                similarRecords?: unknown[] | undefined;
            } | undefined;
            databaseLookup?: {
                success: boolean;
                data?: unknown;
                rowsAffected?: number | undefined;
            } | undefined;
        }, {
            aiAnalysis?: {
                success: boolean;
                confidence?: number | undefined;
                insights?: string | undefined;
            } | undefined;
            webSearch?: {
                success: boolean;
                count?: number | undefined;
                results?: unknown[] | undefined;
            } | undefined;
            vectorSearch?: {
                success: boolean;
                count?: number | undefined;
                similarRecords?: unknown[] | undefined;
            } | undefined;
            databaseLookup?: {
                success: boolean;
                data?: unknown;
                rowsAffected?: number | undefined;
            } | undefined;
        }>>;
        /**
         * Enrichment metadata
         */
        metadata: z.ZodOptional<z.ZodObject<{
            sourcesUsed: z.ZodArray<z.ZodString, "many">;
            enrichmentTimestamp: z.ZodDate;
            processingTime: z.ZodNumber;
            fieldsAdded: z.ZodNumber;
            dataQualityScore: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            processingTime: number;
            sourcesUsed: string[];
            fieldsAdded: number;
            enrichmentTimestamp: Date;
            dataQualityScore: number;
        }, {
            processingTime: number;
            sourcesUsed: string[];
            fieldsAdded: number;
            enrichmentTimestamp: Date;
            dataQualityScore: number;
        }>>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        metadata?: {
            processingTime: number;
            sourcesUsed: string[];
            fieldsAdded: number;
            enrichmentTimestamp: Date;
            dataQualityScore: number;
        } | undefined;
        enrichedRecord?: Record<string, unknown> | undefined;
        enrichmentResults?: {
            aiAnalysis?: {
                success: boolean;
                confidence?: number | undefined;
                insights?: string | undefined;
            } | undefined;
            webSearch?: {
                success: boolean;
                count?: number | undefined;
                results?: unknown[] | undefined;
            } | undefined;
            vectorSearch?: {
                success: boolean;
                count?: number | undefined;
                similarRecords?: unknown[] | undefined;
            } | undefined;
            databaseLookup?: {
                success: boolean;
                data?: unknown;
                rowsAffected?: number | undefined;
            } | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        metadata?: {
            processingTime: number;
            sourcesUsed: string[];
            fieldsAdded: number;
            enrichmentTimestamp: Date;
            dataQualityScore: number;
        } | undefined;
        enrichedRecord?: Record<string, unknown> | undefined;
        enrichmentResults?: {
            aiAnalysis?: {
                success: boolean;
                confidence?: number | undefined;
                insights?: string | undefined;
            } | undefined;
            webSearch?: {
                success: boolean;
                count?: number | undefined;
                results?: unknown[] | undefined;
            } | undefined;
            vectorSearch?: {
                success: boolean;
                count?: number | undefined;
                similarRecords?: unknown[] | undefined;
            } | undefined;
            databaseLookup?: {
                success: boolean;
                data?: unknown;
                rowsAffected?: number | undefined;
            } | undefined;
        } | undefined;
    }>;
    static readonly shortDescription = "Multi-source data enrichment with AI-powered analysis";
    static readonly longDescription = "\n    Enriches data records by combining multiple data sources and AI analysis.\n\n    Features:\n    - Web search for external information retrieval\n    - Vector similarity search for related records\n    - Database lookup for structured data enrichment\n    - AI-powered analysis and synthesis\n    - Intelligent data merging strategies\n    - Data quality scoring and validation\n\n    Use cases:\n    - CRM record enrichment with external data\n    - Lead scoring with additional context\n    - Product data enhancement from multiple sources\n    - Customer profile enrichment\n    - Research data augmentation\n\n    Process:\n    1. Extract key information from input record\n    2. Query enabled enrichment sources in parallel\n    3. AI analyzes and synthesizes all gathered data\n    4. Merge enriched data with original record\n    5. Calculate data quality score\n    6. Return comprehensive enrichment results\n  ";
    static readonly alias = "enrich-data";
    constructor(params: DataEnrichmentParams, context?: BubbleContext);
    protected performAction(): Promise<DataEnrichmentResult>;
    /**
     * Perform web search enrichment
     */
    private performWebSearch;
    /**
     * Perform vector similarity search
     */
    private performVectorSearch;
    /**
     * Perform database lookup
     */
    private performDatabaseLookup;
    /**
     * Perform AI analysis and synthesis
     */
    private performAIAnalysis;
    /**
     * Generate search query from record
     */
    private generateSearchQueryFromRecord;
    /**
     * Build search URL for different search engines
     */
    private buildSearchUrl;
    /**
     * Extract search results from search engine response
     */
    private extractSearchResults;
    /**
     * Build default AI analysis prompt
     */
    private buildDefaultAnalysisPrompt;
    /**
     * Merge data based on output format
     */
    private mergeData;
    /**
     * Calculate data quality score
     */
    private calculateDataQualityScore;
}
export {};
//# sourceMappingURL=data-enrichment.workflow.d.ts.map