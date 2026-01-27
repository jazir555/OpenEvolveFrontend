/**
 * PARSE DOCUMENT WORKFLOW
 *
 * A comprehensive workflow that converts images and PDFs into structured markdown
 * using AI vision analysis. Preserves document structure, describes charts and images
 * numerically, and maintains formatting and layout information.
 *
 * This workflow combines:
 * 1. PDF to images conversion using pdf-img-convert
 * 2. AI vision analysis for content extraction and markdown generation
 * 3. Structure preservation with table, chart, and image descriptions
 *
 * Returns clean markdown with preserved structure and detailed visual descriptions.
 */
import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Parameters schema for Parse Document workflow
 */
declare const ParseDocumentWorkflowParamsSchema: z.ZodObject<{
    documentData: z.ZodString;
    documentType: z.ZodDefault<z.ZodEnum<["pdf", "image"]>>;
    isFileUrl: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    conversionOptions: z.ZodDefault<z.ZodObject<{
        preserveStructure: z.ZodDefault<z.ZodBoolean>;
        includeVisualDescriptions: z.ZodDefault<z.ZodBoolean>;
        extractNumericalData: z.ZodDefault<z.ZodBoolean>;
        combinePages: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        preserveStructure: boolean;
        includeVisualDescriptions: boolean;
        extractNumericalData: boolean;
        combinePages: boolean;
    }, {
        preserveStructure?: boolean | undefined;
        includeVisualDescriptions?: boolean | undefined;
        extractNumericalData?: boolean | undefined;
        combinePages?: boolean | undefined;
    }>>;
    imageOptions: z.ZodDefault<z.ZodObject<{
        format: z.ZodDefault<z.ZodEnum<["png", "jpeg"]>>;
        quality: z.ZodDefault<z.ZodNumber>;
        dpi: z.ZodDefault<z.ZodNumber>;
        pages: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
    }, "strip", z.ZodTypeAny, {
        format: "png" | "jpeg";
        quality: number;
        dpi: number;
        pages?: number[] | undefined;
    }, {
        format?: "png" | "jpeg" | undefined;
        quality?: number | undefined;
        dpi?: number | undefined;
        pages?: number[] | undefined;
    }>>;
    aiOptions: z.ZodDefault<z.ZodObject<{
        model: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
        temperature: z.ZodDefault<z.ZodNumber>;
        maxTokens: z.ZodDefault<z.ZodNumber>;
        jsonMode: z.ZodDefault<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature: number;
        maxTokens: number;
        jsonMode: boolean;
    }, {
        model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
        jsonMode?: boolean | undefined;
    }>>;
    storageOptions: z.ZodOptional<z.ZodObject<{
        uploadImages: z.ZodDefault<z.ZodBoolean>;
        bucketName: z.ZodOptional<z.ZodString>;
        pageImageUrls: z.ZodOptional<z.ZodArray<z.ZodObject<{
            pageNumber: z.ZodNumber;
            uploadUrl: z.ZodString;
            fileName: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            fileName: string;
            uploadUrl: string;
            pageNumber: number;
        }, {
            fileName: string;
            uploadUrl: string;
            pageNumber: number;
        }>, "many">>;
        userId: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        uploadImages: boolean;
        bucketName?: string | undefined;
        userId?: string | undefined;
        pageImageUrls?: {
            fileName: string;
            uploadUrl: string;
            pageNumber: number;
        }[] | undefined;
    }, {
        bucketName?: string | undefined;
        userId?: string | undefined;
        uploadImages?: boolean | undefined;
        pageImageUrls?: {
            fileName: string;
            uploadUrl: string;
            pageNumber: number;
        }[] | undefined;
    }>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    imageOptions: {
        format: "png" | "jpeg";
        quality: number;
        dpi: number;
        pages?: number[] | undefined;
    };
    aiOptions: {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature: number;
        maxTokens: number;
        jsonMode: boolean;
    };
    documentData: string;
    documentType: "image" | "pdf";
    isFileUrl: boolean;
    conversionOptions: {
        preserveStructure: boolean;
        includeVisualDescriptions: boolean;
        extractNumericalData: boolean;
        combinePages: boolean;
    };
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    storageOptions?: {
        uploadImages: boolean;
        bucketName?: string | undefined;
        userId?: string | undefined;
        pageImageUrls?: {
            fileName: string;
            uploadUrl: string;
            pageNumber: number;
        }[] | undefined;
    } | undefined;
}, {
    documentData: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    imageOptions?: {
        format?: "png" | "jpeg" | undefined;
        quality?: number | undefined;
        dpi?: number | undefined;
        pages?: number[] | undefined;
    } | undefined;
    aiOptions?: {
        model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
        jsonMode?: boolean | undefined;
    } | undefined;
    documentType?: "image" | "pdf" | undefined;
    isFileUrl?: boolean | undefined;
    conversionOptions?: {
        preserveStructure?: boolean | undefined;
        includeVisualDescriptions?: boolean | undefined;
        extractNumericalData?: boolean | undefined;
        combinePages?: boolean | undefined;
    } | undefined;
    storageOptions?: {
        bucketName?: string | undefined;
        userId?: string | undefined;
        uploadImages?: boolean | undefined;
        pageImageUrls?: {
            fileName: string;
            uploadUrl: string;
            pageNumber: number;
        }[] | undefined;
    } | undefined;
}>;
/**
 * Result schema for Parse Document workflow
 */
declare const ParseDocumentWorkflowResultSchema: z.ZodObject<{
    markdown: z.ZodString;
    pages: z.ZodArray<z.ZodObject<{
        pageNumber: z.ZodNumber;
        markdown: z.ZodString;
        hasCharts: z.ZodBoolean;
        hasTables: z.ZodBoolean;
        hasImages: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        pageNumber: number;
        markdown: string;
        hasCharts: boolean;
        hasTables: boolean;
        hasImages: boolean;
    }, {
        pageNumber: number;
        markdown: string;
        hasCharts: boolean;
        hasTables: boolean;
        hasImages: boolean;
    }>, "many">;
    metadata: z.ZodObject<{
        totalPages: z.ZodNumber;
        processedPages: z.ZodNumber;
        hasVisualElements: z.ZodBoolean;
        processingTime: z.ZodNumber;
        imageFormat: z.ZodString;
        imageDpi: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        totalPages: number;
        processingTime: number;
        processedPages: number;
        hasVisualElements: boolean;
        imageFormat: string;
        imageDpi: number;
    }, {
        totalPages: number;
        processingTime: number;
        processedPages: number;
        hasVisualElements: boolean;
        imageFormat: string;
        imageDpi: number;
    }>;
    conversionSummary: z.ZodObject<{
        totalCharacters: z.ZodNumber;
        tablesExtracted: z.ZodNumber;
        chartsDescribed: z.ZodNumber;
        imagesDescribed: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        totalCharacters: number;
        tablesExtracted: number;
        chartsDescribed: number;
        imagesDescribed: number;
    }, {
        totalCharacters: number;
        tablesExtracted: number;
        chartsDescribed: number;
        imagesDescribed: number;
    }>;
    aiAnalysis: z.ZodObject<{
        model: z.ZodString;
        iterations: z.ZodNumber;
        processingTime: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        model: string;
        iterations: number;
        processingTime: number;
    }, {
        model: string;
        iterations: number;
        processingTime: number;
    }>;
    uploadedImages: z.ZodOptional<z.ZodArray<z.ZodObject<{
        pageNumber: z.ZodNumber;
        fileName: z.ZodString;
        fileUrl: z.ZodOptional<z.ZodString>;
        uploaded: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        fileName: string;
        pageNumber: number;
        uploaded: boolean;
        fileUrl?: string | undefined;
    }, {
        fileName: string;
        pageNumber: number;
        uploaded: boolean;
        fileUrl?: string | undefined;
    }>, "many">>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    metadata: {
        totalPages: number;
        processingTime: number;
        processedPages: number;
        hasVisualElements: boolean;
        imageFormat: string;
        imageDpi: number;
    };
    pages: {
        pageNumber: number;
        markdown: string;
        hasCharts: boolean;
        hasTables: boolean;
        hasImages: boolean;
    }[];
    markdown: string;
    aiAnalysis: {
        model: string;
        iterations: number;
        processingTime: number;
    };
    conversionSummary: {
        totalCharacters: number;
        tablesExtracted: number;
        chartsDescribed: number;
        imagesDescribed: number;
    };
    uploadedImages?: {
        fileName: string;
        pageNumber: number;
        uploaded: boolean;
        fileUrl?: string | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    metadata: {
        totalPages: number;
        processingTime: number;
        processedPages: number;
        hasVisualElements: boolean;
        imageFormat: string;
        imageDpi: number;
    };
    pages: {
        pageNumber: number;
        markdown: string;
        hasCharts: boolean;
        hasTables: boolean;
        hasImages: boolean;
    }[];
    markdown: string;
    aiAnalysis: {
        model: string;
        iterations: number;
        processingTime: number;
    };
    conversionSummary: {
        totalCharacters: number;
        tablesExtracted: number;
        chartsDescribed: number;
        imagesDescribed: number;
    };
    uploadedImages?: {
        fileName: string;
        pageNumber: number;
        uploaded: boolean;
        fileUrl?: string | undefined;
    }[] | undefined;
}>;
type ParseDocumentWorkflowParams = z.input<typeof ParseDocumentWorkflowParamsSchema>;
type ParseDocumentWorkflowResult = z.output<typeof ParseDocumentWorkflowResultSchema>;
/**
 * Parse Document Workflow
 * Converts PDFs and images to structured markdown using AI vision analysis
 */
export declare class ParseDocumentWorkflow extends WorkflowBubble<ParseDocumentWorkflowParams, ParseDocumentWorkflowResult> {
    static readonly type: "workflow";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        documentData: z.ZodString;
        documentType: z.ZodDefault<z.ZodEnum<["pdf", "image"]>>;
        isFileUrl: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        conversionOptions: z.ZodDefault<z.ZodObject<{
            preserveStructure: z.ZodDefault<z.ZodBoolean>;
            includeVisualDescriptions: z.ZodDefault<z.ZodBoolean>;
            extractNumericalData: z.ZodDefault<z.ZodBoolean>;
            combinePages: z.ZodDefault<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            preserveStructure: boolean;
            includeVisualDescriptions: boolean;
            extractNumericalData: boolean;
            combinePages: boolean;
        }, {
            preserveStructure?: boolean | undefined;
            includeVisualDescriptions?: boolean | undefined;
            extractNumericalData?: boolean | undefined;
            combinePages?: boolean | undefined;
        }>>;
        imageOptions: z.ZodDefault<z.ZodObject<{
            format: z.ZodDefault<z.ZodEnum<["png", "jpeg"]>>;
            quality: z.ZodDefault<z.ZodNumber>;
            dpi: z.ZodDefault<z.ZodNumber>;
            pages: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
        }, "strip", z.ZodTypeAny, {
            format: "png" | "jpeg";
            quality: number;
            dpi: number;
            pages?: number[] | undefined;
        }, {
            format?: "png" | "jpeg" | undefined;
            quality?: number | undefined;
            dpi?: number | undefined;
            pages?: number[] | undefined;
        }>>;
        aiOptions: z.ZodDefault<z.ZodObject<{
            model: z.ZodDefault<z.ZodEnum<["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-4", "openai/gpt-4-turbo", "openai/gpt-3.5-turbo", "openai/gpt-4o", "google/gemini-2.0-flash-exp", "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-image-preview", "google/gemini-3-pro-preview", "google/gemini-3-pro-image-preview", "google/gemini-3-flash-preview", "anthropic/claude-sonnet-4-5", "anthropic/claude-opus-4-5", "anthropic/claude-opus-4.5", "anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-4-20250514", "anthropic/claude-3-5-sonnet-20241022", "openrouter/x-ai/grok-code-fast-1", "openrouter/x-ai/grok-4.1-fast", "openrouter/z-ai/glm-4.6", "openrouter/anthropic/claude-sonnet-4.5", "openrouter/google/gemini-3-pro-preview", "openrouter/morph/morph-v3-large", "openrouter/openai/gpt-oss-120b", "openrouter/deepseek/deepseek-chat-v3.1", "deepseek/deepseek-chat"]>>;
            temperature: z.ZodDefault<z.ZodNumber>;
            maxTokens: z.ZodDefault<z.ZodNumber>;
            jsonMode: z.ZodDefault<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature: number;
            maxTokens: number;
            jsonMode: boolean;
        }, {
            model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            jsonMode?: boolean | undefined;
        }>>;
        storageOptions: z.ZodOptional<z.ZodObject<{
            uploadImages: z.ZodDefault<z.ZodBoolean>;
            bucketName: z.ZodOptional<z.ZodString>;
            pageImageUrls: z.ZodOptional<z.ZodArray<z.ZodObject<{
                pageNumber: z.ZodNumber;
                uploadUrl: z.ZodString;
                fileName: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                fileName: string;
                uploadUrl: string;
                pageNumber: number;
            }, {
                fileName: string;
                uploadUrl: string;
                pageNumber: number;
            }>, "many">>;
            userId: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            uploadImages: boolean;
            bucketName?: string | undefined;
            userId?: string | undefined;
            pageImageUrls?: {
                fileName: string;
                uploadUrl: string;
                pageNumber: number;
            }[] | undefined;
        }, {
            bucketName?: string | undefined;
            userId?: string | undefined;
            uploadImages?: boolean | undefined;
            pageImageUrls?: {
                fileName: string;
                uploadUrl: string;
                pageNumber: number;
            }[] | undefined;
        }>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        imageOptions: {
            format: "png" | "jpeg";
            quality: number;
            dpi: number;
            pages?: number[] | undefined;
        };
        aiOptions: {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature: number;
            maxTokens: number;
            jsonMode: boolean;
        };
        documentData: string;
        documentType: "image" | "pdf";
        isFileUrl: boolean;
        conversionOptions: {
            preserveStructure: boolean;
            includeVisualDescriptions: boolean;
            extractNumericalData: boolean;
            combinePages: boolean;
        };
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        storageOptions?: {
            uploadImages: boolean;
            bucketName?: string | undefined;
            userId?: string | undefined;
            pageImageUrls?: {
                fileName: string;
                uploadUrl: string;
                pageNumber: number;
            }[] | undefined;
        } | undefined;
    }, {
        documentData: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        imageOptions?: {
            format?: "png" | "jpeg" | undefined;
            quality?: number | undefined;
            dpi?: number | undefined;
            pages?: number[] | undefined;
        } | undefined;
        aiOptions?: {
            model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            jsonMode?: boolean | undefined;
        } | undefined;
        documentType?: "image" | "pdf" | undefined;
        isFileUrl?: boolean | undefined;
        conversionOptions?: {
            preserveStructure?: boolean | undefined;
            includeVisualDescriptions?: boolean | undefined;
            extractNumericalData?: boolean | undefined;
            combinePages?: boolean | undefined;
        } | undefined;
        storageOptions?: {
            bucketName?: string | undefined;
            userId?: string | undefined;
            uploadImages?: boolean | undefined;
            pageImageUrls?: {
                fileName: string;
                uploadUrl: string;
                pageNumber: number;
            }[] | undefined;
        } | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        markdown: z.ZodString;
        pages: z.ZodArray<z.ZodObject<{
            pageNumber: z.ZodNumber;
            markdown: z.ZodString;
            hasCharts: z.ZodBoolean;
            hasTables: z.ZodBoolean;
            hasImages: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            pageNumber: number;
            markdown: string;
            hasCharts: boolean;
            hasTables: boolean;
            hasImages: boolean;
        }, {
            pageNumber: number;
            markdown: string;
            hasCharts: boolean;
            hasTables: boolean;
            hasImages: boolean;
        }>, "many">;
        metadata: z.ZodObject<{
            totalPages: z.ZodNumber;
            processedPages: z.ZodNumber;
            hasVisualElements: z.ZodBoolean;
            processingTime: z.ZodNumber;
            imageFormat: z.ZodString;
            imageDpi: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            totalPages: number;
            processingTime: number;
            processedPages: number;
            hasVisualElements: boolean;
            imageFormat: string;
            imageDpi: number;
        }, {
            totalPages: number;
            processingTime: number;
            processedPages: number;
            hasVisualElements: boolean;
            imageFormat: string;
            imageDpi: number;
        }>;
        conversionSummary: z.ZodObject<{
            totalCharacters: z.ZodNumber;
            tablesExtracted: z.ZodNumber;
            chartsDescribed: z.ZodNumber;
            imagesDescribed: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            totalCharacters: number;
            tablesExtracted: number;
            chartsDescribed: number;
            imagesDescribed: number;
        }, {
            totalCharacters: number;
            tablesExtracted: number;
            chartsDescribed: number;
            imagesDescribed: number;
        }>;
        aiAnalysis: z.ZodObject<{
            model: z.ZodString;
            iterations: z.ZodNumber;
            processingTime: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            model: string;
            iterations: number;
            processingTime: number;
        }, {
            model: string;
            iterations: number;
            processingTime: number;
        }>;
        uploadedImages: z.ZodOptional<z.ZodArray<z.ZodObject<{
            pageNumber: z.ZodNumber;
            fileName: z.ZodString;
            fileUrl: z.ZodOptional<z.ZodString>;
            uploaded: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            fileName: string;
            pageNumber: number;
            uploaded: boolean;
            fileUrl?: string | undefined;
        }, {
            fileName: string;
            pageNumber: number;
            uploaded: boolean;
            fileUrl?: string | undefined;
        }>, "many">>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        metadata: {
            totalPages: number;
            processingTime: number;
            processedPages: number;
            hasVisualElements: boolean;
            imageFormat: string;
            imageDpi: number;
        };
        pages: {
            pageNumber: number;
            markdown: string;
            hasCharts: boolean;
            hasTables: boolean;
            hasImages: boolean;
        }[];
        markdown: string;
        aiAnalysis: {
            model: string;
            iterations: number;
            processingTime: number;
        };
        conversionSummary: {
            totalCharacters: number;
            tablesExtracted: number;
            chartsDescribed: number;
            imagesDescribed: number;
        };
        uploadedImages?: {
            fileName: string;
            pageNumber: number;
            uploaded: boolean;
            fileUrl?: string | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        metadata: {
            totalPages: number;
            processingTime: number;
            processedPages: number;
            hasVisualElements: boolean;
            imageFormat: string;
            imageDpi: number;
        };
        pages: {
            pageNumber: number;
            markdown: string;
            hasCharts: boolean;
            hasTables: boolean;
            hasImages: boolean;
        }[];
        markdown: string;
        aiAnalysis: {
            model: string;
            iterations: number;
            processingTime: number;
        };
        conversionSummary: {
            totalCharacters: number;
            tablesExtracted: number;
            chartsDescribed: number;
            imagesDescribed: number;
        };
        uploadedImages?: {
            fileName: string;
            pageNumber: number;
            uploaded: boolean;
            fileUrl?: string | undefined;
        }[] | undefined;
    }>;
    static readonly shortDescription = "Parse Document workflow: convert PDFs/images to markdown using AI vision";
    static readonly longDescription = "\n    Comprehensive document parsing workflow that converts PDFs and images into structured markdown:\n    \n    **Process:**\n    1. Convert PDFs to high-quality images (if needed)\n    2. Analyze images using AI vision models to extract content\n    3. Generate clean, structured markdown preserving document layout\n    4. Describe charts, tables, and images with numerical data when possible\n    \n    **Features:**\n    - **PDF & Image Support**: Handles both PDF documents and image files\n    - **Structure Preservation**: Maintains headers, lists, paragraphs, and formatting\n    - **Visual Element Analysis**: Describes charts, graphs, tables, and images in detail\n    - **Numerical Data Extraction**: Extracts specific values from charts and tables\n    - **High-Quality Conversion**: Configurable DPI and quality settings\n    - **Per-Page Analysis**: Detailed breakdown of each page's content\n    \n    **Visual Element Handling:**\n    - **Charts & Graphs**: Extract data points, trends, axis labels, percentages\n    - **Tables**: Convert to markdown tables with all visible data\n    - **Images & Diagrams**: Detailed descriptions including any visible text/numbers\n    - **Forms**: Structure field names and any filled values\n    \n    **Output Options:**\n    - Combined markdown document or per-page breakdown\n    - Configurable structure preservation and visual descriptions\n    - Comprehensive metadata and conversion statistics\n    \n    **Common Use Cases:**\n    - **Document Digitization**: Convert scanned PDFs to editable markdown\n    - **Report Analysis**: Extract data from business reports and charts\n    - **Academic Papers**: Preserve structure and extract figures/tables\n    - **Technical Documentation**: Maintain formatting and describe diagrams\n    - **Research Materials**: Extract and structure information from various documents\n    \n    **Input**: PDF or image data + conversion preferences\n    **Output**: Clean markdown with preserved structure and visual descriptions\n  ";
    static readonly alias = "parse-doc";
    constructor(params: ParseDocumentWorkflowParams, context?: BubbleContext);
    protected performAction(context?: BubbleContext): Promise<ParseDocumentWorkflowResult>;
}
export {};
//# sourceMappingURL=parse-document.workflow.d.ts.map