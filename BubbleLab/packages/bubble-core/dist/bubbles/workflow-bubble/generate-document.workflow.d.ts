/**
 * GENERATE DOCUMENT WORKFLOW
 *
 * A comprehensive workflow that converts markdown documents into structured formats
 * like HTML tables, CSV, or Excel files. Uses AI analysis to extract and organize
 * data from unstructured markdown text into tabular formats.
 *
 * This workflow combines:
 * 1. AI agent analysis for data extraction from markdown documents
 * 2. JSON schema generation for structured data representation
 * 3. Format conversion to HTML, CSV, or Excel
 *
 * Returns structured data and downloadable files in the requested format.
 */
import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Parameters schema for Generate Document workflow
 */
declare const GenerateDocumentWorkflowParamsSchema: z.ZodObject<{
    documents: z.ZodArray<z.ZodObject<{
        content: z.ZodString;
        index: z.ZodNumber;
        metadata: z.ZodOptional<z.ZodObject<{
            originalFilename: z.ZodOptional<z.ZodString>;
            pageCount: z.ZodOptional<z.ZodNumber>;
            uploadedImages: z.ZodOptional<z.ZodArray<z.ZodObject<{
                pageNumber: z.ZodNumber;
                fileName: z.ZodString;
                fileUrl: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                fileName: string;
                pageNumber: number;
                fileUrl?: string | undefined;
            }, {
                fileName: string;
                pageNumber: number;
                fileUrl?: string | undefined;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            pageCount?: number | undefined;
            originalFilename?: string | undefined;
            uploadedImages?: {
                fileName: string;
                pageNumber: number;
                fileUrl?: string | undefined;
            }[] | undefined;
        }, {
            pageCount?: number | undefined;
            originalFilename?: string | undefined;
            uploadedImages?: {
                fileName: string;
                pageNumber: number;
                fileUrl?: string | undefined;
            }[] | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        content: string;
        index: number;
        metadata?: {
            pageCount?: number | undefined;
            originalFilename?: string | undefined;
            uploadedImages?: {
                fileName: string;
                pageNumber: number;
                fileUrl?: string | undefined;
            }[] | undefined;
        } | undefined;
    }, {
        content: string;
        index: number;
        metadata?: {
            pageCount?: number | undefined;
            originalFilename?: string | undefined;
            uploadedImages?: {
                fileName: string;
                pageNumber: number;
                fileUrl?: string | undefined;
            }[] | undefined;
        } | undefined;
    }>, "many">;
    outputDescription: z.ZodString;
    outputFormat: z.ZodDefault<z.ZodEnum<["html", "csv", "json"]>>;
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
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    aiOptions: {
        model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
        temperature: number;
        maxTokens: number;
        jsonMode: boolean;
    };
    documents: {
        content: string;
        index: number;
        metadata?: {
            pageCount?: number | undefined;
            originalFilename?: string | undefined;
            uploadedImages?: {
                fileName: string;
                pageNumber: number;
                fileUrl?: string | undefined;
            }[] | undefined;
        } | undefined;
    }[];
    outputDescription: string;
    outputFormat: "html" | "json" | "csv";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    documents: {
        content: string;
        index: number;
        metadata?: {
            pageCount?: number | undefined;
            originalFilename?: string | undefined;
            uploadedImages?: {
                fileName: string;
                pageNumber: number;
                fileUrl?: string | undefined;
            }[] | undefined;
        } | undefined;
    }[];
    outputDescription: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    aiOptions?: {
        model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
        temperature?: number | undefined;
        maxTokens?: number | undefined;
        jsonMode?: boolean | undefined;
    } | undefined;
    outputFormat?: "html" | "json" | "csv" | undefined;
}>;
/**
 * Result schema for Generate Document workflow
 */
declare const GenerateDocumentWorkflowResultSchema: z.ZodObject<{
    columns: z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        type: z.ZodEnum<["string", "number", "integer", "float", "date", "boolean"]>;
        description: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        type: "string" | "number" | "boolean" | "integer" | "float" | "date";
        description: string;
        name: string;
    }, {
        type: "string" | "number" | "boolean" | "integer" | "float" | "date";
        description: string;
        name: string;
    }>, "many">;
    rows: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodNull]>>, "many">;
    metadata: z.ZodObject<{
        totalDocuments: z.ZodNumber;
        totalRows: z.ZodNumber;
        totalColumns: z.ZodNumber;
        processingTime: z.ZodNumber;
        extractedFrom: z.ZodArray<z.ZodString, "many">;
    }, "strip", z.ZodTypeAny, {
        processingTime: number;
        totalDocuments: number;
        totalRows: number;
        totalColumns: number;
        extractedFrom: string[];
    }, {
        processingTime: number;
        totalDocuments: number;
        totalRows: number;
        totalColumns: number;
        extractedFrom: string[];
    }>;
    generatedFiles: z.ZodObject<{
        html: z.ZodOptional<z.ZodString>;
        csv: z.ZodOptional<z.ZodString>;
        json: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        html?: string | undefined;
        json?: string | undefined;
        csv?: string | undefined;
    }, {
        html?: string | undefined;
        json?: string | undefined;
        csv?: string | undefined;
    }>;
    aiAnalysis: z.ZodObject<{
        model: z.ZodString;
        iterations: z.ZodNumber;
        processingTime: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        model: string;
        iterations: number;
        processingTime?: number | undefined;
    }, {
        model: string;
        iterations: number;
        processingTime?: number | undefined;
    }>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    rows: Record<string, string | number | boolean | null>[];
    metadata: {
        processingTime: number;
        totalDocuments: number;
        totalRows: number;
        totalColumns: number;
        extractedFrom: string[];
    };
    aiAnalysis: {
        model: string;
        iterations: number;
        processingTime?: number | undefined;
    };
    columns: {
        type: "string" | "number" | "boolean" | "integer" | "float" | "date";
        description: string;
        name: string;
    }[];
    generatedFiles: {
        html?: string | undefined;
        json?: string | undefined;
        csv?: string | undefined;
    };
}, {
    error: string;
    success: boolean;
    rows: Record<string, string | number | boolean | null>[];
    metadata: {
        processingTime: number;
        totalDocuments: number;
        totalRows: number;
        totalColumns: number;
        extractedFrom: string[];
    };
    aiAnalysis: {
        model: string;
        iterations: number;
        processingTime?: number | undefined;
    };
    columns: {
        type: "string" | "number" | "boolean" | "integer" | "float" | "date";
        description: string;
        name: string;
    }[];
    generatedFiles: {
        html?: string | undefined;
        json?: string | undefined;
        csv?: string | undefined;
    };
}>;
type GenerateDocumentWorkflowParams = z.input<typeof GenerateDocumentWorkflowParamsSchema>;
type GenerateDocumentWorkflowResult = z.output<typeof GenerateDocumentWorkflowResultSchema>;
/**
 * Generate Document Workflow
 * Converts markdown documents into structured formats using AI analysis
 */
export declare class GenerateDocumentWorkflow extends WorkflowBubble<GenerateDocumentWorkflowParams, GenerateDocumentWorkflowResult> {
    static readonly type: "workflow";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        documents: z.ZodArray<z.ZodObject<{
            content: z.ZodString;
            index: z.ZodNumber;
            metadata: z.ZodOptional<z.ZodObject<{
                originalFilename: z.ZodOptional<z.ZodString>;
                pageCount: z.ZodOptional<z.ZodNumber>;
                uploadedImages: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    pageNumber: z.ZodNumber;
                    fileName: z.ZodString;
                    fileUrl: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    fileName: string;
                    pageNumber: number;
                    fileUrl?: string | undefined;
                }, {
                    fileName: string;
                    pageNumber: number;
                    fileUrl?: string | undefined;
                }>, "many">>;
            }, "strip", z.ZodTypeAny, {
                pageCount?: number | undefined;
                originalFilename?: string | undefined;
                uploadedImages?: {
                    fileName: string;
                    pageNumber: number;
                    fileUrl?: string | undefined;
                }[] | undefined;
            }, {
                pageCount?: number | undefined;
                originalFilename?: string | undefined;
                uploadedImages?: {
                    fileName: string;
                    pageNumber: number;
                    fileUrl?: string | undefined;
                }[] | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            content: string;
            index: number;
            metadata?: {
                pageCount?: number | undefined;
                originalFilename?: string | undefined;
                uploadedImages?: {
                    fileName: string;
                    pageNumber: number;
                    fileUrl?: string | undefined;
                }[] | undefined;
            } | undefined;
        }, {
            content: string;
            index: number;
            metadata?: {
                pageCount?: number | undefined;
                originalFilename?: string | undefined;
                uploadedImages?: {
                    fileName: string;
                    pageNumber: number;
                    fileUrl?: string | undefined;
                }[] | undefined;
            } | undefined;
        }>, "many">;
        outputDescription: z.ZodString;
        outputFormat: z.ZodDefault<z.ZodEnum<["html", "csv", "json"]>>;
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
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        aiOptions: {
            model: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat";
            temperature: number;
            maxTokens: number;
            jsonMode: boolean;
        };
        documents: {
            content: string;
            index: number;
            metadata?: {
                pageCount?: number | undefined;
                originalFilename?: string | undefined;
                uploadedImages?: {
                    fileName: string;
                    pageNumber: number;
                    fileUrl?: string | undefined;
                }[] | undefined;
            } | undefined;
        }[];
        outputDescription: string;
        outputFormat: "html" | "json" | "csv";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        documents: {
            content: string;
            index: number;
            metadata?: {
                pageCount?: number | undefined;
                originalFilename?: string | undefined;
                uploadedImages?: {
                    fileName: string;
                    pageNumber: number;
                    fileUrl?: string | undefined;
                }[] | undefined;
            } | undefined;
        }[];
        outputDescription: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        aiOptions?: {
            model?: "openai/gpt-5" | "openai/gpt-5-mini" | "openai/gpt-5.1" | "openai/gpt-5.2" | "openai/gpt-4" | "openai/gpt-4-turbo" | "openai/gpt-3.5-turbo" | "openai/gpt-4o" | "google/gemini-2.0-flash-exp" | "google/gemini-2.5-pro" | "google/gemini-2.5-flash" | "google/gemini-2.5-flash-lite" | "google/gemini-2.5-flash-image-preview" | "google/gemini-3-pro-preview" | "google/gemini-3-pro-image-preview" | "google/gemini-3-flash-preview" | "anthropic/claude-sonnet-4-5" | "anthropic/claude-opus-4-5" | "anthropic/claude-opus-4.5" | "anthropic/claude-haiku-4-5" | "anthropic/claude-sonnet-4-20250514" | "anthropic/claude-3-5-sonnet-20241022" | "openrouter/x-ai/grok-code-fast-1" | "openrouter/x-ai/grok-4.1-fast" | "openrouter/z-ai/glm-4.6" | "openrouter/anthropic/claude-sonnet-4.5" | "openrouter/google/gemini-3-pro-preview" | "openrouter/morph/morph-v3-large" | "openrouter/openai/gpt-oss-120b" | "openrouter/deepseek/deepseek-chat-v3.1" | "deepseek/deepseek-chat" | undefined;
            temperature?: number | undefined;
            maxTokens?: number | undefined;
            jsonMode?: boolean | undefined;
        } | undefined;
        outputFormat?: "html" | "json" | "csv" | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        columns: z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            type: z.ZodEnum<["string", "number", "integer", "float", "date", "boolean"]>;
            description: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type: "string" | "number" | "boolean" | "integer" | "float" | "date";
            description: string;
            name: string;
        }, {
            type: "string" | "number" | "boolean" | "integer" | "float" | "date";
            description: string;
            name: string;
        }>, "many">;
        rows: z.ZodArray<z.ZodRecord<z.ZodString, z.ZodUnion<[z.ZodString, z.ZodNumber, z.ZodBoolean, z.ZodNull]>>, "many">;
        metadata: z.ZodObject<{
            totalDocuments: z.ZodNumber;
            totalRows: z.ZodNumber;
            totalColumns: z.ZodNumber;
            processingTime: z.ZodNumber;
            extractedFrom: z.ZodArray<z.ZodString, "many">;
        }, "strip", z.ZodTypeAny, {
            processingTime: number;
            totalDocuments: number;
            totalRows: number;
            totalColumns: number;
            extractedFrom: string[];
        }, {
            processingTime: number;
            totalDocuments: number;
            totalRows: number;
            totalColumns: number;
            extractedFrom: string[];
        }>;
        generatedFiles: z.ZodObject<{
            html: z.ZodOptional<z.ZodString>;
            csv: z.ZodOptional<z.ZodString>;
            json: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            html?: string | undefined;
            json?: string | undefined;
            csv?: string | undefined;
        }, {
            html?: string | undefined;
            json?: string | undefined;
            csv?: string | undefined;
        }>;
        aiAnalysis: z.ZodObject<{
            model: z.ZodString;
            iterations: z.ZodNumber;
            processingTime: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            model: string;
            iterations: number;
            processingTime?: number | undefined;
        }, {
            model: string;
            iterations: number;
            processingTime?: number | undefined;
        }>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        rows: Record<string, string | number | boolean | null>[];
        metadata: {
            processingTime: number;
            totalDocuments: number;
            totalRows: number;
            totalColumns: number;
            extractedFrom: string[];
        };
        aiAnalysis: {
            model: string;
            iterations: number;
            processingTime?: number | undefined;
        };
        columns: {
            type: "string" | "number" | "boolean" | "integer" | "float" | "date";
            description: string;
            name: string;
        }[];
        generatedFiles: {
            html?: string | undefined;
            json?: string | undefined;
            csv?: string | undefined;
        };
    }, {
        error: string;
        success: boolean;
        rows: Record<string, string | number | boolean | null>[];
        metadata: {
            processingTime: number;
            totalDocuments: number;
            totalRows: number;
            totalColumns: number;
            extractedFrom: string[];
        };
        aiAnalysis: {
            model: string;
            iterations: number;
            processingTime?: number | undefined;
        };
        columns: {
            type: "string" | "number" | "boolean" | "integer" | "float" | "date";
            description: string;
            name: string;
        }[];
        generatedFiles: {
            html?: string | undefined;
            json?: string | undefined;
            csv?: string | undefined;
        };
    }>;
    static readonly shortDescription = "Generate Document workflow: convert markdown to structured formats using AI";
    static readonly longDescription = "\n    Comprehensive document generation workflow that transforms unstructured markdown content into structured data formats:\n    \n    **Process:**\n    1. Analyze markdown documents using AI to understand content and structure\n    2. Extract data points based on user requirements and output description\n    3. Generate consistent column definitions and data rows\n    4. Convert to requested format (HTML table, CSV, JSON)\n    \n    **Features:**\n    - Multi-document processing with content consolidation\n    - AI-powered data extraction and structuring\n    - Flexible output format support (HTML, CSV, JSON)\n    - Configurable AI model selection and parameters\n    - Comprehensive metadata and analysis tracking\n    - Error handling and validation\n    \n    **Common Use Cases:**\n    - **Expense Management**: Extract vendor, amount, date, category from receipts\n    - **Invoice Processing**: Structure billing information into tables\n    - **Contact Lists**: Organize people and contact information\n    - **Inventory Management**: Extract product details and quantities\n    - **Research Data**: Structure findings and references\n    \n    **Input**: Array of markdown documents + output requirements description\n    **Output**: Structured data in requested format with metadata and analysis\n  ";
    static readonly alias = "generate-doc";
    constructor(params: GenerateDocumentWorkflowParams, context?: BubbleContext);
    protected performAction(): Promise<GenerateDocumentWorkflowResult>;
    /**
     * Generate CSV content from columns and rows
     */
    private generateCSV;
    /**
     * Normalize column type to match schema expectations
     */
    private normalizeColumnType;
    /**
     * Normalize row value to ensure it's a primitive type
     */
    private normalizeRowValue;
    /**
     * Generate HTML table from columns and rows
     */
    private generateHTML;
}
export {};
//# sourceMappingURL=generate-document.workflow.d.ts.map