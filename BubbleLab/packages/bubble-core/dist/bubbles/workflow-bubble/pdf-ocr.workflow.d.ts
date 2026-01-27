/**
 * PDF OCR WORKFLOW
 *
 * A comprehensive workflow that converts PDF documents to images and passes them
 * to an AI agent along with discovered form fields to parse and extract schema information.
 *
 * This workflow combines:
 * 1. PDF field discovery using pdf-lib
 * 2. PDF to images conversion using pdf-img-convert
 * 3. AI agent analysis for schema parsing and field extraction
 *
 * Returns structured JSON containing field IDs from discovery and extracted field names
 * with their values from AI analysis.
 */
import { z } from 'zod';
import { WorkflowBubble } from '../../types/workflow-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
/**
 * Parameters schema for PDF OCR workflow using discriminated union for different modes
 */
declare const PDFOcrWorkflowParamsSchema: z.ZodDiscriminatedUnion<"mode", [z.ZodObject<{
    mode: z.ZodLiteral<"identify">;
    pdfData: z.ZodString;
    discoveryOptions: z.ZodDefault<z.ZodObject<{
        targetPage: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        targetPage?: number | undefined;
    }, {
        targetPage?: number | undefined;
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
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    mode: "identify";
    pdfData: string;
    discoveryOptions: {
        targetPage?: number | undefined;
    };
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
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    mode: "identify";
    pdfData: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    discoveryOptions?: {
        targetPage?: number | undefined;
    } | undefined;
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
}>, z.ZodObject<{
    mode: z.ZodLiteral<"autofill">;
    pdfData: z.ZodString;
    clientInformation: z.ZodString;
    discoveryOptions: z.ZodDefault<z.ZodObject<{
        targetPage: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        targetPage?: number | undefined;
    }, {
        targetPage?: number | undefined;
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
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    mode: "autofill";
    pdfData: string;
    discoveryOptions: {
        targetPage?: number | undefined;
    };
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
    clientInformation: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    mode: "autofill";
    pdfData: string;
    clientInformation: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    discoveryOptions?: {
        targetPage?: number | undefined;
    } | undefined;
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
}>]>;
/**
 * Result schema for PDF OCR workflow using discriminated union for different modes
 */
declare const PDFOcrWorkflowResultSchema: z.ZodDiscriminatedUnion<"mode", [z.ZodObject<{
    mode: z.ZodLiteral<"identify">;
    extractedFields: z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        fieldName: z.ZodString;
        confidence: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        id: number;
        fieldName: string;
        confidence: number;
    }, {
        id: number;
        fieldName: string;
        confidence: number;
    }>, "many">;
    discoveryData: z.ZodObject<{
        totalFields: z.ZodNumber;
        fieldsWithCoordinates: z.ZodNumber;
        pages: z.ZodArray<z.ZodNumber, "many">;
    }, "strip", z.ZodTypeAny, {
        pages: number[];
        totalFields: number;
        fieldsWithCoordinates: number;
    }, {
        pages: number[];
        totalFields: number;
        fieldsWithCoordinates: number;
    }>;
    imageData: z.ZodObject<{
        totalPages: z.ZodNumber;
        convertedPages: z.ZodNumber;
        format: z.ZodString;
        dpi: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        format: string;
        dpi: number;
        totalPages: number;
        convertedPages: number;
    }, {
        format: string;
        dpi: number;
        totalPages: number;
        convertedPages: number;
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
    mode: "identify";
    imageData: {
        format: string;
        dpi: number;
        totalPages: number;
        convertedPages: number;
    };
    extractedFields: {
        id: number;
        fieldName: string;
        confidence: number;
    }[];
    discoveryData: {
        pages: number[];
        totalFields: number;
        fieldsWithCoordinates: number;
    };
    aiAnalysis: {
        model: string;
        iterations: number;
        processingTime?: number | undefined;
    };
}, {
    error: string;
    success: boolean;
    mode: "identify";
    imageData: {
        format: string;
        dpi: number;
        totalPages: number;
        convertedPages: number;
    };
    extractedFields: {
        id: number;
        fieldName: string;
        confidence: number;
    }[];
    discoveryData: {
        pages: number[];
        totalFields: number;
        fieldsWithCoordinates: number;
    };
    aiAnalysis: {
        model: string;
        iterations: number;
        processingTime?: number | undefined;
    };
}>, z.ZodObject<{
    mode: z.ZodLiteral<"autofill">;
    extractedFields: z.ZodArray<z.ZodObject<{
        id: z.ZodNumber;
        originalFieldName: z.ZodOptional<z.ZodString>;
        fieldName: z.ZodString;
        value: z.ZodString;
        confidence: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        value: string;
        id: number;
        fieldName: string;
        confidence: number;
        originalFieldName?: string | undefined;
    }, {
        value: string;
        id: number;
        fieldName: string;
        confidence: number;
        originalFieldName?: string | undefined;
    }>, "many">;
    filledPdfData: z.ZodString;
    discoveryData: z.ZodObject<{
        totalFields: z.ZodNumber;
        fieldsWithCoordinates: z.ZodNumber;
        pages: z.ZodArray<z.ZodNumber, "many">;
    }, "strip", z.ZodTypeAny, {
        pages: number[];
        totalFields: number;
        fieldsWithCoordinates: number;
    }, {
        pages: number[];
        totalFields: number;
        fieldsWithCoordinates: number;
    }>;
    imageData: z.ZodObject<{
        totalPages: z.ZodNumber;
        convertedPages: z.ZodNumber;
        format: z.ZodString;
        dpi: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        format: string;
        dpi: number;
        totalPages: number;
        convertedPages: number;
    }, {
        format: string;
        dpi: number;
        totalPages: number;
        convertedPages: number;
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
    fillResults: z.ZodObject<{
        filledFields: z.ZodNumber;
        successfullyFilled: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        filledFields: number;
        successfullyFilled: number;
    }, {
        filledFields: number;
        successfullyFilled: number;
    }>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    mode: "autofill";
    filledPdfData: string;
    imageData: {
        format: string;
        dpi: number;
        totalPages: number;
        convertedPages: number;
    };
    extractedFields: {
        value: string;
        id: number;
        fieldName: string;
        confidence: number;
        originalFieldName?: string | undefined;
    }[];
    discoveryData: {
        pages: number[];
        totalFields: number;
        fieldsWithCoordinates: number;
    };
    aiAnalysis: {
        model: string;
        iterations: number;
        processingTime?: number | undefined;
    };
    fillResults: {
        filledFields: number;
        successfullyFilled: number;
    };
}, {
    error: string;
    success: boolean;
    mode: "autofill";
    filledPdfData: string;
    imageData: {
        format: string;
        dpi: number;
        totalPages: number;
        convertedPages: number;
    };
    extractedFields: {
        value: string;
        id: number;
        fieldName: string;
        confidence: number;
        originalFieldName?: string | undefined;
    }[];
    discoveryData: {
        pages: number[];
        totalFields: number;
        fieldsWithCoordinates: number;
    };
    aiAnalysis: {
        model: string;
        iterations: number;
        processingTime?: number | undefined;
    };
    fillResults: {
        filledFields: number;
        successfullyFilled: number;
    };
}>]>;
type PDFOcrWorkflowParams = z.input<typeof PDFOcrWorkflowParamsSchema>;
type PDFOcrWorkflowResult = z.output<typeof PDFOcrWorkflowResultSchema>;
export type PDFOcrModeResult<T extends PDFOcrWorkflowParams['mode']> = Extract<PDFOcrWorkflowResult, {
    mode: T;
}>;
export type PDFOcrOperationResult<T extends PDFOcrWorkflowParams['mode']> = Extract<PDFOcrWorkflowResult, {
    mode: T;
}>;
/**
 * PDF OCR Workflow
 * Combines PDF field discovery, image conversion, and AI analysis for comprehensive form field extraction
 */
export declare class PDFOcrWorkflow<T extends PDFOcrWorkflowParams = PDFOcrWorkflowParams> extends WorkflowBubble<T, Extract<PDFOcrWorkflowResult, {
    mode: T['mode'];
}>> {
    static readonly type: "workflow";
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodDiscriminatedUnion<"mode", [z.ZodObject<{
        mode: z.ZodLiteral<"identify">;
        pdfData: z.ZodString;
        discoveryOptions: z.ZodDefault<z.ZodObject<{
            targetPage: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            targetPage?: number | undefined;
        }, {
            targetPage?: number | undefined;
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
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        mode: "identify";
        pdfData: string;
        discoveryOptions: {
            targetPage?: number | undefined;
        };
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
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        mode: "identify";
        pdfData: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        discoveryOptions?: {
            targetPage?: number | undefined;
        } | undefined;
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
    }>, z.ZodObject<{
        mode: z.ZodLiteral<"autofill">;
        pdfData: z.ZodString;
        clientInformation: z.ZodString;
        discoveryOptions: z.ZodDefault<z.ZodObject<{
            targetPage: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            targetPage?: number | undefined;
        }, {
            targetPage?: number | undefined;
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
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        mode: "autofill";
        pdfData: string;
        discoveryOptions: {
            targetPage?: number | undefined;
        };
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
        clientInformation: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        mode: "autofill";
        pdfData: string;
        clientInformation: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        discoveryOptions?: {
            targetPage?: number | undefined;
        } | undefined;
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
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"mode", [z.ZodObject<{
        mode: z.ZodLiteral<"identify">;
        extractedFields: z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            fieldName: z.ZodString;
            confidence: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            id: number;
            fieldName: string;
            confidence: number;
        }, {
            id: number;
            fieldName: string;
            confidence: number;
        }>, "many">;
        discoveryData: z.ZodObject<{
            totalFields: z.ZodNumber;
            fieldsWithCoordinates: z.ZodNumber;
            pages: z.ZodArray<z.ZodNumber, "many">;
        }, "strip", z.ZodTypeAny, {
            pages: number[];
            totalFields: number;
            fieldsWithCoordinates: number;
        }, {
            pages: number[];
            totalFields: number;
            fieldsWithCoordinates: number;
        }>;
        imageData: z.ZodObject<{
            totalPages: z.ZodNumber;
            convertedPages: z.ZodNumber;
            format: z.ZodString;
            dpi: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            format: string;
            dpi: number;
            totalPages: number;
            convertedPages: number;
        }, {
            format: string;
            dpi: number;
            totalPages: number;
            convertedPages: number;
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
        mode: "identify";
        imageData: {
            format: string;
            dpi: number;
            totalPages: number;
            convertedPages: number;
        };
        extractedFields: {
            id: number;
            fieldName: string;
            confidence: number;
        }[];
        discoveryData: {
            pages: number[];
            totalFields: number;
            fieldsWithCoordinates: number;
        };
        aiAnalysis: {
            model: string;
            iterations: number;
            processingTime?: number | undefined;
        };
    }, {
        error: string;
        success: boolean;
        mode: "identify";
        imageData: {
            format: string;
            dpi: number;
            totalPages: number;
            convertedPages: number;
        };
        extractedFields: {
            id: number;
            fieldName: string;
            confidence: number;
        }[];
        discoveryData: {
            pages: number[];
            totalFields: number;
            fieldsWithCoordinates: number;
        };
        aiAnalysis: {
            model: string;
            iterations: number;
            processingTime?: number | undefined;
        };
    }>, z.ZodObject<{
        mode: z.ZodLiteral<"autofill">;
        extractedFields: z.ZodArray<z.ZodObject<{
            id: z.ZodNumber;
            originalFieldName: z.ZodOptional<z.ZodString>;
            fieldName: z.ZodString;
            value: z.ZodString;
            confidence: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            value: string;
            id: number;
            fieldName: string;
            confidence: number;
            originalFieldName?: string | undefined;
        }, {
            value: string;
            id: number;
            fieldName: string;
            confidence: number;
            originalFieldName?: string | undefined;
        }>, "many">;
        filledPdfData: z.ZodString;
        discoveryData: z.ZodObject<{
            totalFields: z.ZodNumber;
            fieldsWithCoordinates: z.ZodNumber;
            pages: z.ZodArray<z.ZodNumber, "many">;
        }, "strip", z.ZodTypeAny, {
            pages: number[];
            totalFields: number;
            fieldsWithCoordinates: number;
        }, {
            pages: number[];
            totalFields: number;
            fieldsWithCoordinates: number;
        }>;
        imageData: z.ZodObject<{
            totalPages: z.ZodNumber;
            convertedPages: z.ZodNumber;
            format: z.ZodString;
            dpi: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            format: string;
            dpi: number;
            totalPages: number;
            convertedPages: number;
        }, {
            format: string;
            dpi: number;
            totalPages: number;
            convertedPages: number;
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
        fillResults: z.ZodObject<{
            filledFields: z.ZodNumber;
            successfullyFilled: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            filledFields: number;
            successfullyFilled: number;
        }, {
            filledFields: number;
            successfullyFilled: number;
        }>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        mode: "autofill";
        filledPdfData: string;
        imageData: {
            format: string;
            dpi: number;
            totalPages: number;
            convertedPages: number;
        };
        extractedFields: {
            value: string;
            id: number;
            fieldName: string;
            confidence: number;
            originalFieldName?: string | undefined;
        }[];
        discoveryData: {
            pages: number[];
            totalFields: number;
            fieldsWithCoordinates: number;
        };
        aiAnalysis: {
            model: string;
            iterations: number;
            processingTime?: number | undefined;
        };
        fillResults: {
            filledFields: number;
            successfullyFilled: number;
        };
    }, {
        error: string;
        success: boolean;
        mode: "autofill";
        filledPdfData: string;
        imageData: {
            format: string;
            dpi: number;
            totalPages: number;
            convertedPages: number;
        };
        extractedFields: {
            value: string;
            id: number;
            fieldName: string;
            confidence: number;
            originalFieldName?: string | undefined;
        }[];
        discoveryData: {
            pages: number[];
            totalFields: number;
            fieldsWithCoordinates: number;
        };
        aiAnalysis: {
            model: string;
            iterations: number;
            processingTime?: number | undefined;
        };
        fillResults: {
            filledFields: number;
            successfullyFilled: number;
        };
    }>]>;
    static readonly shortDescription = "PDF OCR workflow: identify fields or autofill forms using AI analysis";
    static readonly longDescription = "\n    Comprehensive PDF OCR workflow with two modes for form field processing:\n    \n    **Identify Mode:**\n    - Discovers and names form fields from PDF documents\n    - Returns field IDs, descriptive names, and confidence scores\n    - Useful for form schema generation and document understanding\n    \n    **Autofill Mode:**\n    - Identifies form fields AND fills them using provided client information\n    - Returns field data with values plus a filled PDF\n    - Uses AI to intelligently map client data to appropriate form fields\n    \n    Process:\n    1. Discover form fields using PyMuPDF (field names, types, coordinates)\n    2. Convert PDF pages to high-quality images using PyMuPDF\n    3. Send images + discovery data + client info (autofill mode) to AI agent\n    4. For autofill mode: Use PDF Form Operations to fill the form with AI-determined values\n    \n    Features:\n    - Two distinct modes: identify vs autofill\n    - Cross-references visual analysis with form field metadata\n    - Supports both fillable PDFs and scanned documents\n    - Generates meaningful field names based on PDF content and context\n    - Intelligent value mapping from client information (autofill mode)\n    - Configurable image quality and AI model selection\n    - Returns confidence scores for field identification accuracy\n    \n    Use cases:\n    - **Identify**: Form schema generation, document structure analysis\n    - **Autofill**: Automated form filling, client onboarding, data entry automation\n    \n    Input: Base64 encoded PDF data + mode + client information (autofill mode)\n    Output: Mode-specific results with field data and optional filled PDF\n  ";
    static readonly alias = "pdf-ocr";
    constructor(params: T, context?: BubbleContext);
    protected performAction(): Promise<Extract<PDFOcrWorkflowResult, {
        mode: T['mode'];
    }>>;
}
export {};
//# sourceMappingURL=pdf-ocr.workflow.d.ts.map