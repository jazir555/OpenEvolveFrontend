import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const WebExtractToolParamsSchema: z.ZodObject<{
    url: z.ZodString;
    prompt: z.ZodString;
    schema: z.ZodString;
    timeout: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    schema: string;
    url: string;
    prompt: string;
    timeout?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    schema: string;
    url: string;
    prompt: string;
    timeout?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>;
declare const WebExtractToolResultSchema: z.ZodObject<{
    url: z.ZodString;
    success: z.ZodBoolean;
    error: z.ZodString;
    extractedData: z.ZodAny;
    metadata: z.ZodOptional<z.ZodObject<{
        extractionTime: z.ZodOptional<z.ZodNumber>;
        pageTitle: z.ZodOptional<z.ZodString>;
        statusCode: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        statusCode?: number | undefined;
        extractionTime?: number | undefined;
        pageTitle?: string | undefined;
    }, {
        statusCode?: number | undefined;
        extractionTime?: number | undefined;
        pageTitle?: string | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    error: string;
    url: string;
    success: boolean;
    metadata?: {
        statusCode?: number | undefined;
        extractionTime?: number | undefined;
        pageTitle?: string | undefined;
    } | undefined;
    extractedData?: any;
}, {
    error: string;
    url: string;
    success: boolean;
    metadata?: {
        statusCode?: number | undefined;
        extractionTime?: number | undefined;
        pageTitle?: string | undefined;
    } | undefined;
    extractedData?: any;
}>;
type WebExtractToolParams = z.output<typeof WebExtractToolParamsSchema>;
type WebExtractToolResult = z.output<typeof WebExtractToolResultSchema>;
type WebExtractToolParamsInput = z.input<typeof WebExtractToolParamsSchema>;
export declare class WebExtractTool extends ToolBubble<WebExtractToolParams, WebExtractToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        url: z.ZodString;
        prompt: z.ZodString;
        schema: z.ZodString;
        timeout: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        schema: string;
        url: string;
        prompt: string;
        timeout?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        schema: string;
        url: string;
        prompt: string;
        timeout?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        url: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
        extractedData: z.ZodAny;
        metadata: z.ZodOptional<z.ZodObject<{
            extractionTime: z.ZodOptional<z.ZodNumber>;
            pageTitle: z.ZodOptional<z.ZodString>;
            statusCode: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            statusCode?: number | undefined;
            extractionTime?: number | undefined;
            pageTitle?: string | undefined;
        }, {
            statusCode?: number | undefined;
            extractionTime?: number | undefined;
            pageTitle?: string | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        url: string;
        success: boolean;
        metadata?: {
            statusCode?: number | undefined;
            extractionTime?: number | undefined;
            pageTitle?: string | undefined;
        } | undefined;
        extractedData?: any;
    }, {
        error: string;
        url: string;
        success: boolean;
        metadata?: {
            statusCode?: number | undefined;
            extractionTime?: number | undefined;
            pageTitle?: string | undefined;
        } | undefined;
        extractedData?: any;
    }>;
    static readonly shortDescription = "Extracts structured data from web pages using Firecrawl AI-powered extraction with custom prompts and schemas";
    static readonly longDescription = "\n    A powerful web data extraction tool that uses Firecrawl's AI-powered extraction API to extract structured data from web pages.\n    \n    \uD83C\uDFAF EXTRACT Features:\n    - AI-powered structured data extraction using natural language prompts\n    - Custom JSON schema validation for extracted data\n    - Handles dynamic content and JavaScript-rendered pages\n    - Precise extraction of specific elements like images, prices, descriptions\n    - Works with complex e-commerce sites and product pages\n    - Requires FIRECRAWL_API_KEY credential\n    \n    Use Cases:\n    - Extract product information (names, prices, images) from e-commerce sites\n    - Gather structured data from listings and catalogs\n    - Extract contact information and business details\n    - Parse article metadata and content structure\n    - Collect specific data points from forms and tables\n    - Extract image URLs, especially for product galleries\n    \n    How it works:\n    1. Provide a URL and a natural language prompt describing what to extract\n    2. Define a JSON schema for the expected data structure\n    3. Firecrawl's AI analyzes the page and extracts matching data\n    4. Returns structured data validated against your schema\n    \n    Example use case:\n    - URL: Uniqlo product page\n    - Prompt: \"Extract the product name, price, and all available product image URLs\"\n    - Schema: {\"name\": \"string\", \"price\": \"number\", \"images\": [\"string\"]}\n    - Result: Structured JSON with the exact data you need\n  ";
    static readonly alias = "extract";
    static readonly type = "tool";
    constructor(params?: WebExtractToolParamsInput, context?: BubbleContext);
    performAction(context?: BubbleContext): Promise<WebExtractToolResult>;
}
export {};
//# sourceMappingURL=web-extract-tool.d.ts.map