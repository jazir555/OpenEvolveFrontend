import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const WebScrapeToolParamsSchema: z.ZodObject<{
    url: z.ZodString;
    format: z.ZodDefault<z.ZodEnum<["markdown"]>>;
    onlyMainContent: z.ZodDefault<z.ZodBoolean>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    format: "markdown";
    url: string;
    onlyMainContent: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    url: string;
    format?: "markdown" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    onlyMainContent?: boolean | undefined;
}>;
declare const WebScrapeToolResultSchema: z.ZodObject<{
    content: z.ZodString;
    title: z.ZodString;
    url: z.ZodString;
    format: z.ZodString;
    success: z.ZodBoolean;
    error: z.ZodString;
    creditsUsed: z.ZodNumber;
    metadata: z.ZodOptional<z.ZodObject<{
        statusCode: z.ZodOptional<z.ZodNumber>;
        loadTime: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        statusCode?: number | undefined;
        loadTime?: number | undefined;
    }, {
        statusCode?: number | undefined;
        loadTime?: number | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    error: string;
    format: string;
    title: string;
    content: string;
    url: string;
    success: boolean;
    creditsUsed: number;
    metadata?: {
        statusCode?: number | undefined;
        loadTime?: number | undefined;
    } | undefined;
}, {
    error: string;
    format: string;
    title: string;
    content: string;
    url: string;
    success: boolean;
    creditsUsed: number;
    metadata?: {
        statusCode?: number | undefined;
        loadTime?: number | undefined;
    } | undefined;
}>;
type WebScrapeToolParams = z.output<typeof WebScrapeToolParamsSchema>;
type WebScrapeToolResult = z.output<typeof WebScrapeToolResultSchema>;
type WebScrapeToolParamsInput = z.input<typeof WebScrapeToolParamsSchema>;
export declare class WebScrapeTool extends ToolBubble<WebScrapeToolParams, WebScrapeToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        url: z.ZodString;
        format: z.ZodDefault<z.ZodEnum<["markdown"]>>;
        onlyMainContent: z.ZodDefault<z.ZodBoolean>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        format: "markdown";
        url: string;
        onlyMainContent: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        url: string;
        format?: "markdown" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        onlyMainContent?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        content: z.ZodString;
        title: z.ZodString;
        url: z.ZodString;
        format: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
        creditsUsed: z.ZodNumber;
        metadata: z.ZodOptional<z.ZodObject<{
            statusCode: z.ZodOptional<z.ZodNumber>;
            loadTime: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            statusCode?: number | undefined;
            loadTime?: number | undefined;
        }, {
            statusCode?: number | undefined;
            loadTime?: number | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        format: string;
        title: string;
        content: string;
        url: string;
        success: boolean;
        creditsUsed: number;
        metadata?: {
            statusCode?: number | undefined;
            loadTime?: number | undefined;
        } | undefined;
    }, {
        error: string;
        format: string;
        title: string;
        content: string;
        url: string;
        success: boolean;
        creditsUsed: number;
        metadata?: {
            statusCode?: number | undefined;
            loadTime?: number | undefined;
        } | undefined;
    }>;
    static readonly shortDescription = "Scrapes content from a single web page. Useful after web-search-tool to get the full content of a page. Also useful if you need to understand a site's structure or content.";
    static readonly longDescription = "\n    A simple and powerful web scraping tool that extracts content from any web page.\n    \n    Features:\n    - Clean content extraction with main content focus\n    - Multiple format support (markdown, html, rawHtml)\n    - Fast and reliable using Firecrawl\n    - Handles JavaScript-rendered pages\n    - Optional browser automation for authentication flows\n    - Custom headers support for session-based scraping\n    - Requires FIRECRAWL_API_KEY credential\n    \n    Basic use cases:\n    - Extract article content for analysis\n    - Scrape product information from e-commerce sites\n    - Get clean text from documentation pages\n    - Extract data from any public web page\n    \n    Advanced use cases (with actions):\n    - Login and scrape protected content\n    - Navigate multi-step authentication flows\n    - Interact with dynamic content requiring clicks/scrolls\n    - Execute custom JavaScript for complex scenarios\n  ";
    static readonly alias = "scrape";
    static readonly type = "tool";
    constructor(params?: WebScrapeToolParamsInput, context?: BubbleContext);
    performAction(): Promise<WebScrapeToolResult>;
}
export {};
//# sourceMappingURL=web-scrape-tool.d.ts.map