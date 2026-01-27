import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const WebCrawlToolParamsSchema: z.ZodObject<{
    url: z.ZodString;
    format: z.ZodDefault<z.ZodEnum<["markdown"]>>;
    onlyMainContent: z.ZodDefault<z.ZodBoolean>;
    maxPages: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    crawlDepth: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    includePaths: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    excludePaths: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    waitFor: z.ZodDefault<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    format: "markdown";
    url: string;
    onlyMainContent: boolean;
    waitFor: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    maxPages?: number | undefined;
    crawlDepth?: number | undefined;
    includePaths?: string[] | undefined;
    excludePaths?: string[] | undefined;
}, {
    url: string;
    format?: "markdown" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    onlyMainContent?: boolean | undefined;
    maxPages?: number | undefined;
    crawlDepth?: number | undefined;
    includePaths?: string[] | undefined;
    excludePaths?: string[] | undefined;
    waitFor?: number | undefined;
}>;
declare const WebCrawlToolResultSchema: z.ZodObject<{
    url: z.ZodString;
    success: z.ZodBoolean;
    error: z.ZodString;
    pages: z.ZodArray<z.ZodObject<{
        url: z.ZodString;
        title: z.ZodOptional<z.ZodString>;
        content: z.ZodString;
        depth: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        content: string;
        url: string;
        title?: string | undefined;
        depth?: number | undefined;
    }, {
        content: string;
        url: string;
        title?: string | undefined;
        depth?: number | undefined;
    }>, "many">;
    totalPages: z.ZodNumber;
    creditsUsed: z.ZodNumber;
    metadata: z.ZodOptional<z.ZodObject<{
        loadTime: z.ZodOptional<z.ZodNumber>;
        crawlDepth: z.ZodOptional<z.ZodNumber>;
        maxPagesReached: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        loadTime?: number | undefined;
        crawlDepth?: number | undefined;
        maxPagesReached?: boolean | undefined;
    }, {
        loadTime?: number | undefined;
        crawlDepth?: number | undefined;
        maxPagesReached?: boolean | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    error: string;
    url: string;
    success: boolean;
    pages: {
        content: string;
        url: string;
        title?: string | undefined;
        depth?: number | undefined;
    }[];
    totalPages: number;
    creditsUsed: number;
    metadata?: {
        loadTime?: number | undefined;
        crawlDepth?: number | undefined;
        maxPagesReached?: boolean | undefined;
    } | undefined;
}, {
    error: string;
    url: string;
    success: boolean;
    pages: {
        content: string;
        url: string;
        title?: string | undefined;
        depth?: number | undefined;
    }[];
    totalPages: number;
    creditsUsed: number;
    metadata?: {
        loadTime?: number | undefined;
        crawlDepth?: number | undefined;
        maxPagesReached?: boolean | undefined;
    } | undefined;
}>;
type WebCrawlToolParams = z.input<typeof WebCrawlToolParamsSchema>;
type WebCrawlToolResult = z.output<typeof WebCrawlToolResultSchema>;
type WebCrawlToolParamsInput = z.input<typeof WebCrawlToolParamsSchema>;
export declare class WebCrawlTool extends ToolBubble<WebCrawlToolParams, WebCrawlToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        url: z.ZodString;
        format: z.ZodDefault<z.ZodEnum<["markdown"]>>;
        onlyMainContent: z.ZodDefault<z.ZodBoolean>;
        maxPages: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        crawlDepth: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        includePaths: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        excludePaths: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        waitFor: z.ZodDefault<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        format: "markdown";
        url: string;
        onlyMainContent: boolean;
        waitFor: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        maxPages?: number | undefined;
        crawlDepth?: number | undefined;
        includePaths?: string[] | undefined;
        excludePaths?: string[] | undefined;
    }, {
        url: string;
        format?: "markdown" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        onlyMainContent?: boolean | undefined;
        maxPages?: number | undefined;
        crawlDepth?: number | undefined;
        includePaths?: string[] | undefined;
        excludePaths?: string[] | undefined;
        waitFor?: number | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        url: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
        pages: z.ZodArray<z.ZodObject<{
            url: z.ZodString;
            title: z.ZodOptional<z.ZodString>;
            content: z.ZodString;
            depth: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            content: string;
            url: string;
            title?: string | undefined;
            depth?: number | undefined;
        }, {
            content: string;
            url: string;
            title?: string | undefined;
            depth?: number | undefined;
        }>, "many">;
        totalPages: z.ZodNumber;
        creditsUsed: z.ZodNumber;
        metadata: z.ZodOptional<z.ZodObject<{
            loadTime: z.ZodOptional<z.ZodNumber>;
            crawlDepth: z.ZodOptional<z.ZodNumber>;
            maxPagesReached: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            loadTime?: number | undefined;
            crawlDepth?: number | undefined;
            maxPagesReached?: boolean | undefined;
        }, {
            loadTime?: number | undefined;
            crawlDepth?: number | undefined;
            maxPagesReached?: boolean | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        url: string;
        success: boolean;
        pages: {
            content: string;
            url: string;
            title?: string | undefined;
            depth?: number | undefined;
        }[];
        totalPages: number;
        creditsUsed: number;
        metadata?: {
            loadTime?: number | undefined;
            crawlDepth?: number | undefined;
            maxPagesReached?: boolean | undefined;
        } | undefined;
    }, {
        error: string;
        url: string;
        success: boolean;
        pages: {
            content: string;
            url: string;
            title?: string | undefined;
            depth?: number | undefined;
        }[];
        totalPages: number;
        creditsUsed: number;
        metadata?: {
            loadTime?: number | undefined;
            crawlDepth?: number | undefined;
            maxPagesReached?: boolean | undefined;
        } | undefined;
    }>;
    static readonly shortDescription = "Multi-page web crawling tool for exploring entire websites and subdomains.";
    static readonly longDescription = "\n    A powerful web crawling tool that can systematically explore websites and extract content from multiple pages.\n    \n    \uD83D\uDD77\uFE0F CRAWL Features:\n    - Recursively crawl websites and subdomains\n    - Configurable crawl depth and page limits (up to 100 pages)\n    - URL pattern filtering (include/exclude paths)\n    - Multiple format support (markdown, html, links, rawHtml)\n    - Main content focus filtering\n    - Discover and extract content from entire sites\n    \n    Technical Features:\n    - Handles JavaScript-rendered pages and dynamic content\n    - Robust error handling and retry mechanisms\n    - Configurable wait times for dynamic content\n    - Requires FIRECRAWL_API_KEY credential\n    \n    Use Cases:\n    - Site mapping and competitive analysis\n    - Documentation aggregation across multiple pages  \n    - Content analysis and research across domains\n    - SEO analysis and site structure discovery\n    - Building comprehensive datasets from websites\n  ";
    static readonly alias = "crawl";
    static readonly type = "tool";
    constructor(params?: WebCrawlToolParamsInput, context?: BubbleContext);
    performAction(context?: BubbleContext): Promise<WebCrawlToolResult>;
    /**
     * Execute crawl operation - multi-page site exploration
     */
    private executeCrawl;
}
export {};
//# sourceMappingURL=web-crawl-tool.d.ts.map