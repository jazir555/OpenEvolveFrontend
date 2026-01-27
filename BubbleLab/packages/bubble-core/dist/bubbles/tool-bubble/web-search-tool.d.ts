import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const WebSearchToolParamsSchema: z.ZodObject<{
    query: z.ZodString;
    limit: z.ZodDefault<z.ZodNumber>;
    categories: z.ZodDefault<z.ZodArray<z.ZodEnum<["research", "pdf", "github"]>, "many">>;
    location: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    query: string;
    limit: number;
    categories: ("github" | "pdf" | "research")[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    location?: string | undefined;
}, {
    query: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    location?: string | undefined;
    categories?: ("github" | "pdf" | "research")[] | undefined;
}>;
declare const WebSearchToolResultSchema: z.ZodObject<{
    results: z.ZodArray<z.ZodObject<{
        title: z.ZodString;
        url: z.ZodString;
        content: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        title: string;
        content: string;
        url: string;
    }, {
        title: string;
        content: string;
        url: string;
    }>, "many">;
    query: z.ZodString;
    totalResults: z.ZodNumber;
    searchEngine: z.ZodString;
    creditsUsed: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    query: string;
    results: {
        title: string;
        content: string;
        url: string;
    }[];
    totalResults: number;
    searchEngine: string;
    creditsUsed: number;
}, {
    error: string;
    success: boolean;
    query: string;
    results: {
        title: string;
        content: string;
        url: string;
    }[];
    totalResults: number;
    searchEngine: string;
    creditsUsed: number;
}>;
type WebSearchToolParams = z.output<typeof WebSearchToolParamsSchema>;
type WebSearchToolResult = z.output<typeof WebSearchToolResultSchema>;
type WebSearchToolParamsInput = z.input<typeof WebSearchToolParamsSchema>;
export declare class WebSearchTool extends ToolBubble<WebSearchToolParams, WebSearchToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        query: z.ZodString;
        limit: z.ZodDefault<z.ZodNumber>;
        categories: z.ZodDefault<z.ZodArray<z.ZodEnum<["research", "pdf", "github"]>, "many">>;
        location: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        query: string;
        limit: number;
        categories: ("github" | "pdf" | "research")[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        location?: string | undefined;
    }, {
        query: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        location?: string | undefined;
        categories?: ("github" | "pdf" | "research")[] | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        results: z.ZodArray<z.ZodObject<{
            title: z.ZodString;
            url: z.ZodString;
            content: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            title: string;
            content: string;
            url: string;
        }, {
            title: string;
            content: string;
            url: string;
        }>, "many">;
        query: z.ZodString;
        totalResults: z.ZodNumber;
        searchEngine: z.ZodString;
        creditsUsed: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        query: string;
        results: {
            title: string;
            content: string;
            url: string;
        }[];
        totalResults: number;
        searchEngine: string;
        creditsUsed: number;
    }, {
        error: string;
        success: boolean;
        query: string;
        results: {
            title: string;
            content: string;
            url: string;
        }[];
        totalResults: number;
        searchEngine: string;
        creditsUsed: number;
    }>;
    static readonly shortDescription = "Performs web searches using Firecrawl to find current information from the web";
    static readonly longDescription = "\n    A comprehensive web search tool that uses Firecrawl to find current information from the web.\n    \n    Features:\n    - High-quality web search results with content extraction\n    - Configurable result limits (1-20 results)\n    - Location-based search for regional results\n    - Clean, structured content extraction from search results\n    - Requires FIRECRAWL_API_KEY credential\n    \n    Use cases:\n    - Finding current events and news\n    - Researching topics with web content\n    - Getting up-to-date information for decision making\n    - Answering questions that require web knowledge\n    - Market research and competitive analysis\n    - Real-time data gathering from the web\n  ";
    static readonly alias = "websearch";
    static readonly type = "tool";
    constructor(params?: WebSearchToolParamsInput, context?: BubbleContext);
    performAction(context?: BubbleContext): Promise<WebSearchToolResult>;
}
export {};
//# sourceMappingURL=web-search-tool.d.ts.map