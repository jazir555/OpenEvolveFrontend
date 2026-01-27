import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const RedditPostSchema: z.ZodObject<{
    title: z.ZodString;
    url: z.ZodString;
    author: z.ZodString;
    score: z.ZodNumber;
    numComments: z.ZodNumber;
    createdUtc: z.ZodNumber;
    postUrl: z.ZodString;
    selftext: z.ZodString;
    subreddit: z.ZodString;
    postHint: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    isSelf: z.ZodBoolean;
    thumbnail: z.ZodOptional<z.ZodString>;
    domain: z.ZodOptional<z.ZodString>;
    flair: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    title: string;
    url: string;
    author: string;
    subreddit: string;
    score: number;
    numComments: number;
    createdUtc: number;
    postUrl: string;
    selftext: string;
    isSelf: boolean;
    domain?: string | undefined;
    thumbnail?: string | undefined;
    postHint?: string | null | undefined;
    flair?: string | undefined;
}, {
    title: string;
    url: string;
    author: string;
    subreddit: string;
    score: number;
    numComments: number;
    createdUtc: number;
    postUrl: string;
    selftext: string;
    isSelf: boolean;
    domain?: string | undefined;
    thumbnail?: string | undefined;
    postHint?: string | null | undefined;
    flair?: string | undefined;
}>;
declare const RedditScrapeToolParamsSchema: z.ZodObject<{
    subreddit: z.ZodPipeline<z.ZodEffects<z.ZodString, string, string>, z.ZodString>;
    limit: z.ZodDefault<z.ZodNumber>;
    sort: z.ZodDefault<z.ZodEnum<["hot", "new", "top", "rising"]>>;
    timeFilter: z.ZodOptional<z.ZodEnum<["hour", "day", "week", "month", "year", "all"]>>;
    filterToday: z.ZodDefault<z.ZodBoolean>;
    includeStickied: z.ZodDefault<z.ZodBoolean>;
    minScore: z.ZodOptional<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    sort: "hot" | "new" | "top" | "rising";
    limit: number;
    subreddit: string;
    filterToday: boolean;
    includeStickied: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    timeFilter?: "all" | "hour" | "week" | "month" | "year" | "day" | undefined;
    minScore?: number | undefined;
}, {
    subreddit: string;
    sort?: "hot" | "new" | "top" | "rising" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    timeFilter?: "all" | "hour" | "week" | "month" | "year" | "day" | undefined;
    filterToday?: boolean | undefined;
    includeStickied?: boolean | undefined;
    minScore?: number | undefined;
}>;
declare const RedditScrapeToolResultSchema: z.ZodObject<{
    posts: z.ZodArray<z.ZodObject<{
        title: z.ZodString;
        url: z.ZodString;
        author: z.ZodString;
        score: z.ZodNumber;
        numComments: z.ZodNumber;
        createdUtc: z.ZodNumber;
        postUrl: z.ZodString;
        selftext: z.ZodString;
        subreddit: z.ZodString;
        postHint: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        isSelf: z.ZodBoolean;
        thumbnail: z.ZodOptional<z.ZodString>;
        domain: z.ZodOptional<z.ZodString>;
        flair: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        title: string;
        url: string;
        author: string;
        subreddit: string;
        score: number;
        numComments: number;
        createdUtc: number;
        postUrl: string;
        selftext: string;
        isSelf: boolean;
        domain?: string | undefined;
        thumbnail?: string | undefined;
        postHint?: string | null | undefined;
        flair?: string | undefined;
    }, {
        title: string;
        url: string;
        author: string;
        subreddit: string;
        score: number;
        numComments: number;
        createdUtc: number;
        postUrl: string;
        selftext: string;
        isSelf: boolean;
        domain?: string | undefined;
        thumbnail?: string | undefined;
        postHint?: string | null | undefined;
        flair?: string | undefined;
    }>, "many">;
    metadata: z.ZodObject<{
        subreddit: z.ZodString;
        requestedLimit: z.ZodNumber;
        actualCount: z.ZodNumber;
        filteredCount: z.ZodNumber;
        sort: z.ZodString;
        timeFilter: z.ZodOptional<z.ZodString>;
        scrapedAt: z.ZodString;
        apiEndpoint: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        sort: string;
        scrapedAt: string;
        subreddit: string;
        requestedLimit: number;
        actualCount: number;
        filteredCount: number;
        apiEndpoint: string;
        timeFilter?: string | undefined;
    }, {
        sort: string;
        scrapedAt: string;
        subreddit: string;
        requestedLimit: number;
        actualCount: number;
        filteredCount: number;
        apiEndpoint: string;
        timeFilter?: string | undefined;
    }>;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    metadata: {
        sort: string;
        scrapedAt: string;
        subreddit: string;
        requestedLimit: number;
        actualCount: number;
        filteredCount: number;
        apiEndpoint: string;
        timeFilter?: string | undefined;
    };
    posts: {
        title: string;
        url: string;
        author: string;
        subreddit: string;
        score: number;
        numComments: number;
        createdUtc: number;
        postUrl: string;
        selftext: string;
        isSelf: boolean;
        domain?: string | undefined;
        thumbnail?: string | undefined;
        postHint?: string | null | undefined;
        flair?: string | undefined;
    }[];
}, {
    error: string;
    success: boolean;
    metadata: {
        sort: string;
        scrapedAt: string;
        subreddit: string;
        requestedLimit: number;
        actualCount: number;
        filteredCount: number;
        apiEndpoint: string;
        timeFilter?: string | undefined;
    };
    posts: {
        title: string;
        url: string;
        author: string;
        subreddit: string;
        score: number;
        numComments: number;
        createdUtc: number;
        postUrl: string;
        selftext: string;
        isSelf: boolean;
        domain?: string | undefined;
        thumbnail?: string | undefined;
        postHint?: string | null | undefined;
        flair?: string | undefined;
    }[];
}>;
type RedditScrapeToolParams = z.output<typeof RedditScrapeToolParamsSchema>;
type RedditScrapeToolResult = z.output<typeof RedditScrapeToolResultSchema>;
type RedditScrapeToolParamsInput = z.input<typeof RedditScrapeToolParamsSchema>;
type RedditPost = z.output<typeof RedditPostSchema>;
export declare class RedditScrapeTool extends ToolBubble<RedditScrapeToolParams, RedditScrapeToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        subreddit: z.ZodPipeline<z.ZodEffects<z.ZodString, string, string>, z.ZodString>;
        limit: z.ZodDefault<z.ZodNumber>;
        sort: z.ZodDefault<z.ZodEnum<["hot", "new", "top", "rising"]>>;
        timeFilter: z.ZodOptional<z.ZodEnum<["hour", "day", "week", "month", "year", "all"]>>;
        filterToday: z.ZodDefault<z.ZodBoolean>;
        includeStickied: z.ZodDefault<z.ZodBoolean>;
        minScore: z.ZodOptional<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        sort: "hot" | "new" | "top" | "rising";
        limit: number;
        subreddit: string;
        filterToday: boolean;
        includeStickied: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        timeFilter?: "all" | "hour" | "week" | "month" | "year" | "day" | undefined;
        minScore?: number | undefined;
    }, {
        subreddit: string;
        sort?: "hot" | "new" | "top" | "rising" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        timeFilter?: "all" | "hour" | "week" | "month" | "year" | "day" | undefined;
        filterToday?: boolean | undefined;
        includeStickied?: boolean | undefined;
        minScore?: number | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        posts: z.ZodArray<z.ZodObject<{
            title: z.ZodString;
            url: z.ZodString;
            author: z.ZodString;
            score: z.ZodNumber;
            numComments: z.ZodNumber;
            createdUtc: z.ZodNumber;
            postUrl: z.ZodString;
            selftext: z.ZodString;
            subreddit: z.ZodString;
            postHint: z.ZodOptional<z.ZodNullable<z.ZodString>>;
            isSelf: z.ZodBoolean;
            thumbnail: z.ZodOptional<z.ZodString>;
            domain: z.ZodOptional<z.ZodString>;
            flair: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            title: string;
            url: string;
            author: string;
            subreddit: string;
            score: number;
            numComments: number;
            createdUtc: number;
            postUrl: string;
            selftext: string;
            isSelf: boolean;
            domain?: string | undefined;
            thumbnail?: string | undefined;
            postHint?: string | null | undefined;
            flair?: string | undefined;
        }, {
            title: string;
            url: string;
            author: string;
            subreddit: string;
            score: number;
            numComments: number;
            createdUtc: number;
            postUrl: string;
            selftext: string;
            isSelf: boolean;
            domain?: string | undefined;
            thumbnail?: string | undefined;
            postHint?: string | null | undefined;
            flair?: string | undefined;
        }>, "many">;
        metadata: z.ZodObject<{
            subreddit: z.ZodString;
            requestedLimit: z.ZodNumber;
            actualCount: z.ZodNumber;
            filteredCount: z.ZodNumber;
            sort: z.ZodString;
            timeFilter: z.ZodOptional<z.ZodString>;
            scrapedAt: z.ZodString;
            apiEndpoint: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            sort: string;
            scrapedAt: string;
            subreddit: string;
            requestedLimit: number;
            actualCount: number;
            filteredCount: number;
            apiEndpoint: string;
            timeFilter?: string | undefined;
        }, {
            sort: string;
            scrapedAt: string;
            subreddit: string;
            requestedLimit: number;
            actualCount: number;
            filteredCount: number;
            apiEndpoint: string;
            timeFilter?: string | undefined;
        }>;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        metadata: {
            sort: string;
            scrapedAt: string;
            subreddit: string;
            requestedLimit: number;
            actualCount: number;
            filteredCount: number;
            apiEndpoint: string;
            timeFilter?: string | undefined;
        };
        posts: {
            title: string;
            url: string;
            author: string;
            subreddit: string;
            score: number;
            numComments: number;
            createdUtc: number;
            postUrl: string;
            selftext: string;
            isSelf: boolean;
            domain?: string | undefined;
            thumbnail?: string | undefined;
            postHint?: string | null | undefined;
            flair?: string | undefined;
        }[];
    }, {
        error: string;
        success: boolean;
        metadata: {
            sort: string;
            scrapedAt: string;
            subreddit: string;
            requestedLimit: number;
            actualCount: number;
            filteredCount: number;
            apiEndpoint: string;
            timeFilter?: string | undefined;
        };
        posts: {
            title: string;
            url: string;
            author: string;
            subreddit: string;
            score: number;
            numComments: number;
            createdUtc: number;
            postUrl: string;
            selftext: string;
            isSelf: boolean;
            domain?: string | undefined;
            thumbnail?: string | undefined;
            postHint?: string | null | undefined;
            flair?: string | undefined;
        }[];
    }>;
    static readonly shortDescription = "Scrapes posts from any Reddit subreddit with flexible filtering and sorting options";
    static readonly longDescription = "\n    A specialized tool for scraping Reddit posts from any subreddit with comprehensive filtering and sorting capabilities.\n    \n    \uD83D\uDD25 Core Features:\n    - Scrape posts from any public subreddit\n    - Multiple sorting options (hot, new, top, rising)\n    - Flexible post limits (1-1000 posts with pagination)\n    - Time-based filtering for top posts\n    - Today-only filtering option\n    - Score-based filtering\n    - Stickied post inclusion/exclusion\n    \n    \uD83D\uDCCA Post Data Extracted:\n    - Title, author, and content\n    - Upvote scores and comment counts\n    - Creation timestamps and permalinks\n    - Post types (text vs link posts)\n    - External URLs and domains\n    - Thumbnails and flairs\n    - Comprehensive metadata\n    \n    \uD83C\uDFAF Use Cases:\n    - Monitor specific subreddits for trends\n    - Gather posts for content analysis\n    - Track community engagement metrics\n    - Feed Reddit data into other workflows\n    - Research subreddit activity patterns\n    - Content aggregation and curation\n    \n    \u26A1 Technical Features:\n    - Uses Reddit's official JSON API\n    - No authentication required for public posts\n    - Respects Reddit's rate limiting\n    - Handles large subreddits efficiently\n    - Robust error handling and validation\n    - Clean, structured data output\n    \n    Perfect for integration with AI agents, data analysis workflows, and content monitoring systems.\n  ";
    static readonly alias = "reddit";
    static readonly type = "tool";
    constructor(params?: RedditScrapeToolParamsInput, context?: BubbleContext);
    performAction(context?: BubbleContext): Promise<RedditScrapeToolResult>;
    /**
     * Build the Reddit JSON API URL with optional pagination
     */
    private buildRedditApiUrl;
    /**
     * Fetch posts with pagination support (up to 1000 posts)
     * Makes multiple requests if needed, using the 'after' parameter for pagination
     */
    private fetchPostsWithPagination;
    /**
     * Get a random user agent to avoid being blocked
     */
    private getRandomUserAgent;
    /**
     * Fetch data from Reddit's JSON API
     */
    private fetchRedditData;
    /**
     * Parse Reddit JSON response into standardized post objects
     */
    private parseRedditResponse;
    /**
     * Apply various filters to the posts
     */
    private applyFilters;
}
export type { RedditPost, RedditScrapeToolParams, RedditScrapeToolResult };
//# sourceMappingURL=reddit-scrape-tool.d.ts.map