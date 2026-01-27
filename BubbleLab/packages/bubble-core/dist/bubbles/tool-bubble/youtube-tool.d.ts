import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const YouTubeVideoSchema: z.ZodObject<{
    title: z.ZodNullable<z.ZodString>;
    id: z.ZodNullable<z.ZodString>;
    url: z.ZodNullable<z.ZodString>;
    viewCount: z.ZodNullable<z.ZodNumber>;
    likes: z.ZodNullable<z.ZodNumber>;
    date: z.ZodNullable<z.ZodString>;
    channelName: z.ZodNullable<z.ZodString>;
    channelUrl: z.ZodNullable<z.ZodString>;
    subscribers: z.ZodNullable<z.ZodNumber>;
    duration: z.ZodNullable<z.ZodString>;
    description: z.ZodNullable<z.ZodString>;
    comments: z.ZodNullable<z.ZodNumber>;
    thumbnail: z.ZodNullable<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    description: string | null;
    title: string | null;
    date: string | null;
    url: string | null;
    duration: string | null;
    id: string | null;
    comments: number | null;
    thumbnail: string | null;
    viewCount: number | null;
    likes: number | null;
    channelName: string | null;
    channelUrl: string | null;
    subscribers: number | null;
}, {
    description: string | null;
    title: string | null;
    date: string | null;
    url: string | null;
    duration: string | null;
    id: string | null;
    comments: number | null;
    thumbnail: string | null;
    viewCount: number | null;
    likes: number | null;
    channelName: string | null;
    channelUrl: string | null;
    subscribers: number | null;
}>;
declare const YouTubeTranscriptSegmentSchema: z.ZodObject<{
    start: z.ZodNullable<z.ZodString>;
    duration: z.ZodNullable<z.ZodString>;
    text: z.ZodNullable<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    duration: string | null;
    text: string | null;
    start: string | null;
}, {
    duration: string | null;
    text: string | null;
    start: string | null;
}>;
declare const YouTubeToolParamsSchema: z.ZodObject<{
    operation: z.ZodEnum<["searchVideos", "getTranscript", "scrapeChannel"]>;
    searchQueries: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    videoUrls: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    channelUrl: z.ZodOptional<z.ZodString>;
    videoUrl: z.ZodOptional<z.ZodString>;
    maxResults: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    includeShorts: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "searchVideos" | "getTranscript" | "scrapeChannel";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    videoUrl?: string | undefined;
    searchQueries?: string[] | undefined;
    maxResults?: number | undefined;
    channelUrl?: string | undefined;
    videoUrls?: string[] | undefined;
    includeShorts?: boolean | undefined;
}, {
    operation: "searchVideos" | "getTranscript" | "scrapeChannel";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    videoUrl?: string | undefined;
    searchQueries?: string[] | undefined;
    maxResults?: number | undefined;
    channelUrl?: string | undefined;
    videoUrls?: string[] | undefined;
    includeShorts?: boolean | undefined;
}>;
declare const YouTubeToolResultSchema: z.ZodObject<{
    operation: z.ZodEnum<["searchVideos", "getTranscript", "scrapeChannel"]>;
    videos: z.ZodOptional<z.ZodArray<z.ZodObject<{
        title: z.ZodNullable<z.ZodString>;
        id: z.ZodNullable<z.ZodString>;
        url: z.ZodNullable<z.ZodString>;
        viewCount: z.ZodNullable<z.ZodNumber>;
        likes: z.ZodNullable<z.ZodNumber>;
        date: z.ZodNullable<z.ZodString>;
        channelName: z.ZodNullable<z.ZodString>;
        channelUrl: z.ZodNullable<z.ZodString>;
        subscribers: z.ZodNullable<z.ZodNumber>;
        duration: z.ZodNullable<z.ZodString>;
        description: z.ZodNullable<z.ZodString>;
        comments: z.ZodNullable<z.ZodNumber>;
        thumbnail: z.ZodNullable<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        description: string | null;
        title: string | null;
        date: string | null;
        url: string | null;
        duration: string | null;
        id: string | null;
        comments: number | null;
        thumbnail: string | null;
        viewCount: number | null;
        likes: number | null;
        channelName: string | null;
        channelUrl: string | null;
        subscribers: number | null;
    }, {
        description: string | null;
        title: string | null;
        date: string | null;
        url: string | null;
        duration: string | null;
        id: string | null;
        comments: number | null;
        thumbnail: string | null;
        viewCount: number | null;
        likes: number | null;
        channelName: string | null;
        channelUrl: string | null;
        subscribers: number | null;
    }>, "many">>;
    transcript: z.ZodOptional<z.ZodArray<z.ZodObject<{
        start: z.ZodNullable<z.ZodString>;
        duration: z.ZodNullable<z.ZodString>;
        text: z.ZodNullable<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        duration: string | null;
        text: string | null;
        start: string | null;
    }, {
        duration: string | null;
        text: string | null;
        start: string | null;
    }>, "many">>;
    fullTranscriptText: z.ZodOptional<z.ZodString>;
    totalResults: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "searchVideos" | "getTranscript" | "scrapeChannel";
    totalResults: number;
    videos?: {
        description: string | null;
        title: string | null;
        date: string | null;
        url: string | null;
        duration: string | null;
        id: string | null;
        comments: number | null;
        thumbnail: string | null;
        viewCount: number | null;
        likes: number | null;
        channelName: string | null;
        channelUrl: string | null;
        subscribers: number | null;
    }[] | undefined;
    transcript?: {
        duration: string | null;
        text: string | null;
        start: string | null;
    }[] | undefined;
    fullTranscriptText?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "searchVideos" | "getTranscript" | "scrapeChannel";
    totalResults: number;
    videos?: {
        description: string | null;
        title: string | null;
        date: string | null;
        url: string | null;
        duration: string | null;
        id: string | null;
        comments: number | null;
        thumbnail: string | null;
        viewCount: number | null;
        likes: number | null;
        channelName: string | null;
        channelUrl: string | null;
        subscribers: number | null;
    }[] | undefined;
    transcript?: {
        duration: string | null;
        text: string | null;
        start: string | null;
    }[] | undefined;
    fullTranscriptText?: string | undefined;
}>;
type YouTubeToolParams = z.output<typeof YouTubeToolParamsSchema>;
type YouTubeToolResult = z.output<typeof YouTubeToolResultSchema>;
type YouTubeToolParamsInput = z.input<typeof YouTubeToolParamsSchema>;
export type YouTubeVideo = z.output<typeof YouTubeVideoSchema>;
export type YouTubeTranscriptSegment = z.output<typeof YouTubeTranscriptSegmentSchema>;
/**
 * Simple, dev-friendly YouTube tool
 *
 * Provides easy access to YouTube data through three operations:
 * - searchVideos: Search YouTube or scrape video/channel/playlist URLs
 * - getTranscript: Extract video transcripts with timestamps
 * - scrapeChannel: Get all videos from a channel
 */
export declare class YouTubeTool extends ToolBubble<YouTubeToolParams, YouTubeToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        operation: z.ZodEnum<["searchVideos", "getTranscript", "scrapeChannel"]>;
        searchQueries: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        videoUrls: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        channelUrl: z.ZodOptional<z.ZodString>;
        videoUrl: z.ZodOptional<z.ZodString>;
        maxResults: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        includeShorts: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "searchVideos" | "getTranscript" | "scrapeChannel";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        videoUrl?: string | undefined;
        searchQueries?: string[] | undefined;
        maxResults?: number | undefined;
        channelUrl?: string | undefined;
        videoUrls?: string[] | undefined;
        includeShorts?: boolean | undefined;
    }, {
        operation: "searchVideos" | "getTranscript" | "scrapeChannel";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        videoUrl?: string | undefined;
        searchQueries?: string[] | undefined;
        maxResults?: number | undefined;
        channelUrl?: string | undefined;
        videoUrls?: string[] | undefined;
        includeShorts?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        operation: z.ZodEnum<["searchVideos", "getTranscript", "scrapeChannel"]>;
        videos: z.ZodOptional<z.ZodArray<z.ZodObject<{
            title: z.ZodNullable<z.ZodString>;
            id: z.ZodNullable<z.ZodString>;
            url: z.ZodNullable<z.ZodString>;
            viewCount: z.ZodNullable<z.ZodNumber>;
            likes: z.ZodNullable<z.ZodNumber>;
            date: z.ZodNullable<z.ZodString>;
            channelName: z.ZodNullable<z.ZodString>;
            channelUrl: z.ZodNullable<z.ZodString>;
            subscribers: z.ZodNullable<z.ZodNumber>;
            duration: z.ZodNullable<z.ZodString>;
            description: z.ZodNullable<z.ZodString>;
            comments: z.ZodNullable<z.ZodNumber>;
            thumbnail: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            description: string | null;
            title: string | null;
            date: string | null;
            url: string | null;
            duration: string | null;
            id: string | null;
            comments: number | null;
            thumbnail: string | null;
            viewCount: number | null;
            likes: number | null;
            channelName: string | null;
            channelUrl: string | null;
            subscribers: number | null;
        }, {
            description: string | null;
            title: string | null;
            date: string | null;
            url: string | null;
            duration: string | null;
            id: string | null;
            comments: number | null;
            thumbnail: string | null;
            viewCount: number | null;
            likes: number | null;
            channelName: string | null;
            channelUrl: string | null;
            subscribers: number | null;
        }>, "many">>;
        transcript: z.ZodOptional<z.ZodArray<z.ZodObject<{
            start: z.ZodNullable<z.ZodString>;
            duration: z.ZodNullable<z.ZodString>;
            text: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            duration: string | null;
            text: string | null;
            start: string | null;
        }, {
            duration: string | null;
            text: string | null;
            start: string | null;
        }>, "many">>;
        fullTranscriptText: z.ZodOptional<z.ZodString>;
        totalResults: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "searchVideos" | "getTranscript" | "scrapeChannel";
        totalResults: number;
        videos?: {
            description: string | null;
            title: string | null;
            date: string | null;
            url: string | null;
            duration: string | null;
            id: string | null;
            comments: number | null;
            thumbnail: string | null;
            viewCount: number | null;
            likes: number | null;
            channelName: string | null;
            channelUrl: string | null;
            subscribers: number | null;
        }[] | undefined;
        transcript?: {
            duration: string | null;
            text: string | null;
            start: string | null;
        }[] | undefined;
        fullTranscriptText?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "searchVideos" | "getTranscript" | "scrapeChannel";
        totalResults: number;
        videos?: {
            description: string | null;
            title: string | null;
            date: string | null;
            url: string | null;
            duration: string | null;
            id: string | null;
            comments: number | null;
            thumbnail: string | null;
            viewCount: number | null;
            likes: number | null;
            channelName: string | null;
            channelUrl: string | null;
            subscribers: number | null;
        }[] | undefined;
        transcript?: {
            duration: string | null;
            text: string | null;
            start: string | null;
        }[] | undefined;
        fullTranscriptText?: string | undefined;
    }>;
    static readonly shortDescription = "Search YouTube videos, extract transcripts, and scrape channel content with a simple interface";
    static readonly longDescription = "\n    Universal YouTube tool for video search, transcript extraction, and channel scraping.\n\n    **OPERATIONS:**\n    1. **searchVideos**: Search YouTube or scrape specific URLs\n       - Search by keywords\n       - Scrape videos, channels, playlists, or hashtags\n       - Get video metadata, views, likes, channel info\n\n    2. **getTranscript**: Extract video transcripts\n       - Get timestamped transcript segments\n       - Full transcript text available\n       - Perfect for content analysis or subtitles\n\n    3. **scrapeChannel**: Get all videos from a channel\n       - Fetch recent videos from any channel\n       - Sort by newest, popular, or oldest\n       - Get comprehensive video data\n\n    **WHEN TO USE THIS TOOL:**\n    - YouTube video search and discovery\n    - Content analysis and research\n    - Transcript extraction for AI/ML\n    - Channel monitoring and tracking\n    - Video metadata collection\n\n    **Simple Interface:**\n    Just specify the operation and provide search terms or URLs.\n    The tool handles service selection, data transformation, and error handling.\n\n    **What you get:**\n    - Clean, structured video data\n    - Timestamped transcripts\n    - Channel and engagement metrics\n    - Ready for analysis or processing\n  ";
    static readonly alias = "yt";
    static readonly type = "tool";
    constructor(params?: YouTubeToolParamsInput, context?: BubbleContext, instanceId?: string);
    performAction(): Promise<YouTubeToolResult>;
    /**
     * Create error result
     */
    private createErrorResult;
    /**
     * Handle searchVideos operation
     */
    private handleSearchVideos;
    /**
     * Handle getTranscript operation
     */
    private handleGetTranscript;
    /**
     * Handle scrapeChannel operation
     */
    private handleScrapeChannel;
    /**
     * Transform Apify video data to unified format
     */
    private transformVideos;
    /**
     * Create empty video object
     */
    private createEmptyVideo;
}
export {};
//# sourceMappingURL=youtube-tool.d.ts.map