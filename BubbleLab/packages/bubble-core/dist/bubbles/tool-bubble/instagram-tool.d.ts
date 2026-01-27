import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const InstagramPostSchema: z.ZodObject<{
    url: z.ZodNullable<z.ZodString>;
    caption: z.ZodNullable<z.ZodString>;
    likesCount: z.ZodNullable<z.ZodNumber>;
    commentsCount: z.ZodNullable<z.ZodNumber>;
    ownerUsername: z.ZodNullable<z.ZodString>;
    timestamp: z.ZodNullable<z.ZodString>;
    type: z.ZodNullable<z.ZodEnum<["image", "video", "carousel"]>>;
    displayUrl: z.ZodNullable<z.ZodString>;
    hashtags: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    type: "image" | "video" | "carousel" | null;
    url: string | null;
    timestamp: string | null;
    caption: string | null;
    hashtags: string[] | null;
    commentsCount: number | null;
    displayUrl: string | null;
    likesCount: number | null;
    ownerUsername: string | null;
}, {
    type: "image" | "video" | "carousel" | null;
    url: string | null;
    timestamp: string | null;
    caption: string | null;
    hashtags: string[] | null;
    commentsCount: number | null;
    displayUrl: string | null;
    likesCount: number | null;
    ownerUsername: string | null;
}>;
declare const InstagramProfileSchema: z.ZodObject<{
    username: z.ZodString;
    fullName: z.ZodNullable<z.ZodString>;
    bio: z.ZodNullable<z.ZodString>;
    followersCount: z.ZodNullable<z.ZodNumber>;
    followingCount: z.ZodNullable<z.ZodNumber>;
    postsCount: z.ZodNullable<z.ZodNumber>;
    isVerified: z.ZodNullable<z.ZodBoolean>;
    profilePicUrl: z.ZodNullable<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    username: string;
    fullName: string | null;
    followersCount: number | null;
    postsCount: number | null;
    profilePicUrl: string | null;
    isVerified: boolean | null;
    bio: string | null;
    followingCount: number | null;
}, {
    username: string;
    fullName: string | null;
    followersCount: number | null;
    postsCount: number | null;
    profilePicUrl: string | null;
    isVerified: boolean | null;
    bio: string | null;
    followingCount: number | null;
}>;
declare const InstagramToolParamsSchema: z.ZodObject<{
    operation: z.ZodEnum<["scrapeProfile", "scrapeHashtag"]>;
    profiles: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    hashtags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    limit: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "scrapeProfile" | "scrapeHashtag";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    hashtags?: string[] | undefined;
    profiles?: string[] | undefined;
}, {
    operation: "scrapeProfile" | "scrapeHashtag";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    hashtags?: string[] | undefined;
    profiles?: string[] | undefined;
}>;
declare const InstagramToolResultSchema: z.ZodObject<{
    operation: z.ZodEnum<["scrapeProfile", "scrapeHashtag"]>;
    posts: z.ZodArray<z.ZodObject<{
        url: z.ZodNullable<z.ZodString>;
        caption: z.ZodNullable<z.ZodString>;
        likesCount: z.ZodNullable<z.ZodNumber>;
        commentsCount: z.ZodNullable<z.ZodNumber>;
        ownerUsername: z.ZodNullable<z.ZodString>;
        timestamp: z.ZodNullable<z.ZodString>;
        type: z.ZodNullable<z.ZodEnum<["image", "video", "carousel"]>>;
        displayUrl: z.ZodNullable<z.ZodString>;
        hashtags: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        type: "image" | "video" | "carousel" | null;
        url: string | null;
        timestamp: string | null;
        caption: string | null;
        hashtags: string[] | null;
        commentsCount: number | null;
        displayUrl: string | null;
        likesCount: number | null;
        ownerUsername: string | null;
    }, {
        type: "image" | "video" | "carousel" | null;
        url: string | null;
        timestamp: string | null;
        caption: string | null;
        hashtags: string[] | null;
        commentsCount: number | null;
        displayUrl: string | null;
        likesCount: number | null;
        ownerUsername: string | null;
    }>, "many">;
    profiles: z.ZodOptional<z.ZodArray<z.ZodObject<{
        username: z.ZodString;
        fullName: z.ZodNullable<z.ZodString>;
        bio: z.ZodNullable<z.ZodString>;
        followersCount: z.ZodNullable<z.ZodNumber>;
        followingCount: z.ZodNullable<z.ZodNumber>;
        postsCount: z.ZodNullable<z.ZodNumber>;
        isVerified: z.ZodNullable<z.ZodBoolean>;
        profilePicUrl: z.ZodNullable<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        username: string;
        fullName: string | null;
        followersCount: number | null;
        postsCount: number | null;
        profilePicUrl: string | null;
        isVerified: boolean | null;
        bio: string | null;
        followingCount: number | null;
    }, {
        username: string;
        fullName: string | null;
        followersCount: number | null;
        postsCount: number | null;
        profilePicUrl: string | null;
        isVerified: boolean | null;
        bio: string | null;
        followingCount: number | null;
    }>, "many">>;
    scrapedHashtags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    scrapedProfiles: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    totalPosts: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "scrapeProfile" | "scrapeHashtag";
    posts: {
        type: "image" | "video" | "carousel" | null;
        url: string | null;
        timestamp: string | null;
        caption: string | null;
        hashtags: string[] | null;
        commentsCount: number | null;
        displayUrl: string | null;
        likesCount: number | null;
        ownerUsername: string | null;
    }[];
    totalPosts: number;
    profiles?: {
        username: string;
        fullName: string | null;
        followersCount: number | null;
        postsCount: number | null;
        profilePicUrl: string | null;
        isVerified: boolean | null;
        bio: string | null;
        followingCount: number | null;
    }[] | undefined;
    scrapedHashtags?: string[] | undefined;
    scrapedProfiles?: string[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "scrapeProfile" | "scrapeHashtag";
    posts: {
        type: "image" | "video" | "carousel" | null;
        url: string | null;
        timestamp: string | null;
        caption: string | null;
        hashtags: string[] | null;
        commentsCount: number | null;
        displayUrl: string | null;
        likesCount: number | null;
        ownerUsername: string | null;
    }[];
    totalPosts: number;
    profiles?: {
        username: string;
        fullName: string | null;
        followersCount: number | null;
        postsCount: number | null;
        profilePicUrl: string | null;
        isVerified: boolean | null;
        bio: string | null;
        followingCount: number | null;
    }[] | undefined;
    scrapedHashtags?: string[] | undefined;
    scrapedProfiles?: string[] | undefined;
}>;
type InstagramToolParams = z.output<typeof InstagramToolParamsSchema>;
type InstagramToolResult = z.output<typeof InstagramToolResultSchema>;
type InstagramToolParamsInput = z.input<typeof InstagramToolParamsSchema>;
export type InstagramPost = z.output<typeof InstagramPostSchema>;
export type InstagramProfile = z.output<typeof InstagramProfileSchema>;
export type InstagramOperationResult<T extends InstagramToolParams['operation']> = Extract<InstagramToolResult, {
    operation: T;
}>;
/**
 * Generic Instagram scraping tool with unified interface
 *
 * This tool abstracts away the underlying scraping service (currently Apify)
 * and provides a simple, opinionated interface for Instagram data extraction.
 *
 * Supports two operations:
 * - scrapeProfile: Scrape user profiles and their posts
 * - scrapeHashtag: Scrape posts by hashtag
 *
 * Future versions can add support for other services (BrightData, custom scrapers)
 * while maintaining the same interface.
 */
export declare class InstagramTool extends ToolBubble<InstagramToolParams, InstagramToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        operation: z.ZodEnum<["scrapeProfile", "scrapeHashtag"]>;
        profiles: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        hashtags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        limit: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "scrapeProfile" | "scrapeHashtag";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        hashtags?: string[] | undefined;
        profiles?: string[] | undefined;
    }, {
        operation: "scrapeProfile" | "scrapeHashtag";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        hashtags?: string[] | undefined;
        profiles?: string[] | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        operation: z.ZodEnum<["scrapeProfile", "scrapeHashtag"]>;
        posts: z.ZodArray<z.ZodObject<{
            url: z.ZodNullable<z.ZodString>;
            caption: z.ZodNullable<z.ZodString>;
            likesCount: z.ZodNullable<z.ZodNumber>;
            commentsCount: z.ZodNullable<z.ZodNumber>;
            ownerUsername: z.ZodNullable<z.ZodString>;
            timestamp: z.ZodNullable<z.ZodString>;
            type: z.ZodNullable<z.ZodEnum<["image", "video", "carousel"]>>;
            displayUrl: z.ZodNullable<z.ZodString>;
            hashtags: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            type: "image" | "video" | "carousel" | null;
            url: string | null;
            timestamp: string | null;
            caption: string | null;
            hashtags: string[] | null;
            commentsCount: number | null;
            displayUrl: string | null;
            likesCount: number | null;
            ownerUsername: string | null;
        }, {
            type: "image" | "video" | "carousel" | null;
            url: string | null;
            timestamp: string | null;
            caption: string | null;
            hashtags: string[] | null;
            commentsCount: number | null;
            displayUrl: string | null;
            likesCount: number | null;
            ownerUsername: string | null;
        }>, "many">;
        profiles: z.ZodOptional<z.ZodArray<z.ZodObject<{
            username: z.ZodString;
            fullName: z.ZodNullable<z.ZodString>;
            bio: z.ZodNullable<z.ZodString>;
            followersCount: z.ZodNullable<z.ZodNumber>;
            followingCount: z.ZodNullable<z.ZodNumber>;
            postsCount: z.ZodNullable<z.ZodNumber>;
            isVerified: z.ZodNullable<z.ZodBoolean>;
            profilePicUrl: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            username: string;
            fullName: string | null;
            followersCount: number | null;
            postsCount: number | null;
            profilePicUrl: string | null;
            isVerified: boolean | null;
            bio: string | null;
            followingCount: number | null;
        }, {
            username: string;
            fullName: string | null;
            followersCount: number | null;
            postsCount: number | null;
            profilePicUrl: string | null;
            isVerified: boolean | null;
            bio: string | null;
            followingCount: number | null;
        }>, "many">>;
        scrapedHashtags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        scrapedProfiles: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        totalPosts: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "scrapeProfile" | "scrapeHashtag";
        posts: {
            type: "image" | "video" | "carousel" | null;
            url: string | null;
            timestamp: string | null;
            caption: string | null;
            hashtags: string[] | null;
            commentsCount: number | null;
            displayUrl: string | null;
            likesCount: number | null;
            ownerUsername: string | null;
        }[];
        totalPosts: number;
        profiles?: {
            username: string;
            fullName: string | null;
            followersCount: number | null;
            postsCount: number | null;
            profilePicUrl: string | null;
            isVerified: boolean | null;
            bio: string | null;
            followingCount: number | null;
        }[] | undefined;
        scrapedHashtags?: string[] | undefined;
        scrapedProfiles?: string[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "scrapeProfile" | "scrapeHashtag";
        posts: {
            type: "image" | "video" | "carousel" | null;
            url: string | null;
            timestamp: string | null;
            caption: string | null;
            hashtags: string[] | null;
            commentsCount: number | null;
            displayUrl: string | null;
            likesCount: number | null;
            ownerUsername: string | null;
        }[];
        totalPosts: number;
        profiles?: {
            username: string;
            fullName: string | null;
            followersCount: number | null;
            postsCount: number | null;
            profilePicUrl: string | null;
            isVerified: boolean | null;
            bio: string | null;
            followingCount: number | null;
        }[] | undefined;
        scrapedHashtags?: string[] | undefined;
        scrapedProfiles?: string[] | undefined;
    }>;
    static readonly shortDescription = "Scrape Instagram profiles and posts with a simple, unified interface. Works with individual user profiles and hashtags.";
    static readonly longDescription = "\n    Universal Instagram scraping tool that provides a simple, opinionated interface for extracting Instagram data.\n    \n    **OPERATIONS:**\n    1. **scrapeProfile**: Scrape user profiles and their posts\n       - Get profile information (bio, followers, verified status)\n       - Fetch recent posts from specific users\n       - Track influencer or brand accounts\n    \n    2. **scrapeHashtag**: Scrape posts by hashtag\n       - Find trending content by hashtag\n       - Monitor brand mentions and campaigns\n       - Research hashtag performance\n    \n    **WHEN TO USE THIS TOOL:**\n    - **Any Instagram scraping task** - profiles, posts, hashtags, engagement data\n    - **Social media research** - influencer analysis, competitor monitoring\n    - **Content gathering** - posts, captions, hashtags, engagement metrics\n    - **Market research** - brand mentions, user sentiment on Instagram\n    - **Trend analysis** - hashtag tracking, viral content discovery\n    \n    **DO NOT USE research-agent-tool or web-scrape-tool for Instagram** - This tool is specifically optimized for Instagram and provides:\n    - Unified data format across all Instagram sources\n    - Automatic service selection and optimization\n    - Rate limiting and reliability handling\n    - Clean, structured data ready for analysis\n    \n    **Simple Interface:**\n    Just specify the operation and provide Instagram usernames/URLs or hashtags to get back clean, structured data.\n    The tool automatically handles:\n    - URL normalization (accepts usernames, profile URLs, hashtag URLs)\n    - Service selection (currently Apify, future: multiple sources)\n    - Data transformation to unified format\n    - Error handling and retries\n    \n    **What you get:**\n    - Posts with captions, likes, comments, timestamps\n    - Profile information (for scrapeProfile operation)\n    - Hashtags and engagement metrics\n    - Owner information\n    \n    **Use cases:**\n    - Influencer analysis and discovery\n    - Brand monitoring and sentiment analysis\n    - Competitor research on Instagram\n    - Content strategy and trend analysis\n    - Market research through Instagram data\n    - Campaign performance tracking\n    - Hashtag research and optimization\n    \n    The tool uses best-available services behind the scenes while maintaining a consistent, simple interface.\n  ";
    static readonly alias = "ig";
    static readonly type = "tool";
    constructor(params?: InstagramToolParamsInput, context?: BubbleContext);
    performAction(): Promise<InstagramToolResult>;
    /**
     * Create an error result
     */
    private createErrorResult;
    /**
     * Handle scrapeProfile operation
     */
    private handleScrapeProfile;
    /**
     * Handle scrapeHashtag operation
     */
    private handleScrapeHashtag;
    /**
     * Scrape hashtags using Apify service
     * This is the current implementation - future versions could add other services
     */
    private scrapeWithApifyHashtags;
    /**
     * Normalize various profile inputs to Instagram URLs
     * Accepts: @username, username, https://instagram.com/username/
     */
    private normalizeProfiles;
    /**
     * Normalize hashtags for Apify actor
     * Removes # symbol and cleans format to match Apify requirements
     */
    private normalizeHashtags;
    /**
     * Scrape profiles using Apify service
     * This is the current implementation - future versions could add other services
     * Always fetches both profile details and posts for maximum flexibility
     */
    private scrapeWithApifyProfiles;
    /**
     * Extract username from Instagram URL
     */
    private extractUsername;
    /**
     * Normalize post type to standard enum
     */
    private normalizePostType;
    /**
     * Extract posts from Apify results
     * Handles both 'details' and 'posts' resultsType formats
     */
    private extractPosts;
    /**
     * Extract profile information from Apify results
     * Handles the 'details' resultsType format
     */
    private extractProfileInfo;
    /**
     * Extract posts from hashtag scraper results
     * Hashtag scraper returns posts directly (not nested)
     */
    private extractHashtagPosts;
}
export {};
//# sourceMappingURL=instagram-tool.d.ts.map