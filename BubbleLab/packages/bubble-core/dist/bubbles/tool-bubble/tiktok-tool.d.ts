import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const TikTokToolParamsSchema: z.ZodObject<{
    operation: z.ZodEnum<["scrapeProfile", "scrapeHashtag", "scrapeVideo"]>;
    profiles: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    hashtags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    videoUrls: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    limit: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    shouldDownloadVideos: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "scrapeProfile" | "scrapeHashtag" | "scrapeVideo";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    hashtags?: string[] | undefined;
    profiles?: string[] | undefined;
    shouldDownloadVideos?: boolean | undefined;
    videoUrls?: string[] | undefined;
}, {
    operation: "scrapeProfile" | "scrapeHashtag" | "scrapeVideo";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    hashtags?: string[] | undefined;
    profiles?: string[] | undefined;
    shouldDownloadVideos?: boolean | undefined;
    videoUrls?: string[] | undefined;
}>;
declare const TikTokToolResultSchema: z.ZodObject<{
    operation: z.ZodEnum<["scrapeProfile", "scrapeHashtag", "scrapeVideo"]>;
    videos: z.ZodArray<z.ZodObject<{
        id: z.ZodNullable<z.ZodString>;
        text: z.ZodNullable<z.ZodString>;
        createTime: z.ZodNullable<z.ZodNumber>;
        createTimeISO: z.ZodNullable<z.ZodString>;
        author: z.ZodNullable<z.ZodObject<{
            id: z.ZodNullable<z.ZodString>;
            uniqueId: z.ZodNullable<z.ZodString>;
            nickname: z.ZodNullable<z.ZodString>;
            avatarThumb: z.ZodNullable<z.ZodString>;
            signature: z.ZodNullable<z.ZodString>;
            verified: z.ZodNullable<z.ZodBoolean>;
            followerCount: z.ZodNullable<z.ZodNumber>;
            followingCount: z.ZodNullable<z.ZodNumber>;
            videoCount: z.ZodNullable<z.ZodNumber>;
            heartCount: z.ZodNullable<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            id: string | null;
            verified: boolean | null;
            signature: string | null;
            followingCount: number | null;
            uniqueId: string | null;
            nickname: string | null;
            avatarThumb: string | null;
            followerCount: number | null;
            videoCount: number | null;
            heartCount: number | null;
        }, {
            id: string | null;
            verified: boolean | null;
            signature: string | null;
            followingCount: number | null;
            uniqueId: string | null;
            nickname: string | null;
            avatarThumb: string | null;
            followerCount: number | null;
            videoCount: number | null;
            heartCount: number | null;
        }>>;
        stats: z.ZodNullable<z.ZodObject<{
            diggCount: z.ZodNullable<z.ZodNumber>;
            shareCount: z.ZodNullable<z.ZodNumber>;
            commentCount: z.ZodNullable<z.ZodNumber>;
            playCount: z.ZodNullable<z.ZodNumber>;
            collectCount: z.ZodNullable<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            collectCount: number | null;
            commentCount: number | null;
            diggCount: number | null;
            playCount: number | null;
            shareCount: number | null;
        }, {
            collectCount: number | null;
            commentCount: number | null;
            diggCount: number | null;
            playCount: number | null;
            shareCount: number | null;
        }>>;
        videoUrl: z.ZodNullable<z.ZodString>;
        webVideoUrl: z.ZodNullable<z.ZodString>;
        covers: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
        hashtags: z.ZodNullable<z.ZodArray<z.ZodObject<{
            name: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string | null;
        }, {
            name: string | null;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        text: string | null;
        id: string | null;
        hashtags: {
            name: string | null;
        }[] | null;
        videoUrl: string | null;
        author: {
            id: string | null;
            verified: boolean | null;
            signature: string | null;
            followingCount: number | null;
            uniqueId: string | null;
            nickname: string | null;
            avatarThumb: string | null;
            followerCount: number | null;
            videoCount: number | null;
            heartCount: number | null;
        } | null;
        stats: {
            collectCount: number | null;
            commentCount: number | null;
            diggCount: number | null;
            playCount: number | null;
            shareCount: number | null;
        } | null;
        createTime: number | null;
        createTimeISO: string | null;
        webVideoUrl: string | null;
        covers: string[] | null;
    }, {
        text: string | null;
        id: string | null;
        hashtags: {
            name: string | null;
        }[] | null;
        videoUrl: string | null;
        author: {
            id: string | null;
            verified: boolean | null;
            signature: string | null;
            followingCount: number | null;
            uniqueId: string | null;
            nickname: string | null;
            avatarThumb: string | null;
            followerCount: number | null;
            videoCount: number | null;
            heartCount: number | null;
        } | null;
        stats: {
            collectCount: number | null;
            commentCount: number | null;
            diggCount: number | null;
            playCount: number | null;
            shareCount: number | null;
        } | null;
        createTime: number | null;
        createTimeISO: string | null;
        webVideoUrl: string | null;
        covers: string[] | null;
    }>, "many">;
    totalVideos: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "scrapeProfile" | "scrapeHashtag" | "scrapeVideo";
    videos: {
        text: string | null;
        id: string | null;
        hashtags: {
            name: string | null;
        }[] | null;
        videoUrl: string | null;
        author: {
            id: string | null;
            verified: boolean | null;
            signature: string | null;
            followingCount: number | null;
            uniqueId: string | null;
            nickname: string | null;
            avatarThumb: string | null;
            followerCount: number | null;
            videoCount: number | null;
            heartCount: number | null;
        } | null;
        stats: {
            collectCount: number | null;
            commentCount: number | null;
            diggCount: number | null;
            playCount: number | null;
            shareCount: number | null;
        } | null;
        createTime: number | null;
        createTimeISO: string | null;
        webVideoUrl: string | null;
        covers: string[] | null;
    }[];
    totalVideos: number;
}, {
    error: string;
    success: boolean;
    operation: "scrapeProfile" | "scrapeHashtag" | "scrapeVideo";
    videos: {
        text: string | null;
        id: string | null;
        hashtags: {
            name: string | null;
        }[] | null;
        videoUrl: string | null;
        author: {
            id: string | null;
            verified: boolean | null;
            signature: string | null;
            followingCount: number | null;
            uniqueId: string | null;
            nickname: string | null;
            avatarThumb: string | null;
            followerCount: number | null;
            videoCount: number | null;
            heartCount: number | null;
        } | null;
        stats: {
            collectCount: number | null;
            commentCount: number | null;
            diggCount: number | null;
            playCount: number | null;
            shareCount: number | null;
        } | null;
        createTime: number | null;
        createTimeISO: string | null;
        webVideoUrl: string | null;
        covers: string[] | null;
    }[];
    totalVideos: number;
}>;
type TikTokToolParams = z.output<typeof TikTokToolParamsSchema>;
type TikTokToolResult = z.output<typeof TikTokToolResultSchema>;
type TikTokToolParamsInput = z.input<typeof TikTokToolParamsSchema>;
export declare class TikTokTool extends ToolBubble<TikTokToolParams, TikTokToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        operation: z.ZodEnum<["scrapeProfile", "scrapeHashtag", "scrapeVideo"]>;
        profiles: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        hashtags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        videoUrls: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        limit: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        shouldDownloadVideos: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "scrapeProfile" | "scrapeHashtag" | "scrapeVideo";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        hashtags?: string[] | undefined;
        profiles?: string[] | undefined;
        shouldDownloadVideos?: boolean | undefined;
        videoUrls?: string[] | undefined;
    }, {
        operation: "scrapeProfile" | "scrapeHashtag" | "scrapeVideo";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        hashtags?: string[] | undefined;
        profiles?: string[] | undefined;
        shouldDownloadVideos?: boolean | undefined;
        videoUrls?: string[] | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        operation: z.ZodEnum<["scrapeProfile", "scrapeHashtag", "scrapeVideo"]>;
        videos: z.ZodArray<z.ZodObject<{
            id: z.ZodNullable<z.ZodString>;
            text: z.ZodNullable<z.ZodString>;
            createTime: z.ZodNullable<z.ZodNumber>;
            createTimeISO: z.ZodNullable<z.ZodString>;
            author: z.ZodNullable<z.ZodObject<{
                id: z.ZodNullable<z.ZodString>;
                uniqueId: z.ZodNullable<z.ZodString>;
                nickname: z.ZodNullable<z.ZodString>;
                avatarThumb: z.ZodNullable<z.ZodString>;
                signature: z.ZodNullable<z.ZodString>;
                verified: z.ZodNullable<z.ZodBoolean>;
                followerCount: z.ZodNullable<z.ZodNumber>;
                followingCount: z.ZodNullable<z.ZodNumber>;
                videoCount: z.ZodNullable<z.ZodNumber>;
                heartCount: z.ZodNullable<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                id: string | null;
                verified: boolean | null;
                signature: string | null;
                followingCount: number | null;
                uniqueId: string | null;
                nickname: string | null;
                avatarThumb: string | null;
                followerCount: number | null;
                videoCount: number | null;
                heartCount: number | null;
            }, {
                id: string | null;
                verified: boolean | null;
                signature: string | null;
                followingCount: number | null;
                uniqueId: string | null;
                nickname: string | null;
                avatarThumb: string | null;
                followerCount: number | null;
                videoCount: number | null;
                heartCount: number | null;
            }>>;
            stats: z.ZodNullable<z.ZodObject<{
                diggCount: z.ZodNullable<z.ZodNumber>;
                shareCount: z.ZodNullable<z.ZodNumber>;
                commentCount: z.ZodNullable<z.ZodNumber>;
                playCount: z.ZodNullable<z.ZodNumber>;
                collectCount: z.ZodNullable<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                collectCount: number | null;
                commentCount: number | null;
                diggCount: number | null;
                playCount: number | null;
                shareCount: number | null;
            }, {
                collectCount: number | null;
                commentCount: number | null;
                diggCount: number | null;
                playCount: number | null;
                shareCount: number | null;
            }>>;
            videoUrl: z.ZodNullable<z.ZodString>;
            webVideoUrl: z.ZodNullable<z.ZodString>;
            covers: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
            hashtags: z.ZodNullable<z.ZodArray<z.ZodObject<{
                name: z.ZodNullable<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                name: string | null;
            }, {
                name: string | null;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            text: string | null;
            id: string | null;
            hashtags: {
                name: string | null;
            }[] | null;
            videoUrl: string | null;
            author: {
                id: string | null;
                verified: boolean | null;
                signature: string | null;
                followingCount: number | null;
                uniqueId: string | null;
                nickname: string | null;
                avatarThumb: string | null;
                followerCount: number | null;
                videoCount: number | null;
                heartCount: number | null;
            } | null;
            stats: {
                collectCount: number | null;
                commentCount: number | null;
                diggCount: number | null;
                playCount: number | null;
                shareCount: number | null;
            } | null;
            createTime: number | null;
            createTimeISO: string | null;
            webVideoUrl: string | null;
            covers: string[] | null;
        }, {
            text: string | null;
            id: string | null;
            hashtags: {
                name: string | null;
            }[] | null;
            videoUrl: string | null;
            author: {
                id: string | null;
                verified: boolean | null;
                signature: string | null;
                followingCount: number | null;
                uniqueId: string | null;
                nickname: string | null;
                avatarThumb: string | null;
                followerCount: number | null;
                videoCount: number | null;
                heartCount: number | null;
            } | null;
            stats: {
                collectCount: number | null;
                commentCount: number | null;
                diggCount: number | null;
                playCount: number | null;
                shareCount: number | null;
            } | null;
            createTime: number | null;
            createTimeISO: string | null;
            webVideoUrl: string | null;
            covers: string[] | null;
        }>, "many">;
        totalVideos: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "scrapeProfile" | "scrapeHashtag" | "scrapeVideo";
        videos: {
            text: string | null;
            id: string | null;
            hashtags: {
                name: string | null;
            }[] | null;
            videoUrl: string | null;
            author: {
                id: string | null;
                verified: boolean | null;
                signature: string | null;
                followingCount: number | null;
                uniqueId: string | null;
                nickname: string | null;
                avatarThumb: string | null;
                followerCount: number | null;
                videoCount: number | null;
                heartCount: number | null;
            } | null;
            stats: {
                collectCount: number | null;
                commentCount: number | null;
                diggCount: number | null;
                playCount: number | null;
                shareCount: number | null;
            } | null;
            createTime: number | null;
            createTimeISO: string | null;
            webVideoUrl: string | null;
            covers: string[] | null;
        }[];
        totalVideos: number;
    }, {
        error: string;
        success: boolean;
        operation: "scrapeProfile" | "scrapeHashtag" | "scrapeVideo";
        videos: {
            text: string | null;
            id: string | null;
            hashtags: {
                name: string | null;
            }[] | null;
            videoUrl: string | null;
            author: {
                id: string | null;
                verified: boolean | null;
                signature: string | null;
                followingCount: number | null;
                uniqueId: string | null;
                nickname: string | null;
                avatarThumb: string | null;
                followerCount: number | null;
                videoCount: number | null;
                heartCount: number | null;
            } | null;
            stats: {
                collectCount: number | null;
                commentCount: number | null;
                diggCount: number | null;
                playCount: number | null;
                shareCount: number | null;
            } | null;
            createTime: number | null;
            createTimeISO: string | null;
            webVideoUrl: string | null;
            covers: string[] | null;
        }[];
        totalVideos: number;
    }>;
    static readonly shortDescription = "Scrape TikTok profiles, videos, and hashtags.";
    static readonly longDescription = "\n    Universal TikTok scraping tool.\n    \n    Operations:\n    - scrapeProfile: Get videos from user profiles\n    - scrapeHashtag: Get videos by hashtag\n    - scrapeVideo: Get details for specific videos\n    \n    Uses Apify's clockworks/tiktok-scraper.\n  ";
    static readonly alias = "tiktok";
    static readonly type = "tool";
    constructor(params?: TikTokToolParamsInput, context?: BubbleContext);
    performAction(): Promise<TikTokToolResult>;
    private createErrorResult;
    private runScraper;
    private transformVideos;
}
export {};
//# sourceMappingURL=tiktok-tool.d.ts.map