import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const TwitterUserSchema: z.ZodObject<{
    id: z.ZodNullable<z.ZodString>;
    name: z.ZodNullable<z.ZodString>;
    userName: z.ZodNullable<z.ZodString>;
    description: z.ZodNullable<z.ZodString>;
    isVerified: z.ZodNullable<z.ZodBoolean>;
    isBlueVerified: z.ZodNullable<z.ZodBoolean>;
    profilePicture: z.ZodNullable<z.ZodString>;
    followers: z.ZodNullable<z.ZodNumber>;
    following: z.ZodNullable<z.ZodNumber>;
    tweetsCount: z.ZodNullable<z.ZodNumber>;
    url: z.ZodNullable<z.ZodString>;
    createdAt: z.ZodNullable<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    description: string | null;
    name: string | null;
    url: string | null;
    id: string | null;
    following: number | null;
    userName: string | null;
    isVerified: boolean | null;
    isBlueVerified: boolean | null;
    profilePicture: string | null;
    followers: number | null;
    tweetsCount: number | null;
    createdAt: string | null;
}, {
    description: string | null;
    name: string | null;
    url: string | null;
    id: string | null;
    following: number | null;
    userName: string | null;
    isVerified: boolean | null;
    isBlueVerified: boolean | null;
    profilePicture: string | null;
    followers: number | null;
    tweetsCount: number | null;
    createdAt: string | null;
}>;
declare const TwitterTweetSchema: z.ZodObject<{
    id: z.ZodNullable<z.ZodString>;
    url: z.ZodNullable<z.ZodString>;
    text: z.ZodNullable<z.ZodString>;
    author: z.ZodNullable<z.ZodObject<{
        id: z.ZodNullable<z.ZodString>;
        name: z.ZodNullable<z.ZodString>;
        userName: z.ZodNullable<z.ZodString>;
        description: z.ZodNullable<z.ZodString>;
        isVerified: z.ZodNullable<z.ZodBoolean>;
        isBlueVerified: z.ZodNullable<z.ZodBoolean>;
        profilePicture: z.ZodNullable<z.ZodString>;
        followers: z.ZodNullable<z.ZodNumber>;
        following: z.ZodNullable<z.ZodNumber>;
        tweetsCount: z.ZodNullable<z.ZodNumber>;
        url: z.ZodNullable<z.ZodString>;
        createdAt: z.ZodNullable<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        description: string | null;
        name: string | null;
        url: string | null;
        id: string | null;
        following: number | null;
        userName: string | null;
        isVerified: boolean | null;
        isBlueVerified: boolean | null;
        profilePicture: string | null;
        followers: number | null;
        tweetsCount: number | null;
        createdAt: string | null;
    }, {
        description: string | null;
        name: string | null;
        url: string | null;
        id: string | null;
        following: number | null;
        userName: string | null;
        isVerified: boolean | null;
        isBlueVerified: boolean | null;
        profilePicture: string | null;
        followers: number | null;
        tweetsCount: number | null;
        createdAt: string | null;
    }>>;
    createdAt: z.ZodNullable<z.ZodString>;
    stats: z.ZodNullable<z.ZodObject<{
        retweetCount: z.ZodNullable<z.ZodNumber>;
        replyCount: z.ZodNullable<z.ZodNumber>;
        likeCount: z.ZodNullable<z.ZodNumber>;
        quoteCount: z.ZodNullable<z.ZodNumber>;
        viewCount: z.ZodNullable<z.ZodNumber>;
        bookmarkCount: z.ZodNullable<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        viewCount: number | null;
        retweetCount: number | null;
        replyCount: number | null;
        likeCount: number | null;
        quoteCount: number | null;
        bookmarkCount: number | null;
    }, {
        viewCount: number | null;
        retweetCount: number | null;
        replyCount: number | null;
        likeCount: number | null;
        quoteCount: number | null;
        bookmarkCount: number | null;
    }>>;
    lang: z.ZodNullable<z.ZodString>;
    media: z.ZodNullable<z.ZodArray<z.ZodObject<{
        type: z.ZodNullable<z.ZodString>;
        url: z.ZodNullable<z.ZodString>;
        width: z.ZodNullable<z.ZodNumber>;
        height: z.ZodNullable<z.ZodNumber>;
        duration: z.ZodNullable<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        type: string | null;
        url: string | null;
        duration: number | null;
        width: number | null;
        height: number | null;
    }, {
        type: string | null;
        url: string | null;
        duration: number | null;
        width: number | null;
        height: number | null;
    }>, "many">>;
    entities: z.ZodNullable<z.ZodObject<{
        hashtags: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
        urls: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
        mentions: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        hashtags: string[] | null;
        mentions: string[] | null;
        urls: string[] | null;
    }, {
        hashtags: string[] | null;
        mentions: string[] | null;
        urls: string[] | null;
    }>>;
    isRetweet: z.ZodNullable<z.ZodBoolean>;
    isQuote: z.ZodNullable<z.ZodBoolean>;
    isReply: z.ZodNullable<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    url: string | null;
    text: string | null;
    id: string | null;
    author: {
        description: string | null;
        name: string | null;
        url: string | null;
        id: string | null;
        following: number | null;
        userName: string | null;
        isVerified: boolean | null;
        isBlueVerified: boolean | null;
        profilePicture: string | null;
        followers: number | null;
        tweetsCount: number | null;
        createdAt: string | null;
    } | null;
    stats: {
        viewCount: number | null;
        retweetCount: number | null;
        replyCount: number | null;
        likeCount: number | null;
        quoteCount: number | null;
        bookmarkCount: number | null;
    } | null;
    media: {
        type: string | null;
        url: string | null;
        duration: number | null;
        width: number | null;
        height: number | null;
    }[] | null;
    createdAt: string | null;
    lang: string | null;
    entities: {
        hashtags: string[] | null;
        mentions: string[] | null;
        urls: string[] | null;
    } | null;
    isRetweet: boolean | null;
    isQuote: boolean | null;
    isReply: boolean | null;
}, {
    url: string | null;
    text: string | null;
    id: string | null;
    author: {
        description: string | null;
        name: string | null;
        url: string | null;
        id: string | null;
        following: number | null;
        userName: string | null;
        isVerified: boolean | null;
        isBlueVerified: boolean | null;
        profilePicture: string | null;
        followers: number | null;
        tweetsCount: number | null;
        createdAt: string | null;
    } | null;
    stats: {
        viewCount: number | null;
        retweetCount: number | null;
        replyCount: number | null;
        likeCount: number | null;
        quoteCount: number | null;
        bookmarkCount: number | null;
    } | null;
    media: {
        type: string | null;
        url: string | null;
        duration: number | null;
        width: number | null;
        height: number | null;
    }[] | null;
    createdAt: string | null;
    lang: string | null;
    entities: {
        hashtags: string[] | null;
        mentions: string[] | null;
        urls: string[] | null;
    } | null;
    isRetweet: boolean | null;
    isQuote: boolean | null;
    isReply: boolean | null;
}>;
declare const TwitterToolParamsSchema: z.ZodObject<{
    operation: z.ZodEnum<["scrapeProfile", "search", "scrapeUrl"]>;
    twitterHandles: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    searchTerms: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    startUrls: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    maxItems: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    sort: z.ZodOptional<z.ZodEnum<["Top", "Latest"]>>;
    tweetLanguage: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "search" | "scrapeProfile" | "scrapeUrl";
    sort?: "Top" | "Latest" | undefined;
    maxItems?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    startUrls?: string[] | undefined;
    searchTerms?: string[] | undefined;
    twitterHandles?: string[] | undefined;
    tweetLanguage?: string | undefined;
}, {
    operation: "search" | "scrapeProfile" | "scrapeUrl";
    sort?: "Top" | "Latest" | undefined;
    maxItems?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    startUrls?: string[] | undefined;
    searchTerms?: string[] | undefined;
    twitterHandles?: string[] | undefined;
    tweetLanguage?: string | undefined;
}>;
declare const TwitterToolResultSchema: z.ZodObject<{
    operation: z.ZodEnum<["scrapeProfile", "search", "scrapeUrl"]>;
    tweets: z.ZodArray<z.ZodObject<{
        id: z.ZodNullable<z.ZodString>;
        url: z.ZodNullable<z.ZodString>;
        text: z.ZodNullable<z.ZodString>;
        author: z.ZodNullable<z.ZodObject<{
            id: z.ZodNullable<z.ZodString>;
            name: z.ZodNullable<z.ZodString>;
            userName: z.ZodNullable<z.ZodString>;
            description: z.ZodNullable<z.ZodString>;
            isVerified: z.ZodNullable<z.ZodBoolean>;
            isBlueVerified: z.ZodNullable<z.ZodBoolean>;
            profilePicture: z.ZodNullable<z.ZodString>;
            followers: z.ZodNullable<z.ZodNumber>;
            following: z.ZodNullable<z.ZodNumber>;
            tweetsCount: z.ZodNullable<z.ZodNumber>;
            url: z.ZodNullable<z.ZodString>;
            createdAt: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            description: string | null;
            name: string | null;
            url: string | null;
            id: string | null;
            following: number | null;
            userName: string | null;
            isVerified: boolean | null;
            isBlueVerified: boolean | null;
            profilePicture: string | null;
            followers: number | null;
            tweetsCount: number | null;
            createdAt: string | null;
        }, {
            description: string | null;
            name: string | null;
            url: string | null;
            id: string | null;
            following: number | null;
            userName: string | null;
            isVerified: boolean | null;
            isBlueVerified: boolean | null;
            profilePicture: string | null;
            followers: number | null;
            tweetsCount: number | null;
            createdAt: string | null;
        }>>;
        createdAt: z.ZodNullable<z.ZodString>;
        stats: z.ZodNullable<z.ZodObject<{
            retweetCount: z.ZodNullable<z.ZodNumber>;
            replyCount: z.ZodNullable<z.ZodNumber>;
            likeCount: z.ZodNullable<z.ZodNumber>;
            quoteCount: z.ZodNullable<z.ZodNumber>;
            viewCount: z.ZodNullable<z.ZodNumber>;
            bookmarkCount: z.ZodNullable<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            viewCount: number | null;
            retweetCount: number | null;
            replyCount: number | null;
            likeCount: number | null;
            quoteCount: number | null;
            bookmarkCount: number | null;
        }, {
            viewCount: number | null;
            retweetCount: number | null;
            replyCount: number | null;
            likeCount: number | null;
            quoteCount: number | null;
            bookmarkCount: number | null;
        }>>;
        lang: z.ZodNullable<z.ZodString>;
        media: z.ZodNullable<z.ZodArray<z.ZodObject<{
            type: z.ZodNullable<z.ZodString>;
            url: z.ZodNullable<z.ZodString>;
            width: z.ZodNullable<z.ZodNumber>;
            height: z.ZodNullable<z.ZodNumber>;
            duration: z.ZodNullable<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            type: string | null;
            url: string | null;
            duration: number | null;
            width: number | null;
            height: number | null;
        }, {
            type: string | null;
            url: string | null;
            duration: number | null;
            width: number | null;
            height: number | null;
        }>, "many">>;
        entities: z.ZodNullable<z.ZodObject<{
            hashtags: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
            urls: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
            mentions: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            hashtags: string[] | null;
            mentions: string[] | null;
            urls: string[] | null;
        }, {
            hashtags: string[] | null;
            mentions: string[] | null;
            urls: string[] | null;
        }>>;
        isRetweet: z.ZodNullable<z.ZodBoolean>;
        isQuote: z.ZodNullable<z.ZodBoolean>;
        isReply: z.ZodNullable<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        url: string | null;
        text: string | null;
        id: string | null;
        author: {
            description: string | null;
            name: string | null;
            url: string | null;
            id: string | null;
            following: number | null;
            userName: string | null;
            isVerified: boolean | null;
            isBlueVerified: boolean | null;
            profilePicture: string | null;
            followers: number | null;
            tweetsCount: number | null;
            createdAt: string | null;
        } | null;
        stats: {
            viewCount: number | null;
            retweetCount: number | null;
            replyCount: number | null;
            likeCount: number | null;
            quoteCount: number | null;
            bookmarkCount: number | null;
        } | null;
        media: {
            type: string | null;
            url: string | null;
            duration: number | null;
            width: number | null;
            height: number | null;
        }[] | null;
        createdAt: string | null;
        lang: string | null;
        entities: {
            hashtags: string[] | null;
            mentions: string[] | null;
            urls: string[] | null;
        } | null;
        isRetweet: boolean | null;
        isQuote: boolean | null;
        isReply: boolean | null;
    }, {
        url: string | null;
        text: string | null;
        id: string | null;
        author: {
            description: string | null;
            name: string | null;
            url: string | null;
            id: string | null;
            following: number | null;
            userName: string | null;
            isVerified: boolean | null;
            isBlueVerified: boolean | null;
            profilePicture: string | null;
            followers: number | null;
            tweetsCount: number | null;
            createdAt: string | null;
        } | null;
        stats: {
            viewCount: number | null;
            retweetCount: number | null;
            replyCount: number | null;
            likeCount: number | null;
            quoteCount: number | null;
            bookmarkCount: number | null;
        } | null;
        media: {
            type: string | null;
            url: string | null;
            duration: number | null;
            width: number | null;
            height: number | null;
        }[] | null;
        createdAt: string | null;
        lang: string | null;
        entities: {
            hashtags: string[] | null;
            mentions: string[] | null;
            urls: string[] | null;
        } | null;
        isRetweet: boolean | null;
        isQuote: boolean | null;
        isReply: boolean | null;
    }>, "many">;
    totalTweets: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "search" | "scrapeProfile" | "scrapeUrl";
    tweets: {
        url: string | null;
        text: string | null;
        id: string | null;
        author: {
            description: string | null;
            name: string | null;
            url: string | null;
            id: string | null;
            following: number | null;
            userName: string | null;
            isVerified: boolean | null;
            isBlueVerified: boolean | null;
            profilePicture: string | null;
            followers: number | null;
            tweetsCount: number | null;
            createdAt: string | null;
        } | null;
        stats: {
            viewCount: number | null;
            retweetCount: number | null;
            replyCount: number | null;
            likeCount: number | null;
            quoteCount: number | null;
            bookmarkCount: number | null;
        } | null;
        media: {
            type: string | null;
            url: string | null;
            duration: number | null;
            width: number | null;
            height: number | null;
        }[] | null;
        createdAt: string | null;
        lang: string | null;
        entities: {
            hashtags: string[] | null;
            mentions: string[] | null;
            urls: string[] | null;
        } | null;
        isRetweet: boolean | null;
        isQuote: boolean | null;
        isReply: boolean | null;
    }[];
    totalTweets: number;
}, {
    error: string;
    success: boolean;
    operation: "search" | "scrapeProfile" | "scrapeUrl";
    tweets: {
        url: string | null;
        text: string | null;
        id: string | null;
        author: {
            description: string | null;
            name: string | null;
            url: string | null;
            id: string | null;
            following: number | null;
            userName: string | null;
            isVerified: boolean | null;
            isBlueVerified: boolean | null;
            profilePicture: string | null;
            followers: number | null;
            tweetsCount: number | null;
            createdAt: string | null;
        } | null;
        stats: {
            viewCount: number | null;
            retweetCount: number | null;
            replyCount: number | null;
            likeCount: number | null;
            quoteCount: number | null;
            bookmarkCount: number | null;
        } | null;
        media: {
            type: string | null;
            url: string | null;
            duration: number | null;
            width: number | null;
            height: number | null;
        }[] | null;
        createdAt: string | null;
        lang: string | null;
        entities: {
            hashtags: string[] | null;
            mentions: string[] | null;
            urls: string[] | null;
        } | null;
        isRetweet: boolean | null;
        isQuote: boolean | null;
        isReply: boolean | null;
    }[];
    totalTweets: number;
}>;
type TwitterToolParams = z.output<typeof TwitterToolParamsSchema>;
type TwitterToolResult = z.output<typeof TwitterToolResultSchema>;
type TwitterToolParamsInput = z.input<typeof TwitterToolParamsSchema>;
export type TwitterTweet = z.output<typeof TwitterTweetSchema>;
export type TwitterUser = z.output<typeof TwitterUserSchema>;
export type TwitterOperationResult<T extends TwitterToolParams['operation']> = Extract<TwitterToolResult, {
    operation: T;
}>;
/**
 * Generic Twitter/X scraping tool with unified interface
 *
 * This tool abstracts away the underlying scraping service (currently Apify)
 * and provides a simple, opinionated interface for Twitter data extraction.
 *
 * Supports three operations:
 * - scrapeProfile: Scrape user profiles and their tweets
 * - search: Search for tweets by keywords or hashtags
 * - scrapeUrl: Scrape specific Twitter URLs (tweets, profiles, searches, lists)
 *
 * Future versions can add support for other services (BrightData, custom scrapers)
 * while maintaining the same interface.
 */
export declare class TwitterTool extends ToolBubble<TwitterToolParams, TwitterToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        operation: z.ZodEnum<["scrapeProfile", "search", "scrapeUrl"]>;
        twitterHandles: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        searchTerms: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        startUrls: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        maxItems: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        sort: z.ZodOptional<z.ZodEnum<["Top", "Latest"]>>;
        tweetLanguage: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "search" | "scrapeProfile" | "scrapeUrl";
        sort?: "Top" | "Latest" | undefined;
        maxItems?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        startUrls?: string[] | undefined;
        searchTerms?: string[] | undefined;
        twitterHandles?: string[] | undefined;
        tweetLanguage?: string | undefined;
    }, {
        operation: "search" | "scrapeProfile" | "scrapeUrl";
        sort?: "Top" | "Latest" | undefined;
        maxItems?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        startUrls?: string[] | undefined;
        searchTerms?: string[] | undefined;
        twitterHandles?: string[] | undefined;
        tweetLanguage?: string | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        operation: z.ZodEnum<["scrapeProfile", "search", "scrapeUrl"]>;
        tweets: z.ZodArray<z.ZodObject<{
            id: z.ZodNullable<z.ZodString>;
            url: z.ZodNullable<z.ZodString>;
            text: z.ZodNullable<z.ZodString>;
            author: z.ZodNullable<z.ZodObject<{
                id: z.ZodNullable<z.ZodString>;
                name: z.ZodNullable<z.ZodString>;
                userName: z.ZodNullable<z.ZodString>;
                description: z.ZodNullable<z.ZodString>;
                isVerified: z.ZodNullable<z.ZodBoolean>;
                isBlueVerified: z.ZodNullable<z.ZodBoolean>;
                profilePicture: z.ZodNullable<z.ZodString>;
                followers: z.ZodNullable<z.ZodNumber>;
                following: z.ZodNullable<z.ZodNumber>;
                tweetsCount: z.ZodNullable<z.ZodNumber>;
                url: z.ZodNullable<z.ZodString>;
                createdAt: z.ZodNullable<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                description: string | null;
                name: string | null;
                url: string | null;
                id: string | null;
                following: number | null;
                userName: string | null;
                isVerified: boolean | null;
                isBlueVerified: boolean | null;
                profilePicture: string | null;
                followers: number | null;
                tweetsCount: number | null;
                createdAt: string | null;
            }, {
                description: string | null;
                name: string | null;
                url: string | null;
                id: string | null;
                following: number | null;
                userName: string | null;
                isVerified: boolean | null;
                isBlueVerified: boolean | null;
                profilePicture: string | null;
                followers: number | null;
                tweetsCount: number | null;
                createdAt: string | null;
            }>>;
            createdAt: z.ZodNullable<z.ZodString>;
            stats: z.ZodNullable<z.ZodObject<{
                retweetCount: z.ZodNullable<z.ZodNumber>;
                replyCount: z.ZodNullable<z.ZodNumber>;
                likeCount: z.ZodNullable<z.ZodNumber>;
                quoteCount: z.ZodNullable<z.ZodNumber>;
                viewCount: z.ZodNullable<z.ZodNumber>;
                bookmarkCount: z.ZodNullable<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                viewCount: number | null;
                retweetCount: number | null;
                replyCount: number | null;
                likeCount: number | null;
                quoteCount: number | null;
                bookmarkCount: number | null;
            }, {
                viewCount: number | null;
                retweetCount: number | null;
                replyCount: number | null;
                likeCount: number | null;
                quoteCount: number | null;
                bookmarkCount: number | null;
            }>>;
            lang: z.ZodNullable<z.ZodString>;
            media: z.ZodNullable<z.ZodArray<z.ZodObject<{
                type: z.ZodNullable<z.ZodString>;
                url: z.ZodNullable<z.ZodString>;
                width: z.ZodNullable<z.ZodNumber>;
                height: z.ZodNullable<z.ZodNumber>;
                duration: z.ZodNullable<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                type: string | null;
                url: string | null;
                duration: number | null;
                width: number | null;
                height: number | null;
            }, {
                type: string | null;
                url: string | null;
                duration: number | null;
                width: number | null;
                height: number | null;
            }>, "many">>;
            entities: z.ZodNullable<z.ZodObject<{
                hashtags: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
                urls: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
                mentions: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
            }, "strip", z.ZodTypeAny, {
                hashtags: string[] | null;
                mentions: string[] | null;
                urls: string[] | null;
            }, {
                hashtags: string[] | null;
                mentions: string[] | null;
                urls: string[] | null;
            }>>;
            isRetweet: z.ZodNullable<z.ZodBoolean>;
            isQuote: z.ZodNullable<z.ZodBoolean>;
            isReply: z.ZodNullable<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            url: string | null;
            text: string | null;
            id: string | null;
            author: {
                description: string | null;
                name: string | null;
                url: string | null;
                id: string | null;
                following: number | null;
                userName: string | null;
                isVerified: boolean | null;
                isBlueVerified: boolean | null;
                profilePicture: string | null;
                followers: number | null;
                tweetsCount: number | null;
                createdAt: string | null;
            } | null;
            stats: {
                viewCount: number | null;
                retweetCount: number | null;
                replyCount: number | null;
                likeCount: number | null;
                quoteCount: number | null;
                bookmarkCount: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                duration: number | null;
                width: number | null;
                height: number | null;
            }[] | null;
            createdAt: string | null;
            lang: string | null;
            entities: {
                hashtags: string[] | null;
                mentions: string[] | null;
                urls: string[] | null;
            } | null;
            isRetweet: boolean | null;
            isQuote: boolean | null;
            isReply: boolean | null;
        }, {
            url: string | null;
            text: string | null;
            id: string | null;
            author: {
                description: string | null;
                name: string | null;
                url: string | null;
                id: string | null;
                following: number | null;
                userName: string | null;
                isVerified: boolean | null;
                isBlueVerified: boolean | null;
                profilePicture: string | null;
                followers: number | null;
                tweetsCount: number | null;
                createdAt: string | null;
            } | null;
            stats: {
                viewCount: number | null;
                retweetCount: number | null;
                replyCount: number | null;
                likeCount: number | null;
                quoteCount: number | null;
                bookmarkCount: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                duration: number | null;
                width: number | null;
                height: number | null;
            }[] | null;
            createdAt: string | null;
            lang: string | null;
            entities: {
                hashtags: string[] | null;
                mentions: string[] | null;
                urls: string[] | null;
            } | null;
            isRetweet: boolean | null;
            isQuote: boolean | null;
            isReply: boolean | null;
        }>, "many">;
        totalTweets: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "search" | "scrapeProfile" | "scrapeUrl";
        tweets: {
            url: string | null;
            text: string | null;
            id: string | null;
            author: {
                description: string | null;
                name: string | null;
                url: string | null;
                id: string | null;
                following: number | null;
                userName: string | null;
                isVerified: boolean | null;
                isBlueVerified: boolean | null;
                profilePicture: string | null;
                followers: number | null;
                tweetsCount: number | null;
                createdAt: string | null;
            } | null;
            stats: {
                viewCount: number | null;
                retweetCount: number | null;
                replyCount: number | null;
                likeCount: number | null;
                quoteCount: number | null;
                bookmarkCount: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                duration: number | null;
                width: number | null;
                height: number | null;
            }[] | null;
            createdAt: string | null;
            lang: string | null;
            entities: {
                hashtags: string[] | null;
                mentions: string[] | null;
                urls: string[] | null;
            } | null;
            isRetweet: boolean | null;
            isQuote: boolean | null;
            isReply: boolean | null;
        }[];
        totalTweets: number;
    }, {
        error: string;
        success: boolean;
        operation: "search" | "scrapeProfile" | "scrapeUrl";
        tweets: {
            url: string | null;
            text: string | null;
            id: string | null;
            author: {
                description: string | null;
                name: string | null;
                url: string | null;
                id: string | null;
                following: number | null;
                userName: string | null;
                isVerified: boolean | null;
                isBlueVerified: boolean | null;
                profilePicture: string | null;
                followers: number | null;
                tweetsCount: number | null;
                createdAt: string | null;
            } | null;
            stats: {
                viewCount: number | null;
                retweetCount: number | null;
                replyCount: number | null;
                likeCount: number | null;
                quoteCount: number | null;
                bookmarkCount: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                duration: number | null;
                width: number | null;
                height: number | null;
            }[] | null;
            createdAt: string | null;
            lang: string | null;
            entities: {
                hashtags: string[] | null;
                mentions: string[] | null;
                urls: string[] | null;
            } | null;
            isRetweet: boolean | null;
            isQuote: boolean | null;
            isReply: boolean | null;
        }[];
        totalTweets: number;
    }>;
    static readonly shortDescription = "Scrape Twitter/X profiles, tweets, and search results with a simple, unified interface.";
    static readonly longDescription = "\n    Universal Twitter/X scraping tool that provides a simple, opinionated interface for extracting Twitter data.\n    \n    **OPERATIONS:**\n    1. **scrapeProfile**: Scrape user profiles and their tweets\n       - Get tweets from specific user handles\n       - Track influencer or brand accounts\n       - Monitor user activity and engagement\n    \n    2. **search**: Search for tweets by keywords or hashtags\n       - Find tweets by search terms or hashtags\n       - Monitor brand mentions and campaigns\n       - Research trending topics and conversations\n       - Supports advanced search syntax (see Twitter advanced search)\n    \n    3. **scrapeUrl**: Scrape specific Twitter URLs\n       - Scrape individual tweets, profiles, search results, or lists\n       - Extract data from specific Twitter URLs\n       - Useful for targeted data collection\n    \n    **WHEN TO USE THIS TOOL:**\n    - **Any Twitter scraping task** - profiles, tweets, searches, engagement data\n    - **Social media research** - influencer analysis, competitor monitoring\n    - **Content gathering** - tweets, replies, retweets, engagement metrics\n    - **Market research** - brand mentions, user sentiment on Twitter\n    - **Trend analysis** - hashtag tracking, viral content discovery\n    - **Real-time monitoring** - track conversations and mentions\n    \n    **DO NOT USE research-agent-tool or web-scrape-tool for Twitter** - This tool is specifically optimized for Twitter and provides:\n    - Unified data format across all Twitter sources\n    - Automatic service selection and optimization\n    - Rate limiting and reliability handling\n    - Clean, structured data ready for analysis\n    \n    **Simple Interface:**\n    Just specify the operation and provide Twitter handles, search terms, or URLs to get back clean, structured data.\n    The tool automatically handles:\n    - Handle normalization (accepts handles with or without @)\n    - Service selection (currently Apify, future: multiple sources)\n    - Data transformation to unified format\n    - Error handling and retries\n    \n    **What you get:**\n    - Tweets with text, engagement stats, timestamps\n    - Author information (for scrapeProfile operation)\n    - Hashtags, mentions, and URLs\n    - Media attachments\n    - Language and metadata\n    \n    **Use cases:**\n    - Influencer analysis and discovery\n    - Brand monitoring and sentiment analysis\n    - Competitor research on Twitter\n    - Content strategy and trend analysis\n    - Market research through Twitter data\n    - Campaign performance tracking\n    - Hashtag research and optimization\n    - Real-time event monitoring\n    \n    The tool uses best-available services behind the scenes while maintaining a consistent, simple interface.\n  ";
    static readonly alias = "twitter";
    static readonly type = "tool";
    constructor(params?: TwitterToolParamsInput, context?: BubbleContext);
    performAction(): Promise<TwitterToolResult>;
    /**
     * Create an error result
     */
    private createErrorResult;
    /**
     * Handle scrapeProfile operation
     */
    private handleScrapeProfile;
    /**
     * Handle search operation
     */
    private handleSearch;
    /**
     * Handle scrapeUrl operation
     */
    private handleScrapeUrl;
    /**
     * Scrape profiles using Apify service
     * This is the current implementation - future versions could add other services
     */
    private scrapeWithApifyProfiles;
    /**
     * Search tweets using Apify service
     * This is the current implementation - future versions could add other services
     */
    private scrapeWithApifySearch;
    /**
     * Scrape URLs using Apify service
     * This is the current implementation - future versions could add other services
     */
    private scrapeWithApifyUrls;
    /**
     * Normalize Twitter handles (remove @ if present)
     */
    private normalizeHandles;
    private transformTweets;
}
export {};
//# sourceMappingURL=twitter-tool.d.ts.map