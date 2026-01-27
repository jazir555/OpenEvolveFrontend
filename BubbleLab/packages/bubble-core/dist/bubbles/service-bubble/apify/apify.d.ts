import { z } from 'zod';
import { ServiceBubble } from '../../../types/service-bubble-class.js';
import type { BubbleContext } from '../../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import type { ActorId, ActorOutput, ActorInput } from './types.js';
/**
 * Generic Apify Bubble - Works with ANY Apify Actor
 *
 * This is a universal service bubble that can run any Apify actor.
 * Actor-specific logic and data transformation should be handled by Tool Bubbles.
 *
 * Examples:
 * - InstagramTool uses this to run 'apify/instagram-scraper'
 * - RedditTool could use this to run 'apify/reddit-scraper'
 * - LinkedInTool could use this to run 'apify/linkedin-scraper'
 */
declare const ApifyParamsSchema: z.ZodObject<{
    actorId: z.ZodOptional<z.ZodString>;
    search: z.ZodOptional<z.ZodString>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    input: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    waitForFinish: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    input: Record<string, unknown>;
    limit: number;
    waitForFinish: boolean;
    search?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    actorId?: string | undefined;
}, {
    input: Record<string, unknown>;
    timeout?: number | undefined;
    search?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    actorId?: string | undefined;
    waitForFinish?: boolean | undefined;
}>;
declare const ApifyResultSchema: z.ZodObject<{
    runId: z.ZodString;
    status: z.ZodString;
    datasetId: z.ZodOptional<z.ZodString>;
    items: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    itemsCount: z.ZodOptional<z.ZodNumber>;
    consoleUrl: z.ZodString;
    success: z.ZodBoolean;
    error: z.ZodString;
    discoveredActors: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        inputSchemaUrl: z.ZodString;
        stars: z.ZodOptional<z.ZodNullable<z.ZodNumber>>;
        usage: z.ZodOptional<z.ZodNullable<z.ZodObject<{
            totalRuns: z.ZodOptional<z.ZodNumber>;
            usersCount: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            totalRuns?: number | undefined;
            usersCount?: number | undefined;
        }, {
            totalRuns?: number | undefined;
            usersCount?: number | undefined;
        }>>>;
        requiresRental: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        inputSchemaUrl: string;
        description?: string | null | undefined;
        stars?: number | null | undefined;
        usage?: {
            totalRuns?: number | undefined;
            usersCount?: number | undefined;
        } | null | undefined;
        requiresRental?: boolean | undefined;
    }, {
        name: string;
        id: string;
        inputSchemaUrl: string;
        description?: string | null | undefined;
        stars?: number | null | undefined;
        usage?: {
            totalRuns?: number | undefined;
            usersCount?: number | undefined;
        } | null | undefined;
        requiresRental?: boolean | undefined;
    }>, "many">>;
}, "strip", z.ZodTypeAny, {
    error: string;
    status: string;
    success: boolean;
    runId: string;
    consoleUrl: string;
    items?: unknown[] | undefined;
    datasetId?: string | undefined;
    itemsCount?: number | undefined;
    discoveredActors?: {
        name: string;
        id: string;
        inputSchemaUrl: string;
        description?: string | null | undefined;
        stars?: number | null | undefined;
        usage?: {
            totalRuns?: number | undefined;
            usersCount?: number | undefined;
        } | null | undefined;
        requiresRental?: boolean | undefined;
    }[] | undefined;
}, {
    error: string;
    status: string;
    success: boolean;
    runId: string;
    consoleUrl: string;
    items?: unknown[] | undefined;
    datasetId?: string | undefined;
    itemsCount?: number | undefined;
    discoveredActors?: {
        name: string;
        id: string;
        inputSchemaUrl: string;
        description?: string | null | undefined;
        stars?: number | null | undefined;
        usage?: {
            totalRuns?: number | undefined;
            usersCount?: number | undefined;
        } | null | undefined;
        requiresRental?: boolean | undefined;
    }[] | undefined;
}>;
export type ApifyParamsInput = z.input<typeof ApifyParamsSchema>;
export type ApifyActorInput = Record<string, unknown>;
type ApifyParams = z.output<typeof ApifyParamsSchema>;
type ApifyResult = z.output<typeof ApifyResultSchema>;
type TypedApifyInput<T extends string> = T extends ActorId ? ActorInput<T> : Record<string, unknown>;
type TypedApifyResult<T extends string> = T extends ActorId ? Omit<ApifyResult, 'items'> & {
    items?: ActorOutput<T>[];
} : ApifyResult;
type TypedApifyParams<T extends string> = Omit<ApifyParams, 'input'> & {
    input: TypedApifyInput<T>;
};
export type TypedApifyParamsInput<T extends string> = Omit<ApifyParamsInput, 'input'> & {
    input: TypedApifyInput<T>;
};
/**
 * Apify Bubble - Universal Apify Actor Integration
 *
 * Provides integration with the Apify platform for running web scraping and automation actors.
 *
 * @template T - Actor ID type
 */
export declare class ApifyBubble<T extends string = string> extends ServiceBubble<TypedApifyParams<T>, TypedApifyResult<T>> {
    static readonly service = "apify";
    static readonly authType: "apikey";
    static readonly bubbleName = "apify";
    static readonly type: "service";
    static readonly schema: z.ZodObject<{
        actorId: z.ZodOptional<z.ZodString>;
        search: z.ZodOptional<z.ZodString>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        input: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        waitForFinish: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        input: Record<string, unknown>;
        limit: number;
        waitForFinish: boolean;
        search?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        actorId?: string | undefined;
    }, {
        input: Record<string, unknown>;
        timeout?: number | undefined;
        search?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        actorId?: string | undefined;
        waitForFinish?: boolean | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        runId: z.ZodString;
        status: z.ZodString;
        datasetId: z.ZodOptional<z.ZodString>;
        items: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        itemsCount: z.ZodOptional<z.ZodNumber>;
        consoleUrl: z.ZodString;
        success: z.ZodBoolean;
        error: z.ZodString;
        discoveredActors: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            description: z.ZodOptional<z.ZodNullable<z.ZodString>>;
            inputSchemaUrl: z.ZodString;
            stars: z.ZodOptional<z.ZodNullable<z.ZodNumber>>;
            usage: z.ZodOptional<z.ZodNullable<z.ZodObject<{
                totalRuns: z.ZodOptional<z.ZodNumber>;
                usersCount: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                totalRuns?: number | undefined;
                usersCount?: number | undefined;
            }, {
                totalRuns?: number | undefined;
                usersCount?: number | undefined;
            }>>>;
            requiresRental: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            inputSchemaUrl: string;
            description?: string | null | undefined;
            stars?: number | null | undefined;
            usage?: {
                totalRuns?: number | undefined;
                usersCount?: number | undefined;
            } | null | undefined;
            requiresRental?: boolean | undefined;
        }, {
            name: string;
            id: string;
            inputSchemaUrl: string;
            description?: string | null | undefined;
            stars?: number | null | undefined;
            usage?: {
                totalRuns?: number | undefined;
                usersCount?: number | undefined;
            } | null | undefined;
            requiresRental?: boolean | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        status: string;
        success: boolean;
        runId: string;
        consoleUrl: string;
        items?: unknown[] | undefined;
        datasetId?: string | undefined;
        itemsCount?: number | undefined;
        discoveredActors?: {
            name: string;
            id: string;
            inputSchemaUrl: string;
            description?: string | null | undefined;
            stars?: number | null | undefined;
            usage?: {
                totalRuns?: number | undefined;
                usersCount?: number | undefined;
            } | null | undefined;
            requiresRental?: boolean | undefined;
        }[] | undefined;
    }, {
        error: string;
        status: string;
        success: boolean;
        runId: string;
        consoleUrl: string;
        items?: unknown[] | undefined;
        datasetId?: string | undefined;
        itemsCount?: number | undefined;
        discoveredActors?: {
            name: string;
            id: string;
            inputSchemaUrl: string;
            description?: string | null | undefined;
            stars?: number | null | undefined;
            usage?: {
                totalRuns?: number | undefined;
                usersCount?: number | undefined;
            } | null | undefined;
            requiresRental?: boolean | undefined;
        }[] | undefined;
    }>;
    static readonly shortDescription = "Discover and run specialized Apify actors for complex web scraping tasks not covered by standard tools";
    static readonly longDescription = "\n    Universal integration with Apify platform for running any Apify actor.\n\n    This is a generic service bubble that can execute any Apify actor with any input.\n    Actor-specific logic and data transformation should be handled by Tool Bubbles.\n\n    Integrated Actors, use them through instagram-tool, reddit-tool, linkedin-tool, youtube-tool, tiktok-tool, twitter-tool, google-maps-tool, etc, not directly:\n    - apify/instagram-scraper - Instagram posts, profiles, hashtags\n    - apify/instagram-hashtag-scraper - Instagram hashtag posts\n    - apimaestro/linkedin-profile-posts - LinkedIn profile posts and activity\n    - apimaestro/linkedin-posts-search-scraper-no-cookies - Search LinkedIn posts by keyword\n    - curious_coder/linkedin-jobs-scraper - LinkedIn job postings\n    - streamers/youtube-scraper - YouTube videos and channels\n    - pintostudio/youtube-transcript-scraper - YouTube video transcripts\n    - clockworks/tiktok-scraper - TikTok profiles, videos, hashtags\n    - apidojo/tweet-scraper - Twitter/X profiles, tweets, search results\n    - compass/crawler-google-places - Google Maps business listings and reviews\n    - IMPORTANT: For other actors, use discovery mode to find the actor and its page, then use the web scrape tool to scrape the input schema page to get the input/output schema details.\n\n    Discovery Mode:\n    - Provide a \"search\" parameter to discover available actors\n    - Optionally set \"limit\" to control the number of results (default: 20, max: 100)\n    - Returns actor information including input schemas, descriptions, and metadata\n    - This mode is specifically designed for discovering available actors and their capabilities\n    - Example: { search: \"google flights prices\", limit: 10 } to find Google flights related actors\n\n    Use cases:\n    - Discovering available actors and their schemas then\n    - IMPORTANT: Specific scraping tasks that are not covered by the supported actors and seems hard to do through normal scraping by going to actor https://apify.com/$owner/$actorid/input-schema page and scrape the input schema details.\n\n    DO NOT Use:\n    - Media generation tasks (e.g., image generation, video generation, audio generation, etc.)\n\n  ";
    static readonly alias = "scrape";
    /**
     * Create a new Apify Bubble instance
     * @param params - Operation parameters
     * @param context - Bubble execution context
     * @param instanceId - Optional instance identifier
     */
    constructor(params: TypedApifyParamsInput<T>, context?: BubbleContext, instanceId?: string);
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    protected performAction(context?: BubbleContext): Promise<TypedApifyResult<T>>;
    private startActorRun;
    private waitForActorCompletion;
    private getRunStatus;
    private fetchDatasetItems;
    /**
     * Discovery mode: Search for available Apify actors and return their information
     * This is a special mode activated when the "search" parameter is provided
     */
    private discoverActors;
}
export {};
//# sourceMappingURL=apify.d.ts.map