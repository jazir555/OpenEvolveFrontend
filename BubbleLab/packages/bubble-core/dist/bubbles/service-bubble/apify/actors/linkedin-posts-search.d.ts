import { z } from 'zod';
export declare const LinkedInPostsSearchInputSchema: z.ZodObject<{
    keyword: z.ZodString;
    sort_type: z.ZodOptional<z.ZodDefault<z.ZodEnum<["relevance", "date_posted"]>>>;
    page_number: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    date_filter: z.ZodOptional<z.ZodDefault<z.ZodEnum<["", "past-24h", "past-week", "past-month"]>>>;
    limit: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
}, "strip", z.ZodTypeAny, {
    keyword: string;
    limit?: number | undefined;
    page_number?: number | undefined;
    sort_type?: "relevance" | "date_posted" | undefined;
    date_filter?: "" | "past-24h" | "past-week" | "past-month" | undefined;
}, {
    keyword: string;
    limit?: number | undefined;
    page_number?: number | undefined;
    sort_type?: "relevance" | "date_posted" | undefined;
    date_filter?: "" | "past-24h" | "past-week" | "past-month" | undefined;
}>;
export declare const LinkedInPostsSearchOutputSchema: z.ZodObject<{
    activity_id: z.ZodOptional<z.ZodString>;
    post_url: z.ZodOptional<z.ZodString>;
    text: z.ZodOptional<z.ZodString>;
    full_urn: z.ZodOptional<z.ZodString>;
    author: z.ZodOptional<z.ZodObject<{
        name: z.ZodOptional<z.ZodString>;
        headline: z.ZodOptional<z.ZodString>;
        profile_id: z.ZodOptional<z.ZodString>;
        profile_url: z.ZodOptional<z.ZodString>;
        image_url: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name?: string | undefined;
        image_url?: string | undefined;
        headline?: string | undefined;
        profile_url?: string | undefined;
        profile_id?: string | undefined;
    }, {
        name?: string | undefined;
        image_url?: string | undefined;
        headline?: string | undefined;
        profile_url?: string | undefined;
        profile_id?: string | undefined;
    }>>;
    stats: z.ZodOptional<z.ZodObject<{
        total_reactions: z.ZodOptional<z.ZodNumber>;
        comments: z.ZodOptional<z.ZodNumber>;
        shares: z.ZodOptional<z.ZodNumber>;
        reactions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodOptional<z.ZodString>;
            count: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            type?: string | undefined;
            count?: number | undefined;
        }, {
            type?: string | undefined;
            count?: number | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        reactions?: {
            type?: string | undefined;
            count?: number | undefined;
        }[] | undefined;
        shares?: number | undefined;
        total_reactions?: number | undefined;
        comments?: number | undefined;
    }, {
        reactions?: {
            type?: string | undefined;
            count?: number | undefined;
        }[] | undefined;
        shares?: number | undefined;
        total_reactions?: number | undefined;
        comments?: number | undefined;
    }>>;
    posted_at: z.ZodOptional<z.ZodObject<{
        display_text: z.ZodOptional<z.ZodString>;
        date: z.ZodOptional<z.ZodString>;
        timestamp: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        date?: string | undefined;
        timestamp?: number | undefined;
        display_text?: string | undefined;
    }, {
        date?: string | undefined;
        timestamp?: number | undefined;
        display_text?: string | undefined;
    }>>;
    hashtags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    content: z.ZodOptional<z.ZodObject<{
        type: z.ZodOptional<z.ZodString>;
        article: z.ZodOptional<z.ZodObject<{
            url: z.ZodOptional<z.ZodNullable<z.ZodString>>;
            title: z.ZodOptional<z.ZodNullable<z.ZodString>>;
            subtitle: z.ZodOptional<z.ZodNullable<z.ZodString>>;
            thumbnail: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        }, "strip", z.ZodTypeAny, {
            title?: string | null | undefined;
            url?: string | null | undefined;
            subtitle?: string | null | undefined;
            thumbnail?: string | null | undefined;
        }, {
            title?: string | null | undefined;
            url?: string | null | undefined;
            subtitle?: string | null | undefined;
            thumbnail?: string | null | undefined;
        }>>;
        url: z.ZodOptional<z.ZodString>;
        thumbnail_url: z.ZodOptional<z.ZodString>;
        duration_ms: z.ZodOptional<z.ZodNumber>;
        text: z.ZodOptional<z.ZodString>;
        images: z.ZodOptional<z.ZodArray<z.ZodObject<{
            url: z.ZodOptional<z.ZodString>;
            width: z.ZodOptional<z.ZodNumber>;
            height: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            url?: string | undefined;
            width?: number | undefined;
            height?: number | undefined;
        }, {
            url?: string | undefined;
            width?: number | undefined;
            height?: number | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        type?: string | undefined;
        url?: string | undefined;
        images?: {
            url?: string | undefined;
            width?: number | undefined;
            height?: number | undefined;
        }[] | undefined;
        text?: string | undefined;
        article?: {
            title?: string | null | undefined;
            url?: string | null | undefined;
            subtitle?: string | null | undefined;
            thumbnail?: string | null | undefined;
        } | undefined;
        thumbnail_url?: string | undefined;
        duration_ms?: number | undefined;
    }, {
        type?: string | undefined;
        url?: string | undefined;
        images?: {
            url?: string | undefined;
            width?: number | undefined;
            height?: number | undefined;
        }[] | undefined;
        text?: string | undefined;
        article?: {
            title?: string | null | undefined;
            url?: string | null | undefined;
            subtitle?: string | null | undefined;
            thumbnail?: string | null | undefined;
        } | undefined;
        thumbnail_url?: string | undefined;
        duration_ms?: number | undefined;
    }>>;
    is_reshare: z.ZodOptional<z.ZodBoolean>;
    metadata: z.ZodOptional<z.ZodObject<{
        total_count: z.ZodOptional<z.ZodNumber>;
        count: z.ZodOptional<z.ZodNumber>;
        page: z.ZodOptional<z.ZodNumber>;
        page_size: z.ZodOptional<z.ZodNumber>;
        total_pages: z.ZodOptional<z.ZodNumber>;
        has_next_page: z.ZodOptional<z.ZodBoolean>;
        has_prev_page: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        count?: number | undefined;
        total_count?: number | undefined;
        page?: number | undefined;
        page_size?: number | undefined;
        total_pages?: number | undefined;
        has_next_page?: boolean | undefined;
        has_prev_page?: boolean | undefined;
    }, {
        count?: number | undefined;
        total_count?: number | undefined;
        page?: number | undefined;
        page_size?: number | undefined;
        total_pages?: number | undefined;
        has_next_page?: boolean | undefined;
        has_prev_page?: boolean | undefined;
    }>>;
    search_input: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    content?: {
        type?: string | undefined;
        url?: string | undefined;
        images?: {
            url?: string | undefined;
            width?: number | undefined;
            height?: number | undefined;
        }[] | undefined;
        text?: string | undefined;
        article?: {
            title?: string | null | undefined;
            url?: string | null | undefined;
            subtitle?: string | null | undefined;
            thumbnail?: string | null | undefined;
        } | undefined;
        thumbnail_url?: string | undefined;
        duration_ms?: number | undefined;
    } | undefined;
    text?: string | undefined;
    metadata?: {
        count?: number | undefined;
        total_count?: number | undefined;
        page?: number | undefined;
        page_size?: number | undefined;
        total_pages?: number | undefined;
        has_next_page?: boolean | undefined;
        has_prev_page?: boolean | undefined;
    } | undefined;
    hashtags?: string[] | undefined;
    full_urn?: string | undefined;
    posted_at?: {
        date?: string | undefined;
        timestamp?: number | undefined;
        display_text?: string | undefined;
    } | undefined;
    author?: {
        name?: string | undefined;
        image_url?: string | undefined;
        headline?: string | undefined;
        profile_url?: string | undefined;
        profile_id?: string | undefined;
    } | undefined;
    stats?: {
        reactions?: {
            type?: string | undefined;
            count?: number | undefined;
        }[] | undefined;
        shares?: number | undefined;
        total_reactions?: number | undefined;
        comments?: number | undefined;
    } | undefined;
    activity_id?: string | undefined;
    post_url?: string | undefined;
    is_reshare?: boolean | undefined;
    search_input?: string | undefined;
}, {
    content?: {
        type?: string | undefined;
        url?: string | undefined;
        images?: {
            url?: string | undefined;
            width?: number | undefined;
            height?: number | undefined;
        }[] | undefined;
        text?: string | undefined;
        article?: {
            title?: string | null | undefined;
            url?: string | null | undefined;
            subtitle?: string | null | undefined;
            thumbnail?: string | null | undefined;
        } | undefined;
        thumbnail_url?: string | undefined;
        duration_ms?: number | undefined;
    } | undefined;
    text?: string | undefined;
    metadata?: {
        count?: number | undefined;
        total_count?: number | undefined;
        page?: number | undefined;
        page_size?: number | undefined;
        total_pages?: number | undefined;
        has_next_page?: boolean | undefined;
        has_prev_page?: boolean | undefined;
    } | undefined;
    hashtags?: string[] | undefined;
    full_urn?: string | undefined;
    posted_at?: {
        date?: string | undefined;
        timestamp?: number | undefined;
        display_text?: string | undefined;
    } | undefined;
    author?: {
        name?: string | undefined;
        image_url?: string | undefined;
        headline?: string | undefined;
        profile_url?: string | undefined;
        profile_id?: string | undefined;
    } | undefined;
    stats?: {
        reactions?: {
            type?: string | undefined;
            count?: number | undefined;
        }[] | undefined;
        shares?: number | undefined;
        total_reactions?: number | undefined;
        comments?: number | undefined;
    } | undefined;
    activity_id?: string | undefined;
    post_url?: string | undefined;
    is_reshare?: boolean | undefined;
    search_input?: string | undefined;
}>;
export type LinkedInPostsSearchInput = z.output<typeof LinkedInPostsSearchInputSchema>;
export type LinkedInPostsSearchOutput = z.output<typeof LinkedInPostsSearchOutputSchema>;
//# sourceMappingURL=linkedin-posts-search.d.ts.map