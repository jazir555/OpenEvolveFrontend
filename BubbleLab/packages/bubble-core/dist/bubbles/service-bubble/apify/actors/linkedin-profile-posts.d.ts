import { z } from 'zod';
export declare const LinkedInProfilePostsInputSchema: z.ZodObject<{
    username: z.ZodString;
    page_number: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    limit: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
}, "strip", z.ZodTypeAny, {
    username: string;
    limit?: number | undefined;
    page_number?: number | undefined;
}, {
    username: string;
    limit?: number | undefined;
    page_number?: number | undefined;
}>;
declare const LinkedInURNSchema: z.ZodObject<{
    activity_urn: z.ZodOptional<z.ZodString>;
    share_urn: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    ugcPost_urn: z.ZodOptional<z.ZodNullable<z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    activity_urn?: string | undefined;
    share_urn?: string | null | undefined;
    ugcPost_urn?: string | null | undefined;
}, {
    activity_urn?: string | undefined;
    share_urn?: string | null | undefined;
    ugcPost_urn?: string | null | undefined;
}>;
declare const LinkedInAuthorSchema: z.ZodObject<{
    first_name: z.ZodOptional<z.ZodString>;
    last_name: z.ZodOptional<z.ZodString>;
    headline: z.ZodOptional<z.ZodString>;
    username: z.ZodOptional<z.ZodString>;
    profile_url: z.ZodOptional<z.ZodString>;
    profile_picture: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    username?: string | undefined;
    first_name?: string | undefined;
    last_name?: string | undefined;
    headline?: string | undefined;
    profile_url?: string | undefined;
    profile_picture?: string | undefined;
}, {
    username?: string | undefined;
    first_name?: string | undefined;
    last_name?: string | undefined;
    headline?: string | undefined;
    profile_url?: string | undefined;
    profile_picture?: string | undefined;
}>;
declare const LinkedInStatsSchema: z.ZodObject<{
    total_reactions: z.ZodOptional<z.ZodNumber>;
    like: z.ZodOptional<z.ZodNumber>;
    support: z.ZodOptional<z.ZodNumber>;
    love: z.ZodOptional<z.ZodNumber>;
    insight: z.ZodOptional<z.ZodNumber>;
    celebrate: z.ZodOptional<z.ZodNumber>;
    funny: z.ZodOptional<z.ZodNumber>;
    comments: z.ZodOptional<z.ZodNumber>;
    reposts: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    total_reactions?: number | undefined;
    like?: number | undefined;
    support?: number | undefined;
    love?: number | undefined;
    insight?: number | undefined;
    celebrate?: number | undefined;
    funny?: number | undefined;
    comments?: number | undefined;
    reposts?: number | undefined;
}, {
    total_reactions?: number | undefined;
    like?: number | undefined;
    support?: number | undefined;
    love?: number | undefined;
    insight?: number | undefined;
    celebrate?: number | undefined;
    funny?: number | undefined;
    comments?: number | undefined;
    reposts?: number | undefined;
}>;
declare const LinkedInPostedAtSchema: z.ZodObject<{
    date: z.ZodOptional<z.ZodString>;
    relative: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    date?: string | undefined;
    timestamp?: number | undefined;
    relative?: string | undefined;
}, {
    date?: string | undefined;
    timestamp?: number | undefined;
    relative?: string | undefined;
}>;
declare const LinkedInPostSchema: z.ZodObject<{
    urn: z.ZodOptional<z.ZodObject<{
        activity_urn: z.ZodOptional<z.ZodString>;
        share_urn: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        ugcPost_urn: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        activity_urn?: string | undefined;
        share_urn?: string | null | undefined;
        ugcPost_urn?: string | null | undefined;
    }, {
        activity_urn?: string | undefined;
        share_urn?: string | null | undefined;
        ugcPost_urn?: string | null | undefined;
    }>>;
    full_urn: z.ZodOptional<z.ZodString>;
    posted_at: z.ZodOptional<z.ZodObject<{
        date: z.ZodOptional<z.ZodString>;
        relative: z.ZodOptional<z.ZodString>;
        timestamp: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        date?: string | undefined;
        timestamp?: number | undefined;
        relative?: string | undefined;
    }, {
        date?: string | undefined;
        timestamp?: number | undefined;
        relative?: string | undefined;
    }>>;
    text: z.ZodOptional<z.ZodString>;
    url: z.ZodOptional<z.ZodString>;
    post_type: z.ZodOptional<z.ZodString>;
    author: z.ZodOptional<z.ZodObject<{
        first_name: z.ZodOptional<z.ZodString>;
        last_name: z.ZodOptional<z.ZodString>;
        headline: z.ZodOptional<z.ZodString>;
        username: z.ZodOptional<z.ZodString>;
        profile_url: z.ZodOptional<z.ZodString>;
        profile_picture: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        username?: string | undefined;
        first_name?: string | undefined;
        last_name?: string | undefined;
        headline?: string | undefined;
        profile_url?: string | undefined;
        profile_picture?: string | undefined;
    }, {
        username?: string | undefined;
        first_name?: string | undefined;
        last_name?: string | undefined;
        headline?: string | undefined;
        profile_url?: string | undefined;
        profile_picture?: string | undefined;
    }>>;
    stats: z.ZodOptional<z.ZodObject<{
        total_reactions: z.ZodOptional<z.ZodNumber>;
        like: z.ZodOptional<z.ZodNumber>;
        support: z.ZodOptional<z.ZodNumber>;
        love: z.ZodOptional<z.ZodNumber>;
        insight: z.ZodOptional<z.ZodNumber>;
        celebrate: z.ZodOptional<z.ZodNumber>;
        funny: z.ZodOptional<z.ZodNumber>;
        comments: z.ZodOptional<z.ZodNumber>;
        reposts: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        total_reactions?: number | undefined;
        like?: number | undefined;
        support?: number | undefined;
        love?: number | undefined;
        insight?: number | undefined;
        celebrate?: number | undefined;
        funny?: number | undefined;
        comments?: number | undefined;
        reposts?: number | undefined;
    }, {
        total_reactions?: number | undefined;
        like?: number | undefined;
        support?: number | undefined;
        love?: number | undefined;
        insight?: number | undefined;
        celebrate?: number | undefined;
        funny?: number | undefined;
        comments?: number | undefined;
        reposts?: number | undefined;
    }>>;
    media: z.ZodOptional<z.ZodObject<{
        type: z.ZodOptional<z.ZodString>;
        url: z.ZodOptional<z.ZodString>;
        thumbnail: z.ZodOptional<z.ZodString>;
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
        thumbnail?: string | undefined;
    }, {
        type?: string | undefined;
        url?: string | undefined;
        images?: {
            url?: string | undefined;
            width?: number | undefined;
            height?: number | undefined;
        }[] | undefined;
        thumbnail?: string | undefined;
    }>>;
    article: z.ZodOptional<z.ZodObject<{
        url: z.ZodOptional<z.ZodString>;
        title: z.ZodOptional<z.ZodString>;
        subtitle: z.ZodOptional<z.ZodString>;
        thumbnail: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        title?: string | undefined;
        url?: string | undefined;
        subtitle?: string | undefined;
        thumbnail?: string | undefined;
    }, {
        title?: string | undefined;
        url?: string | undefined;
        subtitle?: string | undefined;
        thumbnail?: string | undefined;
    }>>;
    document: z.ZodOptional<z.ZodObject<{
        title: z.ZodOptional<z.ZodString>;
        page_count: z.ZodOptional<z.ZodNumber>;
        url: z.ZodOptional<z.ZodString>;
        thumbnail: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        title?: string | undefined;
        url?: string | undefined;
        thumbnail?: string | undefined;
        page_count?: number | undefined;
    }, {
        title?: string | undefined;
        url?: string | undefined;
        thumbnail?: string | undefined;
        page_count?: number | undefined;
    }>>;
    reshared_post: z.ZodOptional<z.ZodType<any, z.ZodTypeDef, any>>;
    pagination_token: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    url?: string | undefined;
    text?: string | undefined;
    urn?: {
        activity_urn?: string | undefined;
        share_urn?: string | null | undefined;
        ugcPost_urn?: string | null | undefined;
    } | undefined;
    full_urn?: string | undefined;
    posted_at?: {
        date?: string | undefined;
        timestamp?: number | undefined;
        relative?: string | undefined;
    } | undefined;
    post_type?: string | undefined;
    author?: {
        username?: string | undefined;
        first_name?: string | undefined;
        last_name?: string | undefined;
        headline?: string | undefined;
        profile_url?: string | undefined;
        profile_picture?: string | undefined;
    } | undefined;
    stats?: {
        total_reactions?: number | undefined;
        like?: number | undefined;
        support?: number | undefined;
        love?: number | undefined;
        insight?: number | undefined;
        celebrate?: number | undefined;
        funny?: number | undefined;
        comments?: number | undefined;
        reposts?: number | undefined;
    } | undefined;
    media?: {
        type?: string | undefined;
        url?: string | undefined;
        images?: {
            url?: string | undefined;
            width?: number | undefined;
            height?: number | undefined;
        }[] | undefined;
        thumbnail?: string | undefined;
    } | undefined;
    article?: {
        title?: string | undefined;
        url?: string | undefined;
        subtitle?: string | undefined;
        thumbnail?: string | undefined;
    } | undefined;
    document?: {
        title?: string | undefined;
        url?: string | undefined;
        thumbnail?: string | undefined;
        page_count?: number | undefined;
    } | undefined;
    reshared_post?: any;
    pagination_token?: string | undefined;
}, {
    url?: string | undefined;
    text?: string | undefined;
    urn?: {
        activity_urn?: string | undefined;
        share_urn?: string | null | undefined;
        ugcPost_urn?: string | null | undefined;
    } | undefined;
    full_urn?: string | undefined;
    posted_at?: {
        date?: string | undefined;
        timestamp?: number | undefined;
        relative?: string | undefined;
    } | undefined;
    post_type?: string | undefined;
    author?: {
        username?: string | undefined;
        first_name?: string | undefined;
        last_name?: string | undefined;
        headline?: string | undefined;
        profile_url?: string | undefined;
        profile_picture?: string | undefined;
    } | undefined;
    stats?: {
        total_reactions?: number | undefined;
        like?: number | undefined;
        support?: number | undefined;
        love?: number | undefined;
        insight?: number | undefined;
        celebrate?: number | undefined;
        funny?: number | undefined;
        comments?: number | undefined;
        reposts?: number | undefined;
    } | undefined;
    media?: {
        type?: string | undefined;
        url?: string | undefined;
        images?: {
            url?: string | undefined;
            width?: number | undefined;
            height?: number | undefined;
        }[] | undefined;
        thumbnail?: string | undefined;
    } | undefined;
    article?: {
        title?: string | undefined;
        url?: string | undefined;
        subtitle?: string | undefined;
        thumbnail?: string | undefined;
    } | undefined;
    document?: {
        title?: string | undefined;
        url?: string | undefined;
        thumbnail?: string | undefined;
        page_count?: number | undefined;
    } | undefined;
    reshared_post?: any;
    pagination_token?: string | undefined;
}>;
export declare const LinkedInProfilePostsOutputSchema: z.ZodObject<{
    urn: z.ZodOptional<z.ZodObject<{
        activity_urn: z.ZodOptional<z.ZodString>;
        share_urn: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        ugcPost_urn: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        activity_urn?: string | undefined;
        share_urn?: string | null | undefined;
        ugcPost_urn?: string | null | undefined;
    }, {
        activity_urn?: string | undefined;
        share_urn?: string | null | undefined;
        ugcPost_urn?: string | null | undefined;
    }>>;
    full_urn: z.ZodOptional<z.ZodString>;
    posted_at: z.ZodOptional<z.ZodObject<{
        date: z.ZodOptional<z.ZodString>;
        relative: z.ZodOptional<z.ZodString>;
        timestamp: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        date?: string | undefined;
        timestamp?: number | undefined;
        relative?: string | undefined;
    }, {
        date?: string | undefined;
        timestamp?: number | undefined;
        relative?: string | undefined;
    }>>;
    text: z.ZodOptional<z.ZodString>;
    url: z.ZodOptional<z.ZodString>;
    post_type: z.ZodOptional<z.ZodString>;
    author: z.ZodOptional<z.ZodObject<{
        first_name: z.ZodOptional<z.ZodString>;
        last_name: z.ZodOptional<z.ZodString>;
        headline: z.ZodOptional<z.ZodString>;
        username: z.ZodOptional<z.ZodString>;
        profile_url: z.ZodOptional<z.ZodString>;
        profile_picture: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        username?: string | undefined;
        first_name?: string | undefined;
        last_name?: string | undefined;
        headline?: string | undefined;
        profile_url?: string | undefined;
        profile_picture?: string | undefined;
    }, {
        username?: string | undefined;
        first_name?: string | undefined;
        last_name?: string | undefined;
        headline?: string | undefined;
        profile_url?: string | undefined;
        profile_picture?: string | undefined;
    }>>;
    stats: z.ZodOptional<z.ZodObject<{
        total_reactions: z.ZodOptional<z.ZodNumber>;
        like: z.ZodOptional<z.ZodNumber>;
        support: z.ZodOptional<z.ZodNumber>;
        love: z.ZodOptional<z.ZodNumber>;
        insight: z.ZodOptional<z.ZodNumber>;
        celebrate: z.ZodOptional<z.ZodNumber>;
        funny: z.ZodOptional<z.ZodNumber>;
        comments: z.ZodOptional<z.ZodNumber>;
        reposts: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        total_reactions?: number | undefined;
        like?: number | undefined;
        support?: number | undefined;
        love?: number | undefined;
        insight?: number | undefined;
        celebrate?: number | undefined;
        funny?: number | undefined;
        comments?: number | undefined;
        reposts?: number | undefined;
    }, {
        total_reactions?: number | undefined;
        like?: number | undefined;
        support?: number | undefined;
        love?: number | undefined;
        insight?: number | undefined;
        celebrate?: number | undefined;
        funny?: number | undefined;
        comments?: number | undefined;
        reposts?: number | undefined;
    }>>;
    media: z.ZodOptional<z.ZodObject<{
        type: z.ZodOptional<z.ZodString>;
        url: z.ZodOptional<z.ZodString>;
        thumbnail: z.ZodOptional<z.ZodString>;
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
        thumbnail?: string | undefined;
    }, {
        type?: string | undefined;
        url?: string | undefined;
        images?: {
            url?: string | undefined;
            width?: number | undefined;
            height?: number | undefined;
        }[] | undefined;
        thumbnail?: string | undefined;
    }>>;
    article: z.ZodOptional<z.ZodObject<{
        url: z.ZodOptional<z.ZodString>;
        title: z.ZodOptional<z.ZodString>;
        subtitle: z.ZodOptional<z.ZodString>;
        thumbnail: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        title?: string | undefined;
        url?: string | undefined;
        subtitle?: string | undefined;
        thumbnail?: string | undefined;
    }, {
        title?: string | undefined;
        url?: string | undefined;
        subtitle?: string | undefined;
        thumbnail?: string | undefined;
    }>>;
    document: z.ZodOptional<z.ZodObject<{
        title: z.ZodOptional<z.ZodString>;
        page_count: z.ZodOptional<z.ZodNumber>;
        url: z.ZodOptional<z.ZodString>;
        thumbnail: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        title?: string | undefined;
        url?: string | undefined;
        thumbnail?: string | undefined;
        page_count?: number | undefined;
    }, {
        title?: string | undefined;
        url?: string | undefined;
        thumbnail?: string | undefined;
        page_count?: number | undefined;
    }>>;
    reshared_post: z.ZodOptional<z.ZodType<any, z.ZodTypeDef, any>>;
    pagination_token: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    url?: string | undefined;
    text?: string | undefined;
    urn?: {
        activity_urn?: string | undefined;
        share_urn?: string | null | undefined;
        ugcPost_urn?: string | null | undefined;
    } | undefined;
    full_urn?: string | undefined;
    posted_at?: {
        date?: string | undefined;
        timestamp?: number | undefined;
        relative?: string | undefined;
    } | undefined;
    post_type?: string | undefined;
    author?: {
        username?: string | undefined;
        first_name?: string | undefined;
        last_name?: string | undefined;
        headline?: string | undefined;
        profile_url?: string | undefined;
        profile_picture?: string | undefined;
    } | undefined;
    stats?: {
        total_reactions?: number | undefined;
        like?: number | undefined;
        support?: number | undefined;
        love?: number | undefined;
        insight?: number | undefined;
        celebrate?: number | undefined;
        funny?: number | undefined;
        comments?: number | undefined;
        reposts?: number | undefined;
    } | undefined;
    media?: {
        type?: string | undefined;
        url?: string | undefined;
        images?: {
            url?: string | undefined;
            width?: number | undefined;
            height?: number | undefined;
        }[] | undefined;
        thumbnail?: string | undefined;
    } | undefined;
    article?: {
        title?: string | undefined;
        url?: string | undefined;
        subtitle?: string | undefined;
        thumbnail?: string | undefined;
    } | undefined;
    document?: {
        title?: string | undefined;
        url?: string | undefined;
        thumbnail?: string | undefined;
        page_count?: number | undefined;
    } | undefined;
    reshared_post?: any;
    pagination_token?: string | undefined;
}, {
    url?: string | undefined;
    text?: string | undefined;
    urn?: {
        activity_urn?: string | undefined;
        share_urn?: string | null | undefined;
        ugcPost_urn?: string | null | undefined;
    } | undefined;
    full_urn?: string | undefined;
    posted_at?: {
        date?: string | undefined;
        timestamp?: number | undefined;
        relative?: string | undefined;
    } | undefined;
    post_type?: string | undefined;
    author?: {
        username?: string | undefined;
        first_name?: string | undefined;
        last_name?: string | undefined;
        headline?: string | undefined;
        profile_url?: string | undefined;
        profile_picture?: string | undefined;
    } | undefined;
    stats?: {
        total_reactions?: number | undefined;
        like?: number | undefined;
        support?: number | undefined;
        love?: number | undefined;
        insight?: number | undefined;
        celebrate?: number | undefined;
        funny?: number | undefined;
        comments?: number | undefined;
        reposts?: number | undefined;
    } | undefined;
    media?: {
        type?: string | undefined;
        url?: string | undefined;
        images?: {
            url?: string | undefined;
            width?: number | undefined;
            height?: number | undefined;
        }[] | undefined;
        thumbnail?: string | undefined;
    } | undefined;
    article?: {
        title?: string | undefined;
        url?: string | undefined;
        subtitle?: string | undefined;
        thumbnail?: string | undefined;
    } | undefined;
    document?: {
        title?: string | undefined;
        url?: string | undefined;
        thumbnail?: string | undefined;
        page_count?: number | undefined;
    } | undefined;
    reshared_post?: any;
    pagination_token?: string | undefined;
}>;
export type LinkedInPost = z.output<typeof LinkedInPostSchema>;
export type LinkedInAuthor = z.output<typeof LinkedInAuthorSchema>;
export type LinkedInStats = z.output<typeof LinkedInStatsSchema>;
export type LinkedInPostedAt = z.output<typeof LinkedInPostedAtSchema>;
export type LinkedInURN = z.output<typeof LinkedInURNSchema>;
export {};
//# sourceMappingURL=linkedin-profile-posts.d.ts.map