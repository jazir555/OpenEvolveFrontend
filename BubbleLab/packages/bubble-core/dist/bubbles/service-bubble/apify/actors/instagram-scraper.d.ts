import { z } from 'zod';
export declare const InstagramScraperInputSchema: z.ZodObject<{
    directUrls: z.ZodArray<z.ZodString, "many">;
    resultsType: z.ZodDefault<z.ZodEnum<["posts", "details"]>>;
    resultsLimit: z.ZodDefault<z.ZodNumber>;
    includeStories: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    includeHighlights: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
}, "strip", z.ZodTypeAny, {
    directUrls: string[];
    resultsType: "posts" | "details";
    resultsLimit: number;
    includeStories?: boolean | undefined;
    includeHighlights?: boolean | undefined;
}, {
    directUrls: string[];
    resultsType?: "posts" | "details" | undefined;
    resultsLimit?: number | undefined;
    includeStories?: boolean | undefined;
    includeHighlights?: boolean | undefined;
}>;
export declare const InstagramPostSchema: z.ZodObject<{
    id: z.ZodOptional<z.ZodString>;
    type: z.ZodOptional<z.ZodString>;
    shortCode: z.ZodOptional<z.ZodString>;
    caption: z.ZodOptional<z.ZodString>;
    hashtags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    mentions: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    url: z.ZodOptional<z.ZodString>;
    commentsCount: z.ZodOptional<z.ZodNumber>;
    dimensionsHeight: z.ZodOptional<z.ZodNumber>;
    dimensionsWidth: z.ZodOptional<z.ZodNumber>;
    displayUrl: z.ZodOptional<z.ZodString>;
    images: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    videoUrl: z.ZodOptional<z.ZodString>;
    alt: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    likesCount: z.ZodOptional<z.ZodNumber>;
    videoViewCount: z.ZodOptional<z.ZodNumber>;
    timestamp: z.ZodOptional<z.ZodString>;
    childPosts: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    ownerUsername: z.ZodOptional<z.ZodString>;
    ownerId: z.ZodOptional<z.ZodString>;
    productType: z.ZodOptional<z.ZodString>;
    taggedUsers: z.ZodOptional<z.ZodArray<z.ZodObject<{
        full_name: z.ZodOptional<z.ZodString>;
        id: z.ZodOptional<z.ZodString>;
        is_verified: z.ZodOptional<z.ZodBoolean>;
        profile_pic_url: z.ZodOptional<z.ZodString>;
        username: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        username?: string | undefined;
        id?: string | undefined;
        full_name?: string | undefined;
        is_verified?: boolean | undefined;
        profile_pic_url?: string | undefined;
    }, {
        username?: string | undefined;
        id?: string | undefined;
        full_name?: string | undefined;
        is_verified?: boolean | undefined;
        profile_pic_url?: string | undefined;
    }>, "many">>;
    isCommentsDisabled: z.ZodOptional<z.ZodBoolean>;
    location: z.ZodOptional<z.ZodNullable<z.ZodObject<{
        name: z.ZodOptional<z.ZodString>;
        id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name?: string | undefined;
        id?: string | undefined;
    }, {
        name?: string | undefined;
        id?: string | undefined;
    }>>>;
}, "strip", z.ZodTypeAny, {
    type?: string | undefined;
    url?: string | undefined;
    images?: string[] | undefined;
    timestamp?: string | undefined;
    id?: string | undefined;
    location?: {
        name?: string | undefined;
        id?: string | undefined;
    } | null | undefined;
    shortCode?: string | undefined;
    caption?: string | undefined;
    hashtags?: string[] | undefined;
    mentions?: string[] | undefined;
    commentsCount?: number | undefined;
    dimensionsHeight?: number | undefined;
    dimensionsWidth?: number | undefined;
    displayUrl?: string | undefined;
    videoUrl?: string | undefined;
    alt?: string | null | undefined;
    likesCount?: number | undefined;
    videoViewCount?: number | undefined;
    childPosts?: unknown[] | undefined;
    ownerUsername?: string | undefined;
    ownerId?: string | undefined;
    productType?: string | undefined;
    taggedUsers?: {
        username?: string | undefined;
        id?: string | undefined;
        full_name?: string | undefined;
        is_verified?: boolean | undefined;
        profile_pic_url?: string | undefined;
    }[] | undefined;
    isCommentsDisabled?: boolean | undefined;
}, {
    type?: string | undefined;
    url?: string | undefined;
    images?: string[] | undefined;
    timestamp?: string | undefined;
    id?: string | undefined;
    location?: {
        name?: string | undefined;
        id?: string | undefined;
    } | null | undefined;
    shortCode?: string | undefined;
    caption?: string | undefined;
    hashtags?: string[] | undefined;
    mentions?: string[] | undefined;
    commentsCount?: number | undefined;
    dimensionsHeight?: number | undefined;
    dimensionsWidth?: number | undefined;
    displayUrl?: string | undefined;
    videoUrl?: string | undefined;
    alt?: string | null | undefined;
    likesCount?: number | undefined;
    videoViewCount?: number | undefined;
    childPosts?: unknown[] | undefined;
    ownerUsername?: string | undefined;
    ownerId?: string | undefined;
    productType?: string | undefined;
    taggedUsers?: {
        username?: string | undefined;
        id?: string | undefined;
        full_name?: string | undefined;
        is_verified?: boolean | undefined;
        profile_pic_url?: string | undefined;
    }[] | undefined;
    isCommentsDisabled?: boolean | undefined;
}>;
export declare const InstagramScraperItemSchema: z.ZodObject<{
    inputUrl: z.ZodOptional<z.ZodString>;
    id: z.ZodOptional<z.ZodString>;
    username: z.ZodOptional<z.ZodString>;
    url: z.ZodOptional<z.ZodString>;
    fullName: z.ZodOptional<z.ZodString>;
    biography: z.ZodOptional<z.ZodString>;
    externalUrls: z.ZodOptional<z.ZodArray<z.ZodObject<{
        title: z.ZodOptional<z.ZodString>;
        lynx_url: z.ZodOptional<z.ZodString>;
        url: z.ZodOptional<z.ZodString>;
        link_type: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        title?: string | undefined;
        url?: string | undefined;
        lynx_url?: string | undefined;
        link_type?: string | undefined;
    }, {
        title?: string | undefined;
        url?: string | undefined;
        lynx_url?: string | undefined;
        link_type?: string | undefined;
    }>, "many">>;
    externalUrl: z.ZodOptional<z.ZodString>;
    externalUrlShimmed: z.ZodOptional<z.ZodString>;
    followersCount: z.ZodOptional<z.ZodNumber>;
    followsCount: z.ZodOptional<z.ZodNumber>;
    postsCount: z.ZodOptional<z.ZodNumber>;
    hasChannel: z.ZodOptional<z.ZodBoolean>;
    highlightReelCount: z.ZodOptional<z.ZodNumber>;
    isBusinessAccount: z.ZodOptional<z.ZodBoolean>;
    joinedRecently: z.ZodOptional<z.ZodBoolean>;
    businessCategoryName: z.ZodOptional<z.ZodString>;
    private: z.ZodOptional<z.ZodBoolean>;
    verified: z.ZodOptional<z.ZodBoolean>;
    profilePicUrl: z.ZodOptional<z.ZodString>;
    profilePicUrlHD: z.ZodOptional<z.ZodString>;
    igtvVideoCount: z.ZodOptional<z.ZodNumber>;
    latestIgtvVideos: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    relatedProfiles: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    latestPosts: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodOptional<z.ZodString>;
        type: z.ZodOptional<z.ZodString>;
        shortCode: z.ZodOptional<z.ZodString>;
        caption: z.ZodOptional<z.ZodString>;
        hashtags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        mentions: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        url: z.ZodOptional<z.ZodString>;
        commentsCount: z.ZodOptional<z.ZodNumber>;
        dimensionsHeight: z.ZodOptional<z.ZodNumber>;
        dimensionsWidth: z.ZodOptional<z.ZodNumber>;
        displayUrl: z.ZodOptional<z.ZodString>;
        images: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        videoUrl: z.ZodOptional<z.ZodString>;
        alt: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        likesCount: z.ZodOptional<z.ZodNumber>;
        videoViewCount: z.ZodOptional<z.ZodNumber>;
        timestamp: z.ZodOptional<z.ZodString>;
        childPosts: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        ownerUsername: z.ZodOptional<z.ZodString>;
        ownerId: z.ZodOptional<z.ZodString>;
        productType: z.ZodOptional<z.ZodString>;
        taggedUsers: z.ZodOptional<z.ZodArray<z.ZodObject<{
            full_name: z.ZodOptional<z.ZodString>;
            id: z.ZodOptional<z.ZodString>;
            is_verified: z.ZodOptional<z.ZodBoolean>;
            profile_pic_url: z.ZodOptional<z.ZodString>;
            username: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            username?: string | undefined;
            id?: string | undefined;
            full_name?: string | undefined;
            is_verified?: boolean | undefined;
            profile_pic_url?: string | undefined;
        }, {
            username?: string | undefined;
            id?: string | undefined;
            full_name?: string | undefined;
            is_verified?: boolean | undefined;
            profile_pic_url?: string | undefined;
        }>, "many">>;
        isCommentsDisabled: z.ZodOptional<z.ZodBoolean>;
        location: z.ZodOptional<z.ZodNullable<z.ZodObject<{
            name: z.ZodOptional<z.ZodString>;
            id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name?: string | undefined;
            id?: string | undefined;
        }, {
            name?: string | undefined;
            id?: string | undefined;
        }>>>;
    }, "strip", z.ZodTypeAny, {
        type?: string | undefined;
        url?: string | undefined;
        images?: string[] | undefined;
        timestamp?: string | undefined;
        id?: string | undefined;
        location?: {
            name?: string | undefined;
            id?: string | undefined;
        } | null | undefined;
        shortCode?: string | undefined;
        caption?: string | undefined;
        hashtags?: string[] | undefined;
        mentions?: string[] | undefined;
        commentsCount?: number | undefined;
        dimensionsHeight?: number | undefined;
        dimensionsWidth?: number | undefined;
        displayUrl?: string | undefined;
        videoUrl?: string | undefined;
        alt?: string | null | undefined;
        likesCount?: number | undefined;
        videoViewCount?: number | undefined;
        childPosts?: unknown[] | undefined;
        ownerUsername?: string | undefined;
        ownerId?: string | undefined;
        productType?: string | undefined;
        taggedUsers?: {
            username?: string | undefined;
            id?: string | undefined;
            full_name?: string | undefined;
            is_verified?: boolean | undefined;
            profile_pic_url?: string | undefined;
        }[] | undefined;
        isCommentsDisabled?: boolean | undefined;
    }, {
        type?: string | undefined;
        url?: string | undefined;
        images?: string[] | undefined;
        timestamp?: string | undefined;
        id?: string | undefined;
        location?: {
            name?: string | undefined;
            id?: string | undefined;
        } | null | undefined;
        shortCode?: string | undefined;
        caption?: string | undefined;
        hashtags?: string[] | undefined;
        mentions?: string[] | undefined;
        commentsCount?: number | undefined;
        dimensionsHeight?: number | undefined;
        dimensionsWidth?: number | undefined;
        displayUrl?: string | undefined;
        videoUrl?: string | undefined;
        alt?: string | null | undefined;
        likesCount?: number | undefined;
        videoViewCount?: number | undefined;
        childPosts?: unknown[] | undefined;
        ownerUsername?: string | undefined;
        ownerId?: string | undefined;
        productType?: string | undefined;
        taggedUsers?: {
            username?: string | undefined;
            id?: string | undefined;
            full_name?: string | undefined;
            is_verified?: boolean | undefined;
            profile_pic_url?: string | undefined;
        }[] | undefined;
        isCommentsDisabled?: boolean | undefined;
    }>, "many">>;
    stories: z.ZodOptional<z.ZodArray<z.ZodObject<{
        url: z.ZodOptional<z.ZodString>;
        timestamp: z.ZodOptional<z.ZodString>;
        type: z.ZodOptional<z.ZodEnum<["image", "video"]>>;
        viewsCount: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        type?: "image" | "video" | undefined;
        url?: string | undefined;
        timestamp?: string | undefined;
        viewsCount?: number | undefined;
    }, {
        type?: "image" | "video" | undefined;
        url?: string | undefined;
        timestamp?: string | undefined;
        viewsCount?: number | undefined;
    }>, "many">>;
    highlights: z.ZodOptional<z.ZodArray<z.ZodObject<{
        title: z.ZodOptional<z.ZodString>;
        coverUrl: z.ZodOptional<z.ZodString>;
        itemsCount: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        title?: string | undefined;
        itemsCount?: number | undefined;
        coverUrl?: string | undefined;
    }, {
        title?: string | undefined;
        itemsCount?: number | undefined;
        coverUrl?: string | undefined;
    }>, "many">>;
}, "strip", z.ZodTypeAny, {
    url?: string | undefined;
    username?: string | undefined;
    id?: string | undefined;
    private?: boolean | undefined;
    inputUrl?: string | undefined;
    fullName?: string | undefined;
    biography?: string | undefined;
    externalUrls?: {
        title?: string | undefined;
        url?: string | undefined;
        lynx_url?: string | undefined;
        link_type?: string | undefined;
    }[] | undefined;
    externalUrl?: string | undefined;
    externalUrlShimmed?: string | undefined;
    followersCount?: number | undefined;
    followsCount?: number | undefined;
    postsCount?: number | undefined;
    hasChannel?: boolean | undefined;
    highlightReelCount?: number | undefined;
    isBusinessAccount?: boolean | undefined;
    joinedRecently?: boolean | undefined;
    businessCategoryName?: string | undefined;
    verified?: boolean | undefined;
    profilePicUrl?: string | undefined;
    profilePicUrlHD?: string | undefined;
    igtvVideoCount?: number | undefined;
    latestIgtvVideos?: unknown[] | undefined;
    relatedProfiles?: unknown[] | undefined;
    latestPosts?: {
        type?: string | undefined;
        url?: string | undefined;
        images?: string[] | undefined;
        timestamp?: string | undefined;
        id?: string | undefined;
        location?: {
            name?: string | undefined;
            id?: string | undefined;
        } | null | undefined;
        shortCode?: string | undefined;
        caption?: string | undefined;
        hashtags?: string[] | undefined;
        mentions?: string[] | undefined;
        commentsCount?: number | undefined;
        dimensionsHeight?: number | undefined;
        dimensionsWidth?: number | undefined;
        displayUrl?: string | undefined;
        videoUrl?: string | undefined;
        alt?: string | null | undefined;
        likesCount?: number | undefined;
        videoViewCount?: number | undefined;
        childPosts?: unknown[] | undefined;
        ownerUsername?: string | undefined;
        ownerId?: string | undefined;
        productType?: string | undefined;
        taggedUsers?: {
            username?: string | undefined;
            id?: string | undefined;
            full_name?: string | undefined;
            is_verified?: boolean | undefined;
            profile_pic_url?: string | undefined;
        }[] | undefined;
        isCommentsDisabled?: boolean | undefined;
    }[] | undefined;
    stories?: {
        type?: "image" | "video" | undefined;
        url?: string | undefined;
        timestamp?: string | undefined;
        viewsCount?: number | undefined;
    }[] | undefined;
    highlights?: {
        title?: string | undefined;
        itemsCount?: number | undefined;
        coverUrl?: string | undefined;
    }[] | undefined;
}, {
    url?: string | undefined;
    username?: string | undefined;
    id?: string | undefined;
    private?: boolean | undefined;
    inputUrl?: string | undefined;
    fullName?: string | undefined;
    biography?: string | undefined;
    externalUrls?: {
        title?: string | undefined;
        url?: string | undefined;
        lynx_url?: string | undefined;
        link_type?: string | undefined;
    }[] | undefined;
    externalUrl?: string | undefined;
    externalUrlShimmed?: string | undefined;
    followersCount?: number | undefined;
    followsCount?: number | undefined;
    postsCount?: number | undefined;
    hasChannel?: boolean | undefined;
    highlightReelCount?: number | undefined;
    isBusinessAccount?: boolean | undefined;
    joinedRecently?: boolean | undefined;
    businessCategoryName?: string | undefined;
    verified?: boolean | undefined;
    profilePicUrl?: string | undefined;
    profilePicUrlHD?: string | undefined;
    igtvVideoCount?: number | undefined;
    latestIgtvVideos?: unknown[] | undefined;
    relatedProfiles?: unknown[] | undefined;
    latestPosts?: {
        type?: string | undefined;
        url?: string | undefined;
        images?: string[] | undefined;
        timestamp?: string | undefined;
        id?: string | undefined;
        location?: {
            name?: string | undefined;
            id?: string | undefined;
        } | null | undefined;
        shortCode?: string | undefined;
        caption?: string | undefined;
        hashtags?: string[] | undefined;
        mentions?: string[] | undefined;
        commentsCount?: number | undefined;
        dimensionsHeight?: number | undefined;
        dimensionsWidth?: number | undefined;
        displayUrl?: string | undefined;
        videoUrl?: string | undefined;
        alt?: string | null | undefined;
        likesCount?: number | undefined;
        videoViewCount?: number | undefined;
        childPosts?: unknown[] | undefined;
        ownerUsername?: string | undefined;
        ownerId?: string | undefined;
        productType?: string | undefined;
        taggedUsers?: {
            username?: string | undefined;
            id?: string | undefined;
            full_name?: string | undefined;
            is_verified?: boolean | undefined;
            profile_pic_url?: string | undefined;
        }[] | undefined;
        isCommentsDisabled?: boolean | undefined;
    }[] | undefined;
    stories?: {
        type?: "image" | "video" | undefined;
        url?: string | undefined;
        timestamp?: string | undefined;
        viewsCount?: number | undefined;
    }[] | undefined;
    highlights?: {
        title?: string | undefined;
        itemsCount?: number | undefined;
        coverUrl?: string | undefined;
    }[] | undefined;
}>;
//# sourceMappingURL=instagram-scraper.d.ts.map