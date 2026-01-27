import { z } from 'zod';
export declare const TikTokScraperInputSchema: z.ZodObject<{
    hashtags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    resultsPerPage: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    profiles: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    profileScrapeSections: z.ZodOptional<z.ZodDefault<z.ZodArray<z.ZodEnum<["videos", "reposts"]>, "many">>>;
    profileSorting: z.ZodOptional<z.ZodDefault<z.ZodEnum<["latest", "popular", "oldest"]>>>;
    excludePinnedPosts: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    oldestPostDateUnified: z.ZodOptional<z.ZodString>;
    newestPostDate: z.ZodOptional<z.ZodString>;
    mostDiggs: z.ZodOptional<z.ZodNumber>;
    leastDiggs: z.ZodOptional<z.ZodNumber>;
    maxFollowersPerProfile: z.ZodOptional<z.ZodNumber>;
    maxFollowingPerProfile: z.ZodOptional<z.ZodNumber>;
    searchQueries: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    searchSection: z.ZodOptional<z.ZodDefault<z.ZodEnum<["", "/video", "/user"]>>>;
    maxProfilesPerQuery: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    searchSorting: z.ZodOptional<z.ZodDefault<z.ZodEnum<["0", "1", "3"]>>>;
    searchDatePosted: z.ZodOptional<z.ZodDefault<z.ZodEnum<["0", "1", "2", "3", "4", "5"]>>>;
    postURLs: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    scrapeRelatedVideos: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    shouldDownloadVideos: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    shouldDownloadCovers: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    shouldDownloadSubtitles: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    shouldDownloadSlideshowImages: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    shouldDownloadAvatars: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    shouldDownloadMusicCovers: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
    videoKvStoreIdOrName: z.ZodOptional<z.ZodString>;
    commentsPerPost: z.ZodOptional<z.ZodNumber>;
    maxRepliesPerComment: z.ZodOptional<z.ZodNumber>;
    proxyCountryCode: z.ZodOptional<z.ZodDefault<z.ZodEnum<["None", "AF", "AL", "DZ", "AS", "AD", "AO", "AI", "AG", "AR", "AM", "AU", "AT", "AZ", "BS", "BH", "BD", "BB", "BY", "BE", "BZ", "BJ", "BM", "BT", "BO", "BA", "BW", "BR", "VG", "BN", "BG", "BF", "BI", "KH", "CM", "CA", "CV", "KY", "TD", "CL", "CO", "CK", "CR", "HR", "CY", "CZ", "CD", "DK", "DJ", "DO", "EC", "EG", "SV", "EE", "ET", "FK", "FJ", "FI", "FR", "PF", "GA", "GE", "DE", "GH", "GI", "GR", "GL", "GD", "GP", "GT", "GN", "GW", "GY", "HN", "HK", "HU", "IS", "IN", "ID", "IQ", "IE", "IM", "IL", "IT", "CI", "JM", "JP", "JE", "KZ", "KE", "XK", "KW", "LA", "LV", "LB", "LS", "LR", "LY", "LT", "LU", "MO", "MG", "MW", "MY", "MV", "ML", "MT", "MH", "MQ", "MR", "MU", "MX", "MD", "MC", "MN", "ME", "MA", "MZ", "MM", "NA", "NR", "NP", "NL", "NZ", "NI", "NG", "MK", "NO", "OM", "PK", "PS", "PA", "PG", "PY", "PE", "PH", "PL", "PT", "PR", "QA", "CG", "RO", "RU", "RW", "RE", "KN", "LC", "MF", "PM", "VC", "SM", "SA", "SN", "RS", "SL", "SG", "SX", "SK", "SB", "SO", "ZA", "KR", "ES", "LK", "SR", "SZ", "SE", "CH", "TW", "TJ", "TZ", "TH", "TG", "TO", "TT", "TN", "TR", "TM", "TC", "TV", "VI", "UG", "UA", "AE", "GB", "US", "UY", "VE", "VN", "WF", "YE", "ZM", "ZW", "AX"]>>>;
}, "strip", z.ZodTypeAny, {
    hashtags?: string[] | undefined;
    searchQueries?: string[] | undefined;
    resultsPerPage?: number | undefined;
    profiles?: string[] | undefined;
    profileScrapeSections?: ("reposts" | "videos")[] | undefined;
    profileSorting?: "latest" | "oldest" | "popular" | undefined;
    excludePinnedPosts?: boolean | undefined;
    oldestPostDateUnified?: string | undefined;
    newestPostDate?: string | undefined;
    mostDiggs?: number | undefined;
    leastDiggs?: number | undefined;
    maxFollowersPerProfile?: number | undefined;
    maxFollowingPerProfile?: number | undefined;
    searchSection?: "" | "/video" | "/user" | undefined;
    maxProfilesPerQuery?: number | undefined;
    searchSorting?: "0" | "1" | "3" | undefined;
    searchDatePosted?: "0" | "1" | "2" | "3" | "4" | "5" | undefined;
    postURLs?: string[] | undefined;
    scrapeRelatedVideos?: boolean | undefined;
    shouldDownloadVideos?: boolean | undefined;
    shouldDownloadCovers?: boolean | undefined;
    shouldDownloadSubtitles?: boolean | undefined;
    shouldDownloadSlideshowImages?: boolean | undefined;
    shouldDownloadAvatars?: boolean | undefined;
    shouldDownloadMusicCovers?: boolean | undefined;
    videoKvStoreIdOrName?: string | undefined;
    commentsPerPost?: number | undefined;
    maxRepliesPerComment?: number | undefined;
    proxyCountryCode?: "None" | "AF" | "AL" | "DZ" | "AS" | "AD" | "AO" | "AI" | "AG" | "AR" | "AM" | "AU" | "AT" | "AZ" | "BS" | "BH" | "BD" | "BB" | "BY" | "BE" | "BZ" | "BJ" | "BM" | "BT" | "BO" | "BA" | "BW" | "BR" | "VG" | "BN" | "BG" | "BF" | "BI" | "KH" | "CM" | "CA" | "CV" | "KY" | "TD" | "CL" | "CO" | "CK" | "CR" | "HR" | "CY" | "CZ" | "CD" | "DK" | "DJ" | "DO" | "EC" | "EG" | "SV" | "EE" | "ET" | "FK" | "FJ" | "FI" | "FR" | "PF" | "GA" | "GE" | "DE" | "GH" | "GI" | "GR" | "GL" | "GD" | "GP" | "GT" | "GN" | "GW" | "GY" | "HN" | "HK" | "HU" | "IS" | "IN" | "ID" | "IQ" | "IE" | "IM" | "IL" | "IT" | "CI" | "JM" | "JP" | "JE" | "KZ" | "KE" | "XK" | "KW" | "LA" | "LV" | "LB" | "LS" | "LR" | "LY" | "LT" | "LU" | "MO" | "MG" | "MW" | "MY" | "MV" | "ML" | "MT" | "MH" | "MQ" | "MR" | "MU" | "MX" | "MD" | "MC" | "MN" | "ME" | "MA" | "MZ" | "MM" | "NA" | "NR" | "NP" | "NL" | "NZ" | "NI" | "NG" | "MK" | "NO" | "OM" | "PK" | "PS" | "PA" | "PG" | "PY" | "PE" | "PH" | "PL" | "PT" | "PR" | "QA" | "CG" | "RO" | "RU" | "RW" | "RE" | "KN" | "LC" | "MF" | "PM" | "VC" | "SM" | "SA" | "SN" | "RS" | "SL" | "SG" | "SX" | "SK" | "SB" | "SO" | "ZA" | "KR" | "ES" | "LK" | "SR" | "SZ" | "SE" | "CH" | "TW" | "TJ" | "TZ" | "TH" | "TG" | "TO" | "TT" | "TN" | "TR" | "TM" | "TC" | "TV" | "VI" | "UG" | "UA" | "AE" | "GB" | "US" | "UY" | "VE" | "VN" | "WF" | "YE" | "ZM" | "ZW" | "AX" | undefined;
}, {
    hashtags?: string[] | undefined;
    searchQueries?: string[] | undefined;
    resultsPerPage?: number | undefined;
    profiles?: string[] | undefined;
    profileScrapeSections?: ("reposts" | "videos")[] | undefined;
    profileSorting?: "latest" | "oldest" | "popular" | undefined;
    excludePinnedPosts?: boolean | undefined;
    oldestPostDateUnified?: string | undefined;
    newestPostDate?: string | undefined;
    mostDiggs?: number | undefined;
    leastDiggs?: number | undefined;
    maxFollowersPerProfile?: number | undefined;
    maxFollowingPerProfile?: number | undefined;
    searchSection?: "" | "/video" | "/user" | undefined;
    maxProfilesPerQuery?: number | undefined;
    searchSorting?: "0" | "1" | "3" | undefined;
    searchDatePosted?: "0" | "1" | "2" | "3" | "4" | "5" | undefined;
    postURLs?: string[] | undefined;
    scrapeRelatedVideos?: boolean | undefined;
    shouldDownloadVideos?: boolean | undefined;
    shouldDownloadCovers?: boolean | undefined;
    shouldDownloadSubtitles?: boolean | undefined;
    shouldDownloadSlideshowImages?: boolean | undefined;
    shouldDownloadAvatars?: boolean | undefined;
    shouldDownloadMusicCovers?: boolean | undefined;
    videoKvStoreIdOrName?: string | undefined;
    commentsPerPost?: number | undefined;
    maxRepliesPerComment?: number | undefined;
    proxyCountryCode?: "None" | "AF" | "AL" | "DZ" | "AS" | "AD" | "AO" | "AI" | "AG" | "AR" | "AM" | "AU" | "AT" | "AZ" | "BS" | "BH" | "BD" | "BB" | "BY" | "BE" | "BZ" | "BJ" | "BM" | "BT" | "BO" | "BA" | "BW" | "BR" | "VG" | "BN" | "BG" | "BF" | "BI" | "KH" | "CM" | "CA" | "CV" | "KY" | "TD" | "CL" | "CO" | "CK" | "CR" | "HR" | "CY" | "CZ" | "CD" | "DK" | "DJ" | "DO" | "EC" | "EG" | "SV" | "EE" | "ET" | "FK" | "FJ" | "FI" | "FR" | "PF" | "GA" | "GE" | "DE" | "GH" | "GI" | "GR" | "GL" | "GD" | "GP" | "GT" | "GN" | "GW" | "GY" | "HN" | "HK" | "HU" | "IS" | "IN" | "ID" | "IQ" | "IE" | "IM" | "IL" | "IT" | "CI" | "JM" | "JP" | "JE" | "KZ" | "KE" | "XK" | "KW" | "LA" | "LV" | "LB" | "LS" | "LR" | "LY" | "LT" | "LU" | "MO" | "MG" | "MW" | "MY" | "MV" | "ML" | "MT" | "MH" | "MQ" | "MR" | "MU" | "MX" | "MD" | "MC" | "MN" | "ME" | "MA" | "MZ" | "MM" | "NA" | "NR" | "NP" | "NL" | "NZ" | "NI" | "NG" | "MK" | "NO" | "OM" | "PK" | "PS" | "PA" | "PG" | "PY" | "PE" | "PH" | "PL" | "PT" | "PR" | "QA" | "CG" | "RO" | "RU" | "RW" | "RE" | "KN" | "LC" | "MF" | "PM" | "VC" | "SM" | "SA" | "SN" | "RS" | "SL" | "SG" | "SX" | "SK" | "SB" | "SO" | "ZA" | "KR" | "ES" | "LK" | "SR" | "SZ" | "SE" | "CH" | "TW" | "TJ" | "TZ" | "TH" | "TG" | "TO" | "TT" | "TN" | "TR" | "TM" | "TC" | "TV" | "VI" | "UG" | "UA" | "AE" | "GB" | "US" | "UY" | "VE" | "VN" | "WF" | "YE" | "ZM" | "ZW" | "AX" | undefined;
}>;
export declare const TikTokVideoSchema: z.ZodObject<{
    authorMeta: z.ZodOptional<z.ZodObject<{
        avatar: z.ZodOptional<z.ZodString>;
        bioLink: z.ZodOptional<z.ZodNull>;
        digg: z.ZodOptional<z.ZodNumber>;
        fans: z.ZodOptional<z.ZodNumber>;
        followDatasetUrl: z.ZodOptional<z.ZodNull>;
        following: z.ZodOptional<z.ZodNumber>;
        friends: z.ZodOptional<z.ZodNumber>;
        heart: z.ZodOptional<z.ZodNumber>;
        id: z.ZodOptional<z.ZodString>;
        name: z.ZodOptional<z.ZodString>;
        nickName: z.ZodOptional<z.ZodString>;
        originalAvatarUrl: z.ZodOptional<z.ZodString>;
        privateAccount: z.ZodOptional<z.ZodBoolean>;
        profileUrl: z.ZodOptional<z.ZodString>;
        signature: z.ZodOptional<z.ZodString>;
        verified: z.ZodOptional<z.ZodBoolean>;
        video: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        name?: string | undefined;
        id?: string | undefined;
        verified?: boolean | undefined;
        video?: number | undefined;
        avatar?: string | undefined;
        bioLink?: null | undefined;
        digg?: number | undefined;
        fans?: number | undefined;
        followDatasetUrl?: null | undefined;
        following?: number | undefined;
        friends?: number | undefined;
        heart?: number | undefined;
        nickName?: string | undefined;
        originalAvatarUrl?: string | undefined;
        privateAccount?: boolean | undefined;
        profileUrl?: string | undefined;
        signature?: string | undefined;
    }, {
        name?: string | undefined;
        id?: string | undefined;
        verified?: boolean | undefined;
        video?: number | undefined;
        avatar?: string | undefined;
        bioLink?: null | undefined;
        digg?: number | undefined;
        fans?: number | undefined;
        followDatasetUrl?: null | undefined;
        following?: number | undefined;
        friends?: number | undefined;
        heart?: number | undefined;
        nickName?: string | undefined;
        originalAvatarUrl?: string | undefined;
        privateAccount?: boolean | undefined;
        profileUrl?: string | undefined;
        signature?: string | undefined;
    }>>;
    collectCount: z.ZodOptional<z.ZodNumber>;
    commentCount: z.ZodOptional<z.ZodNumber>;
    commentsDatasetUrl: z.ZodOptional<z.ZodNull>;
    createTime: z.ZodOptional<z.ZodNumber>;
    createTimeISO: z.ZodOptional<z.ZodString>;
    detailedMentions: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodOptional<z.ZodString>;
        name: z.ZodOptional<z.ZodString>;
        nickName: z.ZodOptional<z.ZodString>;
        profileUrl: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name?: string | undefined;
        id?: string | undefined;
        nickName?: string | undefined;
        profileUrl?: string | undefined;
    }, {
        name?: string | undefined;
        id?: string | undefined;
        nickName?: string | undefined;
        profileUrl?: string | undefined;
    }>, "many">>;
    diggCount: z.ZodOptional<z.ZodNumber>;
    effectStickers: z.ZodOptional<z.ZodArray<z.ZodObject<{
        ID: z.ZodOptional<z.ZodString>;
        name: z.ZodOptional<z.ZodString>;
        stickerStats: z.ZodOptional<z.ZodObject<{
            useCount: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            useCount?: number | undefined;
        }, {
            useCount?: number | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        name?: string | undefined;
        ID?: string | undefined;
        stickerStats?: {
            useCount?: number | undefined;
        } | undefined;
    }, {
        name?: string | undefined;
        ID?: string | undefined;
        stickerStats?: {
            useCount?: number | undefined;
        } | undefined;
    }>, "many">>;
    hashtags: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name?: string | undefined;
    }, {
        name?: string | undefined;
    }>, "many">>;
    id: z.ZodOptional<z.ZodString>;
    input: z.ZodOptional<z.ZodString>;
    isAd: z.ZodOptional<z.ZodBoolean>;
    isPinned: z.ZodOptional<z.ZodBoolean>;
    isSlideshow: z.ZodOptional<z.ZodBoolean>;
    isSponsored: z.ZodOptional<z.ZodBoolean>;
    mediaUrls: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    mentions: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    musicMeta: z.ZodOptional<z.ZodObject<{
        coverMediumUrl: z.ZodOptional<z.ZodString>;
        musicAuthor: z.ZodOptional<z.ZodString>;
        musicId: z.ZodOptional<z.ZodString>;
        musicName: z.ZodOptional<z.ZodString>;
        musicOriginal: z.ZodOptional<z.ZodBoolean>;
        originalCoverMediumUrl: z.ZodOptional<z.ZodString>;
        playUrl: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        coverMediumUrl?: string | undefined;
        musicAuthor?: string | undefined;
        musicId?: string | undefined;
        musicName?: string | undefined;
        musicOriginal?: boolean | undefined;
        originalCoverMediumUrl?: string | undefined;
        playUrl?: string | undefined;
    }, {
        coverMediumUrl?: string | undefined;
        musicAuthor?: string | undefined;
        musicId?: string | undefined;
        musicName?: string | undefined;
        musicOriginal?: boolean | undefined;
        originalCoverMediumUrl?: string | undefined;
        playUrl?: string | undefined;
    }>>;
    playCount: z.ZodOptional<z.ZodNumber>;
    repostCount: z.ZodOptional<z.ZodNumber>;
    searchHashtag: z.ZodOptional<z.ZodObject<{
        name: z.ZodOptional<z.ZodString>;
        views: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        name?: string | undefined;
        views?: number | undefined;
    }, {
        name?: string | undefined;
        views?: number | undefined;
    }>>;
    shareCount: z.ZodOptional<z.ZodNumber>;
    text: z.ZodOptional<z.ZodString>;
    textLanguage: z.ZodOptional<z.ZodString>;
    videoMeta: z.ZodOptional<z.ZodObject<{
        coverUrl: z.ZodOptional<z.ZodString>;
        definition: z.ZodOptional<z.ZodString>;
        duration: z.ZodOptional<z.ZodNumber>;
        format: z.ZodOptional<z.ZodString>;
        height: z.ZodOptional<z.ZodNumber>;
        originalCoverUrl: z.ZodOptional<z.ZodString>;
        subtitleLinks: z.ZodOptional<z.ZodArray<z.ZodObject<{
            language: z.ZodOptional<z.ZodString>;
            downloadLink: z.ZodOptional<z.ZodString>;
            tiktokLink: z.ZodOptional<z.ZodString>;
            source: z.ZodOptional<z.ZodString>;
            sourceUnabbreviated: z.ZodOptional<z.ZodString>;
            version: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            language?: string | undefined;
            downloadLink?: string | undefined;
            tiktokLink?: string | undefined;
            source?: string | undefined;
            sourceUnabbreviated?: string | undefined;
            version?: string | undefined;
        }, {
            language?: string | undefined;
            downloadLink?: string | undefined;
            tiktokLink?: string | undefined;
            source?: string | undefined;
            sourceUnabbreviated?: string | undefined;
            version?: string | undefined;
        }>, "many">>;
        width: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        format?: string | undefined;
        duration?: number | undefined;
        coverUrl?: string | undefined;
        width?: number | undefined;
        height?: number | undefined;
        definition?: string | undefined;
        originalCoverUrl?: string | undefined;
        subtitleLinks?: {
            language?: string | undefined;
            downloadLink?: string | undefined;
            tiktokLink?: string | undefined;
            source?: string | undefined;
            sourceUnabbreviated?: string | undefined;
            version?: string | undefined;
        }[] | undefined;
    }, {
        format?: string | undefined;
        duration?: number | undefined;
        coverUrl?: string | undefined;
        width?: number | undefined;
        height?: number | undefined;
        definition?: string | undefined;
        originalCoverUrl?: string | undefined;
        subtitleLinks?: {
            language?: string | undefined;
            downloadLink?: string | undefined;
            tiktokLink?: string | undefined;
            source?: string | undefined;
            sourceUnabbreviated?: string | undefined;
            version?: string | undefined;
        }[] | undefined;
    }>>;
    webVideoUrl: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    input?: string | undefined;
    text?: string | undefined;
    id?: string | undefined;
    hashtags?: {
        name?: string | undefined;
    }[] | undefined;
    mentions?: string[] | undefined;
    isSponsored?: boolean | undefined;
    authorMeta?: {
        name?: string | undefined;
        id?: string | undefined;
        verified?: boolean | undefined;
        video?: number | undefined;
        avatar?: string | undefined;
        bioLink?: null | undefined;
        digg?: number | undefined;
        fans?: number | undefined;
        followDatasetUrl?: null | undefined;
        following?: number | undefined;
        friends?: number | undefined;
        heart?: number | undefined;
        nickName?: string | undefined;
        originalAvatarUrl?: string | undefined;
        privateAccount?: boolean | undefined;
        profileUrl?: string | undefined;
        signature?: string | undefined;
    } | undefined;
    collectCount?: number | undefined;
    commentCount?: number | undefined;
    commentsDatasetUrl?: null | undefined;
    createTime?: number | undefined;
    createTimeISO?: string | undefined;
    detailedMentions?: {
        name?: string | undefined;
        id?: string | undefined;
        nickName?: string | undefined;
        profileUrl?: string | undefined;
    }[] | undefined;
    diggCount?: number | undefined;
    effectStickers?: {
        name?: string | undefined;
        ID?: string | undefined;
        stickerStats?: {
            useCount?: number | undefined;
        } | undefined;
    }[] | undefined;
    isAd?: boolean | undefined;
    isPinned?: boolean | undefined;
    isSlideshow?: boolean | undefined;
    mediaUrls?: string[] | undefined;
    musicMeta?: {
        coverMediumUrl?: string | undefined;
        musicAuthor?: string | undefined;
        musicId?: string | undefined;
        musicName?: string | undefined;
        musicOriginal?: boolean | undefined;
        originalCoverMediumUrl?: string | undefined;
        playUrl?: string | undefined;
    } | undefined;
    playCount?: number | undefined;
    repostCount?: number | undefined;
    searchHashtag?: {
        name?: string | undefined;
        views?: number | undefined;
    } | undefined;
    shareCount?: number | undefined;
    textLanguage?: string | undefined;
    videoMeta?: {
        format?: string | undefined;
        duration?: number | undefined;
        coverUrl?: string | undefined;
        width?: number | undefined;
        height?: number | undefined;
        definition?: string | undefined;
        originalCoverUrl?: string | undefined;
        subtitleLinks?: {
            language?: string | undefined;
            downloadLink?: string | undefined;
            tiktokLink?: string | undefined;
            source?: string | undefined;
            sourceUnabbreviated?: string | undefined;
            version?: string | undefined;
        }[] | undefined;
    } | undefined;
    webVideoUrl?: string | undefined;
}, {
    input?: string | undefined;
    text?: string | undefined;
    id?: string | undefined;
    hashtags?: {
        name?: string | undefined;
    }[] | undefined;
    mentions?: string[] | undefined;
    isSponsored?: boolean | undefined;
    authorMeta?: {
        name?: string | undefined;
        id?: string | undefined;
        verified?: boolean | undefined;
        video?: number | undefined;
        avatar?: string | undefined;
        bioLink?: null | undefined;
        digg?: number | undefined;
        fans?: number | undefined;
        followDatasetUrl?: null | undefined;
        following?: number | undefined;
        friends?: number | undefined;
        heart?: number | undefined;
        nickName?: string | undefined;
        originalAvatarUrl?: string | undefined;
        privateAccount?: boolean | undefined;
        profileUrl?: string | undefined;
        signature?: string | undefined;
    } | undefined;
    collectCount?: number | undefined;
    commentCount?: number | undefined;
    commentsDatasetUrl?: null | undefined;
    createTime?: number | undefined;
    createTimeISO?: string | undefined;
    detailedMentions?: {
        name?: string | undefined;
        id?: string | undefined;
        nickName?: string | undefined;
        profileUrl?: string | undefined;
    }[] | undefined;
    diggCount?: number | undefined;
    effectStickers?: {
        name?: string | undefined;
        ID?: string | undefined;
        stickerStats?: {
            useCount?: number | undefined;
        } | undefined;
    }[] | undefined;
    isAd?: boolean | undefined;
    isPinned?: boolean | undefined;
    isSlideshow?: boolean | undefined;
    mediaUrls?: string[] | undefined;
    musicMeta?: {
        coverMediumUrl?: string | undefined;
        musicAuthor?: string | undefined;
        musicId?: string | undefined;
        musicName?: string | undefined;
        musicOriginal?: boolean | undefined;
        originalCoverMediumUrl?: string | undefined;
        playUrl?: string | undefined;
    } | undefined;
    playCount?: number | undefined;
    repostCount?: number | undefined;
    searchHashtag?: {
        name?: string | undefined;
        views?: number | undefined;
    } | undefined;
    shareCount?: number | undefined;
    textLanguage?: string | undefined;
    videoMeta?: {
        format?: string | undefined;
        duration?: number | undefined;
        coverUrl?: string | undefined;
        width?: number | undefined;
        height?: number | undefined;
        definition?: string | undefined;
        originalCoverUrl?: string | undefined;
        subtitleLinks?: {
            language?: string | undefined;
            downloadLink?: string | undefined;
            tiktokLink?: string | undefined;
            source?: string | undefined;
            sourceUnabbreviated?: string | undefined;
            version?: string | undefined;
        }[] | undefined;
    } | undefined;
    webVideoUrl?: string | undefined;
}>;
export type TikTokScraperInput = z.output<typeof TikTokScraperInputSchema>;
export type TikTokVideo = z.output<typeof TikTokVideoSchema>;
//# sourceMappingURL=tiktok-scraper.d.ts.map