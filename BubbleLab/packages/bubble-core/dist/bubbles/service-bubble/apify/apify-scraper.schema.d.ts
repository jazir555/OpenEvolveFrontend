export declare const APIFY_ACTOR_SCHEMAS: {
    'apify/instagram-scraper': {
        input: import("zod").ZodObject<{
            directUrls: import("zod").ZodArray<import("zod").ZodString, "many">;
            resultsType: import("zod").ZodDefault<import("zod").ZodEnum<["posts", "details"]>>;
            resultsLimit: import("zod").ZodDefault<import("zod").ZodNumber>;
            includeStories: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
            includeHighlights: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
        }, "strip", import("zod").ZodTypeAny, {
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
        output: import("zod").ZodObject<{
            inputUrl: import("zod").ZodOptional<import("zod").ZodString>;
            id: import("zod").ZodOptional<import("zod").ZodString>;
            username: import("zod").ZodOptional<import("zod").ZodString>;
            url: import("zod").ZodOptional<import("zod").ZodString>;
            fullName: import("zod").ZodOptional<import("zod").ZodString>;
            biography: import("zod").ZodOptional<import("zod").ZodString>;
            externalUrls: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                title: import("zod").ZodOptional<import("zod").ZodString>;
                lynx_url: import("zod").ZodOptional<import("zod").ZodString>;
                url: import("zod").ZodOptional<import("zod").ZodString>;
                link_type: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
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
            externalUrl: import("zod").ZodOptional<import("zod").ZodString>;
            externalUrlShimmed: import("zod").ZodOptional<import("zod").ZodString>;
            followersCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            followsCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            postsCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            hasChannel: import("zod").ZodOptional<import("zod").ZodBoolean>;
            highlightReelCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            isBusinessAccount: import("zod").ZodOptional<import("zod").ZodBoolean>;
            joinedRecently: import("zod").ZodOptional<import("zod").ZodBoolean>;
            businessCategoryName: import("zod").ZodOptional<import("zod").ZodString>;
            private: import("zod").ZodOptional<import("zod").ZodBoolean>;
            verified: import("zod").ZodOptional<import("zod").ZodBoolean>;
            profilePicUrl: import("zod").ZodOptional<import("zod").ZodString>;
            profilePicUrlHD: import("zod").ZodOptional<import("zod").ZodString>;
            igtvVideoCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            latestIgtvVideos: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodUnknown, "many">>;
            relatedProfiles: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodUnknown, "many">>;
            latestPosts: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                id: import("zod").ZodOptional<import("zod").ZodString>;
                type: import("zod").ZodOptional<import("zod").ZodString>;
                shortCode: import("zod").ZodOptional<import("zod").ZodString>;
                caption: import("zod").ZodOptional<import("zod").ZodString>;
                hashtags: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
                mentions: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
                url: import("zod").ZodOptional<import("zod").ZodString>;
                commentsCount: import("zod").ZodOptional<import("zod").ZodNumber>;
                dimensionsHeight: import("zod").ZodOptional<import("zod").ZodNumber>;
                dimensionsWidth: import("zod").ZodOptional<import("zod").ZodNumber>;
                displayUrl: import("zod").ZodOptional<import("zod").ZodString>;
                images: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
                videoUrl: import("zod").ZodOptional<import("zod").ZodString>;
                alt: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                likesCount: import("zod").ZodOptional<import("zod").ZodNumber>;
                videoViewCount: import("zod").ZodOptional<import("zod").ZodNumber>;
                timestamp: import("zod").ZodOptional<import("zod").ZodString>;
                childPosts: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodUnknown, "many">>;
                ownerUsername: import("zod").ZodOptional<import("zod").ZodString>;
                ownerId: import("zod").ZodOptional<import("zod").ZodString>;
                productType: import("zod").ZodOptional<import("zod").ZodString>;
                taggedUsers: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                    full_name: import("zod").ZodOptional<import("zod").ZodString>;
                    id: import("zod").ZodOptional<import("zod").ZodString>;
                    is_verified: import("zod").ZodOptional<import("zod").ZodBoolean>;
                    profile_pic_url: import("zod").ZodOptional<import("zod").ZodString>;
                    username: import("zod").ZodOptional<import("zod").ZodString>;
                }, "strip", import("zod").ZodTypeAny, {
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
                isCommentsDisabled: import("zod").ZodOptional<import("zod").ZodBoolean>;
                location: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodObject<{
                    name: import("zod").ZodOptional<import("zod").ZodString>;
                    id: import("zod").ZodOptional<import("zod").ZodString>;
                }, "strip", import("zod").ZodTypeAny, {
                    name?: string | undefined;
                    id?: string | undefined;
                }, {
                    name?: string | undefined;
                    id?: string | undefined;
                }>>>;
            }, "strip", import("zod").ZodTypeAny, {
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
            stories: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                url: import("zod").ZodOptional<import("zod").ZodString>;
                timestamp: import("zod").ZodOptional<import("zod").ZodString>;
                type: import("zod").ZodOptional<import("zod").ZodEnum<["image", "video"]>>;
                viewsCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            }, "strip", import("zod").ZodTypeAny, {
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
            highlights: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                title: import("zod").ZodOptional<import("zod").ZodString>;
                coverUrl: import("zod").ZodOptional<import("zod").ZodString>;
                itemsCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            }, "strip", import("zod").ZodTypeAny, {
                title?: string | undefined;
                itemsCount?: number | undefined;
                coverUrl?: string | undefined;
            }, {
                title?: string | undefined;
                itemsCount?: number | undefined;
                coverUrl?: string | undefined;
            }>, "many">>;
        }, "strip", import("zod").ZodTypeAny, {
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
        description: string;
        documentation: string;
        category: string;
    };
    'apify/instagram-hashtag-scraper': {
        input: import("zod").ZodObject<{
            hashtags: import("zod").ZodArray<import("zod").ZodString, "many">;
            resultsLimit: import("zod").ZodDefault<import("zod").ZodNumber>;
            addParentData: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
        }, "strip", import("zod").ZodTypeAny, {
            resultsLimit: number;
            hashtags: string[];
            addParentData?: boolean | undefined;
        }, {
            hashtags: string[];
            resultsLimit?: number | undefined;
            addParentData?: boolean | undefined;
        }>;
        output: import("zod").ZodObject<{
            id: import("zod").ZodOptional<import("zod").ZodString>;
            type: import("zod").ZodOptional<import("zod").ZodString>;
            shortCode: import("zod").ZodOptional<import("zod").ZodString>;
            caption: import("zod").ZodOptional<import("zod").ZodString>;
            hashtags: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            mentions: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            url: import("zod").ZodOptional<import("zod").ZodString>;
            commentsCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            dimensionsHeight: import("zod").ZodOptional<import("zod").ZodNumber>;
            dimensionsWidth: import("zod").ZodOptional<import("zod").ZodNumber>;
            displayUrl: import("zod").ZodOptional<import("zod").ZodString>;
            images: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            videoUrl: import("zod").ZodOptional<import("zod").ZodString>;
            alt: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
            likesCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            videoViewCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            timestamp: import("zod").ZodOptional<import("zod").ZodString>;
            childPosts: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodUnknown, "many">>;
            ownerUsername: import("zod").ZodOptional<import("zod").ZodString>;
            ownerId: import("zod").ZodOptional<import("zod").ZodString>;
            productType: import("zod").ZodOptional<import("zod").ZodString>;
            taggedUsers: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                full_name: import("zod").ZodOptional<import("zod").ZodString>;
                id: import("zod").ZodOptional<import("zod").ZodString>;
                is_verified: import("zod").ZodOptional<import("zod").ZodBoolean>;
                profile_pic_url: import("zod").ZodOptional<import("zod").ZodString>;
                username: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
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
            isCommentsDisabled: import("zod").ZodOptional<import("zod").ZodBoolean>;
            location: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodObject<{
                name: import("zod").ZodOptional<import("zod").ZodString>;
                id: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
                name?: string | undefined;
                id?: string | undefined;
            }, {
                name?: string | undefined;
                id?: string | undefined;
            }>>>;
        } & {
            inputUrl: import("zod").ZodOptional<import("zod").ZodString>;
            locationName: import("zod").ZodOptional<import("zod").ZodString>;
            locationId: import("zod").ZodOptional<import("zod").ZodString>;
            ownerFullName: import("zod").ZodOptional<import("zod").ZodString>;
            isSponsored: import("zod").ZodOptional<import("zod").ZodBoolean>;
            firstComment: import("zod").ZodOptional<import("zod").ZodString>;
            latestComments: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodUnknown, "many">>;
            musicInfo: import("zod").ZodOptional<import("zod").ZodObject<{
                audio_canonical_id: import("zod").ZodOptional<import("zod").ZodString>;
                audio_type: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                music_info: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodObject<{
                    music_asset_info: import("zod").ZodOptional<import("zod").ZodObject<{
                        allows_saving: import("zod").ZodOptional<import("zod").ZodBoolean>;
                        artist_id: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                        audio_id: import("zod").ZodOptional<import("zod").ZodString>;
                        cover_artwork_thumbnail_uri: import("zod").ZodOptional<import("zod").ZodString>;
                        cover_artwork_uri: import("zod").ZodOptional<import("zod").ZodString>;
                        dark_message: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                        display_artist: import("zod").ZodOptional<import("zod").ZodString>;
                        duration_in_ms: import("zod").ZodOptional<import("zod").ZodNumber>;
                        fast_start_progressive_download_url: import("zod").ZodOptional<import("zod").ZodString>;
                        has_lyrics: import("zod").ZodOptional<import("zod").ZodBoolean>;
                        highlight_start_times_in_ms: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodNumber, "many">>;
                        id: import("zod").ZodOptional<import("zod").ZodString>;
                        ig_username: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                        is_eligible_for_audio_effects: import("zod").ZodOptional<import("zod").ZodBoolean>;
                        is_eligible_for_vinyl_sticker: import("zod").ZodOptional<import("zod").ZodBoolean>;
                        is_explicit: import("zod").ZodOptional<import("zod").ZodBoolean>;
                        licensed_music_subtype: import("zod").ZodOptional<import("zod").ZodString>;
                        lyrics: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                        progressive_download_url: import("zod").ZodOptional<import("zod").ZodString>;
                        reactive_audio_download_url: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                        sanitized_title: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                        song_monetization_info: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodUnknown>>;
                        spotify_track_metadata: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodUnknown>>;
                        subtitle: import("zod").ZodOptional<import("zod").ZodString>;
                        title: import("zod").ZodOptional<import("zod").ZodString>;
                        web_30s_preview_download_url: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                    }, "strip", import("zod").ZodTypeAny, {
                        title?: string | undefined;
                        id?: string | undefined;
                        allows_saving?: boolean | undefined;
                        artist_id?: string | null | undefined;
                        audio_id?: string | undefined;
                        cover_artwork_thumbnail_uri?: string | undefined;
                        cover_artwork_uri?: string | undefined;
                        dark_message?: string | null | undefined;
                        display_artist?: string | undefined;
                        duration_in_ms?: number | undefined;
                        fast_start_progressive_download_url?: string | undefined;
                        has_lyrics?: boolean | undefined;
                        highlight_start_times_in_ms?: number[] | undefined;
                        ig_username?: string | null | undefined;
                        is_eligible_for_audio_effects?: boolean | undefined;
                        is_eligible_for_vinyl_sticker?: boolean | undefined;
                        is_explicit?: boolean | undefined;
                        licensed_music_subtype?: string | undefined;
                        lyrics?: string | null | undefined;
                        progressive_download_url?: string | undefined;
                        reactive_audio_download_url?: string | null | undefined;
                        sanitized_title?: string | null | undefined;
                        song_monetization_info?: unknown;
                        spotify_track_metadata?: unknown;
                        subtitle?: string | undefined;
                        web_30s_preview_download_url?: string | null | undefined;
                    }, {
                        title?: string | undefined;
                        id?: string | undefined;
                        allows_saving?: boolean | undefined;
                        artist_id?: string | null | undefined;
                        audio_id?: string | undefined;
                        cover_artwork_thumbnail_uri?: string | undefined;
                        cover_artwork_uri?: string | undefined;
                        dark_message?: string | null | undefined;
                        display_artist?: string | undefined;
                        duration_in_ms?: number | undefined;
                        fast_start_progressive_download_url?: string | undefined;
                        has_lyrics?: boolean | undefined;
                        highlight_start_times_in_ms?: number[] | undefined;
                        ig_username?: string | null | undefined;
                        is_eligible_for_audio_effects?: boolean | undefined;
                        is_eligible_for_vinyl_sticker?: boolean | undefined;
                        is_explicit?: boolean | undefined;
                        licensed_music_subtype?: string | undefined;
                        lyrics?: string | null | undefined;
                        progressive_download_url?: string | undefined;
                        reactive_audio_download_url?: string | null | undefined;
                        sanitized_title?: string | null | undefined;
                        song_monetization_info?: unknown;
                        spotify_track_metadata?: unknown;
                        subtitle?: string | undefined;
                        web_30s_preview_download_url?: string | null | undefined;
                    }>>;
                    music_consumption_info: import("zod").ZodOptional<import("zod").ZodObject<{
                        allow_media_creation_with_music: import("zod").ZodOptional<import("zod").ZodBoolean>;
                        audio_asset_start_time_in_ms: import("zod").ZodOptional<import("zod").ZodNumber>;
                        audio_filter_infos: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodUnknown, "many">>;
                        audio_muting_info: import("zod").ZodOptional<import("zod").ZodObject<{
                            allow_audio_editing: import("zod").ZodOptional<import("zod").ZodBoolean>;
                            mute_audio: import("zod").ZodOptional<import("zod").ZodBoolean>;
                            mute_reason_str: import("zod").ZodOptional<import("zod").ZodString>;
                            show_muted_audio_toast: import("zod").ZodOptional<import("zod").ZodBoolean>;
                        }, "strip", import("zod").ZodTypeAny, {
                            allow_audio_editing?: boolean | undefined;
                            mute_audio?: boolean | undefined;
                            mute_reason_str?: string | undefined;
                            show_muted_audio_toast?: boolean | undefined;
                        }, {
                            allow_audio_editing?: boolean | undefined;
                            mute_audio?: boolean | undefined;
                            mute_reason_str?: string | undefined;
                            show_muted_audio_toast?: boolean | undefined;
                        }>>;
                        contains_lyrics: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodBoolean>>;
                        derived_content_id: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                        derived_content_start_time_in_composition_in_ms: import("zod").ZodOptional<import("zod").ZodNumber>;
                        display_labels: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodUnknown>>;
                        formatted_clips_media_count: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                        ig_artist: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodObject<{
                            full_name: import("zod").ZodOptional<import("zod").ZodString>;
                            id: import("zod").ZodOptional<import("zod").ZodString>;
                            is_private: import("zod").ZodOptional<import("zod").ZodBoolean>;
                            is_verified: import("zod").ZodOptional<import("zod").ZodBoolean>;
                            profile_pic_id: import("zod").ZodOptional<import("zod").ZodString>;
                            profile_pic_url: import("zod").ZodOptional<import("zod").ZodString>;
                            username: import("zod").ZodOptional<import("zod").ZodString>;
                        }, "strip", import("zod").ZodTypeAny, {
                            username?: string | undefined;
                            id?: string | undefined;
                            is_private?: boolean | undefined;
                            full_name?: string | undefined;
                            is_verified?: boolean | undefined;
                            profile_pic_url?: string | undefined;
                            profile_pic_id?: string | undefined;
                        }, {
                            username?: string | undefined;
                            id?: string | undefined;
                            is_private?: boolean | undefined;
                            full_name?: string | undefined;
                            is_verified?: boolean | undefined;
                            profile_pic_url?: string | undefined;
                            profile_pic_id?: string | undefined;
                        }>>>;
                        is_bookmarked: import("zod").ZodOptional<import("zod").ZodBoolean>;
                        is_trending_in_clips: import("zod").ZodOptional<import("zod").ZodBoolean>;
                        music_creation_restriction_reason: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                        overlap_duration_in_ms: import("zod").ZodOptional<import("zod").ZodNumber>;
                        placeholder_profile_pic_url: import("zod").ZodOptional<import("zod").ZodString>;
                        previous_trend_rank: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodNumber>>;
                        should_allow_music_editing: import("zod").ZodOptional<import("zod").ZodBoolean>;
                        should_mute_audio: import("zod").ZodOptional<import("zod").ZodBoolean>;
                        should_mute_audio_reason: import("zod").ZodOptional<import("zod").ZodString>;
                        should_mute_audio_reason_type: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                        trend_rank: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodNumber>>;
                        user_notes: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodUnknown>>;
                    }, "strip", import("zod").ZodTypeAny, {
                        allow_media_creation_with_music?: boolean | undefined;
                        audio_asset_start_time_in_ms?: number | undefined;
                        audio_filter_infos?: unknown[] | undefined;
                        audio_muting_info?: {
                            allow_audio_editing?: boolean | undefined;
                            mute_audio?: boolean | undefined;
                            mute_reason_str?: string | undefined;
                            show_muted_audio_toast?: boolean | undefined;
                        } | undefined;
                        contains_lyrics?: boolean | null | undefined;
                        derived_content_id?: string | null | undefined;
                        derived_content_start_time_in_composition_in_ms?: number | undefined;
                        display_labels?: unknown;
                        formatted_clips_media_count?: string | null | undefined;
                        ig_artist?: {
                            username?: string | undefined;
                            id?: string | undefined;
                            is_private?: boolean | undefined;
                            full_name?: string | undefined;
                            is_verified?: boolean | undefined;
                            profile_pic_url?: string | undefined;
                            profile_pic_id?: string | undefined;
                        } | null | undefined;
                        is_bookmarked?: boolean | undefined;
                        is_trending_in_clips?: boolean | undefined;
                        music_creation_restriction_reason?: string | null | undefined;
                        overlap_duration_in_ms?: number | undefined;
                        placeholder_profile_pic_url?: string | undefined;
                        previous_trend_rank?: number | null | undefined;
                        should_allow_music_editing?: boolean | undefined;
                        should_mute_audio?: boolean | undefined;
                        should_mute_audio_reason?: string | undefined;
                        should_mute_audio_reason_type?: string | null | undefined;
                        trend_rank?: number | null | undefined;
                        user_notes?: unknown;
                    }, {
                        allow_media_creation_with_music?: boolean | undefined;
                        audio_asset_start_time_in_ms?: number | undefined;
                        audio_filter_infos?: unknown[] | undefined;
                        audio_muting_info?: {
                            allow_audio_editing?: boolean | undefined;
                            mute_audio?: boolean | undefined;
                            mute_reason_str?: string | undefined;
                            show_muted_audio_toast?: boolean | undefined;
                        } | undefined;
                        contains_lyrics?: boolean | null | undefined;
                        derived_content_id?: string | null | undefined;
                        derived_content_start_time_in_composition_in_ms?: number | undefined;
                        display_labels?: unknown;
                        formatted_clips_media_count?: string | null | undefined;
                        ig_artist?: {
                            username?: string | undefined;
                            id?: string | undefined;
                            is_private?: boolean | undefined;
                            full_name?: string | undefined;
                            is_verified?: boolean | undefined;
                            profile_pic_url?: string | undefined;
                            profile_pic_id?: string | undefined;
                        } | null | undefined;
                        is_bookmarked?: boolean | undefined;
                        is_trending_in_clips?: boolean | undefined;
                        music_creation_restriction_reason?: string | null | undefined;
                        overlap_duration_in_ms?: number | undefined;
                        placeholder_profile_pic_url?: string | undefined;
                        previous_trend_rank?: number | null | undefined;
                        should_allow_music_editing?: boolean | undefined;
                        should_mute_audio?: boolean | undefined;
                        should_mute_audio_reason?: string | undefined;
                        should_mute_audio_reason_type?: string | null | undefined;
                        trend_rank?: number | null | undefined;
                        user_notes?: unknown;
                    }>>;
                }, "strip", import("zod").ZodTypeAny, {
                    music_asset_info?: {
                        title?: string | undefined;
                        id?: string | undefined;
                        allows_saving?: boolean | undefined;
                        artist_id?: string | null | undefined;
                        audio_id?: string | undefined;
                        cover_artwork_thumbnail_uri?: string | undefined;
                        cover_artwork_uri?: string | undefined;
                        dark_message?: string | null | undefined;
                        display_artist?: string | undefined;
                        duration_in_ms?: number | undefined;
                        fast_start_progressive_download_url?: string | undefined;
                        has_lyrics?: boolean | undefined;
                        highlight_start_times_in_ms?: number[] | undefined;
                        ig_username?: string | null | undefined;
                        is_eligible_for_audio_effects?: boolean | undefined;
                        is_eligible_for_vinyl_sticker?: boolean | undefined;
                        is_explicit?: boolean | undefined;
                        licensed_music_subtype?: string | undefined;
                        lyrics?: string | null | undefined;
                        progressive_download_url?: string | undefined;
                        reactive_audio_download_url?: string | null | undefined;
                        sanitized_title?: string | null | undefined;
                        song_monetization_info?: unknown;
                        spotify_track_metadata?: unknown;
                        subtitle?: string | undefined;
                        web_30s_preview_download_url?: string | null | undefined;
                    } | undefined;
                    music_consumption_info?: {
                        allow_media_creation_with_music?: boolean | undefined;
                        audio_asset_start_time_in_ms?: number | undefined;
                        audio_filter_infos?: unknown[] | undefined;
                        audio_muting_info?: {
                            allow_audio_editing?: boolean | undefined;
                            mute_audio?: boolean | undefined;
                            mute_reason_str?: string | undefined;
                            show_muted_audio_toast?: boolean | undefined;
                        } | undefined;
                        contains_lyrics?: boolean | null | undefined;
                        derived_content_id?: string | null | undefined;
                        derived_content_start_time_in_composition_in_ms?: number | undefined;
                        display_labels?: unknown;
                        formatted_clips_media_count?: string | null | undefined;
                        ig_artist?: {
                            username?: string | undefined;
                            id?: string | undefined;
                            is_private?: boolean | undefined;
                            full_name?: string | undefined;
                            is_verified?: boolean | undefined;
                            profile_pic_url?: string | undefined;
                            profile_pic_id?: string | undefined;
                        } | null | undefined;
                        is_bookmarked?: boolean | undefined;
                        is_trending_in_clips?: boolean | undefined;
                        music_creation_restriction_reason?: string | null | undefined;
                        overlap_duration_in_ms?: number | undefined;
                        placeholder_profile_pic_url?: string | undefined;
                        previous_trend_rank?: number | null | undefined;
                        should_allow_music_editing?: boolean | undefined;
                        should_mute_audio?: boolean | undefined;
                        should_mute_audio_reason?: string | undefined;
                        should_mute_audio_reason_type?: string | null | undefined;
                        trend_rank?: number | null | undefined;
                        user_notes?: unknown;
                    } | undefined;
                }, {
                    music_asset_info?: {
                        title?: string | undefined;
                        id?: string | undefined;
                        allows_saving?: boolean | undefined;
                        artist_id?: string | null | undefined;
                        audio_id?: string | undefined;
                        cover_artwork_thumbnail_uri?: string | undefined;
                        cover_artwork_uri?: string | undefined;
                        dark_message?: string | null | undefined;
                        display_artist?: string | undefined;
                        duration_in_ms?: number | undefined;
                        fast_start_progressive_download_url?: string | undefined;
                        has_lyrics?: boolean | undefined;
                        highlight_start_times_in_ms?: number[] | undefined;
                        ig_username?: string | null | undefined;
                        is_eligible_for_audio_effects?: boolean | undefined;
                        is_eligible_for_vinyl_sticker?: boolean | undefined;
                        is_explicit?: boolean | undefined;
                        licensed_music_subtype?: string | undefined;
                        lyrics?: string | null | undefined;
                        progressive_download_url?: string | undefined;
                        reactive_audio_download_url?: string | null | undefined;
                        sanitized_title?: string | null | undefined;
                        song_monetization_info?: unknown;
                        spotify_track_metadata?: unknown;
                        subtitle?: string | undefined;
                        web_30s_preview_download_url?: string | null | undefined;
                    } | undefined;
                    music_consumption_info?: {
                        allow_media_creation_with_music?: boolean | undefined;
                        audio_asset_start_time_in_ms?: number | undefined;
                        audio_filter_infos?: unknown[] | undefined;
                        audio_muting_info?: {
                            allow_audio_editing?: boolean | undefined;
                            mute_audio?: boolean | undefined;
                            mute_reason_str?: string | undefined;
                            show_muted_audio_toast?: boolean | undefined;
                        } | undefined;
                        contains_lyrics?: boolean | null | undefined;
                        derived_content_id?: string | null | undefined;
                        derived_content_start_time_in_composition_in_ms?: number | undefined;
                        display_labels?: unknown;
                        formatted_clips_media_count?: string | null | undefined;
                        ig_artist?: {
                            username?: string | undefined;
                            id?: string | undefined;
                            is_private?: boolean | undefined;
                            full_name?: string | undefined;
                            is_verified?: boolean | undefined;
                            profile_pic_url?: string | undefined;
                            profile_pic_id?: string | undefined;
                        } | null | undefined;
                        is_bookmarked?: boolean | undefined;
                        is_trending_in_clips?: boolean | undefined;
                        music_creation_restriction_reason?: string | null | undefined;
                        overlap_duration_in_ms?: number | undefined;
                        placeholder_profile_pic_url?: string | undefined;
                        previous_trend_rank?: number | null | undefined;
                        should_allow_music_editing?: boolean | undefined;
                        should_mute_audio?: boolean | undefined;
                        should_mute_audio_reason?: string | undefined;
                        should_mute_audio_reason_type?: string | null | undefined;
                        trend_rank?: number | null | undefined;
                        user_notes?: unknown;
                    } | undefined;
                }>>>;
                original_sound_info: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodUnknown>>;
                pinned_media_ids: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodArray<import("zod").ZodUnknown, "many">>>;
            }, "strip", import("zod").ZodTypeAny, {
                audio_canonical_id?: string | undefined;
                audio_type?: string | null | undefined;
                music_info?: {
                    music_asset_info?: {
                        title?: string | undefined;
                        id?: string | undefined;
                        allows_saving?: boolean | undefined;
                        artist_id?: string | null | undefined;
                        audio_id?: string | undefined;
                        cover_artwork_thumbnail_uri?: string | undefined;
                        cover_artwork_uri?: string | undefined;
                        dark_message?: string | null | undefined;
                        display_artist?: string | undefined;
                        duration_in_ms?: number | undefined;
                        fast_start_progressive_download_url?: string | undefined;
                        has_lyrics?: boolean | undefined;
                        highlight_start_times_in_ms?: number[] | undefined;
                        ig_username?: string | null | undefined;
                        is_eligible_for_audio_effects?: boolean | undefined;
                        is_eligible_for_vinyl_sticker?: boolean | undefined;
                        is_explicit?: boolean | undefined;
                        licensed_music_subtype?: string | undefined;
                        lyrics?: string | null | undefined;
                        progressive_download_url?: string | undefined;
                        reactive_audio_download_url?: string | null | undefined;
                        sanitized_title?: string | null | undefined;
                        song_monetization_info?: unknown;
                        spotify_track_metadata?: unknown;
                        subtitle?: string | undefined;
                        web_30s_preview_download_url?: string | null | undefined;
                    } | undefined;
                    music_consumption_info?: {
                        allow_media_creation_with_music?: boolean | undefined;
                        audio_asset_start_time_in_ms?: number | undefined;
                        audio_filter_infos?: unknown[] | undefined;
                        audio_muting_info?: {
                            allow_audio_editing?: boolean | undefined;
                            mute_audio?: boolean | undefined;
                            mute_reason_str?: string | undefined;
                            show_muted_audio_toast?: boolean | undefined;
                        } | undefined;
                        contains_lyrics?: boolean | null | undefined;
                        derived_content_id?: string | null | undefined;
                        derived_content_start_time_in_composition_in_ms?: number | undefined;
                        display_labels?: unknown;
                        formatted_clips_media_count?: string | null | undefined;
                        ig_artist?: {
                            username?: string | undefined;
                            id?: string | undefined;
                            is_private?: boolean | undefined;
                            full_name?: string | undefined;
                            is_verified?: boolean | undefined;
                            profile_pic_url?: string | undefined;
                            profile_pic_id?: string | undefined;
                        } | null | undefined;
                        is_bookmarked?: boolean | undefined;
                        is_trending_in_clips?: boolean | undefined;
                        music_creation_restriction_reason?: string | null | undefined;
                        overlap_duration_in_ms?: number | undefined;
                        placeholder_profile_pic_url?: string | undefined;
                        previous_trend_rank?: number | null | undefined;
                        should_allow_music_editing?: boolean | undefined;
                        should_mute_audio?: boolean | undefined;
                        should_mute_audio_reason?: string | undefined;
                        should_mute_audio_reason_type?: string | null | undefined;
                        trend_rank?: number | null | undefined;
                        user_notes?: unknown;
                    } | undefined;
                } | null | undefined;
                original_sound_info?: unknown;
                pinned_media_ids?: unknown[] | null | undefined;
            }, {
                audio_canonical_id?: string | undefined;
                audio_type?: string | null | undefined;
                music_info?: {
                    music_asset_info?: {
                        title?: string | undefined;
                        id?: string | undefined;
                        allows_saving?: boolean | undefined;
                        artist_id?: string | null | undefined;
                        audio_id?: string | undefined;
                        cover_artwork_thumbnail_uri?: string | undefined;
                        cover_artwork_uri?: string | undefined;
                        dark_message?: string | null | undefined;
                        display_artist?: string | undefined;
                        duration_in_ms?: number | undefined;
                        fast_start_progressive_download_url?: string | undefined;
                        has_lyrics?: boolean | undefined;
                        highlight_start_times_in_ms?: number[] | undefined;
                        ig_username?: string | null | undefined;
                        is_eligible_for_audio_effects?: boolean | undefined;
                        is_eligible_for_vinyl_sticker?: boolean | undefined;
                        is_explicit?: boolean | undefined;
                        licensed_music_subtype?: string | undefined;
                        lyrics?: string | null | undefined;
                        progressive_download_url?: string | undefined;
                        reactive_audio_download_url?: string | null | undefined;
                        sanitized_title?: string | null | undefined;
                        song_monetization_info?: unknown;
                        spotify_track_metadata?: unknown;
                        subtitle?: string | undefined;
                        web_30s_preview_download_url?: string | null | undefined;
                    } | undefined;
                    music_consumption_info?: {
                        allow_media_creation_with_music?: boolean | undefined;
                        audio_asset_start_time_in_ms?: number | undefined;
                        audio_filter_infos?: unknown[] | undefined;
                        audio_muting_info?: {
                            allow_audio_editing?: boolean | undefined;
                            mute_audio?: boolean | undefined;
                            mute_reason_str?: string | undefined;
                            show_muted_audio_toast?: boolean | undefined;
                        } | undefined;
                        contains_lyrics?: boolean | null | undefined;
                        derived_content_id?: string | null | undefined;
                        derived_content_start_time_in_composition_in_ms?: number | undefined;
                        display_labels?: unknown;
                        formatted_clips_media_count?: string | null | undefined;
                        ig_artist?: {
                            username?: string | undefined;
                            id?: string | undefined;
                            is_private?: boolean | undefined;
                            full_name?: string | undefined;
                            is_verified?: boolean | undefined;
                            profile_pic_url?: string | undefined;
                            profile_pic_id?: string | undefined;
                        } | null | undefined;
                        is_bookmarked?: boolean | undefined;
                        is_trending_in_clips?: boolean | undefined;
                        music_creation_restriction_reason?: string | null | undefined;
                        overlap_duration_in_ms?: number | undefined;
                        placeholder_profile_pic_url?: string | undefined;
                        previous_trend_rank?: number | null | undefined;
                        should_allow_music_editing?: boolean | undefined;
                        should_mute_audio?: boolean | undefined;
                        should_mute_audio_reason?: string | undefined;
                        should_mute_audio_reason_type?: string | null | undefined;
                        trend_rank?: number | null | undefined;
                        user_notes?: unknown;
                    } | undefined;
                } | null | undefined;
                original_sound_info?: unknown;
                pinned_media_ids?: unknown[] | null | undefined;
            }>>;
        }, "strip", import("zod").ZodTypeAny, {
            type?: string | undefined;
            url?: string | undefined;
            images?: string[] | undefined;
            timestamp?: string | undefined;
            id?: string | undefined;
            location?: {
                name?: string | undefined;
                id?: string | undefined;
            } | null | undefined;
            inputUrl?: string | undefined;
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
            locationName?: string | undefined;
            locationId?: string | undefined;
            ownerFullName?: string | undefined;
            isSponsored?: boolean | undefined;
            firstComment?: string | undefined;
            latestComments?: unknown[] | undefined;
            musicInfo?: {
                audio_canonical_id?: string | undefined;
                audio_type?: string | null | undefined;
                music_info?: {
                    music_asset_info?: {
                        title?: string | undefined;
                        id?: string | undefined;
                        allows_saving?: boolean | undefined;
                        artist_id?: string | null | undefined;
                        audio_id?: string | undefined;
                        cover_artwork_thumbnail_uri?: string | undefined;
                        cover_artwork_uri?: string | undefined;
                        dark_message?: string | null | undefined;
                        display_artist?: string | undefined;
                        duration_in_ms?: number | undefined;
                        fast_start_progressive_download_url?: string | undefined;
                        has_lyrics?: boolean | undefined;
                        highlight_start_times_in_ms?: number[] | undefined;
                        ig_username?: string | null | undefined;
                        is_eligible_for_audio_effects?: boolean | undefined;
                        is_eligible_for_vinyl_sticker?: boolean | undefined;
                        is_explicit?: boolean | undefined;
                        licensed_music_subtype?: string | undefined;
                        lyrics?: string | null | undefined;
                        progressive_download_url?: string | undefined;
                        reactive_audio_download_url?: string | null | undefined;
                        sanitized_title?: string | null | undefined;
                        song_monetization_info?: unknown;
                        spotify_track_metadata?: unknown;
                        subtitle?: string | undefined;
                        web_30s_preview_download_url?: string | null | undefined;
                    } | undefined;
                    music_consumption_info?: {
                        allow_media_creation_with_music?: boolean | undefined;
                        audio_asset_start_time_in_ms?: number | undefined;
                        audio_filter_infos?: unknown[] | undefined;
                        audio_muting_info?: {
                            allow_audio_editing?: boolean | undefined;
                            mute_audio?: boolean | undefined;
                            mute_reason_str?: string | undefined;
                            show_muted_audio_toast?: boolean | undefined;
                        } | undefined;
                        contains_lyrics?: boolean | null | undefined;
                        derived_content_id?: string | null | undefined;
                        derived_content_start_time_in_composition_in_ms?: number | undefined;
                        display_labels?: unknown;
                        formatted_clips_media_count?: string | null | undefined;
                        ig_artist?: {
                            username?: string | undefined;
                            id?: string | undefined;
                            is_private?: boolean | undefined;
                            full_name?: string | undefined;
                            is_verified?: boolean | undefined;
                            profile_pic_url?: string | undefined;
                            profile_pic_id?: string | undefined;
                        } | null | undefined;
                        is_bookmarked?: boolean | undefined;
                        is_trending_in_clips?: boolean | undefined;
                        music_creation_restriction_reason?: string | null | undefined;
                        overlap_duration_in_ms?: number | undefined;
                        placeholder_profile_pic_url?: string | undefined;
                        previous_trend_rank?: number | null | undefined;
                        should_allow_music_editing?: boolean | undefined;
                        should_mute_audio?: boolean | undefined;
                        should_mute_audio_reason?: string | undefined;
                        should_mute_audio_reason_type?: string | null | undefined;
                        trend_rank?: number | null | undefined;
                        user_notes?: unknown;
                    } | undefined;
                } | null | undefined;
                original_sound_info?: unknown;
                pinned_media_ids?: unknown[] | null | undefined;
            } | undefined;
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
            inputUrl?: string | undefined;
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
            locationName?: string | undefined;
            locationId?: string | undefined;
            ownerFullName?: string | undefined;
            isSponsored?: boolean | undefined;
            firstComment?: string | undefined;
            latestComments?: unknown[] | undefined;
            musicInfo?: {
                audio_canonical_id?: string | undefined;
                audio_type?: string | null | undefined;
                music_info?: {
                    music_asset_info?: {
                        title?: string | undefined;
                        id?: string | undefined;
                        allows_saving?: boolean | undefined;
                        artist_id?: string | null | undefined;
                        audio_id?: string | undefined;
                        cover_artwork_thumbnail_uri?: string | undefined;
                        cover_artwork_uri?: string | undefined;
                        dark_message?: string | null | undefined;
                        display_artist?: string | undefined;
                        duration_in_ms?: number | undefined;
                        fast_start_progressive_download_url?: string | undefined;
                        has_lyrics?: boolean | undefined;
                        highlight_start_times_in_ms?: number[] | undefined;
                        ig_username?: string | null | undefined;
                        is_eligible_for_audio_effects?: boolean | undefined;
                        is_eligible_for_vinyl_sticker?: boolean | undefined;
                        is_explicit?: boolean | undefined;
                        licensed_music_subtype?: string | undefined;
                        lyrics?: string | null | undefined;
                        progressive_download_url?: string | undefined;
                        reactive_audio_download_url?: string | null | undefined;
                        sanitized_title?: string | null | undefined;
                        song_monetization_info?: unknown;
                        spotify_track_metadata?: unknown;
                        subtitle?: string | undefined;
                        web_30s_preview_download_url?: string | null | undefined;
                    } | undefined;
                    music_consumption_info?: {
                        allow_media_creation_with_music?: boolean | undefined;
                        audio_asset_start_time_in_ms?: number | undefined;
                        audio_filter_infos?: unknown[] | undefined;
                        audio_muting_info?: {
                            allow_audio_editing?: boolean | undefined;
                            mute_audio?: boolean | undefined;
                            mute_reason_str?: string | undefined;
                            show_muted_audio_toast?: boolean | undefined;
                        } | undefined;
                        contains_lyrics?: boolean | null | undefined;
                        derived_content_id?: string | null | undefined;
                        derived_content_start_time_in_composition_in_ms?: number | undefined;
                        display_labels?: unknown;
                        formatted_clips_media_count?: string | null | undefined;
                        ig_artist?: {
                            username?: string | undefined;
                            id?: string | undefined;
                            is_private?: boolean | undefined;
                            full_name?: string | undefined;
                            is_verified?: boolean | undefined;
                            profile_pic_url?: string | undefined;
                            profile_pic_id?: string | undefined;
                        } | null | undefined;
                        is_bookmarked?: boolean | undefined;
                        is_trending_in_clips?: boolean | undefined;
                        music_creation_restriction_reason?: string | null | undefined;
                        overlap_duration_in_ms?: number | undefined;
                        placeholder_profile_pic_url?: string | undefined;
                        previous_trend_rank?: number | null | undefined;
                        should_allow_music_editing?: boolean | undefined;
                        should_mute_audio?: boolean | undefined;
                        should_mute_audio_reason?: string | undefined;
                        should_mute_audio_reason_type?: string | null | undefined;
                        trend_rank?: number | null | undefined;
                        user_notes?: unknown;
                    } | undefined;
                } | null | undefined;
                original_sound_info?: unknown;
                pinned_media_ids?: unknown[] | null | undefined;
            } | undefined;
        }>;
        description: string;
        documentation: string;
        category: string;
    };
    'apimaestro/linkedin-profile-posts': {
        input: import("zod").ZodObject<{
            username: import("zod").ZodString;
            page_number: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodNumber>>;
            limit: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodNumber>>;
        }, "strip", import("zod").ZodTypeAny, {
            username: string;
            limit?: number | undefined;
            page_number?: number | undefined;
        }, {
            username: string;
            limit?: number | undefined;
            page_number?: number | undefined;
        }>;
        output: import("zod").ZodObject<{
            urn: import("zod").ZodOptional<import("zod").ZodObject<{
                activity_urn: import("zod").ZodOptional<import("zod").ZodString>;
                share_urn: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                ugcPost_urn: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
            }, "strip", import("zod").ZodTypeAny, {
                activity_urn?: string | undefined;
                share_urn?: string | null | undefined;
                ugcPost_urn?: string | null | undefined;
            }, {
                activity_urn?: string | undefined;
                share_urn?: string | null | undefined;
                ugcPost_urn?: string | null | undefined;
            }>>;
            full_urn: import("zod").ZodOptional<import("zod").ZodString>;
            posted_at: import("zod").ZodOptional<import("zod").ZodObject<{
                date: import("zod").ZodOptional<import("zod").ZodString>;
                relative: import("zod").ZodOptional<import("zod").ZodString>;
                timestamp: import("zod").ZodOptional<import("zod").ZodNumber>;
            }, "strip", import("zod").ZodTypeAny, {
                date?: string | undefined;
                timestamp?: number | undefined;
                relative?: string | undefined;
            }, {
                date?: string | undefined;
                timestamp?: number | undefined;
                relative?: string | undefined;
            }>>;
            text: import("zod").ZodOptional<import("zod").ZodString>;
            url: import("zod").ZodOptional<import("zod").ZodString>;
            post_type: import("zod").ZodOptional<import("zod").ZodString>;
            author: import("zod").ZodOptional<import("zod").ZodObject<{
                first_name: import("zod").ZodOptional<import("zod").ZodString>;
                last_name: import("zod").ZodOptional<import("zod").ZodString>;
                headline: import("zod").ZodOptional<import("zod").ZodString>;
                username: import("zod").ZodOptional<import("zod").ZodString>;
                profile_url: import("zod").ZodOptional<import("zod").ZodString>;
                profile_picture: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
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
            stats: import("zod").ZodOptional<import("zod").ZodObject<{
                total_reactions: import("zod").ZodOptional<import("zod").ZodNumber>;
                like: import("zod").ZodOptional<import("zod").ZodNumber>;
                support: import("zod").ZodOptional<import("zod").ZodNumber>;
                love: import("zod").ZodOptional<import("zod").ZodNumber>;
                insight: import("zod").ZodOptional<import("zod").ZodNumber>;
                celebrate: import("zod").ZodOptional<import("zod").ZodNumber>;
                funny: import("zod").ZodOptional<import("zod").ZodNumber>;
                comments: import("zod").ZodOptional<import("zod").ZodNumber>;
                reposts: import("zod").ZodOptional<import("zod").ZodNumber>;
            }, "strip", import("zod").ZodTypeAny, {
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
            media: import("zod").ZodOptional<import("zod").ZodObject<{
                type: import("zod").ZodOptional<import("zod").ZodString>;
                url: import("zod").ZodOptional<import("zod").ZodString>;
                thumbnail: import("zod").ZodOptional<import("zod").ZodString>;
                images: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                    url: import("zod").ZodOptional<import("zod").ZodString>;
                    width: import("zod").ZodOptional<import("zod").ZodNumber>;
                    height: import("zod").ZodOptional<import("zod").ZodNumber>;
                }, "strip", import("zod").ZodTypeAny, {
                    url?: string | undefined;
                    width?: number | undefined;
                    height?: number | undefined;
                }, {
                    url?: string | undefined;
                    width?: number | undefined;
                    height?: number | undefined;
                }>, "many">>;
            }, "strip", import("zod").ZodTypeAny, {
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
            article: import("zod").ZodOptional<import("zod").ZodObject<{
                url: import("zod").ZodOptional<import("zod").ZodString>;
                title: import("zod").ZodOptional<import("zod").ZodString>;
                subtitle: import("zod").ZodOptional<import("zod").ZodString>;
                thumbnail: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
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
            document: import("zod").ZodOptional<import("zod").ZodObject<{
                title: import("zod").ZodOptional<import("zod").ZodString>;
                page_count: import("zod").ZodOptional<import("zod").ZodNumber>;
                url: import("zod").ZodOptional<import("zod").ZodString>;
                thumbnail: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
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
            reshared_post: import("zod").ZodOptional<import("zod").ZodType<any, import("zod").ZodTypeDef, any>>;
            pagination_token: import("zod").ZodOptional<import("zod").ZodString>;
        }, "strip", import("zod").ZodTypeAny, {
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
        description: string;
        documentation: string;
        category: string;
    };
    'apimaestro/linkedin-posts-search-scraper-no-cookies': {
        input: import("zod").ZodObject<{
            keyword: import("zod").ZodString;
            sort_type: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodEnum<["relevance", "date_posted"]>>>;
            page_number: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodNumber>>;
            date_filter: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodEnum<["", "past-24h", "past-week", "past-month"]>>>;
            limit: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodNumber>>;
        }, "strip", import("zod").ZodTypeAny, {
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
        output: import("zod").ZodObject<{
            activity_id: import("zod").ZodOptional<import("zod").ZodString>;
            post_url: import("zod").ZodOptional<import("zod").ZodString>;
            text: import("zod").ZodOptional<import("zod").ZodString>;
            full_urn: import("zod").ZodOptional<import("zod").ZodString>;
            author: import("zod").ZodOptional<import("zod").ZodObject<{
                name: import("zod").ZodOptional<import("zod").ZodString>;
                headline: import("zod").ZodOptional<import("zod").ZodString>;
                profile_id: import("zod").ZodOptional<import("zod").ZodString>;
                profile_url: import("zod").ZodOptional<import("zod").ZodString>;
                image_url: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
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
            stats: import("zod").ZodOptional<import("zod").ZodObject<{
                total_reactions: import("zod").ZodOptional<import("zod").ZodNumber>;
                comments: import("zod").ZodOptional<import("zod").ZodNumber>;
                shares: import("zod").ZodOptional<import("zod").ZodNumber>;
                reactions: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                    type: import("zod").ZodOptional<import("zod").ZodString>;
                    count: import("zod").ZodOptional<import("zod").ZodNumber>;
                }, "strip", import("zod").ZodTypeAny, {
                    type?: string | undefined;
                    count?: number | undefined;
                }, {
                    type?: string | undefined;
                    count?: number | undefined;
                }>, "many">>;
            }, "strip", import("zod").ZodTypeAny, {
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
            posted_at: import("zod").ZodOptional<import("zod").ZodObject<{
                display_text: import("zod").ZodOptional<import("zod").ZodString>;
                date: import("zod").ZodOptional<import("zod").ZodString>;
                timestamp: import("zod").ZodOptional<import("zod").ZodNumber>;
            }, "strip", import("zod").ZodTypeAny, {
                date?: string | undefined;
                timestamp?: number | undefined;
                display_text?: string | undefined;
            }, {
                date?: string | undefined;
                timestamp?: number | undefined;
                display_text?: string | undefined;
            }>>;
            hashtags: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            content: import("zod").ZodOptional<import("zod").ZodObject<{
                type: import("zod").ZodOptional<import("zod").ZodString>;
                article: import("zod").ZodOptional<import("zod").ZodObject<{
                    url: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                    title: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                    subtitle: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                    thumbnail: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
                }, "strip", import("zod").ZodTypeAny, {
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
                url: import("zod").ZodOptional<import("zod").ZodString>;
                thumbnail_url: import("zod").ZodOptional<import("zod").ZodString>;
                duration_ms: import("zod").ZodOptional<import("zod").ZodNumber>;
                text: import("zod").ZodOptional<import("zod").ZodString>;
                images: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                    url: import("zod").ZodOptional<import("zod").ZodString>;
                    width: import("zod").ZodOptional<import("zod").ZodNumber>;
                    height: import("zod").ZodOptional<import("zod").ZodNumber>;
                }, "strip", import("zod").ZodTypeAny, {
                    url?: string | undefined;
                    width?: number | undefined;
                    height?: number | undefined;
                }, {
                    url?: string | undefined;
                    width?: number | undefined;
                    height?: number | undefined;
                }>, "many">>;
            }, "strip", import("zod").ZodTypeAny, {
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
            is_reshare: import("zod").ZodOptional<import("zod").ZodBoolean>;
            metadata: import("zod").ZodOptional<import("zod").ZodObject<{
                total_count: import("zod").ZodOptional<import("zod").ZodNumber>;
                count: import("zod").ZodOptional<import("zod").ZodNumber>;
                page: import("zod").ZodOptional<import("zod").ZodNumber>;
                page_size: import("zod").ZodOptional<import("zod").ZodNumber>;
                total_pages: import("zod").ZodOptional<import("zod").ZodNumber>;
                has_next_page: import("zod").ZodOptional<import("zod").ZodBoolean>;
                has_prev_page: import("zod").ZodOptional<import("zod").ZodBoolean>;
            }, "strip", import("zod").ZodTypeAny, {
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
            search_input: import("zod").ZodOptional<import("zod").ZodString>;
        }, "strip", import("zod").ZodTypeAny, {
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
        description: string;
        documentation: string;
        category: string;
    };
    'streamers/youtube-scraper': {
        input: import("zod").ZodObject<{
            searchQueries: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            startUrls: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                url: import("zod").ZodString;
            }, "strip", import("zod").ZodTypeAny, {
                url: string;
            }, {
                url: string;
            }>, "many">>>;
            maxResults: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodNumber>>;
            maxResultsShorts: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodNumber>>;
            maxResultStreams: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodNumber>>;
            downloadSubtitles: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodBoolean>>;
            saveSubsToKVS: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodBoolean>>;
            subtitlesLanguage: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["any", "en", "de", "es", "fr", "it", "ja", "ko", "nl", "pt", "ru"]>>>;
            preferAutoGeneratedSubtitles: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodBoolean>>;
            subtitlesFormat: import("zod").ZodDefault<import("zod").ZodOptional<import("zod").ZodEnum<["srt", "vtt", "xml", "plaintext"]>>>;
            sortingOrder: import("zod").ZodOptional<import("zod").ZodEnum<["relevance", "rating", "date", "views"]>>;
            dateFilter: import("zod").ZodOptional<import("zod").ZodEnum<["hour", "today", "week", "month", "year"]>>;
            videoType: import("zod").ZodOptional<import("zod").ZodEnum<["video", "movie"]>>;
            lengthFilter: import("zod").ZodOptional<import("zod").ZodEnum<["under4", "between420", "plus20"]>>;
            isHD: import("zod").ZodOptional<import("zod").ZodBoolean>;
            hasSubtitles: import("zod").ZodOptional<import("zod").ZodBoolean>;
            hasCC: import("zod").ZodOptional<import("zod").ZodBoolean>;
            is3D: import("zod").ZodOptional<import("zod").ZodBoolean>;
            isLive: import("zod").ZodOptional<import("zod").ZodBoolean>;
            isBought: import("zod").ZodOptional<import("zod").ZodBoolean>;
            is4K: import("zod").ZodOptional<import("zod").ZodBoolean>;
            is360: import("zod").ZodOptional<import("zod").ZodBoolean>;
            hasLocation: import("zod").ZodOptional<import("zod").ZodBoolean>;
            isHDR: import("zod").ZodOptional<import("zod").ZodBoolean>;
            isVR180: import("zod").ZodOptional<import("zod").ZodBoolean>;
            oldestPostDate: import("zod").ZodOptional<import("zod").ZodString>;
            sortVideosBy: import("zod").ZodOptional<import("zod").ZodEnum<["NEWEST", "POPULAR", "OLDEST"]>>;
        }, "strip", import("zod").ZodTypeAny, {
            startUrls: {
                url: string;
            }[];
            maxResults: number;
            maxResultsShorts: number;
            maxResultStreams: number;
            downloadSubtitles: boolean;
            saveSubsToKVS: boolean;
            subtitlesLanguage: "any" | "en" | "de" | "es" | "fr" | "it" | "ja" | "ko" | "nl" | "pt" | "ru";
            preferAutoGeneratedSubtitles: boolean;
            subtitlesFormat: "xml" | "srt" | "vtt" | "plaintext";
            searchQueries?: string[] | undefined;
            sortingOrder?: "date" | "relevance" | "rating" | "views" | undefined;
            dateFilter?: "hour" | "today" | "week" | "month" | "year" | undefined;
            videoType?: "video" | "movie" | undefined;
            lengthFilter?: "under4" | "between420" | "plus20" | undefined;
            isHD?: boolean | undefined;
            hasSubtitles?: boolean | undefined;
            hasCC?: boolean | undefined;
            is3D?: boolean | undefined;
            isLive?: boolean | undefined;
            isBought?: boolean | undefined;
            is4K?: boolean | undefined;
            is360?: boolean | undefined;
            hasLocation?: boolean | undefined;
            isHDR?: boolean | undefined;
            isVR180?: boolean | undefined;
            oldestPostDate?: string | undefined;
            sortVideosBy?: "NEWEST" | "POPULAR" | "OLDEST" | undefined;
        }, {
            searchQueries?: string[] | undefined;
            startUrls?: {
                url: string;
            }[] | undefined;
            maxResults?: number | undefined;
            maxResultsShorts?: number | undefined;
            maxResultStreams?: number | undefined;
            downloadSubtitles?: boolean | undefined;
            saveSubsToKVS?: boolean | undefined;
            subtitlesLanguage?: "any" | "en" | "de" | "es" | "fr" | "it" | "ja" | "ko" | "nl" | "pt" | "ru" | undefined;
            preferAutoGeneratedSubtitles?: boolean | undefined;
            subtitlesFormat?: "xml" | "srt" | "vtt" | "plaintext" | undefined;
            sortingOrder?: "date" | "relevance" | "rating" | "views" | undefined;
            dateFilter?: "hour" | "today" | "week" | "month" | "year" | undefined;
            videoType?: "video" | "movie" | undefined;
            lengthFilter?: "under4" | "between420" | "plus20" | undefined;
            isHD?: boolean | undefined;
            hasSubtitles?: boolean | undefined;
            hasCC?: boolean | undefined;
            is3D?: boolean | undefined;
            isLive?: boolean | undefined;
            isBought?: boolean | undefined;
            is4K?: boolean | undefined;
            is360?: boolean | undefined;
            hasLocation?: boolean | undefined;
            isHDR?: boolean | undefined;
            isVR180?: boolean | undefined;
            oldestPostDate?: string | undefined;
            sortVideosBy?: "NEWEST" | "POPULAR" | "OLDEST" | undefined;
        }>;
        output: import("zod").ZodObject<{
            title: import("zod").ZodOptional<import("zod").ZodString>;
            id: import("zod").ZodOptional<import("zod").ZodString>;
            url: import("zod").ZodOptional<import("zod").ZodString>;
            viewCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            date: import("zod").ZodOptional<import("zod").ZodString>;
            likes: import("zod").ZodOptional<import("zod").ZodNumber>;
            channelName: import("zod").ZodOptional<import("zod").ZodString>;
            channelUrl: import("zod").ZodOptional<import("zod").ZodString>;
            numberOfSubscribers: import("zod").ZodOptional<import("zod").ZodNumber>;
            duration: import("zod").ZodOptional<import("zod").ZodString>;
            description: import("zod").ZodOptional<import("zod").ZodString>;
            text: import("zod").ZodOptional<import("zod").ZodString>;
            comments: import("zod").ZodOptional<import("zod").ZodNumber>;
            commentsCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            thumbnail: import("zod").ZodOptional<import("zod").ZodString>;
            thumbnailUrl: import("zod").ZodOptional<import("zod").ZodString>;
            videoType: import("zod").ZodOptional<import("zod").ZodString>;
            tags: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            category: import("zod").ZodOptional<import("zod").ZodString>;
            isLive: import("zod").ZodOptional<import("zod").ZodBoolean>;
            subtitles: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                language: import("zod").ZodOptional<import("zod").ZodString>;
                url: import("zod").ZodOptional<import("zod").ZodString>;
                text: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
                url?: string | undefined;
                text?: string | undefined;
                language?: string | undefined;
            }, {
                url?: string | undefined;
                text?: string | undefined;
                language?: string | undefined;
            }>, "many">>;
        }, "strip", import("zod").ZodTypeAny, {
            description?: string | undefined;
            title?: string | undefined;
            date?: string | undefined;
            url?: string | undefined;
            duration?: string | undefined;
            text?: string | undefined;
            id?: string | undefined;
            tags?: string[] | undefined;
            commentsCount?: number | undefined;
            comments?: number | undefined;
            thumbnail?: string | undefined;
            videoType?: string | undefined;
            isLive?: boolean | undefined;
            viewCount?: number | undefined;
            likes?: number | undefined;
            channelName?: string | undefined;
            channelUrl?: string | undefined;
            numberOfSubscribers?: number | undefined;
            thumbnailUrl?: string | undefined;
            category?: string | undefined;
            subtitles?: {
                url?: string | undefined;
                text?: string | undefined;
                language?: string | undefined;
            }[] | undefined;
        }, {
            description?: string | undefined;
            title?: string | undefined;
            date?: string | undefined;
            url?: string | undefined;
            duration?: string | undefined;
            text?: string | undefined;
            id?: string | undefined;
            tags?: string[] | undefined;
            commentsCount?: number | undefined;
            comments?: number | undefined;
            thumbnail?: string | undefined;
            videoType?: string | undefined;
            isLive?: boolean | undefined;
            viewCount?: number | undefined;
            likes?: number | undefined;
            channelName?: string | undefined;
            channelUrl?: string | undefined;
            numberOfSubscribers?: number | undefined;
            thumbnailUrl?: string | undefined;
            category?: string | undefined;
            subtitles?: {
                url?: string | undefined;
                text?: string | undefined;
                language?: string | undefined;
            }[] | undefined;
        }>;
        description: string;
        documentation: string;
        category: string;
    };
    'pintostudio/youtube-transcript-scraper': {
        input: import("zod").ZodObject<{
            videoUrl: import("zod").ZodString;
        }, "strip", import("zod").ZodTypeAny, {
            videoUrl: string;
        }, {
            videoUrl: string;
        }>;
        output: import("zod").ZodObject<{
            videoUrl: import("zod").ZodOptional<import("zod").ZodString>;
            data: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                start: import("zod").ZodOptional<import("zod").ZodString>;
                dur: import("zod").ZodOptional<import("zod").ZodString>;
                text: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
                text?: string | undefined;
                start?: string | undefined;
                dur?: string | undefined;
            }, {
                text?: string | undefined;
                start?: string | undefined;
                dur?: string | undefined;
            }>, "many">>;
        }, "strip", import("zod").ZodTypeAny, {
            data?: {
                text?: string | undefined;
                start?: string | undefined;
                dur?: string | undefined;
            }[] | undefined;
            videoUrl?: string | undefined;
        }, {
            data?: {
                text?: string | undefined;
                start?: string | undefined;
                dur?: string | undefined;
            }[] | undefined;
            videoUrl?: string | undefined;
        }>;
        description: string;
        documentation: string;
        category: string;
    };
    'curious_coder/linkedin-jobs-scraper': {
        input: import("zod").ZodObject<{
            urls: import("zod").ZodArray<import("zod").ZodString, "many">;
            scrapeCompany: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
            count: import("zod").ZodOptional<import("zod").ZodNumber>;
        }, "strip", import("zod").ZodTypeAny, {
            urls: string[];
            count?: number | undefined;
            scrapeCompany?: boolean | undefined;
        }, {
            urls: string[];
            count?: number | undefined;
            scrapeCompany?: boolean | undefined;
        }>;
        output: import("zod").ZodObject<{
            id: import("zod").ZodOptional<import("zod").ZodString>;
            trackingId: import("zod").ZodOptional<import("zod").ZodString>;
            refId: import("zod").ZodOptional<import("zod").ZodString>;
            link: import("zod").ZodOptional<import("zod").ZodString>;
            title: import("zod").ZodOptional<import("zod").ZodString>;
            companyName: import("zod").ZodOptional<import("zod").ZodString>;
            companyLinkedinUrl: import("zod").ZodOptional<import("zod").ZodString>;
            companyLogo: import("zod").ZodOptional<import("zod").ZodString>;
            location: import("zod").ZodOptional<import("zod").ZodString>;
            salaryInfo: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            postedAt: import("zod").ZodOptional<import("zod").ZodString>;
            benefits: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            descriptionHtml: import("zod").ZodOptional<import("zod").ZodString>;
            applicantsCount: import("zod").ZodOptional<import("zod").ZodString>;
            applyUrl: import("zod").ZodOptional<import("zod").ZodString>;
            salary: import("zod").ZodOptional<import("zod").ZodString>;
            descriptionText: import("zod").ZodOptional<import("zod").ZodString>;
            seniorityLevel: import("zod").ZodOptional<import("zod").ZodString>;
            employmentType: import("zod").ZodOptional<import("zod").ZodString>;
            jobFunction: import("zod").ZodOptional<import("zod").ZodString>;
            industries: import("zod").ZodOptional<import("zod").ZodString>;
            inputUrl: import("zod").ZodOptional<import("zod").ZodString>;
            companyAddress: import("zod").ZodOptional<import("zod").ZodObject<{
                type: import("zod").ZodOptional<import("zod").ZodString>;
                streetAddress: import("zod").ZodOptional<import("zod").ZodString>;
                addressLocality: import("zod").ZodOptional<import("zod").ZodString>;
                addressRegion: import("zod").ZodOptional<import("zod").ZodString>;
                postalCode: import("zod").ZodOptional<import("zod").ZodString>;
                addressCountry: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
                type?: string | undefined;
                streetAddress?: string | undefined;
                addressLocality?: string | undefined;
                addressRegion?: string | undefined;
                postalCode?: string | undefined;
                addressCountry?: string | undefined;
            }, {
                type?: string | undefined;
                streetAddress?: string | undefined;
                addressLocality?: string | undefined;
                addressRegion?: string | undefined;
                postalCode?: string | undefined;
                addressCountry?: string | undefined;
            }>>;
            companyWebsite: import("zod").ZodOptional<import("zod").ZodString>;
            companySlogan: import("zod").ZodOptional<import("zod").ZodString>;
            companyDescription: import("zod").ZodOptional<import("zod").ZodString>;
            companyEmployeesCount: import("zod").ZodOptional<import("zod").ZodNumber>;
        }, "strip", import("zod").ZodTypeAny, {
            title?: string | undefined;
            link?: string | undefined;
            id?: string | undefined;
            location?: string | undefined;
            inputUrl?: string | undefined;
            trackingId?: string | undefined;
            refId?: string | undefined;
            companyName?: string | undefined;
            companyLinkedinUrl?: string | undefined;
            companyLogo?: string | undefined;
            salaryInfo?: string[] | undefined;
            postedAt?: string | undefined;
            benefits?: string[] | undefined;
            descriptionHtml?: string | undefined;
            applicantsCount?: string | undefined;
            applyUrl?: string | undefined;
            salary?: string | undefined;
            descriptionText?: string | undefined;
            seniorityLevel?: string | undefined;
            employmentType?: string | undefined;
            jobFunction?: string | undefined;
            industries?: string | undefined;
            companyAddress?: {
                type?: string | undefined;
                streetAddress?: string | undefined;
                addressLocality?: string | undefined;
                addressRegion?: string | undefined;
                postalCode?: string | undefined;
                addressCountry?: string | undefined;
            } | undefined;
            companyWebsite?: string | undefined;
            companySlogan?: string | undefined;
            companyDescription?: string | undefined;
            companyEmployeesCount?: number | undefined;
        }, {
            title?: string | undefined;
            link?: string | undefined;
            id?: string | undefined;
            location?: string | undefined;
            inputUrl?: string | undefined;
            trackingId?: string | undefined;
            refId?: string | undefined;
            companyName?: string | undefined;
            companyLinkedinUrl?: string | undefined;
            companyLogo?: string | undefined;
            salaryInfo?: string[] | undefined;
            postedAt?: string | undefined;
            benefits?: string[] | undefined;
            descriptionHtml?: string | undefined;
            applicantsCount?: string | undefined;
            applyUrl?: string | undefined;
            salary?: string | undefined;
            descriptionText?: string | undefined;
            seniorityLevel?: string | undefined;
            employmentType?: string | undefined;
            jobFunction?: string | undefined;
            industries?: string | undefined;
            companyAddress?: {
                type?: string | undefined;
                streetAddress?: string | undefined;
                addressLocality?: string | undefined;
                addressRegion?: string | undefined;
                postalCode?: string | undefined;
                addressCountry?: string | undefined;
            } | undefined;
            companyWebsite?: string | undefined;
            companySlogan?: string | undefined;
            companyDescription?: string | undefined;
            companyEmployeesCount?: number | undefined;
        }>;
        description: string;
        documentation: string;
        category: string;
    };
    'clockworks/tiktok-scraper': {
        input: import("zod").ZodObject<{
            hashtags: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            resultsPerPage: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodNumber>>;
            profiles: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            profileScrapeSections: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodArray<import("zod").ZodEnum<["videos", "reposts"]>, "many">>>;
            profileSorting: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodEnum<["latest", "popular", "oldest"]>>>;
            excludePinnedPosts: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
            oldestPostDateUnified: import("zod").ZodOptional<import("zod").ZodString>;
            newestPostDate: import("zod").ZodOptional<import("zod").ZodString>;
            mostDiggs: import("zod").ZodOptional<import("zod").ZodNumber>;
            leastDiggs: import("zod").ZodOptional<import("zod").ZodNumber>;
            maxFollowersPerProfile: import("zod").ZodOptional<import("zod").ZodNumber>;
            maxFollowingPerProfile: import("zod").ZodOptional<import("zod").ZodNumber>;
            searchQueries: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            searchSection: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodEnum<["", "/video", "/user"]>>>;
            maxProfilesPerQuery: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodNumber>>;
            searchSorting: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodEnum<["0", "1", "3"]>>>;
            searchDatePosted: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodEnum<["0", "1", "2", "3", "4", "5"]>>>;
            postURLs: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            scrapeRelatedVideos: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
            shouldDownloadVideos: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
            shouldDownloadCovers: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
            shouldDownloadSubtitles: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
            shouldDownloadSlideshowImages: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
            shouldDownloadAvatars: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
            shouldDownloadMusicCovers: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
            videoKvStoreIdOrName: import("zod").ZodOptional<import("zod").ZodString>;
            commentsPerPost: import("zod").ZodOptional<import("zod").ZodNumber>;
            maxRepliesPerComment: import("zod").ZodOptional<import("zod").ZodNumber>;
            proxyCountryCode: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodEnum<["None", "AF", "AL", "DZ", "AS", "AD", "AO", "AI", "AG", "AR", "AM", "AU", "AT", "AZ", "BS", "BH", "BD", "BB", "BY", "BE", "BZ", "BJ", "BM", "BT", "BO", "BA", "BW", "BR", "VG", "BN", "BG", "BF", "BI", "KH", "CM", "CA", "CV", "KY", "TD", "CL", "CO", "CK", "CR", "HR", "CY", "CZ", "CD", "DK", "DJ", "DO", "EC", "EG", "SV", "EE", "ET", "FK", "FJ", "FI", "FR", "PF", "GA", "GE", "DE", "GH", "GI", "GR", "GL", "GD", "GP", "GT", "GN", "GW", "GY", "HN", "HK", "HU", "IS", "IN", "ID", "IQ", "IE", "IM", "IL", "IT", "CI", "JM", "JP", "JE", "KZ", "KE", "XK", "KW", "LA", "LV", "LB", "LS", "LR", "LY", "LT", "LU", "MO", "MG", "MW", "MY", "MV", "ML", "MT", "MH", "MQ", "MR", "MU", "MX", "MD", "MC", "MN", "ME", "MA", "MZ", "MM", "NA", "NR", "NP", "NL", "NZ", "NI", "NG", "MK", "NO", "OM", "PK", "PS", "PA", "PG", "PY", "PE", "PH", "PL", "PT", "PR", "QA", "CG", "RO", "RU", "RW", "RE", "KN", "LC", "MF", "PM", "VC", "SM", "SA", "SN", "RS", "SL", "SG", "SX", "SK", "SB", "SO", "ZA", "KR", "ES", "LK", "SR", "SZ", "SE", "CH", "TW", "TJ", "TZ", "TH", "TG", "TO", "TT", "TN", "TR", "TM", "TC", "TV", "VI", "UG", "UA", "AE", "GB", "US", "UY", "VE", "VN", "WF", "YE", "ZM", "ZW", "AX"]>>>;
        }, "strip", import("zod").ZodTypeAny, {
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
        output: import("zod").ZodObject<{
            authorMeta: import("zod").ZodOptional<import("zod").ZodObject<{
                avatar: import("zod").ZodOptional<import("zod").ZodString>;
                bioLink: import("zod").ZodOptional<import("zod").ZodNull>;
                digg: import("zod").ZodOptional<import("zod").ZodNumber>;
                fans: import("zod").ZodOptional<import("zod").ZodNumber>;
                followDatasetUrl: import("zod").ZodOptional<import("zod").ZodNull>;
                following: import("zod").ZodOptional<import("zod").ZodNumber>;
                friends: import("zod").ZodOptional<import("zod").ZodNumber>;
                heart: import("zod").ZodOptional<import("zod").ZodNumber>;
                id: import("zod").ZodOptional<import("zod").ZodString>;
                name: import("zod").ZodOptional<import("zod").ZodString>;
                nickName: import("zod").ZodOptional<import("zod").ZodString>;
                originalAvatarUrl: import("zod").ZodOptional<import("zod").ZodString>;
                privateAccount: import("zod").ZodOptional<import("zod").ZodBoolean>;
                profileUrl: import("zod").ZodOptional<import("zod").ZodString>;
                signature: import("zod").ZodOptional<import("zod").ZodString>;
                verified: import("zod").ZodOptional<import("zod").ZodBoolean>;
                video: import("zod").ZodOptional<import("zod").ZodNumber>;
            }, "strip", import("zod").ZodTypeAny, {
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
            collectCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            commentCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            commentsDatasetUrl: import("zod").ZodOptional<import("zod").ZodNull>;
            createTime: import("zod").ZodOptional<import("zod").ZodNumber>;
            createTimeISO: import("zod").ZodOptional<import("zod").ZodString>;
            detailedMentions: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                id: import("zod").ZodOptional<import("zod").ZodString>;
                name: import("zod").ZodOptional<import("zod").ZodString>;
                nickName: import("zod").ZodOptional<import("zod").ZodString>;
                profileUrl: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
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
            diggCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            effectStickers: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                ID: import("zod").ZodOptional<import("zod").ZodString>;
                name: import("zod").ZodOptional<import("zod").ZodString>;
                stickerStats: import("zod").ZodOptional<import("zod").ZodObject<{
                    useCount: import("zod").ZodOptional<import("zod").ZodNumber>;
                }, "strip", import("zod").ZodTypeAny, {
                    useCount?: number | undefined;
                }, {
                    useCount?: number | undefined;
                }>>;
            }, "strip", import("zod").ZodTypeAny, {
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
            hashtags: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                name: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
                name?: string | undefined;
            }, {
                name?: string | undefined;
            }>, "many">>;
            id: import("zod").ZodOptional<import("zod").ZodString>;
            input: import("zod").ZodOptional<import("zod").ZodString>;
            isAd: import("zod").ZodOptional<import("zod").ZodBoolean>;
            isPinned: import("zod").ZodOptional<import("zod").ZodBoolean>;
            isSlideshow: import("zod").ZodOptional<import("zod").ZodBoolean>;
            isSponsored: import("zod").ZodOptional<import("zod").ZodBoolean>;
            mediaUrls: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            mentions: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            musicMeta: import("zod").ZodOptional<import("zod").ZodObject<{
                coverMediumUrl: import("zod").ZodOptional<import("zod").ZodString>;
                musicAuthor: import("zod").ZodOptional<import("zod").ZodString>;
                musicId: import("zod").ZodOptional<import("zod").ZodString>;
                musicName: import("zod").ZodOptional<import("zod").ZodString>;
                musicOriginal: import("zod").ZodOptional<import("zod").ZodBoolean>;
                originalCoverMediumUrl: import("zod").ZodOptional<import("zod").ZodString>;
                playUrl: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
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
            playCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            repostCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            searchHashtag: import("zod").ZodOptional<import("zod").ZodObject<{
                name: import("zod").ZodOptional<import("zod").ZodString>;
                views: import("zod").ZodOptional<import("zod").ZodNumber>;
            }, "strip", import("zod").ZodTypeAny, {
                name?: string | undefined;
                views?: number | undefined;
            }, {
                name?: string | undefined;
                views?: number | undefined;
            }>>;
            shareCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            text: import("zod").ZodOptional<import("zod").ZodString>;
            textLanguage: import("zod").ZodOptional<import("zod").ZodString>;
            videoMeta: import("zod").ZodOptional<import("zod").ZodObject<{
                coverUrl: import("zod").ZodOptional<import("zod").ZodString>;
                definition: import("zod").ZodOptional<import("zod").ZodString>;
                duration: import("zod").ZodOptional<import("zod").ZodNumber>;
                format: import("zod").ZodOptional<import("zod").ZodString>;
                height: import("zod").ZodOptional<import("zod").ZodNumber>;
                originalCoverUrl: import("zod").ZodOptional<import("zod").ZodString>;
                subtitleLinks: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                    language: import("zod").ZodOptional<import("zod").ZodString>;
                    downloadLink: import("zod").ZodOptional<import("zod").ZodString>;
                    tiktokLink: import("zod").ZodOptional<import("zod").ZodString>;
                    source: import("zod").ZodOptional<import("zod").ZodString>;
                    sourceUnabbreviated: import("zod").ZodOptional<import("zod").ZodString>;
                    version: import("zod").ZodOptional<import("zod").ZodString>;
                }, "strip", import("zod").ZodTypeAny, {
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
                width: import("zod").ZodOptional<import("zod").ZodNumber>;
            }, "strip", import("zod").ZodTypeAny, {
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
            webVideoUrl: import("zod").ZodOptional<import("zod").ZodString>;
        }, "strip", import("zod").ZodTypeAny, {
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
        description: string;
        documentation: string;
        category: string;
    };
    'apidojo/tweet-scraper': {
        input: import("zod").ZodObject<{
            startUrls: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            searchTerms: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            twitterHandles: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            conversationIds: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            maxItems: import("zod").ZodOptional<import("zod").ZodNumber>;
            sort: import("zod").ZodOptional<import("zod").ZodEnum<["Top", "Latest"]>>;
            tweetLanguage: import("zod").ZodOptional<import("zod").ZodEnum<["ab", "aa", "af", "ak", "sq", "am", "ar", "an", "hy", "as", "av", "ae", "ay", "az", "bm", "ba", "eu", "be", "bn", "bi", "bs", "br", "bg", "my", "ca", "ch", "ce", "ny", "zh", "cu", "cv", "kw", "co", "cr", "hr", "cs", "da", "dv", "nl", "dz", "en", "eo", "et", "ee", "fo", "fj", "fi", "fr", "fy", "ff", "gd", "gl", "lg", "ka", "de", "el", "kl", "gn", "gu", "ht", "ha", "he", "hz", "hi", "ho", "hu", "is", "io", "ig", "id", "ia", "ie", "iu", "ik", "ga", "it", "ja", "jv", "kn", "kr", "ks", "kk", "km", "ki", "rw", "ky", "kv", "kg", "ko", "kj", "ku", "lo", "la", "lv", "li", "ln", "lt", "lu", "lb", "mk", "mg", "ms", "ml", "mt", "gv", "mi", "mr", "mh", "mn", "na", "nv", "nd", "nr", "ng", "ne", "no", "nb", "nn", "ii", "oc", "oj", "or", "om", "os", "pi", "ps", "fa", "pl", "pt", "pa", "qu", "ro", "rm", "rn", "ru", "se", "sm", "sg", "sa", "sc", "sr", "sn", "sd", "si", "sk", "sl", "so", "st", "es", "su", "sw", "ss", "sv", "tl", "ty", "tg", "ta", "tt", "te", "th", "bo", "ti", "to", "ts", "tn", "tr", "tk", "tw", "ug", "uk", "ur", "uz", "ve", "vi", "vo", "wa", "cy", "wo", "xh", "yi", "yo", "za", "zu"]>>;
        }, "strip", import("zod").ZodTypeAny, {
            sort?: "Top" | "Latest" | undefined;
            maxItems?: number | undefined;
            startUrls?: string[] | undefined;
            searchTerms?: string[] | undefined;
            twitterHandles?: string[] | undefined;
            conversationIds?: string[] | undefined;
            tweetLanguage?: "ts" | "id" | "to" | "en" | "de" | "es" | "fr" | "it" | "ja" | "ko" | "nl" | "pt" | "ru" | "ab" | "aa" | "af" | "ak" | "sq" | "am" | "ar" | "an" | "hy" | "as" | "av" | "ae" | "ay" | "az" | "bm" | "ba" | "eu" | "be" | "bn" | "bi" | "bs" | "br" | "bg" | "my" | "ca" | "ch" | "ce" | "ny" | "zh" | "cu" | "cv" | "kw" | "co" | "cr" | "hr" | "cs" | "da" | "dv" | "dz" | "eo" | "et" | "ee" | "fo" | "fj" | "fi" | "fy" | "ff" | "gd" | "gl" | "lg" | "ka" | "el" | "kl" | "gn" | "gu" | "ht" | "ha" | "he" | "hz" | "hi" | "ho" | "hu" | "is" | "io" | "ig" | "ia" | "ie" | "iu" | "ik" | "ga" | "jv" | "kn" | "kr" | "ks" | "kk" | "km" | "ki" | "rw" | "ky" | "kv" | "kg" | "kj" | "ku" | "lo" | "la" | "lv" | "li" | "ln" | "lt" | "lu" | "lb" | "mk" | "mg" | "ms" | "ml" | "mt" | "gv" | "mi" | "mr" | "mh" | "mn" | "na" | "nv" | "nd" | "nr" | "ng" | "ne" | "no" | "nb" | "nn" | "ii" | "oc" | "oj" | "or" | "om" | "os" | "pi" | "ps" | "fa" | "pl" | "pa" | "qu" | "ro" | "rm" | "rn" | "se" | "sm" | "sg" | "sa" | "sc" | "sr" | "sn" | "sd" | "si" | "sk" | "sl" | "so" | "st" | "su" | "sw" | "ss" | "sv" | "tl" | "ty" | "tg" | "ta" | "tt" | "te" | "th" | "bo" | "ti" | "tn" | "tr" | "tk" | "tw" | "ug" | "uk" | "ur" | "uz" | "ve" | "vi" | "vo" | "wa" | "cy" | "wo" | "xh" | "yi" | "yo" | "za" | "zu" | undefined;
        }, {
            sort?: "Top" | "Latest" | undefined;
            maxItems?: number | undefined;
            startUrls?: string[] | undefined;
            searchTerms?: string[] | undefined;
            twitterHandles?: string[] | undefined;
            conversationIds?: string[] | undefined;
            tweetLanguage?: "ts" | "id" | "to" | "en" | "de" | "es" | "fr" | "it" | "ja" | "ko" | "nl" | "pt" | "ru" | "ab" | "aa" | "af" | "ak" | "sq" | "am" | "ar" | "an" | "hy" | "as" | "av" | "ae" | "ay" | "az" | "bm" | "ba" | "eu" | "be" | "bn" | "bi" | "bs" | "br" | "bg" | "my" | "ca" | "ch" | "ce" | "ny" | "zh" | "cu" | "cv" | "kw" | "co" | "cr" | "hr" | "cs" | "da" | "dv" | "dz" | "eo" | "et" | "ee" | "fo" | "fj" | "fi" | "fy" | "ff" | "gd" | "gl" | "lg" | "ka" | "el" | "kl" | "gn" | "gu" | "ht" | "ha" | "he" | "hz" | "hi" | "ho" | "hu" | "is" | "io" | "ig" | "ia" | "ie" | "iu" | "ik" | "ga" | "jv" | "kn" | "kr" | "ks" | "kk" | "km" | "ki" | "rw" | "ky" | "kv" | "kg" | "kj" | "ku" | "lo" | "la" | "lv" | "li" | "ln" | "lt" | "lu" | "lb" | "mk" | "mg" | "ms" | "ml" | "mt" | "gv" | "mi" | "mr" | "mh" | "mn" | "na" | "nv" | "nd" | "nr" | "ng" | "ne" | "no" | "nb" | "nn" | "ii" | "oc" | "oj" | "or" | "om" | "os" | "pi" | "ps" | "fa" | "pl" | "pa" | "qu" | "ro" | "rm" | "rn" | "se" | "sm" | "sg" | "sa" | "sc" | "sr" | "sn" | "sd" | "si" | "sk" | "sl" | "so" | "st" | "su" | "sw" | "ss" | "sv" | "tl" | "ty" | "tg" | "ta" | "tt" | "te" | "th" | "bo" | "ti" | "tn" | "tr" | "tk" | "tw" | "ug" | "uk" | "ur" | "uz" | "ve" | "vi" | "vo" | "wa" | "cy" | "wo" | "xh" | "yi" | "yo" | "za" | "zu" | undefined;
        }>;
        output: import("zod").ZodObject<{
            id: import("zod").ZodOptional<import("zod").ZodString>;
            url: import("zod").ZodOptional<import("zod").ZodString>;
            text: import("zod").ZodOptional<import("zod").ZodString>;
            author: import("zod").ZodOptional<import("zod").ZodObject<{
                id: import("zod").ZodOptional<import("zod").ZodString>;
                name: import("zod").ZodOptional<import("zod").ZodString>;
                userName: import("zod").ZodOptional<import("zod").ZodString>;
                description: import("zod").ZodOptional<import("zod").ZodString>;
                isVerified: import("zod").ZodOptional<import("zod").ZodBoolean>;
                isBlueVerified: import("zod").ZodOptional<import("zod").ZodBoolean>;
                profilePicture: import("zod").ZodOptional<import("zod").ZodString>;
                followers: import("zod").ZodOptional<import("zod").ZodNumber>;
                following: import("zod").ZodOptional<import("zod").ZodNumber>;
                tweetsCount: import("zod").ZodOptional<import("zod").ZodNumber>;
                url: import("zod").ZodOptional<import("zod").ZodString>;
                createdAt: import("zod").ZodOptional<import("zod").ZodString>;
            }, "strip", import("zod").ZodTypeAny, {
                description?: string | undefined;
                name?: string | undefined;
                url?: string | undefined;
                id?: string | undefined;
                following?: number | undefined;
                userName?: string | undefined;
                isVerified?: boolean | undefined;
                isBlueVerified?: boolean | undefined;
                profilePicture?: string | undefined;
                followers?: number | undefined;
                tweetsCount?: number | undefined;
                createdAt?: string | undefined;
            }, {
                description?: string | undefined;
                name?: string | undefined;
                url?: string | undefined;
                id?: string | undefined;
                following?: number | undefined;
                userName?: string | undefined;
                isVerified?: boolean | undefined;
                isBlueVerified?: boolean | undefined;
                profilePicture?: string | undefined;
                followers?: number | undefined;
                tweetsCount?: number | undefined;
                createdAt?: string | undefined;
            }>>;
            createdAt: import("zod").ZodOptional<import("zod").ZodString>;
            retweetCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            replyCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            likeCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            quoteCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            viewCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            bookmarkCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            lang: import("zod").ZodOptional<import("zod").ZodString>;
            media: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodUnion<[import("zod").ZodString, import("zod").ZodObject<{
                type: import("zod").ZodOptional<import("zod").ZodEnum<["photo", "video", "animated_gif"]>>;
                url: import("zod").ZodOptional<import("zod").ZodString>;
                width: import("zod").ZodOptional<import("zod").ZodNumber>;
                height: import("zod").ZodOptional<import("zod").ZodNumber>;
                duration: import("zod").ZodOptional<import("zod").ZodNumber>;
            }, "strip", import("zod").ZodTypeAny, {
                type?: "video" | "photo" | "animated_gif" | undefined;
                url?: string | undefined;
                duration?: number | undefined;
                width?: number | undefined;
                height?: number | undefined;
            }, {
                type?: "video" | "photo" | "animated_gif" | undefined;
                url?: string | undefined;
                duration?: number | undefined;
                width?: number | undefined;
                height?: number | undefined;
            }>]>, "many">>;
            entities: import("zod").ZodOptional<import("zod").ZodObject<{
                hashtags: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                    text: import("zod").ZodOptional<import("zod").ZodString>;
                }, "strip", import("zod").ZodTypeAny, {
                    text?: string | undefined;
                }, {
                    text?: string | undefined;
                }>, "many">>;
                urls: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                    url: import("zod").ZodOptional<import("zod").ZodString>;
                    expandedUrl: import("zod").ZodOptional<import("zod").ZodString>;
                    displayUrl: import("zod").ZodOptional<import("zod").ZodString>;
                }, "strip", import("zod").ZodTypeAny, {
                    url?: string | undefined;
                    displayUrl?: string | undefined;
                    expandedUrl?: string | undefined;
                }, {
                    url?: string | undefined;
                    displayUrl?: string | undefined;
                    expandedUrl?: string | undefined;
                }>, "many">>;
                userMentions: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                    screenName: import("zod").ZodOptional<import("zod").ZodString>;
                    name: import("zod").ZodOptional<import("zod").ZodString>;
                }, "strip", import("zod").ZodTypeAny, {
                    name?: string | undefined;
                    screenName?: string | undefined;
                }, {
                    name?: string | undefined;
                    screenName?: string | undefined;
                }>, "many">>;
            }, "strip", import("zod").ZodTypeAny, {
                hashtags?: {
                    text?: string | undefined;
                }[] | undefined;
                urls?: {
                    url?: string | undefined;
                    displayUrl?: string | undefined;
                    expandedUrl?: string | undefined;
                }[] | undefined;
                userMentions?: {
                    name?: string | undefined;
                    screenName?: string | undefined;
                }[] | undefined;
            }, {
                hashtags?: {
                    text?: string | undefined;
                }[] | undefined;
                urls?: {
                    url?: string | undefined;
                    displayUrl?: string | undefined;
                    expandedUrl?: string | undefined;
                }[] | undefined;
                userMentions?: {
                    name?: string | undefined;
                    screenName?: string | undefined;
                }[] | undefined;
            }>>;
            isRetweet: import("zod").ZodOptional<import("zod").ZodBoolean>;
            isQuote: import("zod").ZodOptional<import("zod").ZodBoolean>;
            isReply: import("zod").ZodOptional<import("zod").ZodBoolean>;
        }, "strip", import("zod").ZodTypeAny, {
            url?: string | undefined;
            text?: string | undefined;
            id?: string | undefined;
            author?: {
                description?: string | undefined;
                name?: string | undefined;
                url?: string | undefined;
                id?: string | undefined;
                following?: number | undefined;
                userName?: string | undefined;
                isVerified?: boolean | undefined;
                isBlueVerified?: boolean | undefined;
                profilePicture?: string | undefined;
                followers?: number | undefined;
                tweetsCount?: number | undefined;
                createdAt?: string | undefined;
            } | undefined;
            media?: (string | {
                type?: "video" | "photo" | "animated_gif" | undefined;
                url?: string | undefined;
                duration?: number | undefined;
                width?: number | undefined;
                height?: number | undefined;
            })[] | undefined;
            viewCount?: number | undefined;
            createdAt?: string | undefined;
            retweetCount?: number | undefined;
            replyCount?: number | undefined;
            likeCount?: number | undefined;
            quoteCount?: number | undefined;
            bookmarkCount?: number | undefined;
            lang?: string | undefined;
            entities?: {
                hashtags?: {
                    text?: string | undefined;
                }[] | undefined;
                urls?: {
                    url?: string | undefined;
                    displayUrl?: string | undefined;
                    expandedUrl?: string | undefined;
                }[] | undefined;
                userMentions?: {
                    name?: string | undefined;
                    screenName?: string | undefined;
                }[] | undefined;
            } | undefined;
            isRetweet?: boolean | undefined;
            isQuote?: boolean | undefined;
            isReply?: boolean | undefined;
        }, {
            url?: string | undefined;
            text?: string | undefined;
            id?: string | undefined;
            author?: {
                description?: string | undefined;
                name?: string | undefined;
                url?: string | undefined;
                id?: string | undefined;
                following?: number | undefined;
                userName?: string | undefined;
                isVerified?: boolean | undefined;
                isBlueVerified?: boolean | undefined;
                profilePicture?: string | undefined;
                followers?: number | undefined;
                tweetsCount?: number | undefined;
                createdAt?: string | undefined;
            } | undefined;
            media?: (string | {
                type?: "video" | "photo" | "animated_gif" | undefined;
                url?: string | undefined;
                duration?: number | undefined;
                width?: number | undefined;
                height?: number | undefined;
            })[] | undefined;
            viewCount?: number | undefined;
            createdAt?: string | undefined;
            retweetCount?: number | undefined;
            replyCount?: number | undefined;
            likeCount?: number | undefined;
            quoteCount?: number | undefined;
            bookmarkCount?: number | undefined;
            lang?: string | undefined;
            entities?: {
                hashtags?: {
                    text?: string | undefined;
                }[] | undefined;
                urls?: {
                    url?: string | undefined;
                    displayUrl?: string | undefined;
                    expandedUrl?: string | undefined;
                }[] | undefined;
                userMentions?: {
                    name?: string | undefined;
                    screenName?: string | undefined;
                }[] | undefined;
            } | undefined;
            isRetweet?: boolean | undefined;
            isQuote?: boolean | undefined;
            isReply?: boolean | undefined;
        }>;
        description: string;
        documentation: string;
        category: string;
    };
    'compass/crawler-google-places': {
        input: import("zod").ZodObject<{
            searchStringsArray: import("zod").ZodArray<import("zod").ZodString, "many">;
            locationQuery: import("zod").ZodOptional<import("zod").ZodString>;
            maxCrawledPlacesPerSearch: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodNumber>>;
            language: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodString>>;
            onlyDataFromSearchPage: import("zod").ZodOptional<import("zod").ZodDefault<import("zod").ZodBoolean>>;
        }, "strip", import("zod").ZodTypeAny, {
            searchStringsArray: string[];
            language?: string | undefined;
            locationQuery?: string | undefined;
            maxCrawledPlacesPerSearch?: number | undefined;
            onlyDataFromSearchPage?: boolean | undefined;
        }, {
            searchStringsArray: string[];
            language?: string | undefined;
            locationQuery?: string | undefined;
            maxCrawledPlacesPerSearch?: number | undefined;
            onlyDataFromSearchPage?: boolean | undefined;
        }>;
        output: import("zod").ZodObject<{
            title: import("zod").ZodOptional<import("zod").ZodString>;
            description: import("zod").ZodOptional<import("zod").ZodString>;
            price: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
            categoryName: import("zod").ZodOptional<import("zod").ZodString>;
            address: import("zod").ZodOptional<import("zod").ZodString>;
            neighborhood: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
            street: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
            city: import("zod").ZodOptional<import("zod").ZodString>;
            postalCode: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
            state: import("zod").ZodOptional<import("zod").ZodString>;
            countryCode: import("zod").ZodOptional<import("zod").ZodString>;
            website: import("zod").ZodOptional<import("zod").ZodString>;
            phone: import("zod").ZodOptional<import("zod").ZodString>;
            phoneUnformatted: import("zod").ZodOptional<import("zod").ZodString>;
            claimThisBusiness: import("zod").ZodOptional<import("zod").ZodBoolean>;
            location: import("zod").ZodOptional<import("zod").ZodObject<{
                lat: import("zod").ZodNumber;
                lng: import("zod").ZodNumber;
            }, "strip", import("zod").ZodTypeAny, {
                lat: number;
                lng: number;
            }, {
                lat: number;
                lng: number;
            }>>;
            locatedIn: import("zod").ZodOptional<import("zod").ZodString>;
            totalScore: import("zod").ZodOptional<import("zod").ZodNumber>;
            permanentlyClosed: import("zod").ZodOptional<import("zod").ZodBoolean>;
            temporarilyClosed: import("zod").ZodOptional<import("zod").ZodBoolean>;
            placeId: import("zod").ZodOptional<import("zod").ZodString>;
            categories: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            fid: import("zod").ZodOptional<import("zod").ZodString>;
            cid: import("zod").ZodOptional<import("zod").ZodString>;
            reviewsCount: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodNumber>>;
            reviewsDistribution: import("zod").ZodOptional<import("zod").ZodObject<{
                oneStar: import("zod").ZodOptional<import("zod").ZodNumber>;
                twoStar: import("zod").ZodOptional<import("zod").ZodNumber>;
                threeStar: import("zod").ZodOptional<import("zod").ZodNumber>;
                fourStar: import("zod").ZodOptional<import("zod").ZodNumber>;
                fiveStar: import("zod").ZodOptional<import("zod").ZodNumber>;
            }, "strip", import("zod").ZodTypeAny, {
                oneStar?: number | undefined;
                twoStar?: number | undefined;
                threeStar?: number | undefined;
                fourStar?: number | undefined;
                fiveStar?: number | undefined;
            }, {
                oneStar?: number | undefined;
                twoStar?: number | undefined;
                threeStar?: number | undefined;
                fourStar?: number | undefined;
                fiveStar?: number | undefined;
            }>>;
            imagesCount: import("zod").ZodOptional<import("zod").ZodNumber>;
            imageCategories: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            scrapedAt: import("zod").ZodOptional<import("zod").ZodString>;
            googleFoodUrl: import("zod").ZodOptional<import("zod").ZodNullable<import("zod").ZodString>>;
            hotelAds: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodUnknown, "many">>;
            openingHours: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodObject<{
                day: import("zod").ZodString;
                hours: import("zod").ZodString;
            }, "strip", import("zod").ZodTypeAny, {
                day: string;
                hours: string;
            }, {
                day: string;
                hours: string;
            }>, "many">>;
            additionalOpeningHours: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodString, import("zod").ZodArray<import("zod").ZodObject<{
                day: import("zod").ZodString;
                hours: import("zod").ZodString;
            }, "strip", import("zod").ZodTypeAny, {
                day: string;
                hours: string;
            }, {
                day: string;
                hours: string;
            }>, "many">>>;
            peopleAlsoSearch: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            placesTags: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            reviewsTags: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodString, "many">>;
            additionalInfo: import("zod").ZodOptional<import("zod").ZodRecord<import("zod").ZodString, import("zod").ZodArray<import("zod").ZodRecord<import("zod").ZodString, import("zod").ZodBoolean>, "many">>>;
            gasPrices: import("zod").ZodOptional<import("zod").ZodArray<import("zod").ZodUnknown, "many">>;
            url: import("zod").ZodOptional<import("zod").ZodString>;
            searchPageUrl: import("zod").ZodOptional<import("zod").ZodString>;
            searchString: import("zod").ZodOptional<import("zod").ZodString>;
            language: import("zod").ZodOptional<import("zod").ZodString>;
            rank: import("zod").ZodOptional<import("zod").ZodNumber>;
            isAdvertisement: import("zod").ZodOptional<import("zod").ZodBoolean>;
            imageUrl: import("zod").ZodOptional<import("zod").ZodString>;
            kgmid: import("zod").ZodOptional<import("zod").ZodString>;
        }, "strip", import("zod").ZodTypeAny, {
            description?: string | undefined;
            title?: string | undefined;
            url?: string | undefined;
            phone?: string | undefined;
            location?: {
                lat: number;
                lng: number;
            } | undefined;
            language?: string | undefined;
            postalCode?: string | null | undefined;
            price?: string | null | undefined;
            categoryName?: string | undefined;
            address?: string | undefined;
            neighborhood?: string | null | undefined;
            street?: string | null | undefined;
            city?: string | undefined;
            state?: string | undefined;
            countryCode?: string | undefined;
            website?: string | undefined;
            phoneUnformatted?: string | undefined;
            claimThisBusiness?: boolean | undefined;
            locatedIn?: string | undefined;
            totalScore?: number | undefined;
            permanentlyClosed?: boolean | undefined;
            temporarilyClosed?: boolean | undefined;
            placeId?: string | undefined;
            categories?: string[] | undefined;
            fid?: string | undefined;
            cid?: string | undefined;
            reviewsCount?: number | null | undefined;
            reviewsDistribution?: {
                oneStar?: number | undefined;
                twoStar?: number | undefined;
                threeStar?: number | undefined;
                fourStar?: number | undefined;
                fiveStar?: number | undefined;
            } | undefined;
            imagesCount?: number | undefined;
            imageCategories?: string[] | undefined;
            scrapedAt?: string | undefined;
            googleFoodUrl?: string | null | undefined;
            hotelAds?: unknown[] | undefined;
            openingHours?: {
                day: string;
                hours: string;
            }[] | undefined;
            additionalOpeningHours?: Record<string, {
                day: string;
                hours: string;
            }[]> | undefined;
            peopleAlsoSearch?: string[] | undefined;
            placesTags?: string[] | undefined;
            reviewsTags?: string[] | undefined;
            additionalInfo?: Record<string, Record<string, boolean>[]> | undefined;
            gasPrices?: unknown[] | undefined;
            searchPageUrl?: string | undefined;
            searchString?: string | undefined;
            rank?: number | undefined;
            isAdvertisement?: boolean | undefined;
            imageUrl?: string | undefined;
            kgmid?: string | undefined;
        }, {
            description?: string | undefined;
            title?: string | undefined;
            url?: string | undefined;
            phone?: string | undefined;
            location?: {
                lat: number;
                lng: number;
            } | undefined;
            language?: string | undefined;
            postalCode?: string | null | undefined;
            price?: string | null | undefined;
            categoryName?: string | undefined;
            address?: string | undefined;
            neighborhood?: string | null | undefined;
            street?: string | null | undefined;
            city?: string | undefined;
            state?: string | undefined;
            countryCode?: string | undefined;
            website?: string | undefined;
            phoneUnformatted?: string | undefined;
            claimThisBusiness?: boolean | undefined;
            locatedIn?: string | undefined;
            totalScore?: number | undefined;
            permanentlyClosed?: boolean | undefined;
            temporarilyClosed?: boolean | undefined;
            placeId?: string | undefined;
            categories?: string[] | undefined;
            fid?: string | undefined;
            cid?: string | undefined;
            reviewsCount?: number | null | undefined;
            reviewsDistribution?: {
                oneStar?: number | undefined;
                twoStar?: number | undefined;
                threeStar?: number | undefined;
                fourStar?: number | undefined;
                fiveStar?: number | undefined;
            } | undefined;
            imagesCount?: number | undefined;
            imageCategories?: string[] | undefined;
            scrapedAt?: string | undefined;
            googleFoodUrl?: string | null | undefined;
            hotelAds?: unknown[] | undefined;
            openingHours?: {
                day: string;
                hours: string;
            }[] | undefined;
            additionalOpeningHours?: Record<string, {
                day: string;
                hours: string;
            }[]> | undefined;
            peopleAlsoSearch?: string[] | undefined;
            placesTags?: string[] | undefined;
            reviewsTags?: string[] | undefined;
            additionalInfo?: Record<string, Record<string, boolean>[]> | undefined;
            gasPrices?: unknown[] | undefined;
            searchPageUrl?: string | undefined;
            searchString?: string | undefined;
            rank?: number | undefined;
            isAdvertisement?: boolean | undefined;
            imageUrl?: string | undefined;
            kgmid?: string | undefined;
        }>;
        description: string;
        documentation: string;
        category: string;
    };
};
//# sourceMappingURL=apify-scraper.schema.d.ts.map