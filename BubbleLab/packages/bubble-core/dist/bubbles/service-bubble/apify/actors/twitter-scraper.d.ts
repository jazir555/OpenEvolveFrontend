import { z } from 'zod';
export declare const TwitterScraperInputSchema: z.ZodObject<{
    startUrls: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    searchTerms: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    twitterHandles: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    conversationIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    maxItems: z.ZodOptional<z.ZodNumber>;
    sort: z.ZodOptional<z.ZodEnum<["Top", "Latest"]>>;
    tweetLanguage: z.ZodOptional<z.ZodEnum<["ab", "aa", "af", "ak", "sq", "am", "ar", "an", "hy", "as", "av", "ae", "ay", "az", "bm", "ba", "eu", "be", "bn", "bi", "bs", "br", "bg", "my", "ca", "ch", "ce", "ny", "zh", "cu", "cv", "kw", "co", "cr", "hr", "cs", "da", "dv", "nl", "dz", "en", "eo", "et", "ee", "fo", "fj", "fi", "fr", "fy", "ff", "gd", "gl", "lg", "ka", "de", "el", "kl", "gn", "gu", "ht", "ha", "he", "hz", "hi", "ho", "hu", "is", "io", "ig", "id", "ia", "ie", "iu", "ik", "ga", "it", "ja", "jv", "kn", "kr", "ks", "kk", "km", "ki", "rw", "ky", "kv", "kg", "ko", "kj", "ku", "lo", "la", "lv", "li", "ln", "lt", "lu", "lb", "mk", "mg", "ms", "ml", "mt", "gv", "mi", "mr", "mh", "mn", "na", "nv", "nd", "nr", "ng", "ne", "no", "nb", "nn", "ii", "oc", "oj", "or", "om", "os", "pi", "ps", "fa", "pl", "pt", "pa", "qu", "ro", "rm", "rn", "ru", "se", "sm", "sg", "sa", "sc", "sr", "sn", "sd", "si", "sk", "sl", "so", "st", "es", "su", "sw", "ss", "sv", "tl", "ty", "tg", "ta", "tt", "te", "th", "bo", "ti", "to", "ts", "tn", "tr", "tk", "tw", "ug", "uk", "ur", "uz", "ve", "vi", "vo", "wa", "cy", "wo", "xh", "yi", "yo", "za", "zu"]>>;
}, "strip", z.ZodTypeAny, {
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
export declare const TwitterTweetSchema: z.ZodObject<{
    id: z.ZodOptional<z.ZodString>;
    url: z.ZodOptional<z.ZodString>;
    text: z.ZodOptional<z.ZodString>;
    author: z.ZodOptional<z.ZodObject<{
        id: z.ZodOptional<z.ZodString>;
        name: z.ZodOptional<z.ZodString>;
        userName: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        isVerified: z.ZodOptional<z.ZodBoolean>;
        isBlueVerified: z.ZodOptional<z.ZodBoolean>;
        profilePicture: z.ZodOptional<z.ZodString>;
        followers: z.ZodOptional<z.ZodNumber>;
        following: z.ZodOptional<z.ZodNumber>;
        tweetsCount: z.ZodOptional<z.ZodNumber>;
        url: z.ZodOptional<z.ZodString>;
        createdAt: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
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
    createdAt: z.ZodOptional<z.ZodString>;
    retweetCount: z.ZodOptional<z.ZodNumber>;
    replyCount: z.ZodOptional<z.ZodNumber>;
    likeCount: z.ZodOptional<z.ZodNumber>;
    quoteCount: z.ZodOptional<z.ZodNumber>;
    viewCount: z.ZodOptional<z.ZodNumber>;
    bookmarkCount: z.ZodOptional<z.ZodNumber>;
    lang: z.ZodOptional<z.ZodString>;
    media: z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodObject<{
        type: z.ZodOptional<z.ZodEnum<["photo", "video", "animated_gif"]>>;
        url: z.ZodOptional<z.ZodString>;
        width: z.ZodOptional<z.ZodNumber>;
        height: z.ZodOptional<z.ZodNumber>;
        duration: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
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
    entities: z.ZodOptional<z.ZodObject<{
        hashtags: z.ZodOptional<z.ZodArray<z.ZodObject<{
            text: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            text?: string | undefined;
        }, {
            text?: string | undefined;
        }>, "many">>;
        urls: z.ZodOptional<z.ZodArray<z.ZodObject<{
            url: z.ZodOptional<z.ZodString>;
            expandedUrl: z.ZodOptional<z.ZodString>;
            displayUrl: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            url?: string | undefined;
            displayUrl?: string | undefined;
            expandedUrl?: string | undefined;
        }, {
            url?: string | undefined;
            displayUrl?: string | undefined;
            expandedUrl?: string | undefined;
        }>, "many">>;
        userMentions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            screenName: z.ZodOptional<z.ZodString>;
            name: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name?: string | undefined;
            screenName?: string | undefined;
        }, {
            name?: string | undefined;
            screenName?: string | undefined;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
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
    isRetweet: z.ZodOptional<z.ZodBoolean>;
    isQuote: z.ZodOptional<z.ZodBoolean>;
    isReply: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
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
export type TwitterScraperInput = z.output<typeof TwitterScraperInputSchema>;
export type TwitterTweet = z.output<typeof TwitterTweetSchema>;
//# sourceMappingURL=twitter-scraper.d.ts.map