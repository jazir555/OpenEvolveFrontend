import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const LinkedInAuthorSchema: z.ZodObject<{
    firstName: z.ZodNullable<z.ZodString>;
    lastName: z.ZodNullable<z.ZodString>;
    headline: z.ZodNullable<z.ZodString>;
    username: z.ZodNullable<z.ZodString>;
    profileUrl: z.ZodNullable<z.ZodString>;
    profilePicture: z.ZodNullable<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    username: string | null;
    headline: string | null;
    profileUrl: string | null;
    profilePicture: string | null;
    firstName: string | null;
    lastName: string | null;
}, {
    username: string | null;
    headline: string | null;
    profileUrl: string | null;
    profilePicture: string | null;
    firstName: string | null;
    lastName: string | null;
}>;
declare const LinkedInStatsSchema: z.ZodObject<{
    totalReactions: z.ZodNullable<z.ZodNumber>;
    like: z.ZodNullable<z.ZodNumber>;
    support: z.ZodNullable<z.ZodNumber>;
    love: z.ZodNullable<z.ZodNumber>;
    insight: z.ZodNullable<z.ZodNumber>;
    celebrate: z.ZodNullable<z.ZodNumber>;
    funny: z.ZodNullable<z.ZodNumber>;
    comments: z.ZodNullable<z.ZodNumber>;
    reposts: z.ZodNullable<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    like: number | null;
    support: number | null;
    love: number | null;
    insight: number | null;
    celebrate: number | null;
    funny: number | null;
    comments: number | null;
    reposts: number | null;
    totalReactions: number | null;
}, {
    like: number | null;
    support: number | null;
    love: number | null;
    insight: number | null;
    celebrate: number | null;
    funny: number | null;
    comments: number | null;
    reposts: number | null;
    totalReactions: number | null;
}>;
declare const LinkedInPostSchema: z.ZodObject<{
    urn: z.ZodNullable<z.ZodString>;
    fullUrn: z.ZodNullable<z.ZodString>;
    postedAt: z.ZodNullable<z.ZodObject<{
        date: z.ZodNullable<z.ZodString>;
        relative: z.ZodNullable<z.ZodString>;
        timestamp: z.ZodNullable<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        date: string | null;
        timestamp: number | null;
        relative: string | null;
    }, {
        date: string | null;
        timestamp: number | null;
        relative: string | null;
    }>>;
    text: z.ZodNullable<z.ZodString>;
    url: z.ZodNullable<z.ZodString>;
    postType: z.ZodNullable<z.ZodString>;
    author: z.ZodNullable<z.ZodObject<{
        firstName: z.ZodNullable<z.ZodString>;
        lastName: z.ZodNullable<z.ZodString>;
        headline: z.ZodNullable<z.ZodString>;
        username: z.ZodNullable<z.ZodString>;
        profileUrl: z.ZodNullable<z.ZodString>;
        profilePicture: z.ZodNullable<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        username: string | null;
        headline: string | null;
        profileUrl: string | null;
        profilePicture: string | null;
        firstName: string | null;
        lastName: string | null;
    }, {
        username: string | null;
        headline: string | null;
        profileUrl: string | null;
        profilePicture: string | null;
        firstName: string | null;
        lastName: string | null;
    }>>;
    stats: z.ZodNullable<z.ZodObject<{
        totalReactions: z.ZodNullable<z.ZodNumber>;
        like: z.ZodNullable<z.ZodNumber>;
        support: z.ZodNullable<z.ZodNumber>;
        love: z.ZodNullable<z.ZodNumber>;
        insight: z.ZodNullable<z.ZodNumber>;
        celebrate: z.ZodNullable<z.ZodNumber>;
        funny: z.ZodNullable<z.ZodNumber>;
        comments: z.ZodNullable<z.ZodNumber>;
        reposts: z.ZodNullable<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        like: number | null;
        support: number | null;
        love: number | null;
        insight: number | null;
        celebrate: number | null;
        funny: number | null;
        comments: number | null;
        reposts: number | null;
        totalReactions: number | null;
    }, {
        like: number | null;
        support: number | null;
        love: number | null;
        insight: number | null;
        celebrate: number | null;
        funny: number | null;
        comments: number | null;
        reposts: number | null;
        totalReactions: number | null;
    }>>;
    media: z.ZodNullable<z.ZodObject<{
        type: z.ZodNullable<z.ZodString>;
        url: z.ZodNullable<z.ZodString>;
        thumbnail: z.ZodNullable<z.ZodString>;
        images: z.ZodNullable<z.ZodArray<z.ZodObject<{
            url: z.ZodNullable<z.ZodString>;
            width: z.ZodNullable<z.ZodNumber>;
            height: z.ZodNullable<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            url: string | null;
            width: number | null;
            height: number | null;
        }, {
            url: string | null;
            width: number | null;
            height: number | null;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        type: string | null;
        url: string | null;
        images: {
            url: string | null;
            width: number | null;
            height: number | null;
        }[] | null;
        thumbnail: string | null;
    }, {
        type: string | null;
        url: string | null;
        images: {
            url: string | null;
            width: number | null;
            height: number | null;
        }[] | null;
        thumbnail: string | null;
    }>>;
    article: z.ZodNullable<z.ZodObject<{
        url: z.ZodNullable<z.ZodString>;
        title: z.ZodNullable<z.ZodString>;
        subtitle: z.ZodNullable<z.ZodString>;
        thumbnail: z.ZodNullable<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        title: string | null;
        url: string | null;
        subtitle: string | null;
        thumbnail: string | null;
    }, {
        title: string | null;
        url: string | null;
        subtitle: string | null;
        thumbnail: string | null;
    }>>;
    document: z.ZodNullable<z.ZodObject<{
        title: z.ZodNullable<z.ZodString>;
        pageCount: z.ZodNullable<z.ZodNumber>;
        url: z.ZodNullable<z.ZodString>;
        thumbnail: z.ZodNullable<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        title: string | null;
        url: string | null;
        pageCount: number | null;
        thumbnail: string | null;
    }, {
        title: string | null;
        url: string | null;
        pageCount: number | null;
        thumbnail: string | null;
    }>>;
    resharedPost: z.ZodNullable<z.ZodObject<{
        urn: z.ZodNullable<z.ZodString>;
        postedAt: z.ZodNullable<z.ZodObject<{
            date: z.ZodNullable<z.ZodString>;
            relative: z.ZodNullable<z.ZodString>;
            timestamp: z.ZodNullable<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            date: string | null;
            timestamp: number | null;
            relative: string | null;
        }, {
            date: string | null;
            timestamp: number | null;
            relative: string | null;
        }>>;
        text: z.ZodNullable<z.ZodString>;
        url: z.ZodNullable<z.ZodString>;
        postType: z.ZodNullable<z.ZodString>;
        author: z.ZodNullable<z.ZodObject<{
            firstName: z.ZodNullable<z.ZodString>;
            lastName: z.ZodNullable<z.ZodString>;
            headline: z.ZodNullable<z.ZodString>;
            username: z.ZodNullable<z.ZodString>;
            profileUrl: z.ZodNullable<z.ZodString>;
            profilePicture: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            username: string | null;
            headline: string | null;
            profileUrl: string | null;
            profilePicture: string | null;
            firstName: string | null;
            lastName: string | null;
        }, {
            username: string | null;
            headline: string | null;
            profileUrl: string | null;
            profilePicture: string | null;
            firstName: string | null;
            lastName: string | null;
        }>>;
        stats: z.ZodNullable<z.ZodObject<{
            totalReactions: z.ZodNullable<z.ZodNumber>;
            like: z.ZodNullable<z.ZodNumber>;
            support: z.ZodNullable<z.ZodNumber>;
            love: z.ZodNullable<z.ZodNumber>;
            insight: z.ZodNullable<z.ZodNumber>;
            celebrate: z.ZodNullable<z.ZodNumber>;
            funny: z.ZodNullable<z.ZodNumber>;
            comments: z.ZodNullable<z.ZodNumber>;
            reposts: z.ZodNullable<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            like: number | null;
            support: number | null;
            love: number | null;
            insight: number | null;
            celebrate: number | null;
            funny: number | null;
            comments: number | null;
            reposts: number | null;
            totalReactions: number | null;
        }, {
            like: number | null;
            support: number | null;
            love: number | null;
            insight: number | null;
            celebrate: number | null;
            funny: number | null;
            comments: number | null;
            reposts: number | null;
            totalReactions: number | null;
        }>>;
        media: z.ZodNullable<z.ZodObject<{
            type: z.ZodNullable<z.ZodString>;
            url: z.ZodNullable<z.ZodString>;
            thumbnail: z.ZodNullable<z.ZodString>;
            images: z.ZodNullable<z.ZodArray<z.ZodObject<{
                url: z.ZodNullable<z.ZodString>;
                width: z.ZodNullable<z.ZodNumber>;
                height: z.ZodNullable<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                url: string | null;
                width: number | null;
                height: number | null;
            }, {
                url: string | null;
                width: number | null;
                height: number | null;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            type: string | null;
            url: string | null;
            images: {
                url: string | null;
                width: number | null;
                height: number | null;
            }[] | null;
            thumbnail: string | null;
        }, {
            type: string | null;
            url: string | null;
            images: {
                url: string | null;
                width: number | null;
                height: number | null;
            }[] | null;
            thumbnail: string | null;
        }>>;
    }, "strip", z.ZodTypeAny, {
        url: string | null;
        text: string | null;
        urn: string | null;
        author: {
            username: string | null;
            headline: string | null;
            profileUrl: string | null;
            profilePicture: string | null;
            firstName: string | null;
            lastName: string | null;
        } | null;
        stats: {
            like: number | null;
            support: number | null;
            love: number | null;
            insight: number | null;
            celebrate: number | null;
            funny: number | null;
            comments: number | null;
            reposts: number | null;
            totalReactions: number | null;
        } | null;
        media: {
            type: string | null;
            url: string | null;
            images: {
                url: string | null;
                width: number | null;
                height: number | null;
            }[] | null;
            thumbnail: string | null;
        } | null;
        postedAt: {
            date: string | null;
            timestamp: number | null;
            relative: string | null;
        } | null;
        postType: string | null;
    }, {
        url: string | null;
        text: string | null;
        urn: string | null;
        author: {
            username: string | null;
            headline: string | null;
            profileUrl: string | null;
            profilePicture: string | null;
            firstName: string | null;
            lastName: string | null;
        } | null;
        stats: {
            like: number | null;
            support: number | null;
            love: number | null;
            insight: number | null;
            celebrate: number | null;
            funny: number | null;
            comments: number | null;
            reposts: number | null;
            totalReactions: number | null;
        } | null;
        media: {
            type: string | null;
            url: string | null;
            images: {
                url: string | null;
                width: number | null;
                height: number | null;
            }[] | null;
            thumbnail: string | null;
        } | null;
        postedAt: {
            date: string | null;
            timestamp: number | null;
            relative: string | null;
        } | null;
        postType: string | null;
    }>>;
}, "strip", z.ZodTypeAny, {
    url: string | null;
    text: string | null;
    urn: string | null;
    author: {
        username: string | null;
        headline: string | null;
        profileUrl: string | null;
        profilePicture: string | null;
        firstName: string | null;
        lastName: string | null;
    } | null;
    stats: {
        like: number | null;
        support: number | null;
        love: number | null;
        insight: number | null;
        celebrate: number | null;
        funny: number | null;
        comments: number | null;
        reposts: number | null;
        totalReactions: number | null;
    } | null;
    media: {
        type: string | null;
        url: string | null;
        images: {
            url: string | null;
            width: number | null;
            height: number | null;
        }[] | null;
        thumbnail: string | null;
    } | null;
    article: {
        title: string | null;
        url: string | null;
        subtitle: string | null;
        thumbnail: string | null;
    } | null;
    document: {
        title: string | null;
        url: string | null;
        pageCount: number | null;
        thumbnail: string | null;
    } | null;
    postedAt: {
        date: string | null;
        timestamp: number | null;
        relative: string | null;
    } | null;
    fullUrn: string | null;
    postType: string | null;
    resharedPost: {
        url: string | null;
        text: string | null;
        urn: string | null;
        author: {
            username: string | null;
            headline: string | null;
            profileUrl: string | null;
            profilePicture: string | null;
            firstName: string | null;
            lastName: string | null;
        } | null;
        stats: {
            like: number | null;
            support: number | null;
            love: number | null;
            insight: number | null;
            celebrate: number | null;
            funny: number | null;
            comments: number | null;
            reposts: number | null;
            totalReactions: number | null;
        } | null;
        media: {
            type: string | null;
            url: string | null;
            images: {
                url: string | null;
                width: number | null;
                height: number | null;
            }[] | null;
            thumbnail: string | null;
        } | null;
        postedAt: {
            date: string | null;
            timestamp: number | null;
            relative: string | null;
        } | null;
        postType: string | null;
    } | null;
}, {
    url: string | null;
    text: string | null;
    urn: string | null;
    author: {
        username: string | null;
        headline: string | null;
        profileUrl: string | null;
        profilePicture: string | null;
        firstName: string | null;
        lastName: string | null;
    } | null;
    stats: {
        like: number | null;
        support: number | null;
        love: number | null;
        insight: number | null;
        celebrate: number | null;
        funny: number | null;
        comments: number | null;
        reposts: number | null;
        totalReactions: number | null;
    } | null;
    media: {
        type: string | null;
        url: string | null;
        images: {
            url: string | null;
            width: number | null;
            height: number | null;
        }[] | null;
        thumbnail: string | null;
    } | null;
    article: {
        title: string | null;
        url: string | null;
        subtitle: string | null;
        thumbnail: string | null;
    } | null;
    document: {
        title: string | null;
        url: string | null;
        pageCount: number | null;
        thumbnail: string | null;
    } | null;
    postedAt: {
        date: string | null;
        timestamp: number | null;
        relative: string | null;
    } | null;
    fullUrn: string | null;
    postType: string | null;
    resharedPost: {
        url: string | null;
        text: string | null;
        urn: string | null;
        author: {
            username: string | null;
            headline: string | null;
            profileUrl: string | null;
            profilePicture: string | null;
            firstName: string | null;
            lastName: string | null;
        } | null;
        stats: {
            like: number | null;
            support: number | null;
            love: number | null;
            insight: number | null;
            celebrate: number | null;
            funny: number | null;
            comments: number | null;
            reposts: number | null;
            totalReactions: number | null;
        } | null;
        media: {
            type: string | null;
            url: string | null;
            images: {
                url: string | null;
                width: number | null;
                height: number | null;
            }[] | null;
            thumbnail: string | null;
        } | null;
        postedAt: {
            date: string | null;
            timestamp: number | null;
            relative: string | null;
        } | null;
        postType: string | null;
    } | null;
}>;
declare const LinkedInJobSchema: z.ZodObject<{
    id: z.ZodNullable<z.ZodString>;
    title: z.ZodNullable<z.ZodString>;
    company: z.ZodNullable<z.ZodObject<{
        name: z.ZodNullable<z.ZodString>;
        url: z.ZodNullable<z.ZodString>;
        logo: z.ZodNullable<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name: string | null;
        url: string | null;
        logo: string | null;
    }, {
        name: string | null;
        url: string | null;
        logo: string | null;
    }>>;
    location: z.ZodNullable<z.ZodString>;
    description: z.ZodNullable<z.ZodString>;
    employmentType: z.ZodNullable<z.ZodString>;
    seniorityLevel: z.ZodNullable<z.ZodString>;
    postedAt: z.ZodNullable<z.ZodString>;
    url: z.ZodNullable<z.ZodString>;
    applyUrl: z.ZodNullable<z.ZodString>;
    salary: z.ZodNullable<z.ZodObject<{
        from: z.ZodNullable<z.ZodNumber>;
        to: z.ZodNullable<z.ZodNumber>;
        currency: z.ZodNullable<z.ZodString>;
        period: z.ZodNullable<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        from: number | null;
        to: number | null;
        currency: string | null;
        period: string | null;
    }, {
        from: number | null;
        to: number | null;
        currency: string | null;
        period: string | null;
    }>>;
    skills: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    description: string | null;
    title: string | null;
    url: string | null;
    id: string | null;
    location: string | null;
    postedAt: string | null;
    applyUrl: string | null;
    salary: {
        from: number | null;
        to: number | null;
        currency: string | null;
        period: string | null;
    } | null;
    seniorityLevel: string | null;
    employmentType: string | null;
    company: {
        name: string | null;
        url: string | null;
        logo: string | null;
    } | null;
    skills: string[] | null;
}, {
    description: string | null;
    title: string | null;
    url: string | null;
    id: string | null;
    location: string | null;
    postedAt: string | null;
    applyUrl: string | null;
    salary: {
        from: number | null;
        to: number | null;
        currency: string | null;
        period: string | null;
    } | null;
    seniorityLevel: string | null;
    employmentType: string | null;
    company: {
        name: string | null;
        url: string | null;
        logo: string | null;
    } | null;
    skills: string[] | null;
}>;
declare const LinkedInToolParamsSchema: z.ZodObject<{
    operation: z.ZodEnum<["scrapePosts", "searchPosts", "scrapeJobs"]>;
    username: z.ZodOptional<z.ZodString>;
    keyword: z.ZodOptional<z.ZodString>;
    location: z.ZodOptional<z.ZodString>;
    jobType: z.ZodOptional<z.ZodArray<z.ZodEnum<["full-time", "part-time", "contract", "temporary", "internship"]>, "many">>;
    workplaceType: z.ZodOptional<z.ZodArray<z.ZodEnum<["on-site", "remote", "hybrid"]>, "many">>;
    experienceLevel: z.ZodOptional<z.ZodArray<z.ZodEnum<["internship", "entry-level", "associate", "mid-senior", "director", "executive"]>, "many">>;
    sortBy: z.ZodOptional<z.ZodDefault<z.ZodEnum<["relevance", "date_posted"]>>>;
    dateFilter: z.ZodOptional<z.ZodDefault<z.ZodEnum<["", "past-24h", "past-week", "past-month"]>>>;
    limit: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    pageNumber: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "scrapePosts" | "searchPosts" | "scrapeJobs";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    username?: string | undefined;
    limit?: number | undefined;
    pageNumber?: number | undefined;
    location?: string | undefined;
    keyword?: string | undefined;
    dateFilter?: "" | "past-24h" | "past-week" | "past-month" | undefined;
    jobType?: ("temporary" | "full-time" | "part-time" | "contract" | "internship")[] | undefined;
    workplaceType?: ("on-site" | "remote" | "hybrid")[] | undefined;
    experienceLevel?: ("internship" | "entry-level" | "associate" | "mid-senior" | "director" | "executive")[] | undefined;
    sortBy?: "relevance" | "date_posted" | undefined;
}, {
    operation: "scrapePosts" | "searchPosts" | "scrapeJobs";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    username?: string | undefined;
    limit?: number | undefined;
    pageNumber?: number | undefined;
    location?: string | undefined;
    keyword?: string | undefined;
    dateFilter?: "" | "past-24h" | "past-week" | "past-month" | undefined;
    jobType?: ("temporary" | "full-time" | "part-time" | "contract" | "internship")[] | undefined;
    workplaceType?: ("on-site" | "remote" | "hybrid")[] | undefined;
    experienceLevel?: ("internship" | "entry-level" | "associate" | "mid-senior" | "director" | "executive")[] | undefined;
    sortBy?: "relevance" | "date_posted" | undefined;
}>;
declare const LinkedInToolResultSchema: z.ZodObject<{
    operation: z.ZodEnum<["scrapePosts", "searchPosts", "scrapeJobs"]>;
    jobs: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodNullable<z.ZodString>;
        title: z.ZodNullable<z.ZodString>;
        company: z.ZodNullable<z.ZodObject<{
            name: z.ZodNullable<z.ZodString>;
            url: z.ZodNullable<z.ZodString>;
            logo: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string | null;
            url: string | null;
            logo: string | null;
        }, {
            name: string | null;
            url: string | null;
            logo: string | null;
        }>>;
        location: z.ZodNullable<z.ZodString>;
        description: z.ZodNullable<z.ZodString>;
        employmentType: z.ZodNullable<z.ZodString>;
        seniorityLevel: z.ZodNullable<z.ZodString>;
        postedAt: z.ZodNullable<z.ZodString>;
        url: z.ZodNullable<z.ZodString>;
        applyUrl: z.ZodNullable<z.ZodString>;
        salary: z.ZodNullable<z.ZodObject<{
            from: z.ZodNullable<z.ZodNumber>;
            to: z.ZodNullable<z.ZodNumber>;
            currency: z.ZodNullable<z.ZodString>;
            period: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            from: number | null;
            to: number | null;
            currency: string | null;
            period: string | null;
        }, {
            from: number | null;
            to: number | null;
            currency: string | null;
            period: string | null;
        }>>;
        skills: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        description: string | null;
        title: string | null;
        url: string | null;
        id: string | null;
        location: string | null;
        postedAt: string | null;
        applyUrl: string | null;
        salary: {
            from: number | null;
            to: number | null;
            currency: string | null;
            period: string | null;
        } | null;
        seniorityLevel: string | null;
        employmentType: string | null;
        company: {
            name: string | null;
            url: string | null;
            logo: string | null;
        } | null;
        skills: string[] | null;
    }, {
        description: string | null;
        title: string | null;
        url: string | null;
        id: string | null;
        location: string | null;
        postedAt: string | null;
        applyUrl: string | null;
        salary: {
            from: number | null;
            to: number | null;
            currency: string | null;
            period: string | null;
        } | null;
        seniorityLevel: string | null;
        employmentType: string | null;
        company: {
            name: string | null;
            url: string | null;
            logo: string | null;
        } | null;
        skills: string[] | null;
    }>, "many">>;
    posts: z.ZodArray<z.ZodObject<{
        urn: z.ZodNullable<z.ZodString>;
        fullUrn: z.ZodNullable<z.ZodString>;
        postedAt: z.ZodNullable<z.ZodObject<{
            date: z.ZodNullable<z.ZodString>;
            relative: z.ZodNullable<z.ZodString>;
            timestamp: z.ZodNullable<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            date: string | null;
            timestamp: number | null;
            relative: string | null;
        }, {
            date: string | null;
            timestamp: number | null;
            relative: string | null;
        }>>;
        text: z.ZodNullable<z.ZodString>;
        url: z.ZodNullable<z.ZodString>;
        postType: z.ZodNullable<z.ZodString>;
        author: z.ZodNullable<z.ZodObject<{
            firstName: z.ZodNullable<z.ZodString>;
            lastName: z.ZodNullable<z.ZodString>;
            headline: z.ZodNullable<z.ZodString>;
            username: z.ZodNullable<z.ZodString>;
            profileUrl: z.ZodNullable<z.ZodString>;
            profilePicture: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            username: string | null;
            headline: string | null;
            profileUrl: string | null;
            profilePicture: string | null;
            firstName: string | null;
            lastName: string | null;
        }, {
            username: string | null;
            headline: string | null;
            profileUrl: string | null;
            profilePicture: string | null;
            firstName: string | null;
            lastName: string | null;
        }>>;
        stats: z.ZodNullable<z.ZodObject<{
            totalReactions: z.ZodNullable<z.ZodNumber>;
            like: z.ZodNullable<z.ZodNumber>;
            support: z.ZodNullable<z.ZodNumber>;
            love: z.ZodNullable<z.ZodNumber>;
            insight: z.ZodNullable<z.ZodNumber>;
            celebrate: z.ZodNullable<z.ZodNumber>;
            funny: z.ZodNullable<z.ZodNumber>;
            comments: z.ZodNullable<z.ZodNumber>;
            reposts: z.ZodNullable<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            like: number | null;
            support: number | null;
            love: number | null;
            insight: number | null;
            celebrate: number | null;
            funny: number | null;
            comments: number | null;
            reposts: number | null;
            totalReactions: number | null;
        }, {
            like: number | null;
            support: number | null;
            love: number | null;
            insight: number | null;
            celebrate: number | null;
            funny: number | null;
            comments: number | null;
            reposts: number | null;
            totalReactions: number | null;
        }>>;
        media: z.ZodNullable<z.ZodObject<{
            type: z.ZodNullable<z.ZodString>;
            url: z.ZodNullable<z.ZodString>;
            thumbnail: z.ZodNullable<z.ZodString>;
            images: z.ZodNullable<z.ZodArray<z.ZodObject<{
                url: z.ZodNullable<z.ZodString>;
                width: z.ZodNullable<z.ZodNumber>;
                height: z.ZodNullable<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                url: string | null;
                width: number | null;
                height: number | null;
            }, {
                url: string | null;
                width: number | null;
                height: number | null;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            type: string | null;
            url: string | null;
            images: {
                url: string | null;
                width: number | null;
                height: number | null;
            }[] | null;
            thumbnail: string | null;
        }, {
            type: string | null;
            url: string | null;
            images: {
                url: string | null;
                width: number | null;
                height: number | null;
            }[] | null;
            thumbnail: string | null;
        }>>;
        article: z.ZodNullable<z.ZodObject<{
            url: z.ZodNullable<z.ZodString>;
            title: z.ZodNullable<z.ZodString>;
            subtitle: z.ZodNullable<z.ZodString>;
            thumbnail: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            title: string | null;
            url: string | null;
            subtitle: string | null;
            thumbnail: string | null;
        }, {
            title: string | null;
            url: string | null;
            subtitle: string | null;
            thumbnail: string | null;
        }>>;
        document: z.ZodNullable<z.ZodObject<{
            title: z.ZodNullable<z.ZodString>;
            pageCount: z.ZodNullable<z.ZodNumber>;
            url: z.ZodNullable<z.ZodString>;
            thumbnail: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            title: string | null;
            url: string | null;
            pageCount: number | null;
            thumbnail: string | null;
        }, {
            title: string | null;
            url: string | null;
            pageCount: number | null;
            thumbnail: string | null;
        }>>;
        resharedPost: z.ZodNullable<z.ZodObject<{
            urn: z.ZodNullable<z.ZodString>;
            postedAt: z.ZodNullable<z.ZodObject<{
                date: z.ZodNullable<z.ZodString>;
                relative: z.ZodNullable<z.ZodString>;
                timestamp: z.ZodNullable<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            }, {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            }>>;
            text: z.ZodNullable<z.ZodString>;
            url: z.ZodNullable<z.ZodString>;
            postType: z.ZodNullable<z.ZodString>;
            author: z.ZodNullable<z.ZodObject<{
                firstName: z.ZodNullable<z.ZodString>;
                lastName: z.ZodNullable<z.ZodString>;
                headline: z.ZodNullable<z.ZodString>;
                username: z.ZodNullable<z.ZodString>;
                profileUrl: z.ZodNullable<z.ZodString>;
                profilePicture: z.ZodNullable<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            }, {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            }>>;
            stats: z.ZodNullable<z.ZodObject<{
                totalReactions: z.ZodNullable<z.ZodNumber>;
                like: z.ZodNullable<z.ZodNumber>;
                support: z.ZodNullable<z.ZodNumber>;
                love: z.ZodNullable<z.ZodNumber>;
                insight: z.ZodNullable<z.ZodNumber>;
                celebrate: z.ZodNullable<z.ZodNumber>;
                funny: z.ZodNullable<z.ZodNumber>;
                comments: z.ZodNullable<z.ZodNumber>;
                reposts: z.ZodNullable<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            }, {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            }>>;
            media: z.ZodNullable<z.ZodObject<{
                type: z.ZodNullable<z.ZodString>;
                url: z.ZodNullable<z.ZodString>;
                thumbnail: z.ZodNullable<z.ZodString>;
                images: z.ZodNullable<z.ZodArray<z.ZodObject<{
                    url: z.ZodNullable<z.ZodString>;
                    width: z.ZodNullable<z.ZodNumber>;
                    height: z.ZodNullable<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }, {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }>, "many">>;
            }, "strip", z.ZodTypeAny, {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            }, {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            }>>;
        }, "strip", z.ZodTypeAny, {
            url: string | null;
            text: string | null;
            urn: string | null;
            author: {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            } | null;
            stats: {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            } | null;
            postedAt: {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            } | null;
            postType: string | null;
        }, {
            url: string | null;
            text: string | null;
            urn: string | null;
            author: {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            } | null;
            stats: {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            } | null;
            postedAt: {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            } | null;
            postType: string | null;
        }>>;
    }, "strip", z.ZodTypeAny, {
        url: string | null;
        text: string | null;
        urn: string | null;
        author: {
            username: string | null;
            headline: string | null;
            profileUrl: string | null;
            profilePicture: string | null;
            firstName: string | null;
            lastName: string | null;
        } | null;
        stats: {
            like: number | null;
            support: number | null;
            love: number | null;
            insight: number | null;
            celebrate: number | null;
            funny: number | null;
            comments: number | null;
            reposts: number | null;
            totalReactions: number | null;
        } | null;
        media: {
            type: string | null;
            url: string | null;
            images: {
                url: string | null;
                width: number | null;
                height: number | null;
            }[] | null;
            thumbnail: string | null;
        } | null;
        article: {
            title: string | null;
            url: string | null;
            subtitle: string | null;
            thumbnail: string | null;
        } | null;
        document: {
            title: string | null;
            url: string | null;
            pageCount: number | null;
            thumbnail: string | null;
        } | null;
        postedAt: {
            date: string | null;
            timestamp: number | null;
            relative: string | null;
        } | null;
        fullUrn: string | null;
        postType: string | null;
        resharedPost: {
            url: string | null;
            text: string | null;
            urn: string | null;
            author: {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            } | null;
            stats: {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            } | null;
            postedAt: {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            } | null;
            postType: string | null;
        } | null;
    }, {
        url: string | null;
        text: string | null;
        urn: string | null;
        author: {
            username: string | null;
            headline: string | null;
            profileUrl: string | null;
            profilePicture: string | null;
            firstName: string | null;
            lastName: string | null;
        } | null;
        stats: {
            like: number | null;
            support: number | null;
            love: number | null;
            insight: number | null;
            celebrate: number | null;
            funny: number | null;
            comments: number | null;
            reposts: number | null;
            totalReactions: number | null;
        } | null;
        media: {
            type: string | null;
            url: string | null;
            images: {
                url: string | null;
                width: number | null;
                height: number | null;
            }[] | null;
            thumbnail: string | null;
        } | null;
        article: {
            title: string | null;
            url: string | null;
            subtitle: string | null;
            thumbnail: string | null;
        } | null;
        document: {
            title: string | null;
            url: string | null;
            pageCount: number | null;
            thumbnail: string | null;
        } | null;
        postedAt: {
            date: string | null;
            timestamp: number | null;
            relative: string | null;
        } | null;
        fullUrn: string | null;
        postType: string | null;
        resharedPost: {
            url: string | null;
            text: string | null;
            urn: string | null;
            author: {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            } | null;
            stats: {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            } | null;
            postedAt: {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            } | null;
            postType: string | null;
        } | null;
    }>, "many">;
    username: z.ZodOptional<z.ZodString>;
    paginationToken: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    keyword: z.ZodOptional<z.ZodString>;
    totalResults: z.ZodOptional<z.ZodNullable<z.ZodNumber>>;
    hasNextPage: z.ZodOptional<z.ZodNullable<z.ZodBoolean>>;
    totalPosts: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "scrapePosts" | "searchPosts" | "scrapeJobs";
    posts: {
        url: string | null;
        text: string | null;
        urn: string | null;
        author: {
            username: string | null;
            headline: string | null;
            profileUrl: string | null;
            profilePicture: string | null;
            firstName: string | null;
            lastName: string | null;
        } | null;
        stats: {
            like: number | null;
            support: number | null;
            love: number | null;
            insight: number | null;
            celebrate: number | null;
            funny: number | null;
            comments: number | null;
            reposts: number | null;
            totalReactions: number | null;
        } | null;
        media: {
            type: string | null;
            url: string | null;
            images: {
                url: string | null;
                width: number | null;
                height: number | null;
            }[] | null;
            thumbnail: string | null;
        } | null;
        article: {
            title: string | null;
            url: string | null;
            subtitle: string | null;
            thumbnail: string | null;
        } | null;
        document: {
            title: string | null;
            url: string | null;
            pageCount: number | null;
            thumbnail: string | null;
        } | null;
        postedAt: {
            date: string | null;
            timestamp: number | null;
            relative: string | null;
        } | null;
        fullUrn: string | null;
        postType: string | null;
        resharedPost: {
            url: string | null;
            text: string | null;
            urn: string | null;
            author: {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            } | null;
            stats: {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            } | null;
            postedAt: {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            } | null;
            postType: string | null;
        } | null;
    }[];
    totalPosts: number;
    username?: string | undefined;
    keyword?: string | undefined;
    jobs?: {
        description: string | null;
        title: string | null;
        url: string | null;
        id: string | null;
        location: string | null;
        postedAt: string | null;
        applyUrl: string | null;
        salary: {
            from: number | null;
            to: number | null;
            currency: string | null;
            period: string | null;
        } | null;
        seniorityLevel: string | null;
        employmentType: string | null;
        company: {
            name: string | null;
            url: string | null;
            logo: string | null;
        } | null;
        skills: string[] | null;
    }[] | undefined;
    totalResults?: number | null | undefined;
    paginationToken?: string | null | undefined;
    hasNextPage?: boolean | null | undefined;
}, {
    error: string;
    success: boolean;
    operation: "scrapePosts" | "searchPosts" | "scrapeJobs";
    posts: {
        url: string | null;
        text: string | null;
        urn: string | null;
        author: {
            username: string | null;
            headline: string | null;
            profileUrl: string | null;
            profilePicture: string | null;
            firstName: string | null;
            lastName: string | null;
        } | null;
        stats: {
            like: number | null;
            support: number | null;
            love: number | null;
            insight: number | null;
            celebrate: number | null;
            funny: number | null;
            comments: number | null;
            reposts: number | null;
            totalReactions: number | null;
        } | null;
        media: {
            type: string | null;
            url: string | null;
            images: {
                url: string | null;
                width: number | null;
                height: number | null;
            }[] | null;
            thumbnail: string | null;
        } | null;
        article: {
            title: string | null;
            url: string | null;
            subtitle: string | null;
            thumbnail: string | null;
        } | null;
        document: {
            title: string | null;
            url: string | null;
            pageCount: number | null;
            thumbnail: string | null;
        } | null;
        postedAt: {
            date: string | null;
            timestamp: number | null;
            relative: string | null;
        } | null;
        fullUrn: string | null;
        postType: string | null;
        resharedPost: {
            url: string | null;
            text: string | null;
            urn: string | null;
            author: {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            } | null;
            stats: {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            } | null;
            postedAt: {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            } | null;
            postType: string | null;
        } | null;
    }[];
    totalPosts: number;
    username?: string | undefined;
    keyword?: string | undefined;
    jobs?: {
        description: string | null;
        title: string | null;
        url: string | null;
        id: string | null;
        location: string | null;
        postedAt: string | null;
        applyUrl: string | null;
        salary: {
            from: number | null;
            to: number | null;
            currency: string | null;
            period: string | null;
        } | null;
        seniorityLevel: string | null;
        employmentType: string | null;
        company: {
            name: string | null;
            url: string | null;
            logo: string | null;
        } | null;
        skills: string[] | null;
    }[] | undefined;
    totalResults?: number | null | undefined;
    paginationToken?: string | null | undefined;
    hasNextPage?: boolean | null | undefined;
}>;
type LinkedInToolParams = z.output<typeof LinkedInToolParamsSchema>;
type LinkedInToolResult = z.output<typeof LinkedInToolResultSchema>;
type LinkedInToolParamsInput = z.input<typeof LinkedInToolParamsSchema>;
export type LinkedInPost = z.output<typeof LinkedInPostSchema>;
export type LinkedInJob = z.output<typeof LinkedInJobSchema>;
export type LinkedInAuthor = z.output<typeof LinkedInAuthorSchema>;
export type LinkedInStats = z.output<typeof LinkedInStatsSchema>;
/**
 * LinkedIn scraping tool with multiple operations
 *
 * This tool provides a simple interface for scraping LinkedIn data.
 *
 * Operations:
 * 1. scrapePosts - Scrape posts from a specific LinkedIn profile
 * 2. searchPosts - Search for LinkedIn posts by keyword
 *
 * Features:
 * - Get complete post metadata (text, engagement stats, media, etc.)
 * - Support for all post types (regular, quotes, articles, documents)
 * - Pagination support
 * - Date filtering for search
 */
export declare class LinkedInTool extends ToolBubble<LinkedInToolParams, LinkedInToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        operation: z.ZodEnum<["scrapePosts", "searchPosts", "scrapeJobs"]>;
        username: z.ZodOptional<z.ZodString>;
        keyword: z.ZodOptional<z.ZodString>;
        location: z.ZodOptional<z.ZodString>;
        jobType: z.ZodOptional<z.ZodArray<z.ZodEnum<["full-time", "part-time", "contract", "temporary", "internship"]>, "many">>;
        workplaceType: z.ZodOptional<z.ZodArray<z.ZodEnum<["on-site", "remote", "hybrid"]>, "many">>;
        experienceLevel: z.ZodOptional<z.ZodArray<z.ZodEnum<["internship", "entry-level", "associate", "mid-senior", "director", "executive"]>, "many">>;
        sortBy: z.ZodOptional<z.ZodDefault<z.ZodEnum<["relevance", "date_posted"]>>>;
        dateFilter: z.ZodOptional<z.ZodDefault<z.ZodEnum<["", "past-24h", "past-week", "past-month"]>>>;
        limit: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        pageNumber: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "scrapePosts" | "searchPosts" | "scrapeJobs";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        username?: string | undefined;
        limit?: number | undefined;
        pageNumber?: number | undefined;
        location?: string | undefined;
        keyword?: string | undefined;
        dateFilter?: "" | "past-24h" | "past-week" | "past-month" | undefined;
        jobType?: ("temporary" | "full-time" | "part-time" | "contract" | "internship")[] | undefined;
        workplaceType?: ("on-site" | "remote" | "hybrid")[] | undefined;
        experienceLevel?: ("internship" | "entry-level" | "associate" | "mid-senior" | "director" | "executive")[] | undefined;
        sortBy?: "relevance" | "date_posted" | undefined;
    }, {
        operation: "scrapePosts" | "searchPosts" | "scrapeJobs";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        username?: string | undefined;
        limit?: number | undefined;
        pageNumber?: number | undefined;
        location?: string | undefined;
        keyword?: string | undefined;
        dateFilter?: "" | "past-24h" | "past-week" | "past-month" | undefined;
        jobType?: ("temporary" | "full-time" | "part-time" | "contract" | "internship")[] | undefined;
        workplaceType?: ("on-site" | "remote" | "hybrid")[] | undefined;
        experienceLevel?: ("internship" | "entry-level" | "associate" | "mid-senior" | "director" | "executive")[] | undefined;
        sortBy?: "relevance" | "date_posted" | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        operation: z.ZodEnum<["scrapePosts", "searchPosts", "scrapeJobs"]>;
        jobs: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodNullable<z.ZodString>;
            title: z.ZodNullable<z.ZodString>;
            company: z.ZodNullable<z.ZodObject<{
                name: z.ZodNullable<z.ZodString>;
                url: z.ZodNullable<z.ZodString>;
                logo: z.ZodNullable<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                name: string | null;
                url: string | null;
                logo: string | null;
            }, {
                name: string | null;
                url: string | null;
                logo: string | null;
            }>>;
            location: z.ZodNullable<z.ZodString>;
            description: z.ZodNullable<z.ZodString>;
            employmentType: z.ZodNullable<z.ZodString>;
            seniorityLevel: z.ZodNullable<z.ZodString>;
            postedAt: z.ZodNullable<z.ZodString>;
            url: z.ZodNullable<z.ZodString>;
            applyUrl: z.ZodNullable<z.ZodString>;
            salary: z.ZodNullable<z.ZodObject<{
                from: z.ZodNullable<z.ZodNumber>;
                to: z.ZodNullable<z.ZodNumber>;
                currency: z.ZodNullable<z.ZodString>;
                period: z.ZodNullable<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                from: number | null;
                to: number | null;
                currency: string | null;
                period: string | null;
            }, {
                from: number | null;
                to: number | null;
                currency: string | null;
                period: string | null;
            }>>;
            skills: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            description: string | null;
            title: string | null;
            url: string | null;
            id: string | null;
            location: string | null;
            postedAt: string | null;
            applyUrl: string | null;
            salary: {
                from: number | null;
                to: number | null;
                currency: string | null;
                period: string | null;
            } | null;
            seniorityLevel: string | null;
            employmentType: string | null;
            company: {
                name: string | null;
                url: string | null;
                logo: string | null;
            } | null;
            skills: string[] | null;
        }, {
            description: string | null;
            title: string | null;
            url: string | null;
            id: string | null;
            location: string | null;
            postedAt: string | null;
            applyUrl: string | null;
            salary: {
                from: number | null;
                to: number | null;
                currency: string | null;
                period: string | null;
            } | null;
            seniorityLevel: string | null;
            employmentType: string | null;
            company: {
                name: string | null;
                url: string | null;
                logo: string | null;
            } | null;
            skills: string[] | null;
        }>, "many">>;
        posts: z.ZodArray<z.ZodObject<{
            urn: z.ZodNullable<z.ZodString>;
            fullUrn: z.ZodNullable<z.ZodString>;
            postedAt: z.ZodNullable<z.ZodObject<{
                date: z.ZodNullable<z.ZodString>;
                relative: z.ZodNullable<z.ZodString>;
                timestamp: z.ZodNullable<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            }, {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            }>>;
            text: z.ZodNullable<z.ZodString>;
            url: z.ZodNullable<z.ZodString>;
            postType: z.ZodNullable<z.ZodString>;
            author: z.ZodNullable<z.ZodObject<{
                firstName: z.ZodNullable<z.ZodString>;
                lastName: z.ZodNullable<z.ZodString>;
                headline: z.ZodNullable<z.ZodString>;
                username: z.ZodNullable<z.ZodString>;
                profileUrl: z.ZodNullable<z.ZodString>;
                profilePicture: z.ZodNullable<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            }, {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            }>>;
            stats: z.ZodNullable<z.ZodObject<{
                totalReactions: z.ZodNullable<z.ZodNumber>;
                like: z.ZodNullable<z.ZodNumber>;
                support: z.ZodNullable<z.ZodNumber>;
                love: z.ZodNullable<z.ZodNumber>;
                insight: z.ZodNullable<z.ZodNumber>;
                celebrate: z.ZodNullable<z.ZodNumber>;
                funny: z.ZodNullable<z.ZodNumber>;
                comments: z.ZodNullable<z.ZodNumber>;
                reposts: z.ZodNullable<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            }, {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            }>>;
            media: z.ZodNullable<z.ZodObject<{
                type: z.ZodNullable<z.ZodString>;
                url: z.ZodNullable<z.ZodString>;
                thumbnail: z.ZodNullable<z.ZodString>;
                images: z.ZodNullable<z.ZodArray<z.ZodObject<{
                    url: z.ZodNullable<z.ZodString>;
                    width: z.ZodNullable<z.ZodNumber>;
                    height: z.ZodNullable<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }, {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }>, "many">>;
            }, "strip", z.ZodTypeAny, {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            }, {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            }>>;
            article: z.ZodNullable<z.ZodObject<{
                url: z.ZodNullable<z.ZodString>;
                title: z.ZodNullable<z.ZodString>;
                subtitle: z.ZodNullable<z.ZodString>;
                thumbnail: z.ZodNullable<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                title: string | null;
                url: string | null;
                subtitle: string | null;
                thumbnail: string | null;
            }, {
                title: string | null;
                url: string | null;
                subtitle: string | null;
                thumbnail: string | null;
            }>>;
            document: z.ZodNullable<z.ZodObject<{
                title: z.ZodNullable<z.ZodString>;
                pageCount: z.ZodNullable<z.ZodNumber>;
                url: z.ZodNullable<z.ZodString>;
                thumbnail: z.ZodNullable<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                title: string | null;
                url: string | null;
                pageCount: number | null;
                thumbnail: string | null;
            }, {
                title: string | null;
                url: string | null;
                pageCount: number | null;
                thumbnail: string | null;
            }>>;
            resharedPost: z.ZodNullable<z.ZodObject<{
                urn: z.ZodNullable<z.ZodString>;
                postedAt: z.ZodNullable<z.ZodObject<{
                    date: z.ZodNullable<z.ZodString>;
                    relative: z.ZodNullable<z.ZodString>;
                    timestamp: z.ZodNullable<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    date: string | null;
                    timestamp: number | null;
                    relative: string | null;
                }, {
                    date: string | null;
                    timestamp: number | null;
                    relative: string | null;
                }>>;
                text: z.ZodNullable<z.ZodString>;
                url: z.ZodNullable<z.ZodString>;
                postType: z.ZodNullable<z.ZodString>;
                author: z.ZodNullable<z.ZodObject<{
                    firstName: z.ZodNullable<z.ZodString>;
                    lastName: z.ZodNullable<z.ZodString>;
                    headline: z.ZodNullable<z.ZodString>;
                    username: z.ZodNullable<z.ZodString>;
                    profileUrl: z.ZodNullable<z.ZodString>;
                    profilePicture: z.ZodNullable<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    username: string | null;
                    headline: string | null;
                    profileUrl: string | null;
                    profilePicture: string | null;
                    firstName: string | null;
                    lastName: string | null;
                }, {
                    username: string | null;
                    headline: string | null;
                    profileUrl: string | null;
                    profilePicture: string | null;
                    firstName: string | null;
                    lastName: string | null;
                }>>;
                stats: z.ZodNullable<z.ZodObject<{
                    totalReactions: z.ZodNullable<z.ZodNumber>;
                    like: z.ZodNullable<z.ZodNumber>;
                    support: z.ZodNullable<z.ZodNumber>;
                    love: z.ZodNullable<z.ZodNumber>;
                    insight: z.ZodNullable<z.ZodNumber>;
                    celebrate: z.ZodNullable<z.ZodNumber>;
                    funny: z.ZodNullable<z.ZodNumber>;
                    comments: z.ZodNullable<z.ZodNumber>;
                    reposts: z.ZodNullable<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    like: number | null;
                    support: number | null;
                    love: number | null;
                    insight: number | null;
                    celebrate: number | null;
                    funny: number | null;
                    comments: number | null;
                    reposts: number | null;
                    totalReactions: number | null;
                }, {
                    like: number | null;
                    support: number | null;
                    love: number | null;
                    insight: number | null;
                    celebrate: number | null;
                    funny: number | null;
                    comments: number | null;
                    reposts: number | null;
                    totalReactions: number | null;
                }>>;
                media: z.ZodNullable<z.ZodObject<{
                    type: z.ZodNullable<z.ZodString>;
                    url: z.ZodNullable<z.ZodString>;
                    thumbnail: z.ZodNullable<z.ZodString>;
                    images: z.ZodNullable<z.ZodArray<z.ZodObject<{
                        url: z.ZodNullable<z.ZodString>;
                        width: z.ZodNullable<z.ZodNumber>;
                        height: z.ZodNullable<z.ZodNumber>;
                    }, "strip", z.ZodTypeAny, {
                        url: string | null;
                        width: number | null;
                        height: number | null;
                    }, {
                        url: string | null;
                        width: number | null;
                        height: number | null;
                    }>, "many">>;
                }, "strip", z.ZodTypeAny, {
                    type: string | null;
                    url: string | null;
                    images: {
                        url: string | null;
                        width: number | null;
                        height: number | null;
                    }[] | null;
                    thumbnail: string | null;
                }, {
                    type: string | null;
                    url: string | null;
                    images: {
                        url: string | null;
                        width: number | null;
                        height: number | null;
                    }[] | null;
                    thumbnail: string | null;
                }>>;
            }, "strip", z.ZodTypeAny, {
                url: string | null;
                text: string | null;
                urn: string | null;
                author: {
                    username: string | null;
                    headline: string | null;
                    profileUrl: string | null;
                    profilePicture: string | null;
                    firstName: string | null;
                    lastName: string | null;
                } | null;
                stats: {
                    like: number | null;
                    support: number | null;
                    love: number | null;
                    insight: number | null;
                    celebrate: number | null;
                    funny: number | null;
                    comments: number | null;
                    reposts: number | null;
                    totalReactions: number | null;
                } | null;
                media: {
                    type: string | null;
                    url: string | null;
                    images: {
                        url: string | null;
                        width: number | null;
                        height: number | null;
                    }[] | null;
                    thumbnail: string | null;
                } | null;
                postedAt: {
                    date: string | null;
                    timestamp: number | null;
                    relative: string | null;
                } | null;
                postType: string | null;
            }, {
                url: string | null;
                text: string | null;
                urn: string | null;
                author: {
                    username: string | null;
                    headline: string | null;
                    profileUrl: string | null;
                    profilePicture: string | null;
                    firstName: string | null;
                    lastName: string | null;
                } | null;
                stats: {
                    like: number | null;
                    support: number | null;
                    love: number | null;
                    insight: number | null;
                    celebrate: number | null;
                    funny: number | null;
                    comments: number | null;
                    reposts: number | null;
                    totalReactions: number | null;
                } | null;
                media: {
                    type: string | null;
                    url: string | null;
                    images: {
                        url: string | null;
                        width: number | null;
                        height: number | null;
                    }[] | null;
                    thumbnail: string | null;
                } | null;
                postedAt: {
                    date: string | null;
                    timestamp: number | null;
                    relative: string | null;
                } | null;
                postType: string | null;
            }>>;
        }, "strip", z.ZodTypeAny, {
            url: string | null;
            text: string | null;
            urn: string | null;
            author: {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            } | null;
            stats: {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            } | null;
            article: {
                title: string | null;
                url: string | null;
                subtitle: string | null;
                thumbnail: string | null;
            } | null;
            document: {
                title: string | null;
                url: string | null;
                pageCount: number | null;
                thumbnail: string | null;
            } | null;
            postedAt: {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            } | null;
            fullUrn: string | null;
            postType: string | null;
            resharedPost: {
                url: string | null;
                text: string | null;
                urn: string | null;
                author: {
                    username: string | null;
                    headline: string | null;
                    profileUrl: string | null;
                    profilePicture: string | null;
                    firstName: string | null;
                    lastName: string | null;
                } | null;
                stats: {
                    like: number | null;
                    support: number | null;
                    love: number | null;
                    insight: number | null;
                    celebrate: number | null;
                    funny: number | null;
                    comments: number | null;
                    reposts: number | null;
                    totalReactions: number | null;
                } | null;
                media: {
                    type: string | null;
                    url: string | null;
                    images: {
                        url: string | null;
                        width: number | null;
                        height: number | null;
                    }[] | null;
                    thumbnail: string | null;
                } | null;
                postedAt: {
                    date: string | null;
                    timestamp: number | null;
                    relative: string | null;
                } | null;
                postType: string | null;
            } | null;
        }, {
            url: string | null;
            text: string | null;
            urn: string | null;
            author: {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            } | null;
            stats: {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            } | null;
            article: {
                title: string | null;
                url: string | null;
                subtitle: string | null;
                thumbnail: string | null;
            } | null;
            document: {
                title: string | null;
                url: string | null;
                pageCount: number | null;
                thumbnail: string | null;
            } | null;
            postedAt: {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            } | null;
            fullUrn: string | null;
            postType: string | null;
            resharedPost: {
                url: string | null;
                text: string | null;
                urn: string | null;
                author: {
                    username: string | null;
                    headline: string | null;
                    profileUrl: string | null;
                    profilePicture: string | null;
                    firstName: string | null;
                    lastName: string | null;
                } | null;
                stats: {
                    like: number | null;
                    support: number | null;
                    love: number | null;
                    insight: number | null;
                    celebrate: number | null;
                    funny: number | null;
                    comments: number | null;
                    reposts: number | null;
                    totalReactions: number | null;
                } | null;
                media: {
                    type: string | null;
                    url: string | null;
                    images: {
                        url: string | null;
                        width: number | null;
                        height: number | null;
                    }[] | null;
                    thumbnail: string | null;
                } | null;
                postedAt: {
                    date: string | null;
                    timestamp: number | null;
                    relative: string | null;
                } | null;
                postType: string | null;
            } | null;
        }>, "many">;
        username: z.ZodOptional<z.ZodString>;
        paginationToken: z.ZodOptional<z.ZodNullable<z.ZodString>>;
        keyword: z.ZodOptional<z.ZodString>;
        totalResults: z.ZodOptional<z.ZodNullable<z.ZodNumber>>;
        hasNextPage: z.ZodOptional<z.ZodNullable<z.ZodBoolean>>;
        totalPosts: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "scrapePosts" | "searchPosts" | "scrapeJobs";
        posts: {
            url: string | null;
            text: string | null;
            urn: string | null;
            author: {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            } | null;
            stats: {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            } | null;
            article: {
                title: string | null;
                url: string | null;
                subtitle: string | null;
                thumbnail: string | null;
            } | null;
            document: {
                title: string | null;
                url: string | null;
                pageCount: number | null;
                thumbnail: string | null;
            } | null;
            postedAt: {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            } | null;
            fullUrn: string | null;
            postType: string | null;
            resharedPost: {
                url: string | null;
                text: string | null;
                urn: string | null;
                author: {
                    username: string | null;
                    headline: string | null;
                    profileUrl: string | null;
                    profilePicture: string | null;
                    firstName: string | null;
                    lastName: string | null;
                } | null;
                stats: {
                    like: number | null;
                    support: number | null;
                    love: number | null;
                    insight: number | null;
                    celebrate: number | null;
                    funny: number | null;
                    comments: number | null;
                    reposts: number | null;
                    totalReactions: number | null;
                } | null;
                media: {
                    type: string | null;
                    url: string | null;
                    images: {
                        url: string | null;
                        width: number | null;
                        height: number | null;
                    }[] | null;
                    thumbnail: string | null;
                } | null;
                postedAt: {
                    date: string | null;
                    timestamp: number | null;
                    relative: string | null;
                } | null;
                postType: string | null;
            } | null;
        }[];
        totalPosts: number;
        username?: string | undefined;
        keyword?: string | undefined;
        jobs?: {
            description: string | null;
            title: string | null;
            url: string | null;
            id: string | null;
            location: string | null;
            postedAt: string | null;
            applyUrl: string | null;
            salary: {
                from: number | null;
                to: number | null;
                currency: string | null;
                period: string | null;
            } | null;
            seniorityLevel: string | null;
            employmentType: string | null;
            company: {
                name: string | null;
                url: string | null;
                logo: string | null;
            } | null;
            skills: string[] | null;
        }[] | undefined;
        totalResults?: number | null | undefined;
        paginationToken?: string | null | undefined;
        hasNextPage?: boolean | null | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "scrapePosts" | "searchPosts" | "scrapeJobs";
        posts: {
            url: string | null;
            text: string | null;
            urn: string | null;
            author: {
                username: string | null;
                headline: string | null;
                profileUrl: string | null;
                profilePicture: string | null;
                firstName: string | null;
                lastName: string | null;
            } | null;
            stats: {
                like: number | null;
                support: number | null;
                love: number | null;
                insight: number | null;
                celebrate: number | null;
                funny: number | null;
                comments: number | null;
                reposts: number | null;
                totalReactions: number | null;
            } | null;
            media: {
                type: string | null;
                url: string | null;
                images: {
                    url: string | null;
                    width: number | null;
                    height: number | null;
                }[] | null;
                thumbnail: string | null;
            } | null;
            article: {
                title: string | null;
                url: string | null;
                subtitle: string | null;
                thumbnail: string | null;
            } | null;
            document: {
                title: string | null;
                url: string | null;
                pageCount: number | null;
                thumbnail: string | null;
            } | null;
            postedAt: {
                date: string | null;
                timestamp: number | null;
                relative: string | null;
            } | null;
            fullUrn: string | null;
            postType: string | null;
            resharedPost: {
                url: string | null;
                text: string | null;
                urn: string | null;
                author: {
                    username: string | null;
                    headline: string | null;
                    profileUrl: string | null;
                    profilePicture: string | null;
                    firstName: string | null;
                    lastName: string | null;
                } | null;
                stats: {
                    like: number | null;
                    support: number | null;
                    love: number | null;
                    insight: number | null;
                    celebrate: number | null;
                    funny: number | null;
                    comments: number | null;
                    reposts: number | null;
                    totalReactions: number | null;
                } | null;
                media: {
                    type: string | null;
                    url: string | null;
                    images: {
                        url: string | null;
                        width: number | null;
                        height: number | null;
                    }[] | null;
                    thumbnail: string | null;
                } | null;
                postedAt: {
                    date: string | null;
                    timestamp: number | null;
                    relative: string | null;
                } | null;
                postType: string | null;
            } | null;
        }[];
        totalPosts: number;
        username?: string | undefined;
        keyword?: string | undefined;
        jobs?: {
            description: string | null;
            title: string | null;
            url: string | null;
            id: string | null;
            location: string | null;
            postedAt: string | null;
            applyUrl: string | null;
            salary: {
                from: number | null;
                to: number | null;
                currency: string | null;
                period: string | null;
            } | null;
            seniorityLevel: string | null;
            employmentType: string | null;
            company: {
                name: string | null;
                url: string | null;
                logo: string | null;
            } | null;
            skills: string[] | null;
        }[] | undefined;
        totalResults?: number | null | undefined;
        paginationToken?: string | null | undefined;
        hasNextPage?: boolean | null | undefined;
    }>;
    static readonly shortDescription = "Scrape LinkedIn posts by profile or search by keyword. Get engagement metrics, media, and complete metadata.";
    static readonly longDescription = "\n    Universal LinkedIn scraping tool for extracting posts and activity data.\n    \n    **OPERATIONS:**\n    1. **scrapePosts**: Scrape posts from a LinkedIn profile\n       - Get posts from specific users\n       - Extract author information and post metadata\n       - Track reactions, comments, and reposts\n       - Support for articles, documents, and reshared content\n    \n    2. **searchPosts**: Search LinkedIn posts by keyword\n       - Find posts across all of LinkedIn by keyword\n       - Filter by date (past 24h, week, month)\n       - Sort by relevance or date\n       - Perfect for monitoring topics, trends, and mentions\n    \n    **WHEN TO USE THIS TOOL:**\n    - **LinkedIn profile research** - analyze someone's LinkedIn activity\n    - **Content strategy** - research what content performs well\n    - **Influencer analysis** - track thought leaders and their engagement\n    - **Competitive intelligence** - monitor competitor LinkedIn presence\n    - **Lead generation** - identify active LinkedIn users in your space\n    - **Social listening** - track discussions and trends on LinkedIn\n    - **Job Market Analysis** - scrape job postings and salary data\n    \n    **DO NOT USE research-agent-tool or web-scrape-tool for LinkedIn** - This tool is specifically optimized for LinkedIn and provides:\n    - Clean, structured post data ready for analysis\n    - Complete engagement metrics (reactions, comments, reposts)\n    - Support for all LinkedIn post types\n    - Automatic pagination handling\n    - Rate limiting and reliability\n    \n    **Simple Interface:**\n    Just provide a LinkedIn username to get back all their recent posts with complete metadata.\n    The tool automatically handles:\n    - Authentication with Apify\n    - Data transformation to unified format\n    - Error handling and retries\n    - Pagination token management\n    \n    **What you get:**\n    - Post text and metadata (URN, URL, type, timestamp)\n    - Complete engagement statistics (likes, comments, reposts, all reaction types)\n    - Author information (name, headline, profile URL, picture)\n    - Media content (images, videos, documents, articles)\n    - Reshared post data (for quote posts)\n    \n    **Use cases:**\n    - Influencer and thought leader tracking\n    - Content performance analysis\n    - Competitive research on LinkedIn\n    - Lead generation and prospecting\n    - Brand monitoring and reputation management\n    - Recruitment and talent sourcing\n    - Partnership and collaboration discovery\n    - Job market research and salary analysis\n\n    The tool uses Apify's LinkedIn scrapers behind the scenes while maintaining a clean, consistent interface.\n  ";
    static readonly alias = "li";
    static readonly type = "tool";
    constructor(params?: LinkedInToolParamsInput, context?: BubbleContext);
    performAction(): Promise<LinkedInToolResult>;
    /**
     * Create an error result
     */
    private createErrorResult;
    /**
     * Handle scrapePosts operation
     */
    private handleScrapePosts;
    /**
     * Transform LinkedIn posts from Apify format to unified format
     */
    private transformPosts;
    /**
     * Handle searchPosts operation
     */
    private handleSearchPosts;
    /**
     * Transform search results to unified post format
     */
    private transformSearchResults;
    /**
     * Helper to get reaction count by type from reactions array
     */
    private getReactionCount;
    /**
     * Handle scrapeJobs operation
     */
    private handleScrapeJobs;
    private transformJobs;
}
export {};
//# sourceMappingURL=linkedin-tool.d.ts.map