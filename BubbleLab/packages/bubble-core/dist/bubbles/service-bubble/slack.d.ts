import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const SlackParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"send_message">;
    channel: z.ZodString;
    text: z.ZodString;
    username: z.ZodOptional<z.ZodString>;
    icon_emoji: z.ZodOptional<z.ZodString>;
    icon_url: z.ZodOptional<z.ZodString>;
    attachments: z.ZodOptional<z.ZodArray<z.ZodObject<{
        color: z.ZodOptional<z.ZodString>;
        pretext: z.ZodOptional<z.ZodString>;
        author_name: z.ZodOptional<z.ZodString>;
        author_link: z.ZodOptional<z.ZodString>;
        author_icon: z.ZodOptional<z.ZodString>;
        title: z.ZodOptional<z.ZodString>;
        title_link: z.ZodOptional<z.ZodString>;
        text: z.ZodOptional<z.ZodString>;
        fields: z.ZodOptional<z.ZodArray<z.ZodObject<{
            title: z.ZodString;
            value: z.ZodString;
            short: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            title: string;
            short?: boolean | undefined;
        }, {
            value: string;
            title: string;
            short?: boolean | undefined;
        }>, "many">>;
        image_url: z.ZodOptional<z.ZodString>;
        thumb_url: z.ZodOptional<z.ZodString>;
        footer: z.ZodOptional<z.ZodString>;
        footer_icon: z.ZodOptional<z.ZodString>;
        ts: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        title?: string | undefined;
        fields?: {
            value: string;
            title: string;
            short?: boolean | undefined;
        }[] | undefined;
        text?: string | undefined;
        color?: string | undefined;
        pretext?: string | undefined;
        author_name?: string | undefined;
        author_link?: string | undefined;
        author_icon?: string | undefined;
        title_link?: string | undefined;
        image_url?: string | undefined;
        thumb_url?: string | undefined;
        footer?: string | undefined;
        footer_icon?: string | undefined;
        ts?: number | undefined;
    }, {
        title?: string | undefined;
        fields?: {
            value: string;
            title: string;
            short?: boolean | undefined;
        }[] | undefined;
        text?: string | undefined;
        color?: string | undefined;
        pretext?: string | undefined;
        author_name?: string | undefined;
        author_link?: string | undefined;
        author_icon?: string | undefined;
        title_link?: string | undefined;
        image_url?: string | undefined;
        thumb_url?: string | undefined;
        footer?: string | undefined;
        footer_icon?: string | undefined;
        ts?: number | undefined;
    }>, "many">>;
    blocks: z.ZodOptional<z.ZodArray<z.ZodObject<{
        type: z.ZodString;
        text: z.ZodOptional<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>>;
        elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
            text: z.ZodOptional<z.ZodString>;
            image_url: z.ZodOptional<z.ZodString>;
            alt_text: z.ZodOptional<z.ZodString>;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }>, "many">>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        type: z.ZodString;
        text: z.ZodOptional<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>>;
        elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
            text: z.ZodOptional<z.ZodString>;
            image_url: z.ZodOptional<z.ZodString>;
            alt_text: z.ZodOptional<z.ZodString>;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }>, "many">>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        type: z.ZodString;
        text: z.ZodOptional<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>>;
        elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
            text: z.ZodOptional<z.ZodString>;
            image_url: z.ZodOptional<z.ZodString>;
            alt_text: z.ZodOptional<z.ZodString>;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }>, "many">>;
    }, z.ZodTypeAny, "passthrough">>, "many">>;
    thread_ts: z.ZodOptional<z.ZodString>;
    reply_broadcast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    unfurl_links: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    unfurl_media: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
}, "strip", z.ZodTypeAny, {
    operation: "send_message";
    channel: string;
    text: string;
    reply_broadcast: boolean;
    unfurl_links: boolean;
    unfurl_media: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    username?: string | undefined;
    icon_emoji?: string | undefined;
    icon_url?: string | undefined;
    attachments?: {
        title?: string | undefined;
        fields?: {
            value: string;
            title: string;
            short?: boolean | undefined;
        }[] | undefined;
        text?: string | undefined;
        color?: string | undefined;
        pretext?: string | undefined;
        author_name?: string | undefined;
        author_link?: string | undefined;
        author_icon?: string | undefined;
        title_link?: string | undefined;
        image_url?: string | undefined;
        thumb_url?: string | undefined;
        footer?: string | undefined;
        footer_icon?: string | undefined;
        ts?: number | undefined;
    }[] | undefined;
    blocks?: z.objectOutputType<{
        type: z.ZodString;
        text: z.ZodOptional<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>>;
        elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
            text: z.ZodOptional<z.ZodString>;
            image_url: z.ZodOptional<z.ZodString>;
            alt_text: z.ZodOptional<z.ZodString>;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }>, "many">>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
    thread_ts?: string | undefined;
}, {
    operation: "send_message";
    channel: string;
    text: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    username?: string | undefined;
    icon_emoji?: string | undefined;
    icon_url?: string | undefined;
    attachments?: {
        title?: string | undefined;
        fields?: {
            value: string;
            title: string;
            short?: boolean | undefined;
        }[] | undefined;
        text?: string | undefined;
        color?: string | undefined;
        pretext?: string | undefined;
        author_name?: string | undefined;
        author_link?: string | undefined;
        author_icon?: string | undefined;
        title_link?: string | undefined;
        image_url?: string | undefined;
        thumb_url?: string | undefined;
        footer?: string | undefined;
        footer_icon?: string | undefined;
        ts?: number | undefined;
    }[] | undefined;
    blocks?: z.objectInputType<{
        type: z.ZodString;
        text: z.ZodOptional<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>>;
        elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
            text: z.ZodOptional<z.ZodString>;
            image_url: z.ZodOptional<z.ZodString>;
            alt_text: z.ZodOptional<z.ZodString>;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }>, "many">>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
    thread_ts?: string | undefined;
    reply_broadcast?: boolean | undefined;
    unfurl_links?: boolean | undefined;
    unfurl_media?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_channels">;
    types: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodEnum<["public_channel", "private_channel", "mpim", "im"]>, "many">>>;
    exclude_archived: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    cursor: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_channels";
    types: ("public_channel" | "private_channel" | "mpim" | "im")[];
    exclude_archived: boolean;
    limit: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cursor?: string | undefined;
}, {
    operation: "list_channels";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    types?: ("public_channel" | "private_channel" | "mpim" | "im")[] | undefined;
    exclude_archived?: boolean | undefined;
    limit?: number | undefined;
    cursor?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_channel_info">;
    channel: z.ZodString;
    include_locale: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_channel_info";
    channel: string;
    include_locale: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_channel_info";
    channel: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    include_locale?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_user_info">;
    user: z.ZodString;
    include_locale: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_user_info";
    include_locale: boolean;
    user: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_user_info";
    user: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    include_locale?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_users">;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    cursor: z.ZodOptional<z.ZodString>;
    include_locale: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_users";
    limit: number;
    include_locale: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cursor?: string | undefined;
}, {
    operation: "list_users";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    cursor?: string | undefined;
    include_locale?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_conversation_history">;
    channel: z.ZodString;
    latest: z.ZodOptional<z.ZodString>;
    oldest: z.ZodOptional<z.ZodString>;
    inclusive: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    cursor: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    inclusive: boolean;
    operation: "get_conversation_history";
    channel: string;
    limit: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cursor?: string | undefined;
    latest?: string | undefined;
    oldest?: string | undefined;
}, {
    operation: "get_conversation_history";
    channel: string;
    inclusive?: boolean | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    cursor?: string | undefined;
    latest?: string | undefined;
    oldest?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_thread_replies">;
    channel: z.ZodString;
    ts: z.ZodString;
    latest: z.ZodOptional<z.ZodString>;
    oldest: z.ZodOptional<z.ZodString>;
    inclusive: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    cursor: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    inclusive: boolean;
    operation: "get_thread_replies";
    channel: string;
    ts: string;
    limit: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cursor?: string | undefined;
    latest?: string | undefined;
    oldest?: string | undefined;
}, {
    operation: "get_thread_replies";
    channel: string;
    ts: string;
    inclusive?: boolean | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    cursor?: string | undefined;
    latest?: string | undefined;
    oldest?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_message">;
    channel: z.ZodString;
    ts: z.ZodString;
    text: z.ZodOptional<z.ZodString>;
    attachments: z.ZodOptional<z.ZodArray<z.ZodObject<{
        color: z.ZodOptional<z.ZodString>;
        pretext: z.ZodOptional<z.ZodString>;
        author_name: z.ZodOptional<z.ZodString>;
        author_link: z.ZodOptional<z.ZodString>;
        author_icon: z.ZodOptional<z.ZodString>;
        title: z.ZodOptional<z.ZodString>;
        title_link: z.ZodOptional<z.ZodString>;
        text: z.ZodOptional<z.ZodString>;
        fields: z.ZodOptional<z.ZodArray<z.ZodObject<{
            title: z.ZodString;
            value: z.ZodString;
            short: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            value: string;
            title: string;
            short?: boolean | undefined;
        }, {
            value: string;
            title: string;
            short?: boolean | undefined;
        }>, "many">>;
        image_url: z.ZodOptional<z.ZodString>;
        thumb_url: z.ZodOptional<z.ZodString>;
        footer: z.ZodOptional<z.ZodString>;
        footer_icon: z.ZodOptional<z.ZodString>;
        ts: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        title?: string | undefined;
        fields?: {
            value: string;
            title: string;
            short?: boolean | undefined;
        }[] | undefined;
        text?: string | undefined;
        color?: string | undefined;
        pretext?: string | undefined;
        author_name?: string | undefined;
        author_link?: string | undefined;
        author_icon?: string | undefined;
        title_link?: string | undefined;
        image_url?: string | undefined;
        thumb_url?: string | undefined;
        footer?: string | undefined;
        footer_icon?: string | undefined;
        ts?: number | undefined;
    }, {
        title?: string | undefined;
        fields?: {
            value: string;
            title: string;
            short?: boolean | undefined;
        }[] | undefined;
        text?: string | undefined;
        color?: string | undefined;
        pretext?: string | undefined;
        author_name?: string | undefined;
        author_link?: string | undefined;
        author_icon?: string | undefined;
        title_link?: string | undefined;
        image_url?: string | undefined;
        thumb_url?: string | undefined;
        footer?: string | undefined;
        footer_icon?: string | undefined;
        ts?: number | undefined;
    }>, "many">>;
    blocks: z.ZodOptional<z.ZodArray<z.ZodObject<{
        type: z.ZodString;
        text: z.ZodOptional<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>>;
        elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
            text: z.ZodOptional<z.ZodString>;
            image_url: z.ZodOptional<z.ZodString>;
            alt_text: z.ZodOptional<z.ZodString>;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }>, "many">>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        type: z.ZodString;
        text: z.ZodOptional<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>>;
        elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
            text: z.ZodOptional<z.ZodString>;
            image_url: z.ZodOptional<z.ZodString>;
            alt_text: z.ZodOptional<z.ZodString>;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }>, "many">>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        type: z.ZodString;
        text: z.ZodOptional<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>>;
        elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
            text: z.ZodOptional<z.ZodString>;
            image_url: z.ZodOptional<z.ZodString>;
            alt_text: z.ZodOptional<z.ZodString>;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }>, "many">>;
    }, z.ZodTypeAny, "passthrough">>, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "update_message";
    channel: string;
    ts: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    text?: string | undefined;
    attachments?: {
        title?: string | undefined;
        fields?: {
            value: string;
            title: string;
            short?: boolean | undefined;
        }[] | undefined;
        text?: string | undefined;
        color?: string | undefined;
        pretext?: string | undefined;
        author_name?: string | undefined;
        author_link?: string | undefined;
        author_icon?: string | undefined;
        title_link?: string | undefined;
        image_url?: string | undefined;
        thumb_url?: string | undefined;
        footer?: string | undefined;
        footer_icon?: string | undefined;
        ts?: number | undefined;
    }[] | undefined;
    blocks?: z.objectOutputType<{
        type: z.ZodString;
        text: z.ZodOptional<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>>;
        elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
            text: z.ZodOptional<z.ZodString>;
            image_url: z.ZodOptional<z.ZodString>;
            alt_text: z.ZodOptional<z.ZodString>;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }>, "many">>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}, {
    operation: "update_message";
    channel: string;
    ts: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    text?: string | undefined;
    attachments?: {
        title?: string | undefined;
        fields?: {
            value: string;
            title: string;
            short?: boolean | undefined;
        }[] | undefined;
        text?: string | undefined;
        color?: string | undefined;
        pretext?: string | undefined;
        author_name?: string | undefined;
        author_link?: string | undefined;
        author_icon?: string | undefined;
        title_link?: string | undefined;
        image_url?: string | undefined;
        thumb_url?: string | undefined;
        footer?: string | undefined;
        footer_icon?: string | undefined;
        ts?: number | undefined;
    }[] | undefined;
    blocks?: z.objectInputType<{
        type: z.ZodString;
        text: z.ZodOptional<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn"]>;
            text: z.ZodString;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }, {
            type: "plain_text" | "mrkdwn";
            text: string;
            emoji?: boolean | undefined;
            verbatim?: boolean | undefined;
        }>>;
        elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
            text: z.ZodOptional<z.ZodString>;
            image_url: z.ZodOptional<z.ZodString>;
            alt_text: z.ZodOptional<z.ZodString>;
            emoji: z.ZodOptional<z.ZodBoolean>;
            verbatim: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }, {
            type: "plain_text" | "mrkdwn" | "image";
            emoji?: boolean | undefined;
            text?: string | undefined;
            image_url?: string | undefined;
            verbatim?: boolean | undefined;
            alt_text?: string | undefined;
        }>, "many">>;
    }, z.ZodTypeAny, "passthrough">[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_message">;
    channel: z.ZodString;
    ts: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_message";
    channel: string;
    ts: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "delete_message";
    channel: string;
    ts: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"add_reaction">;
    name: z.ZodString;
    channel: z.ZodString;
    timestamp: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    operation: "add_reaction";
    channel: string;
    timestamp: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    name: string;
    operation: "add_reaction";
    channel: string;
    timestamp: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"remove_reaction">;
    name: z.ZodString;
    channel: z.ZodString;
    timestamp: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    operation: "remove_reaction";
    channel: string;
    timestamp: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    name: string;
    operation: "remove_reaction";
    channel: string;
    timestamp: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"join_channel">;
    channel: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "join_channel";
    channel: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "join_channel";
    channel: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"upload_file">;
    channel: z.ZodString;
    file_path: z.ZodEffects<z.ZodString, string, string>;
    filename: z.ZodOptional<z.ZodString>;
    title: z.ZodOptional<z.ZodString>;
    initial_comment: z.ZodOptional<z.ZodString>;
    thread_ts: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "upload_file";
    channel: string;
    file_path: string;
    title?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    thread_ts?: string | undefined;
    filename?: string | undefined;
    initial_comment?: string | undefined;
}, {
    operation: "upload_file";
    channel: string;
    file_path: string;
    title?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    thread_ts?: string | undefined;
    filename?: string | undefined;
    initial_comment?: string | undefined;
}>]>;
declare const SlackResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"send_message">;
    ok: z.ZodBoolean;
    channel: z.ZodOptional<z.ZodString>;
    ts: z.ZodOptional<z.ZodString>;
    message: z.ZodOptional<z.ZodObject<{
        type: z.ZodString;
        ts: z.ZodString;
        user: z.ZodOptional<z.ZodString>;
        bot_id: z.ZodOptional<z.ZodString>;
        bot_profile: z.ZodOptional<z.ZodObject<{
            name: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name?: string | undefined;
        }, {
            name?: string | undefined;
        }>>;
        username: z.ZodOptional<z.ZodString>;
        text: z.ZodOptional<z.ZodString>;
        thread_ts: z.ZodOptional<z.ZodString>;
        parent_user_id: z.ZodOptional<z.ZodString>;
        reply_count: z.ZodOptional<z.ZodNumber>;
        reply_users_count: z.ZodOptional<z.ZodNumber>;
        latest_reply: z.ZodOptional<z.ZodString>;
        reply_users: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        is_locked: z.ZodOptional<z.ZodBoolean>;
        subscribed: z.ZodOptional<z.ZodBoolean>;
        attachments: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        reactions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            users: z.ZodArray<z.ZodString, "many">;
            count: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            name: string;
            users: string[];
            count: number;
        }, {
            name: string;
            users: string[];
            count: number;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    }, {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "send_message";
    ok: boolean;
    message?: {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    } | undefined;
    channel?: string | undefined;
    ts?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "send_message";
    ok: boolean;
    message?: {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    } | undefined;
    channel?: string | undefined;
    ts?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_channels">;
    ok: z.ZodBoolean;
    channels: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        is_channel: z.ZodOptional<z.ZodBoolean>;
        is_group: z.ZodOptional<z.ZodBoolean>;
        is_im: z.ZodOptional<z.ZodBoolean>;
        is_mpim: z.ZodOptional<z.ZodBoolean>;
        is_private: z.ZodOptional<z.ZodBoolean>;
        created: z.ZodNumber;
        is_archived: z.ZodBoolean;
        is_general: z.ZodOptional<z.ZodBoolean>;
        unlinked: z.ZodOptional<z.ZodNumber>;
        name_normalized: z.ZodOptional<z.ZodString>;
        is_shared: z.ZodOptional<z.ZodBoolean>;
        is_ext_shared: z.ZodOptional<z.ZodBoolean>;
        is_org_shared: z.ZodOptional<z.ZodBoolean>;
        shared_team_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        pending_shared: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        pending_connected_team_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        is_pending_ext_shared: z.ZodOptional<z.ZodBoolean>;
        is_member: z.ZodOptional<z.ZodBoolean>;
        is_open: z.ZodOptional<z.ZodBoolean>;
        topic: z.ZodOptional<z.ZodObject<{
            value: z.ZodString;
            creator: z.ZodString;
            last_set: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            value: string;
            creator: string;
            last_set: number;
        }, {
            value: string;
            creator: string;
            last_set: number;
        }>>;
        purpose: z.ZodOptional<z.ZodObject<{
            value: z.ZodString;
            creator: z.ZodString;
            last_set: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            value: string;
            creator: string;
            last_set: number;
        }, {
            value: string;
            creator: string;
            last_set: number;
        }>>;
        num_members: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        created: number;
        is_archived: boolean;
        is_channel?: boolean | undefined;
        is_group?: boolean | undefined;
        is_im?: boolean | undefined;
        is_mpim?: boolean | undefined;
        is_private?: boolean | undefined;
        is_general?: boolean | undefined;
        unlinked?: number | undefined;
        name_normalized?: string | undefined;
        is_shared?: boolean | undefined;
        is_ext_shared?: boolean | undefined;
        is_org_shared?: boolean | undefined;
        shared_team_ids?: string[] | undefined;
        pending_shared?: string[] | undefined;
        pending_connected_team_ids?: string[] | undefined;
        is_pending_ext_shared?: boolean | undefined;
        is_member?: boolean | undefined;
        is_open?: boolean | undefined;
        topic?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        purpose?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        num_members?: number | undefined;
    }, {
        name: string;
        id: string;
        created: number;
        is_archived: boolean;
        is_channel?: boolean | undefined;
        is_group?: boolean | undefined;
        is_im?: boolean | undefined;
        is_mpim?: boolean | undefined;
        is_private?: boolean | undefined;
        is_general?: boolean | undefined;
        unlinked?: number | undefined;
        name_normalized?: string | undefined;
        is_shared?: boolean | undefined;
        is_ext_shared?: boolean | undefined;
        is_org_shared?: boolean | undefined;
        shared_team_ids?: string[] | undefined;
        pending_shared?: string[] | undefined;
        pending_connected_team_ids?: string[] | undefined;
        is_pending_ext_shared?: boolean | undefined;
        is_member?: boolean | undefined;
        is_open?: boolean | undefined;
        topic?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        purpose?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        num_members?: number | undefined;
    }>, "many">>;
    response_metadata: z.ZodOptional<z.ZodObject<{
        next_cursor: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        next_cursor: string;
    }, {
        next_cursor: string;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_channels";
    ok: boolean;
    channels?: {
        name: string;
        id: string;
        created: number;
        is_archived: boolean;
        is_channel?: boolean | undefined;
        is_group?: boolean | undefined;
        is_im?: boolean | undefined;
        is_mpim?: boolean | undefined;
        is_private?: boolean | undefined;
        is_general?: boolean | undefined;
        unlinked?: number | undefined;
        name_normalized?: string | undefined;
        is_shared?: boolean | undefined;
        is_ext_shared?: boolean | undefined;
        is_org_shared?: boolean | undefined;
        shared_team_ids?: string[] | undefined;
        pending_shared?: string[] | undefined;
        pending_connected_team_ids?: string[] | undefined;
        is_pending_ext_shared?: boolean | undefined;
        is_member?: boolean | undefined;
        is_open?: boolean | undefined;
        topic?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        purpose?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        num_members?: number | undefined;
    }[] | undefined;
    response_metadata?: {
        next_cursor: string;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_channels";
    ok: boolean;
    channels?: {
        name: string;
        id: string;
        created: number;
        is_archived: boolean;
        is_channel?: boolean | undefined;
        is_group?: boolean | undefined;
        is_im?: boolean | undefined;
        is_mpim?: boolean | undefined;
        is_private?: boolean | undefined;
        is_general?: boolean | undefined;
        unlinked?: number | undefined;
        name_normalized?: string | undefined;
        is_shared?: boolean | undefined;
        is_ext_shared?: boolean | undefined;
        is_org_shared?: boolean | undefined;
        shared_team_ids?: string[] | undefined;
        pending_shared?: string[] | undefined;
        pending_connected_team_ids?: string[] | undefined;
        is_pending_ext_shared?: boolean | undefined;
        is_member?: boolean | undefined;
        is_open?: boolean | undefined;
        topic?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        purpose?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        num_members?: number | undefined;
    }[] | undefined;
    response_metadata?: {
        next_cursor: string;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_channel_info">;
    ok: z.ZodBoolean;
    channel: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        is_channel: z.ZodOptional<z.ZodBoolean>;
        is_group: z.ZodOptional<z.ZodBoolean>;
        is_im: z.ZodOptional<z.ZodBoolean>;
        is_mpim: z.ZodOptional<z.ZodBoolean>;
        is_private: z.ZodOptional<z.ZodBoolean>;
        created: z.ZodNumber;
        is_archived: z.ZodBoolean;
        is_general: z.ZodOptional<z.ZodBoolean>;
        unlinked: z.ZodOptional<z.ZodNumber>;
        name_normalized: z.ZodOptional<z.ZodString>;
        is_shared: z.ZodOptional<z.ZodBoolean>;
        is_ext_shared: z.ZodOptional<z.ZodBoolean>;
        is_org_shared: z.ZodOptional<z.ZodBoolean>;
        shared_team_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        pending_shared: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        pending_connected_team_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        is_pending_ext_shared: z.ZodOptional<z.ZodBoolean>;
        is_member: z.ZodOptional<z.ZodBoolean>;
        is_open: z.ZodOptional<z.ZodBoolean>;
        topic: z.ZodOptional<z.ZodObject<{
            value: z.ZodString;
            creator: z.ZodString;
            last_set: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            value: string;
            creator: string;
            last_set: number;
        }, {
            value: string;
            creator: string;
            last_set: number;
        }>>;
        purpose: z.ZodOptional<z.ZodObject<{
            value: z.ZodString;
            creator: z.ZodString;
            last_set: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            value: string;
            creator: string;
            last_set: number;
        }, {
            value: string;
            creator: string;
            last_set: number;
        }>>;
        num_members: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        created: number;
        is_archived: boolean;
        is_channel?: boolean | undefined;
        is_group?: boolean | undefined;
        is_im?: boolean | undefined;
        is_mpim?: boolean | undefined;
        is_private?: boolean | undefined;
        is_general?: boolean | undefined;
        unlinked?: number | undefined;
        name_normalized?: string | undefined;
        is_shared?: boolean | undefined;
        is_ext_shared?: boolean | undefined;
        is_org_shared?: boolean | undefined;
        shared_team_ids?: string[] | undefined;
        pending_shared?: string[] | undefined;
        pending_connected_team_ids?: string[] | undefined;
        is_pending_ext_shared?: boolean | undefined;
        is_member?: boolean | undefined;
        is_open?: boolean | undefined;
        topic?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        purpose?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        num_members?: number | undefined;
    }, {
        name: string;
        id: string;
        created: number;
        is_archived: boolean;
        is_channel?: boolean | undefined;
        is_group?: boolean | undefined;
        is_im?: boolean | undefined;
        is_mpim?: boolean | undefined;
        is_private?: boolean | undefined;
        is_general?: boolean | undefined;
        unlinked?: number | undefined;
        name_normalized?: string | undefined;
        is_shared?: boolean | undefined;
        is_ext_shared?: boolean | undefined;
        is_org_shared?: boolean | undefined;
        shared_team_ids?: string[] | undefined;
        pending_shared?: string[] | undefined;
        pending_connected_team_ids?: string[] | undefined;
        is_pending_ext_shared?: boolean | undefined;
        is_member?: boolean | undefined;
        is_open?: boolean | undefined;
        topic?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        purpose?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        num_members?: number | undefined;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_channel_info";
    ok: boolean;
    channel?: {
        name: string;
        id: string;
        created: number;
        is_archived: boolean;
        is_channel?: boolean | undefined;
        is_group?: boolean | undefined;
        is_im?: boolean | undefined;
        is_mpim?: boolean | undefined;
        is_private?: boolean | undefined;
        is_general?: boolean | undefined;
        unlinked?: number | undefined;
        name_normalized?: string | undefined;
        is_shared?: boolean | undefined;
        is_ext_shared?: boolean | undefined;
        is_org_shared?: boolean | undefined;
        shared_team_ids?: string[] | undefined;
        pending_shared?: string[] | undefined;
        pending_connected_team_ids?: string[] | undefined;
        is_pending_ext_shared?: boolean | undefined;
        is_member?: boolean | undefined;
        is_open?: boolean | undefined;
        topic?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        purpose?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        num_members?: number | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_channel_info";
    ok: boolean;
    channel?: {
        name: string;
        id: string;
        created: number;
        is_archived: boolean;
        is_channel?: boolean | undefined;
        is_group?: boolean | undefined;
        is_im?: boolean | undefined;
        is_mpim?: boolean | undefined;
        is_private?: boolean | undefined;
        is_general?: boolean | undefined;
        unlinked?: number | undefined;
        name_normalized?: string | undefined;
        is_shared?: boolean | undefined;
        is_ext_shared?: boolean | undefined;
        is_org_shared?: boolean | undefined;
        shared_team_ids?: string[] | undefined;
        pending_shared?: string[] | undefined;
        pending_connected_team_ids?: string[] | undefined;
        is_pending_ext_shared?: boolean | undefined;
        is_member?: boolean | undefined;
        is_open?: boolean | undefined;
        topic?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        purpose?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        num_members?: number | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_user_info">;
    ok: z.ZodBoolean;
    user: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        team_id: z.ZodOptional<z.ZodString>;
        name: z.ZodString;
        deleted: z.ZodOptional<z.ZodBoolean>;
        color: z.ZodOptional<z.ZodString>;
        real_name: z.ZodOptional<z.ZodString>;
        tz: z.ZodOptional<z.ZodString>;
        tz_label: z.ZodOptional<z.ZodString>;
        tz_offset: z.ZodOptional<z.ZodNumber>;
        profile: z.ZodOptional<z.ZodObject<{
            title: z.ZodOptional<z.ZodString>;
            phone: z.ZodOptional<z.ZodString>;
            skype: z.ZodOptional<z.ZodString>;
            real_name: z.ZodOptional<z.ZodString>;
            real_name_normalized: z.ZodOptional<z.ZodString>;
            display_name: z.ZodOptional<z.ZodString>;
            display_name_normalized: z.ZodOptional<z.ZodString>;
            fields: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            status_text: z.ZodOptional<z.ZodString>;
            status_emoji: z.ZodOptional<z.ZodString>;
            status_expiration: z.ZodOptional<z.ZodNumber>;
            avatar_hash: z.ZodOptional<z.ZodString>;
            image_original: z.ZodOptional<z.ZodString>;
            is_custom_image: z.ZodOptional<z.ZodBoolean>;
            email: z.ZodOptional<z.ZodString>;
            first_name: z.ZodOptional<z.ZodString>;
            last_name: z.ZodOptional<z.ZodString>;
            image_24: z.ZodOptional<z.ZodString>;
            image_32: z.ZodOptional<z.ZodString>;
            image_48: z.ZodOptional<z.ZodString>;
            image_72: z.ZodOptional<z.ZodString>;
            image_192: z.ZodOptional<z.ZodString>;
            image_512: z.ZodOptional<z.ZodString>;
            image_1024: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            title?: string | undefined;
            email?: string | undefined;
            fields?: Record<string, unknown> | undefined;
            real_name?: string | undefined;
            phone?: string | undefined;
            skype?: string | undefined;
            real_name_normalized?: string | undefined;
            display_name?: string | undefined;
            display_name_normalized?: string | undefined;
            status_text?: string | undefined;
            status_emoji?: string | undefined;
            status_expiration?: number | undefined;
            avatar_hash?: string | undefined;
            image_original?: string | undefined;
            is_custom_image?: boolean | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            image_24?: string | undefined;
            image_32?: string | undefined;
            image_48?: string | undefined;
            image_72?: string | undefined;
            image_192?: string | undefined;
            image_512?: string | undefined;
            image_1024?: string | undefined;
        }, {
            title?: string | undefined;
            email?: string | undefined;
            fields?: Record<string, unknown> | undefined;
            real_name?: string | undefined;
            phone?: string | undefined;
            skype?: string | undefined;
            real_name_normalized?: string | undefined;
            display_name?: string | undefined;
            display_name_normalized?: string | undefined;
            status_text?: string | undefined;
            status_emoji?: string | undefined;
            status_expiration?: number | undefined;
            avatar_hash?: string | undefined;
            image_original?: string | undefined;
            is_custom_image?: boolean | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            image_24?: string | undefined;
            image_32?: string | undefined;
            image_48?: string | undefined;
            image_72?: string | undefined;
            image_192?: string | undefined;
            image_512?: string | undefined;
            image_1024?: string | undefined;
        }>>;
        is_admin: z.ZodOptional<z.ZodBoolean>;
        is_owner: z.ZodOptional<z.ZodBoolean>;
        is_primary_owner: z.ZodOptional<z.ZodBoolean>;
        is_restricted: z.ZodOptional<z.ZodBoolean>;
        is_ultra_restricted: z.ZodOptional<z.ZodBoolean>;
        is_bot: z.ZodOptional<z.ZodBoolean>;
        is_app_user: z.ZodOptional<z.ZodBoolean>;
        updated: z.ZodOptional<z.ZodNumber>;
        has_2fa: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        color?: string | undefined;
        team_id?: string | undefined;
        deleted?: boolean | undefined;
        real_name?: string | undefined;
        tz?: string | undefined;
        tz_label?: string | undefined;
        tz_offset?: number | undefined;
        profile?: {
            title?: string | undefined;
            email?: string | undefined;
            fields?: Record<string, unknown> | undefined;
            real_name?: string | undefined;
            phone?: string | undefined;
            skype?: string | undefined;
            real_name_normalized?: string | undefined;
            display_name?: string | undefined;
            display_name_normalized?: string | undefined;
            status_text?: string | undefined;
            status_emoji?: string | undefined;
            status_expiration?: number | undefined;
            avatar_hash?: string | undefined;
            image_original?: string | undefined;
            is_custom_image?: boolean | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            image_24?: string | undefined;
            image_32?: string | undefined;
            image_48?: string | undefined;
            image_72?: string | undefined;
            image_192?: string | undefined;
            image_512?: string | undefined;
            image_1024?: string | undefined;
        } | undefined;
        is_admin?: boolean | undefined;
        is_owner?: boolean | undefined;
        is_primary_owner?: boolean | undefined;
        is_restricted?: boolean | undefined;
        is_ultra_restricted?: boolean | undefined;
        is_bot?: boolean | undefined;
        is_app_user?: boolean | undefined;
        updated?: number | undefined;
        has_2fa?: boolean | undefined;
    }, {
        name: string;
        id: string;
        color?: string | undefined;
        team_id?: string | undefined;
        deleted?: boolean | undefined;
        real_name?: string | undefined;
        tz?: string | undefined;
        tz_label?: string | undefined;
        tz_offset?: number | undefined;
        profile?: {
            title?: string | undefined;
            email?: string | undefined;
            fields?: Record<string, unknown> | undefined;
            real_name?: string | undefined;
            phone?: string | undefined;
            skype?: string | undefined;
            real_name_normalized?: string | undefined;
            display_name?: string | undefined;
            display_name_normalized?: string | undefined;
            status_text?: string | undefined;
            status_emoji?: string | undefined;
            status_expiration?: number | undefined;
            avatar_hash?: string | undefined;
            image_original?: string | undefined;
            is_custom_image?: boolean | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            image_24?: string | undefined;
            image_32?: string | undefined;
            image_48?: string | undefined;
            image_72?: string | undefined;
            image_192?: string | undefined;
            image_512?: string | undefined;
            image_1024?: string | undefined;
        } | undefined;
        is_admin?: boolean | undefined;
        is_owner?: boolean | undefined;
        is_primary_owner?: boolean | undefined;
        is_restricted?: boolean | undefined;
        is_ultra_restricted?: boolean | undefined;
        is_bot?: boolean | undefined;
        is_app_user?: boolean | undefined;
        updated?: number | undefined;
        has_2fa?: boolean | undefined;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_user_info";
    ok: boolean;
    user?: {
        name: string;
        id: string;
        color?: string | undefined;
        team_id?: string | undefined;
        deleted?: boolean | undefined;
        real_name?: string | undefined;
        tz?: string | undefined;
        tz_label?: string | undefined;
        tz_offset?: number | undefined;
        profile?: {
            title?: string | undefined;
            email?: string | undefined;
            fields?: Record<string, unknown> | undefined;
            real_name?: string | undefined;
            phone?: string | undefined;
            skype?: string | undefined;
            real_name_normalized?: string | undefined;
            display_name?: string | undefined;
            display_name_normalized?: string | undefined;
            status_text?: string | undefined;
            status_emoji?: string | undefined;
            status_expiration?: number | undefined;
            avatar_hash?: string | undefined;
            image_original?: string | undefined;
            is_custom_image?: boolean | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            image_24?: string | undefined;
            image_32?: string | undefined;
            image_48?: string | undefined;
            image_72?: string | undefined;
            image_192?: string | undefined;
            image_512?: string | undefined;
            image_1024?: string | undefined;
        } | undefined;
        is_admin?: boolean | undefined;
        is_owner?: boolean | undefined;
        is_primary_owner?: boolean | undefined;
        is_restricted?: boolean | undefined;
        is_ultra_restricted?: boolean | undefined;
        is_bot?: boolean | undefined;
        is_app_user?: boolean | undefined;
        updated?: number | undefined;
        has_2fa?: boolean | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_user_info";
    ok: boolean;
    user?: {
        name: string;
        id: string;
        color?: string | undefined;
        team_id?: string | undefined;
        deleted?: boolean | undefined;
        real_name?: string | undefined;
        tz?: string | undefined;
        tz_label?: string | undefined;
        tz_offset?: number | undefined;
        profile?: {
            title?: string | undefined;
            email?: string | undefined;
            fields?: Record<string, unknown> | undefined;
            real_name?: string | undefined;
            phone?: string | undefined;
            skype?: string | undefined;
            real_name_normalized?: string | undefined;
            display_name?: string | undefined;
            display_name_normalized?: string | undefined;
            status_text?: string | undefined;
            status_emoji?: string | undefined;
            status_expiration?: number | undefined;
            avatar_hash?: string | undefined;
            image_original?: string | undefined;
            is_custom_image?: boolean | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            image_24?: string | undefined;
            image_32?: string | undefined;
            image_48?: string | undefined;
            image_72?: string | undefined;
            image_192?: string | undefined;
            image_512?: string | undefined;
            image_1024?: string | undefined;
        } | undefined;
        is_admin?: boolean | undefined;
        is_owner?: boolean | undefined;
        is_primary_owner?: boolean | undefined;
        is_restricted?: boolean | undefined;
        is_ultra_restricted?: boolean | undefined;
        is_bot?: boolean | undefined;
        is_app_user?: boolean | undefined;
        updated?: number | undefined;
        has_2fa?: boolean | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_users">;
    ok: z.ZodBoolean;
    members: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        team_id: z.ZodOptional<z.ZodString>;
        name: z.ZodString;
        deleted: z.ZodOptional<z.ZodBoolean>;
        color: z.ZodOptional<z.ZodString>;
        real_name: z.ZodOptional<z.ZodString>;
        tz: z.ZodOptional<z.ZodString>;
        tz_label: z.ZodOptional<z.ZodString>;
        tz_offset: z.ZodOptional<z.ZodNumber>;
        profile: z.ZodOptional<z.ZodObject<{
            title: z.ZodOptional<z.ZodString>;
            phone: z.ZodOptional<z.ZodString>;
            skype: z.ZodOptional<z.ZodString>;
            real_name: z.ZodOptional<z.ZodString>;
            real_name_normalized: z.ZodOptional<z.ZodString>;
            display_name: z.ZodOptional<z.ZodString>;
            display_name_normalized: z.ZodOptional<z.ZodString>;
            fields: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            status_text: z.ZodOptional<z.ZodString>;
            status_emoji: z.ZodOptional<z.ZodString>;
            status_expiration: z.ZodOptional<z.ZodNumber>;
            avatar_hash: z.ZodOptional<z.ZodString>;
            image_original: z.ZodOptional<z.ZodString>;
            is_custom_image: z.ZodOptional<z.ZodBoolean>;
            email: z.ZodOptional<z.ZodString>;
            first_name: z.ZodOptional<z.ZodString>;
            last_name: z.ZodOptional<z.ZodString>;
            image_24: z.ZodOptional<z.ZodString>;
            image_32: z.ZodOptional<z.ZodString>;
            image_48: z.ZodOptional<z.ZodString>;
            image_72: z.ZodOptional<z.ZodString>;
            image_192: z.ZodOptional<z.ZodString>;
            image_512: z.ZodOptional<z.ZodString>;
            image_1024: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            title?: string | undefined;
            email?: string | undefined;
            fields?: Record<string, unknown> | undefined;
            real_name?: string | undefined;
            phone?: string | undefined;
            skype?: string | undefined;
            real_name_normalized?: string | undefined;
            display_name?: string | undefined;
            display_name_normalized?: string | undefined;
            status_text?: string | undefined;
            status_emoji?: string | undefined;
            status_expiration?: number | undefined;
            avatar_hash?: string | undefined;
            image_original?: string | undefined;
            is_custom_image?: boolean | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            image_24?: string | undefined;
            image_32?: string | undefined;
            image_48?: string | undefined;
            image_72?: string | undefined;
            image_192?: string | undefined;
            image_512?: string | undefined;
            image_1024?: string | undefined;
        }, {
            title?: string | undefined;
            email?: string | undefined;
            fields?: Record<string, unknown> | undefined;
            real_name?: string | undefined;
            phone?: string | undefined;
            skype?: string | undefined;
            real_name_normalized?: string | undefined;
            display_name?: string | undefined;
            display_name_normalized?: string | undefined;
            status_text?: string | undefined;
            status_emoji?: string | undefined;
            status_expiration?: number | undefined;
            avatar_hash?: string | undefined;
            image_original?: string | undefined;
            is_custom_image?: boolean | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            image_24?: string | undefined;
            image_32?: string | undefined;
            image_48?: string | undefined;
            image_72?: string | undefined;
            image_192?: string | undefined;
            image_512?: string | undefined;
            image_1024?: string | undefined;
        }>>;
        is_admin: z.ZodOptional<z.ZodBoolean>;
        is_owner: z.ZodOptional<z.ZodBoolean>;
        is_primary_owner: z.ZodOptional<z.ZodBoolean>;
        is_restricted: z.ZodOptional<z.ZodBoolean>;
        is_ultra_restricted: z.ZodOptional<z.ZodBoolean>;
        is_bot: z.ZodOptional<z.ZodBoolean>;
        is_app_user: z.ZodOptional<z.ZodBoolean>;
        updated: z.ZodOptional<z.ZodNumber>;
        has_2fa: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        color?: string | undefined;
        team_id?: string | undefined;
        deleted?: boolean | undefined;
        real_name?: string | undefined;
        tz?: string | undefined;
        tz_label?: string | undefined;
        tz_offset?: number | undefined;
        profile?: {
            title?: string | undefined;
            email?: string | undefined;
            fields?: Record<string, unknown> | undefined;
            real_name?: string | undefined;
            phone?: string | undefined;
            skype?: string | undefined;
            real_name_normalized?: string | undefined;
            display_name?: string | undefined;
            display_name_normalized?: string | undefined;
            status_text?: string | undefined;
            status_emoji?: string | undefined;
            status_expiration?: number | undefined;
            avatar_hash?: string | undefined;
            image_original?: string | undefined;
            is_custom_image?: boolean | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            image_24?: string | undefined;
            image_32?: string | undefined;
            image_48?: string | undefined;
            image_72?: string | undefined;
            image_192?: string | undefined;
            image_512?: string | undefined;
            image_1024?: string | undefined;
        } | undefined;
        is_admin?: boolean | undefined;
        is_owner?: boolean | undefined;
        is_primary_owner?: boolean | undefined;
        is_restricted?: boolean | undefined;
        is_ultra_restricted?: boolean | undefined;
        is_bot?: boolean | undefined;
        is_app_user?: boolean | undefined;
        updated?: number | undefined;
        has_2fa?: boolean | undefined;
    }, {
        name: string;
        id: string;
        color?: string | undefined;
        team_id?: string | undefined;
        deleted?: boolean | undefined;
        real_name?: string | undefined;
        tz?: string | undefined;
        tz_label?: string | undefined;
        tz_offset?: number | undefined;
        profile?: {
            title?: string | undefined;
            email?: string | undefined;
            fields?: Record<string, unknown> | undefined;
            real_name?: string | undefined;
            phone?: string | undefined;
            skype?: string | undefined;
            real_name_normalized?: string | undefined;
            display_name?: string | undefined;
            display_name_normalized?: string | undefined;
            status_text?: string | undefined;
            status_emoji?: string | undefined;
            status_expiration?: number | undefined;
            avatar_hash?: string | undefined;
            image_original?: string | undefined;
            is_custom_image?: boolean | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            image_24?: string | undefined;
            image_32?: string | undefined;
            image_48?: string | undefined;
            image_72?: string | undefined;
            image_192?: string | undefined;
            image_512?: string | undefined;
            image_1024?: string | undefined;
        } | undefined;
        is_admin?: boolean | undefined;
        is_owner?: boolean | undefined;
        is_primary_owner?: boolean | undefined;
        is_restricted?: boolean | undefined;
        is_ultra_restricted?: boolean | undefined;
        is_bot?: boolean | undefined;
        is_app_user?: boolean | undefined;
        updated?: number | undefined;
        has_2fa?: boolean | undefined;
    }>, "many">>;
    response_metadata: z.ZodOptional<z.ZodObject<{
        next_cursor: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        next_cursor: string;
    }, {
        next_cursor: string;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_users";
    ok: boolean;
    response_metadata?: {
        next_cursor: string;
    } | undefined;
    members?: {
        name: string;
        id: string;
        color?: string | undefined;
        team_id?: string | undefined;
        deleted?: boolean | undefined;
        real_name?: string | undefined;
        tz?: string | undefined;
        tz_label?: string | undefined;
        tz_offset?: number | undefined;
        profile?: {
            title?: string | undefined;
            email?: string | undefined;
            fields?: Record<string, unknown> | undefined;
            real_name?: string | undefined;
            phone?: string | undefined;
            skype?: string | undefined;
            real_name_normalized?: string | undefined;
            display_name?: string | undefined;
            display_name_normalized?: string | undefined;
            status_text?: string | undefined;
            status_emoji?: string | undefined;
            status_expiration?: number | undefined;
            avatar_hash?: string | undefined;
            image_original?: string | undefined;
            is_custom_image?: boolean | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            image_24?: string | undefined;
            image_32?: string | undefined;
            image_48?: string | undefined;
            image_72?: string | undefined;
            image_192?: string | undefined;
            image_512?: string | undefined;
            image_1024?: string | undefined;
        } | undefined;
        is_admin?: boolean | undefined;
        is_owner?: boolean | undefined;
        is_primary_owner?: boolean | undefined;
        is_restricted?: boolean | undefined;
        is_ultra_restricted?: boolean | undefined;
        is_bot?: boolean | undefined;
        is_app_user?: boolean | undefined;
        updated?: number | undefined;
        has_2fa?: boolean | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_users";
    ok: boolean;
    response_metadata?: {
        next_cursor: string;
    } | undefined;
    members?: {
        name: string;
        id: string;
        color?: string | undefined;
        team_id?: string | undefined;
        deleted?: boolean | undefined;
        real_name?: string | undefined;
        tz?: string | undefined;
        tz_label?: string | undefined;
        tz_offset?: number | undefined;
        profile?: {
            title?: string | undefined;
            email?: string | undefined;
            fields?: Record<string, unknown> | undefined;
            real_name?: string | undefined;
            phone?: string | undefined;
            skype?: string | undefined;
            real_name_normalized?: string | undefined;
            display_name?: string | undefined;
            display_name_normalized?: string | undefined;
            status_text?: string | undefined;
            status_emoji?: string | undefined;
            status_expiration?: number | undefined;
            avatar_hash?: string | undefined;
            image_original?: string | undefined;
            is_custom_image?: boolean | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            image_24?: string | undefined;
            image_32?: string | undefined;
            image_48?: string | undefined;
            image_72?: string | undefined;
            image_192?: string | undefined;
            image_512?: string | undefined;
            image_1024?: string | undefined;
        } | undefined;
        is_admin?: boolean | undefined;
        is_owner?: boolean | undefined;
        is_primary_owner?: boolean | undefined;
        is_restricted?: boolean | undefined;
        is_ultra_restricted?: boolean | undefined;
        is_bot?: boolean | undefined;
        is_app_user?: boolean | undefined;
        updated?: number | undefined;
        has_2fa?: boolean | undefined;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_conversation_history">;
    ok: z.ZodBoolean;
    messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
        type: z.ZodString;
        ts: z.ZodString;
        user: z.ZodOptional<z.ZodString>;
        bot_id: z.ZodOptional<z.ZodString>;
        bot_profile: z.ZodOptional<z.ZodObject<{
            name: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name?: string | undefined;
        }, {
            name?: string | undefined;
        }>>;
        username: z.ZodOptional<z.ZodString>;
        text: z.ZodOptional<z.ZodString>;
        thread_ts: z.ZodOptional<z.ZodString>;
        parent_user_id: z.ZodOptional<z.ZodString>;
        reply_count: z.ZodOptional<z.ZodNumber>;
        reply_users_count: z.ZodOptional<z.ZodNumber>;
        latest_reply: z.ZodOptional<z.ZodString>;
        reply_users: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        is_locked: z.ZodOptional<z.ZodBoolean>;
        subscribed: z.ZodOptional<z.ZodBoolean>;
        attachments: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        reactions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            users: z.ZodArray<z.ZodString, "many">;
            count: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            name: string;
            users: string[];
            count: number;
        }, {
            name: string;
            users: string[];
            count: number;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    }, {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    }>, "many">>;
    has_more: z.ZodOptional<z.ZodBoolean>;
    response_metadata: z.ZodOptional<z.ZodObject<{
        next_cursor: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        next_cursor: string;
    }, {
        next_cursor: string;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_conversation_history";
    ok: boolean;
    response_metadata?: {
        next_cursor: string;
    } | undefined;
    messages?: {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    }[] | undefined;
    has_more?: boolean | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_conversation_history";
    ok: boolean;
    response_metadata?: {
        next_cursor: string;
    } | undefined;
    messages?: {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    }[] | undefined;
    has_more?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_thread_replies">;
    ok: z.ZodBoolean;
    messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
        type: z.ZodString;
        ts: z.ZodString;
        user: z.ZodOptional<z.ZodString>;
        bot_id: z.ZodOptional<z.ZodString>;
        bot_profile: z.ZodOptional<z.ZodObject<{
            name: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name?: string | undefined;
        }, {
            name?: string | undefined;
        }>>;
        username: z.ZodOptional<z.ZodString>;
        text: z.ZodOptional<z.ZodString>;
        thread_ts: z.ZodOptional<z.ZodString>;
        parent_user_id: z.ZodOptional<z.ZodString>;
        reply_count: z.ZodOptional<z.ZodNumber>;
        reply_users_count: z.ZodOptional<z.ZodNumber>;
        latest_reply: z.ZodOptional<z.ZodString>;
        reply_users: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        is_locked: z.ZodOptional<z.ZodBoolean>;
        subscribed: z.ZodOptional<z.ZodBoolean>;
        attachments: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        reactions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            users: z.ZodArray<z.ZodString, "many">;
            count: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            name: string;
            users: string[];
            count: number;
        }, {
            name: string;
            users: string[];
            count: number;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    }, {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    }>, "many">>;
    has_more: z.ZodOptional<z.ZodBoolean>;
    response_metadata: z.ZodOptional<z.ZodObject<{
        next_cursor: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        next_cursor: string;
    }, {
        next_cursor: string;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_thread_replies";
    ok: boolean;
    response_metadata?: {
        next_cursor: string;
    } | undefined;
    messages?: {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    }[] | undefined;
    has_more?: boolean | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_thread_replies";
    ok: boolean;
    response_metadata?: {
        next_cursor: string;
    } | undefined;
    messages?: {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    }[] | undefined;
    has_more?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"update_message">;
    ok: z.ZodBoolean;
    channel: z.ZodOptional<z.ZodString>;
    ts: z.ZodOptional<z.ZodString>;
    text: z.ZodOptional<z.ZodString>;
    message: z.ZodOptional<z.ZodObject<{
        type: z.ZodString;
        ts: z.ZodString;
        user: z.ZodOptional<z.ZodString>;
        bot_id: z.ZodOptional<z.ZodString>;
        bot_profile: z.ZodOptional<z.ZodObject<{
            name: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name?: string | undefined;
        }, {
            name?: string | undefined;
        }>>;
        username: z.ZodOptional<z.ZodString>;
        text: z.ZodOptional<z.ZodString>;
        thread_ts: z.ZodOptional<z.ZodString>;
        parent_user_id: z.ZodOptional<z.ZodString>;
        reply_count: z.ZodOptional<z.ZodNumber>;
        reply_users_count: z.ZodOptional<z.ZodNumber>;
        latest_reply: z.ZodOptional<z.ZodString>;
        reply_users: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        is_locked: z.ZodOptional<z.ZodBoolean>;
        subscribed: z.ZodOptional<z.ZodBoolean>;
        attachments: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        reactions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            users: z.ZodArray<z.ZodString, "many">;
            count: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            name: string;
            users: string[];
            count: number;
        }, {
            name: string;
            users: string[];
            count: number;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    }, {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "update_message";
    ok: boolean;
    message?: {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    } | undefined;
    channel?: string | undefined;
    text?: string | undefined;
    ts?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "update_message";
    ok: boolean;
    message?: {
        type: string;
        ts: string;
        text?: string | undefined;
        username?: string | undefined;
        attachments?: unknown[] | undefined;
        blocks?: unknown[] | undefined;
        thread_ts?: string | undefined;
        user?: string | undefined;
        bot_id?: string | undefined;
        bot_profile?: {
            name?: string | undefined;
        } | undefined;
        parent_user_id?: string | undefined;
        reply_count?: number | undefined;
        reply_users_count?: number | undefined;
        latest_reply?: string | undefined;
        reply_users?: string[] | undefined;
        is_locked?: boolean | undefined;
        subscribed?: boolean | undefined;
        reactions?: {
            name: string;
            users: string[];
            count: number;
        }[] | undefined;
    } | undefined;
    channel?: string | undefined;
    text?: string | undefined;
    ts?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_message">;
    ok: z.ZodBoolean;
    channel: z.ZodOptional<z.ZodString>;
    ts: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_message";
    ok: boolean;
    channel?: string | undefined;
    ts?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "delete_message";
    ok: boolean;
    channel?: string | undefined;
    ts?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"add_reaction">;
    ok: z.ZodBoolean;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "add_reaction";
    ok: boolean;
}, {
    error: string;
    success: boolean;
    operation: "add_reaction";
    ok: boolean;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"remove_reaction">;
    ok: z.ZodBoolean;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "remove_reaction";
    ok: boolean;
}, {
    error: string;
    success: boolean;
    operation: "remove_reaction";
    ok: boolean;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"join_channel">;
    ok: z.ZodBoolean;
    channel: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        is_channel: z.ZodOptional<z.ZodBoolean>;
        is_group: z.ZodOptional<z.ZodBoolean>;
        is_im: z.ZodOptional<z.ZodBoolean>;
        is_mpim: z.ZodOptional<z.ZodBoolean>;
        is_private: z.ZodOptional<z.ZodBoolean>;
        created: z.ZodNumber;
        is_archived: z.ZodBoolean;
        is_general: z.ZodOptional<z.ZodBoolean>;
        unlinked: z.ZodOptional<z.ZodNumber>;
        name_normalized: z.ZodOptional<z.ZodString>;
        is_shared: z.ZodOptional<z.ZodBoolean>;
        is_ext_shared: z.ZodOptional<z.ZodBoolean>;
        is_org_shared: z.ZodOptional<z.ZodBoolean>;
        shared_team_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        pending_shared: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        pending_connected_team_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        is_pending_ext_shared: z.ZodOptional<z.ZodBoolean>;
        is_member: z.ZodOptional<z.ZodBoolean>;
        is_open: z.ZodOptional<z.ZodBoolean>;
        topic: z.ZodOptional<z.ZodObject<{
            value: z.ZodString;
            creator: z.ZodString;
            last_set: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            value: string;
            creator: string;
            last_set: number;
        }, {
            value: string;
            creator: string;
            last_set: number;
        }>>;
        purpose: z.ZodOptional<z.ZodObject<{
            value: z.ZodString;
            creator: z.ZodString;
            last_set: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            value: string;
            creator: string;
            last_set: number;
        }, {
            value: string;
            creator: string;
            last_set: number;
        }>>;
        num_members: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        created: number;
        is_archived: boolean;
        is_channel?: boolean | undefined;
        is_group?: boolean | undefined;
        is_im?: boolean | undefined;
        is_mpim?: boolean | undefined;
        is_private?: boolean | undefined;
        is_general?: boolean | undefined;
        unlinked?: number | undefined;
        name_normalized?: string | undefined;
        is_shared?: boolean | undefined;
        is_ext_shared?: boolean | undefined;
        is_org_shared?: boolean | undefined;
        shared_team_ids?: string[] | undefined;
        pending_shared?: string[] | undefined;
        pending_connected_team_ids?: string[] | undefined;
        is_pending_ext_shared?: boolean | undefined;
        is_member?: boolean | undefined;
        is_open?: boolean | undefined;
        topic?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        purpose?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        num_members?: number | undefined;
    }, {
        name: string;
        id: string;
        created: number;
        is_archived: boolean;
        is_channel?: boolean | undefined;
        is_group?: boolean | undefined;
        is_im?: boolean | undefined;
        is_mpim?: boolean | undefined;
        is_private?: boolean | undefined;
        is_general?: boolean | undefined;
        unlinked?: number | undefined;
        name_normalized?: string | undefined;
        is_shared?: boolean | undefined;
        is_ext_shared?: boolean | undefined;
        is_org_shared?: boolean | undefined;
        shared_team_ids?: string[] | undefined;
        pending_shared?: string[] | undefined;
        pending_connected_team_ids?: string[] | undefined;
        is_pending_ext_shared?: boolean | undefined;
        is_member?: boolean | undefined;
        is_open?: boolean | undefined;
        topic?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        purpose?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        num_members?: number | undefined;
    }>>;
    already_in_channel: z.ZodOptional<z.ZodBoolean>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "join_channel";
    ok: boolean;
    channel?: {
        name: string;
        id: string;
        created: number;
        is_archived: boolean;
        is_channel?: boolean | undefined;
        is_group?: boolean | undefined;
        is_im?: boolean | undefined;
        is_mpim?: boolean | undefined;
        is_private?: boolean | undefined;
        is_general?: boolean | undefined;
        unlinked?: number | undefined;
        name_normalized?: string | undefined;
        is_shared?: boolean | undefined;
        is_ext_shared?: boolean | undefined;
        is_org_shared?: boolean | undefined;
        shared_team_ids?: string[] | undefined;
        pending_shared?: string[] | undefined;
        pending_connected_team_ids?: string[] | undefined;
        is_pending_ext_shared?: boolean | undefined;
        is_member?: boolean | undefined;
        is_open?: boolean | undefined;
        topic?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        purpose?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        num_members?: number | undefined;
    } | undefined;
    already_in_channel?: boolean | undefined;
}, {
    error: string;
    success: boolean;
    operation: "join_channel";
    ok: boolean;
    channel?: {
        name: string;
        id: string;
        created: number;
        is_archived: boolean;
        is_channel?: boolean | undefined;
        is_group?: boolean | undefined;
        is_im?: boolean | undefined;
        is_mpim?: boolean | undefined;
        is_private?: boolean | undefined;
        is_general?: boolean | undefined;
        unlinked?: number | undefined;
        name_normalized?: string | undefined;
        is_shared?: boolean | undefined;
        is_ext_shared?: boolean | undefined;
        is_org_shared?: boolean | undefined;
        shared_team_ids?: string[] | undefined;
        pending_shared?: string[] | undefined;
        pending_connected_team_ids?: string[] | undefined;
        is_pending_ext_shared?: boolean | undefined;
        is_member?: boolean | undefined;
        is_open?: boolean | undefined;
        topic?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        purpose?: {
            value: string;
            creator: string;
            last_set: number;
        } | undefined;
        num_members?: number | undefined;
    } | undefined;
    already_in_channel?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"upload_file">;
    ok: z.ZodBoolean;
    file: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        created: z.ZodNumber;
        timestamp: z.ZodNumber;
        name: z.ZodString;
        title: z.ZodOptional<z.ZodString>;
        mimetype: z.ZodString;
        filetype: z.ZodString;
        pretty_type: z.ZodString;
        user: z.ZodString;
        editable: z.ZodBoolean;
        size: z.ZodNumber;
        mode: z.ZodString;
        is_external: z.ZodBoolean;
        external_type: z.ZodString;
        is_public: z.ZodBoolean;
        public_url_shared: z.ZodBoolean;
        display_as_bot: z.ZodBoolean;
        username: z.ZodString;
        url_private: z.ZodString;
        url_private_download: z.ZodString;
        permalink: z.ZodString;
        permalink_public: z.ZodOptional<z.ZodString>;
        shares: z.ZodOptional<z.ZodObject<{
            public: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodObject<{
                reply_users: z.ZodArray<z.ZodString, "many">;
                reply_users_count: z.ZodNumber;
                reply_count: z.ZodNumber;
                ts: z.ZodString;
                channel_name: z.ZodString;
                team_id: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }>, "many">>>;
            private: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodObject<{
                reply_users: z.ZodArray<z.ZodString, "many">;
                reply_users_count: z.ZodNumber;
                reply_count: z.ZodNumber;
                ts: z.ZodString;
                channel_name: z.ZodString;
                team_id: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }>, "many">>>;
        }, "strip", z.ZodTypeAny, {
            public?: Record<string, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }[]> | undefined;
            private?: Record<string, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }[]> | undefined;
        }, {
            public?: Record<string, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }[]> | undefined;
            private?: Record<string, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }[]> | undefined;
        }>>;
        channels: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        groups: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        ims: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        has_rich_preview: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        username: string;
        user: string;
        timestamp: number;
        id: string;
        created: number;
        mimetype: string;
        filetype: string;
        pretty_type: string;
        editable: boolean;
        size: number;
        mode: string;
        is_external: boolean;
        external_type: string;
        is_public: boolean;
        public_url_shared: boolean;
        display_as_bot: boolean;
        url_private: string;
        url_private_download: string;
        permalink: string;
        title?: string | undefined;
        channels?: string[] | undefined;
        permalink_public?: string | undefined;
        shares?: {
            public?: Record<string, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }[]> | undefined;
            private?: Record<string, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }[]> | undefined;
        } | undefined;
        groups?: string[] | undefined;
        ims?: string[] | undefined;
        has_rich_preview?: boolean | undefined;
    }, {
        name: string;
        username: string;
        user: string;
        timestamp: number;
        id: string;
        created: number;
        mimetype: string;
        filetype: string;
        pretty_type: string;
        editable: boolean;
        size: number;
        mode: string;
        is_external: boolean;
        external_type: string;
        is_public: boolean;
        public_url_shared: boolean;
        display_as_bot: boolean;
        url_private: string;
        url_private_download: string;
        permalink: string;
        title?: string | undefined;
        channels?: string[] | undefined;
        permalink_public?: string | undefined;
        shares?: {
            public?: Record<string, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }[]> | undefined;
            private?: Record<string, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }[]> | undefined;
        } | undefined;
        groups?: string[] | undefined;
        ims?: string[] | undefined;
        has_rich_preview?: boolean | undefined;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "upload_file";
    ok: boolean;
    file?: {
        name: string;
        username: string;
        user: string;
        timestamp: number;
        id: string;
        created: number;
        mimetype: string;
        filetype: string;
        pretty_type: string;
        editable: boolean;
        size: number;
        mode: string;
        is_external: boolean;
        external_type: string;
        is_public: boolean;
        public_url_shared: boolean;
        display_as_bot: boolean;
        url_private: string;
        url_private_download: string;
        permalink: string;
        title?: string | undefined;
        channels?: string[] | undefined;
        permalink_public?: string | undefined;
        shares?: {
            public?: Record<string, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }[]> | undefined;
            private?: Record<string, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }[]> | undefined;
        } | undefined;
        groups?: string[] | undefined;
        ims?: string[] | undefined;
        has_rich_preview?: boolean | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "upload_file";
    ok: boolean;
    file?: {
        name: string;
        username: string;
        user: string;
        timestamp: number;
        id: string;
        created: number;
        mimetype: string;
        filetype: string;
        pretty_type: string;
        editable: boolean;
        size: number;
        mode: string;
        is_external: boolean;
        external_type: string;
        is_public: boolean;
        public_url_shared: boolean;
        display_as_bot: boolean;
        url_private: string;
        url_private_download: string;
        permalink: string;
        title?: string | undefined;
        channels?: string[] | undefined;
        permalink_public?: string | undefined;
        shares?: {
            public?: Record<string, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }[]> | undefined;
            private?: Record<string, {
                ts: string;
                reply_count: number;
                reply_users_count: number;
                reply_users: string[];
                team_id: string;
                channel_name: string;
            }[]> | undefined;
        } | undefined;
        groups?: string[] | undefined;
        ims?: string[] | undefined;
        has_rich_preview?: boolean | undefined;
    } | undefined;
}>]>;
type SlackResult = z.output<typeof SlackResultSchema>;
type SlackParams = z.input<typeof SlackParamsSchema>;
export type SlackOperationResult<T extends SlackParams['operation']> = Extract<SlackResult, {
    operation: T;
}>;
export declare class SlackBubble<T extends SlackParams = SlackParams> extends ServiceBubble<T, Extract<SlackResult, {
    operation: T['operation'];
}>> {
    /**
     * Test the validity of the Slack credential by making a test API call
     * @returns Promise that resolves to true if credential is valid, false otherwise
     * @throws ExternalServiceError if Slack API is unreachable
     */
    testCredential(): Promise<boolean>;
    static readonly type: "service";
    static readonly service = "slack";
    static readonly authType: "apikey";
    static readonly bubbleName = "slack";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"send_message">;
        channel: z.ZodString;
        text: z.ZodString;
        username: z.ZodOptional<z.ZodString>;
        icon_emoji: z.ZodOptional<z.ZodString>;
        icon_url: z.ZodOptional<z.ZodString>;
        attachments: z.ZodOptional<z.ZodArray<z.ZodObject<{
            color: z.ZodOptional<z.ZodString>;
            pretext: z.ZodOptional<z.ZodString>;
            author_name: z.ZodOptional<z.ZodString>;
            author_link: z.ZodOptional<z.ZodString>;
            author_icon: z.ZodOptional<z.ZodString>;
            title: z.ZodOptional<z.ZodString>;
            title_link: z.ZodOptional<z.ZodString>;
            text: z.ZodOptional<z.ZodString>;
            fields: z.ZodOptional<z.ZodArray<z.ZodObject<{
                title: z.ZodString;
                value: z.ZodString;
                short: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                title: string;
                short?: boolean | undefined;
            }, {
                value: string;
                title: string;
                short?: boolean | undefined;
            }>, "many">>;
            image_url: z.ZodOptional<z.ZodString>;
            thumb_url: z.ZodOptional<z.ZodString>;
            footer: z.ZodOptional<z.ZodString>;
            footer_icon: z.ZodOptional<z.ZodString>;
            ts: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            title?: string | undefined;
            fields?: {
                value: string;
                title: string;
                short?: boolean | undefined;
            }[] | undefined;
            text?: string | undefined;
            color?: string | undefined;
            pretext?: string | undefined;
            author_name?: string | undefined;
            author_link?: string | undefined;
            author_icon?: string | undefined;
            title_link?: string | undefined;
            image_url?: string | undefined;
            thumb_url?: string | undefined;
            footer?: string | undefined;
            footer_icon?: string | undefined;
            ts?: number | undefined;
        }, {
            title?: string | undefined;
            fields?: {
                value: string;
                title: string;
                short?: boolean | undefined;
            }[] | undefined;
            text?: string | undefined;
            color?: string | undefined;
            pretext?: string | undefined;
            author_name?: string | undefined;
            author_link?: string | undefined;
            author_icon?: string | undefined;
            title_link?: string | undefined;
            image_url?: string | undefined;
            thumb_url?: string | undefined;
            footer?: string | undefined;
            footer_icon?: string | undefined;
            ts?: number | undefined;
        }>, "many">>;
        blocks: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodString;
            text: z.ZodOptional<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>>;
            elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
                text: z.ZodOptional<z.ZodString>;
                image_url: z.ZodOptional<z.ZodString>;
                alt_text: z.ZodOptional<z.ZodString>;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }>, "many">>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            type: z.ZodString;
            text: z.ZodOptional<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>>;
            elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
                text: z.ZodOptional<z.ZodString>;
                image_url: z.ZodOptional<z.ZodString>;
                alt_text: z.ZodOptional<z.ZodString>;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }>, "many">>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            type: z.ZodString;
            text: z.ZodOptional<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>>;
            elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
                text: z.ZodOptional<z.ZodString>;
                image_url: z.ZodOptional<z.ZodString>;
                alt_text: z.ZodOptional<z.ZodString>;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }>, "many">>;
        }, z.ZodTypeAny, "passthrough">>, "many">>;
        thread_ts: z.ZodOptional<z.ZodString>;
        reply_broadcast: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
        unfurl_links: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        unfurl_media: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    }, "strip", z.ZodTypeAny, {
        operation: "send_message";
        channel: string;
        text: string;
        reply_broadcast: boolean;
        unfurl_links: boolean;
        unfurl_media: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        username?: string | undefined;
        icon_emoji?: string | undefined;
        icon_url?: string | undefined;
        attachments?: {
            title?: string | undefined;
            fields?: {
                value: string;
                title: string;
                short?: boolean | undefined;
            }[] | undefined;
            text?: string | undefined;
            color?: string | undefined;
            pretext?: string | undefined;
            author_name?: string | undefined;
            author_link?: string | undefined;
            author_icon?: string | undefined;
            title_link?: string | undefined;
            image_url?: string | undefined;
            thumb_url?: string | undefined;
            footer?: string | undefined;
            footer_icon?: string | undefined;
            ts?: number | undefined;
        }[] | undefined;
        blocks?: z.objectOutputType<{
            type: z.ZodString;
            text: z.ZodOptional<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>>;
            elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
                text: z.ZodOptional<z.ZodString>;
                image_url: z.ZodOptional<z.ZodString>;
                alt_text: z.ZodOptional<z.ZodString>;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }>, "many">>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
        thread_ts?: string | undefined;
    }, {
        operation: "send_message";
        channel: string;
        text: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        username?: string | undefined;
        icon_emoji?: string | undefined;
        icon_url?: string | undefined;
        attachments?: {
            title?: string | undefined;
            fields?: {
                value: string;
                title: string;
                short?: boolean | undefined;
            }[] | undefined;
            text?: string | undefined;
            color?: string | undefined;
            pretext?: string | undefined;
            author_name?: string | undefined;
            author_link?: string | undefined;
            author_icon?: string | undefined;
            title_link?: string | undefined;
            image_url?: string | undefined;
            thumb_url?: string | undefined;
            footer?: string | undefined;
            footer_icon?: string | undefined;
            ts?: number | undefined;
        }[] | undefined;
        blocks?: z.objectInputType<{
            type: z.ZodString;
            text: z.ZodOptional<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>>;
            elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
                text: z.ZodOptional<z.ZodString>;
                image_url: z.ZodOptional<z.ZodString>;
                alt_text: z.ZodOptional<z.ZodString>;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }>, "many">>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
        thread_ts?: string | undefined;
        reply_broadcast?: boolean | undefined;
        unfurl_links?: boolean | undefined;
        unfurl_media?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_channels">;
        types: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodEnum<["public_channel", "private_channel", "mpim", "im"]>, "many">>>;
        exclude_archived: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        cursor: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_channels";
        types: ("public_channel" | "private_channel" | "mpim" | "im")[];
        exclude_archived: boolean;
        limit: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cursor?: string | undefined;
    }, {
        operation: "list_channels";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        types?: ("public_channel" | "private_channel" | "mpim" | "im")[] | undefined;
        exclude_archived?: boolean | undefined;
        limit?: number | undefined;
        cursor?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_channel_info">;
        channel: z.ZodString;
        include_locale: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_channel_info";
        channel: string;
        include_locale: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_channel_info";
        channel: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        include_locale?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_user_info">;
        user: z.ZodString;
        include_locale: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_user_info";
        include_locale: boolean;
        user: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_user_info";
        user: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        include_locale?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_users">;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        cursor: z.ZodOptional<z.ZodString>;
        include_locale: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_users";
        limit: number;
        include_locale: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cursor?: string | undefined;
    }, {
        operation: "list_users";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        cursor?: string | undefined;
        include_locale?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_conversation_history">;
        channel: z.ZodString;
        latest: z.ZodOptional<z.ZodString>;
        oldest: z.ZodOptional<z.ZodString>;
        inclusive: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        cursor: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        inclusive: boolean;
        operation: "get_conversation_history";
        channel: string;
        limit: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cursor?: string | undefined;
        latest?: string | undefined;
        oldest?: string | undefined;
    }, {
        operation: "get_conversation_history";
        channel: string;
        inclusive?: boolean | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        cursor?: string | undefined;
        latest?: string | undefined;
        oldest?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_thread_replies">;
        channel: z.ZodString;
        ts: z.ZodString;
        latest: z.ZodOptional<z.ZodString>;
        oldest: z.ZodOptional<z.ZodString>;
        inclusive: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        cursor: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        inclusive: boolean;
        operation: "get_thread_replies";
        channel: string;
        ts: string;
        limit: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cursor?: string | undefined;
        latest?: string | undefined;
        oldest?: string | undefined;
    }, {
        operation: "get_thread_replies";
        channel: string;
        ts: string;
        inclusive?: boolean | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        cursor?: string | undefined;
        latest?: string | undefined;
        oldest?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_message">;
        channel: z.ZodString;
        ts: z.ZodString;
        text: z.ZodOptional<z.ZodString>;
        attachments: z.ZodOptional<z.ZodArray<z.ZodObject<{
            color: z.ZodOptional<z.ZodString>;
            pretext: z.ZodOptional<z.ZodString>;
            author_name: z.ZodOptional<z.ZodString>;
            author_link: z.ZodOptional<z.ZodString>;
            author_icon: z.ZodOptional<z.ZodString>;
            title: z.ZodOptional<z.ZodString>;
            title_link: z.ZodOptional<z.ZodString>;
            text: z.ZodOptional<z.ZodString>;
            fields: z.ZodOptional<z.ZodArray<z.ZodObject<{
                title: z.ZodString;
                value: z.ZodString;
                short: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                value: string;
                title: string;
                short?: boolean | undefined;
            }, {
                value: string;
                title: string;
                short?: boolean | undefined;
            }>, "many">>;
            image_url: z.ZodOptional<z.ZodString>;
            thumb_url: z.ZodOptional<z.ZodString>;
            footer: z.ZodOptional<z.ZodString>;
            footer_icon: z.ZodOptional<z.ZodString>;
            ts: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            title?: string | undefined;
            fields?: {
                value: string;
                title: string;
                short?: boolean | undefined;
            }[] | undefined;
            text?: string | undefined;
            color?: string | undefined;
            pretext?: string | undefined;
            author_name?: string | undefined;
            author_link?: string | undefined;
            author_icon?: string | undefined;
            title_link?: string | undefined;
            image_url?: string | undefined;
            thumb_url?: string | undefined;
            footer?: string | undefined;
            footer_icon?: string | undefined;
            ts?: number | undefined;
        }, {
            title?: string | undefined;
            fields?: {
                value: string;
                title: string;
                short?: boolean | undefined;
            }[] | undefined;
            text?: string | undefined;
            color?: string | undefined;
            pretext?: string | undefined;
            author_name?: string | undefined;
            author_link?: string | undefined;
            author_icon?: string | undefined;
            title_link?: string | undefined;
            image_url?: string | undefined;
            thumb_url?: string | undefined;
            footer?: string | undefined;
            footer_icon?: string | undefined;
            ts?: number | undefined;
        }>, "many">>;
        blocks: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodString;
            text: z.ZodOptional<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>>;
            elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
                text: z.ZodOptional<z.ZodString>;
                image_url: z.ZodOptional<z.ZodString>;
                alt_text: z.ZodOptional<z.ZodString>;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }>, "many">>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            type: z.ZodString;
            text: z.ZodOptional<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>>;
            elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
                text: z.ZodOptional<z.ZodString>;
                image_url: z.ZodOptional<z.ZodString>;
                alt_text: z.ZodOptional<z.ZodString>;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }>, "many">>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            type: z.ZodString;
            text: z.ZodOptional<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>>;
            elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
                text: z.ZodOptional<z.ZodString>;
                image_url: z.ZodOptional<z.ZodString>;
                alt_text: z.ZodOptional<z.ZodString>;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }>, "many">>;
        }, z.ZodTypeAny, "passthrough">>, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "update_message";
        channel: string;
        ts: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        text?: string | undefined;
        attachments?: {
            title?: string | undefined;
            fields?: {
                value: string;
                title: string;
                short?: boolean | undefined;
            }[] | undefined;
            text?: string | undefined;
            color?: string | undefined;
            pretext?: string | undefined;
            author_name?: string | undefined;
            author_link?: string | undefined;
            author_icon?: string | undefined;
            title_link?: string | undefined;
            image_url?: string | undefined;
            thumb_url?: string | undefined;
            footer?: string | undefined;
            footer_icon?: string | undefined;
            ts?: number | undefined;
        }[] | undefined;
        blocks?: z.objectOutputType<{
            type: z.ZodString;
            text: z.ZodOptional<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>>;
            elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
                text: z.ZodOptional<z.ZodString>;
                image_url: z.ZodOptional<z.ZodString>;
                alt_text: z.ZodOptional<z.ZodString>;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }>, "many">>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }, {
        operation: "update_message";
        channel: string;
        ts: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        text?: string | undefined;
        attachments?: {
            title?: string | undefined;
            fields?: {
                value: string;
                title: string;
                short?: boolean | undefined;
            }[] | undefined;
            text?: string | undefined;
            color?: string | undefined;
            pretext?: string | undefined;
            author_name?: string | undefined;
            author_link?: string | undefined;
            author_icon?: string | undefined;
            title_link?: string | undefined;
            image_url?: string | undefined;
            thumb_url?: string | undefined;
            footer?: string | undefined;
            footer_icon?: string | undefined;
            ts?: number | undefined;
        }[] | undefined;
        blocks?: z.objectInputType<{
            type: z.ZodString;
            text: z.ZodOptional<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn"]>;
                text: z.ZodString;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }, {
                type: "plain_text" | "mrkdwn";
                text: string;
                emoji?: boolean | undefined;
                verbatim?: boolean | undefined;
            }>>;
            elements: z.ZodOptional<z.ZodArray<z.ZodObject<{
                type: z.ZodEnum<["plain_text", "mrkdwn", "image"]>;
                text: z.ZodOptional<z.ZodString>;
                image_url: z.ZodOptional<z.ZodString>;
                alt_text: z.ZodOptional<z.ZodString>;
                emoji: z.ZodOptional<z.ZodBoolean>;
                verbatim: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }, {
                type: "plain_text" | "mrkdwn" | "image";
                emoji?: boolean | undefined;
                text?: string | undefined;
                image_url?: string | undefined;
                verbatim?: boolean | undefined;
                alt_text?: string | undefined;
            }>, "many">>;
        }, z.ZodTypeAny, "passthrough">[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_message">;
        channel: z.ZodString;
        ts: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_message";
        channel: string;
        ts: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_message";
        channel: string;
        ts: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"add_reaction">;
        name: z.ZodString;
        channel: z.ZodString;
        timestamp: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        operation: "add_reaction";
        channel: string;
        timestamp: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        name: string;
        operation: "add_reaction";
        channel: string;
        timestamp: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"remove_reaction">;
        name: z.ZodString;
        channel: z.ZodString;
        timestamp: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        operation: "remove_reaction";
        channel: string;
        timestamp: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        name: string;
        operation: "remove_reaction";
        channel: string;
        timestamp: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"join_channel">;
        channel: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "join_channel";
        channel: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "join_channel";
        channel: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"upload_file">;
        channel: z.ZodString;
        file_path: z.ZodEffects<z.ZodString, string, string>;
        filename: z.ZodOptional<z.ZodString>;
        title: z.ZodOptional<z.ZodString>;
        initial_comment: z.ZodOptional<z.ZodString>;
        thread_ts: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "upload_file";
        channel: string;
        file_path: string;
        title?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        thread_ts?: string | undefined;
        filename?: string | undefined;
        initial_comment?: string | undefined;
    }, {
        operation: "upload_file";
        channel: string;
        file_path: string;
        title?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        thread_ts?: string | undefined;
        filename?: string | undefined;
        initial_comment?: string | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"send_message">;
        ok: z.ZodBoolean;
        channel: z.ZodOptional<z.ZodString>;
        ts: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodObject<{
            type: z.ZodString;
            ts: z.ZodString;
            user: z.ZodOptional<z.ZodString>;
            bot_id: z.ZodOptional<z.ZodString>;
            bot_profile: z.ZodOptional<z.ZodObject<{
                name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                name?: string | undefined;
            }, {
                name?: string | undefined;
            }>>;
            username: z.ZodOptional<z.ZodString>;
            text: z.ZodOptional<z.ZodString>;
            thread_ts: z.ZodOptional<z.ZodString>;
            parent_user_id: z.ZodOptional<z.ZodString>;
            reply_count: z.ZodOptional<z.ZodNumber>;
            reply_users_count: z.ZodOptional<z.ZodNumber>;
            latest_reply: z.ZodOptional<z.ZodString>;
            reply_users: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            is_locked: z.ZodOptional<z.ZodBoolean>;
            subscribed: z.ZodOptional<z.ZodBoolean>;
            attachments: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            reactions: z.ZodOptional<z.ZodArray<z.ZodObject<{
                name: z.ZodString;
                users: z.ZodArray<z.ZodString, "many">;
                count: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                name: string;
                users: string[];
                count: number;
            }, {
                name: string;
                users: string[];
                count: number;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        }, {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "send_message";
        ok: boolean;
        message?: {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        } | undefined;
        channel?: string | undefined;
        ts?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "send_message";
        ok: boolean;
        message?: {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        } | undefined;
        channel?: string | undefined;
        ts?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_channels">;
        ok: z.ZodBoolean;
        channels: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            is_channel: z.ZodOptional<z.ZodBoolean>;
            is_group: z.ZodOptional<z.ZodBoolean>;
            is_im: z.ZodOptional<z.ZodBoolean>;
            is_mpim: z.ZodOptional<z.ZodBoolean>;
            is_private: z.ZodOptional<z.ZodBoolean>;
            created: z.ZodNumber;
            is_archived: z.ZodBoolean;
            is_general: z.ZodOptional<z.ZodBoolean>;
            unlinked: z.ZodOptional<z.ZodNumber>;
            name_normalized: z.ZodOptional<z.ZodString>;
            is_shared: z.ZodOptional<z.ZodBoolean>;
            is_ext_shared: z.ZodOptional<z.ZodBoolean>;
            is_org_shared: z.ZodOptional<z.ZodBoolean>;
            shared_team_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            pending_shared: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            pending_connected_team_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            is_pending_ext_shared: z.ZodOptional<z.ZodBoolean>;
            is_member: z.ZodOptional<z.ZodBoolean>;
            is_open: z.ZodOptional<z.ZodBoolean>;
            topic: z.ZodOptional<z.ZodObject<{
                value: z.ZodString;
                creator: z.ZodString;
                last_set: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                value: string;
                creator: string;
                last_set: number;
            }, {
                value: string;
                creator: string;
                last_set: number;
            }>>;
            purpose: z.ZodOptional<z.ZodObject<{
                value: z.ZodString;
                creator: z.ZodString;
                last_set: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                value: string;
                creator: string;
                last_set: number;
            }, {
                value: string;
                creator: string;
                last_set: number;
            }>>;
            num_members: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            created: number;
            is_archived: boolean;
            is_channel?: boolean | undefined;
            is_group?: boolean | undefined;
            is_im?: boolean | undefined;
            is_mpim?: boolean | undefined;
            is_private?: boolean | undefined;
            is_general?: boolean | undefined;
            unlinked?: number | undefined;
            name_normalized?: string | undefined;
            is_shared?: boolean | undefined;
            is_ext_shared?: boolean | undefined;
            is_org_shared?: boolean | undefined;
            shared_team_ids?: string[] | undefined;
            pending_shared?: string[] | undefined;
            pending_connected_team_ids?: string[] | undefined;
            is_pending_ext_shared?: boolean | undefined;
            is_member?: boolean | undefined;
            is_open?: boolean | undefined;
            topic?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            purpose?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            num_members?: number | undefined;
        }, {
            name: string;
            id: string;
            created: number;
            is_archived: boolean;
            is_channel?: boolean | undefined;
            is_group?: boolean | undefined;
            is_im?: boolean | undefined;
            is_mpim?: boolean | undefined;
            is_private?: boolean | undefined;
            is_general?: boolean | undefined;
            unlinked?: number | undefined;
            name_normalized?: string | undefined;
            is_shared?: boolean | undefined;
            is_ext_shared?: boolean | undefined;
            is_org_shared?: boolean | undefined;
            shared_team_ids?: string[] | undefined;
            pending_shared?: string[] | undefined;
            pending_connected_team_ids?: string[] | undefined;
            is_pending_ext_shared?: boolean | undefined;
            is_member?: boolean | undefined;
            is_open?: boolean | undefined;
            topic?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            purpose?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            num_members?: number | undefined;
        }>, "many">>;
        response_metadata: z.ZodOptional<z.ZodObject<{
            next_cursor: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            next_cursor: string;
        }, {
            next_cursor: string;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_channels";
        ok: boolean;
        channels?: {
            name: string;
            id: string;
            created: number;
            is_archived: boolean;
            is_channel?: boolean | undefined;
            is_group?: boolean | undefined;
            is_im?: boolean | undefined;
            is_mpim?: boolean | undefined;
            is_private?: boolean | undefined;
            is_general?: boolean | undefined;
            unlinked?: number | undefined;
            name_normalized?: string | undefined;
            is_shared?: boolean | undefined;
            is_ext_shared?: boolean | undefined;
            is_org_shared?: boolean | undefined;
            shared_team_ids?: string[] | undefined;
            pending_shared?: string[] | undefined;
            pending_connected_team_ids?: string[] | undefined;
            is_pending_ext_shared?: boolean | undefined;
            is_member?: boolean | undefined;
            is_open?: boolean | undefined;
            topic?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            purpose?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            num_members?: number | undefined;
        }[] | undefined;
        response_metadata?: {
            next_cursor: string;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_channels";
        ok: boolean;
        channels?: {
            name: string;
            id: string;
            created: number;
            is_archived: boolean;
            is_channel?: boolean | undefined;
            is_group?: boolean | undefined;
            is_im?: boolean | undefined;
            is_mpim?: boolean | undefined;
            is_private?: boolean | undefined;
            is_general?: boolean | undefined;
            unlinked?: number | undefined;
            name_normalized?: string | undefined;
            is_shared?: boolean | undefined;
            is_ext_shared?: boolean | undefined;
            is_org_shared?: boolean | undefined;
            shared_team_ids?: string[] | undefined;
            pending_shared?: string[] | undefined;
            pending_connected_team_ids?: string[] | undefined;
            is_pending_ext_shared?: boolean | undefined;
            is_member?: boolean | undefined;
            is_open?: boolean | undefined;
            topic?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            purpose?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            num_members?: number | undefined;
        }[] | undefined;
        response_metadata?: {
            next_cursor: string;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_channel_info">;
        ok: z.ZodBoolean;
        channel: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            is_channel: z.ZodOptional<z.ZodBoolean>;
            is_group: z.ZodOptional<z.ZodBoolean>;
            is_im: z.ZodOptional<z.ZodBoolean>;
            is_mpim: z.ZodOptional<z.ZodBoolean>;
            is_private: z.ZodOptional<z.ZodBoolean>;
            created: z.ZodNumber;
            is_archived: z.ZodBoolean;
            is_general: z.ZodOptional<z.ZodBoolean>;
            unlinked: z.ZodOptional<z.ZodNumber>;
            name_normalized: z.ZodOptional<z.ZodString>;
            is_shared: z.ZodOptional<z.ZodBoolean>;
            is_ext_shared: z.ZodOptional<z.ZodBoolean>;
            is_org_shared: z.ZodOptional<z.ZodBoolean>;
            shared_team_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            pending_shared: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            pending_connected_team_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            is_pending_ext_shared: z.ZodOptional<z.ZodBoolean>;
            is_member: z.ZodOptional<z.ZodBoolean>;
            is_open: z.ZodOptional<z.ZodBoolean>;
            topic: z.ZodOptional<z.ZodObject<{
                value: z.ZodString;
                creator: z.ZodString;
                last_set: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                value: string;
                creator: string;
                last_set: number;
            }, {
                value: string;
                creator: string;
                last_set: number;
            }>>;
            purpose: z.ZodOptional<z.ZodObject<{
                value: z.ZodString;
                creator: z.ZodString;
                last_set: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                value: string;
                creator: string;
                last_set: number;
            }, {
                value: string;
                creator: string;
                last_set: number;
            }>>;
            num_members: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            created: number;
            is_archived: boolean;
            is_channel?: boolean | undefined;
            is_group?: boolean | undefined;
            is_im?: boolean | undefined;
            is_mpim?: boolean | undefined;
            is_private?: boolean | undefined;
            is_general?: boolean | undefined;
            unlinked?: number | undefined;
            name_normalized?: string | undefined;
            is_shared?: boolean | undefined;
            is_ext_shared?: boolean | undefined;
            is_org_shared?: boolean | undefined;
            shared_team_ids?: string[] | undefined;
            pending_shared?: string[] | undefined;
            pending_connected_team_ids?: string[] | undefined;
            is_pending_ext_shared?: boolean | undefined;
            is_member?: boolean | undefined;
            is_open?: boolean | undefined;
            topic?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            purpose?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            num_members?: number | undefined;
        }, {
            name: string;
            id: string;
            created: number;
            is_archived: boolean;
            is_channel?: boolean | undefined;
            is_group?: boolean | undefined;
            is_im?: boolean | undefined;
            is_mpim?: boolean | undefined;
            is_private?: boolean | undefined;
            is_general?: boolean | undefined;
            unlinked?: number | undefined;
            name_normalized?: string | undefined;
            is_shared?: boolean | undefined;
            is_ext_shared?: boolean | undefined;
            is_org_shared?: boolean | undefined;
            shared_team_ids?: string[] | undefined;
            pending_shared?: string[] | undefined;
            pending_connected_team_ids?: string[] | undefined;
            is_pending_ext_shared?: boolean | undefined;
            is_member?: boolean | undefined;
            is_open?: boolean | undefined;
            topic?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            purpose?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            num_members?: number | undefined;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_channel_info";
        ok: boolean;
        channel?: {
            name: string;
            id: string;
            created: number;
            is_archived: boolean;
            is_channel?: boolean | undefined;
            is_group?: boolean | undefined;
            is_im?: boolean | undefined;
            is_mpim?: boolean | undefined;
            is_private?: boolean | undefined;
            is_general?: boolean | undefined;
            unlinked?: number | undefined;
            name_normalized?: string | undefined;
            is_shared?: boolean | undefined;
            is_ext_shared?: boolean | undefined;
            is_org_shared?: boolean | undefined;
            shared_team_ids?: string[] | undefined;
            pending_shared?: string[] | undefined;
            pending_connected_team_ids?: string[] | undefined;
            is_pending_ext_shared?: boolean | undefined;
            is_member?: boolean | undefined;
            is_open?: boolean | undefined;
            topic?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            purpose?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            num_members?: number | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_channel_info";
        ok: boolean;
        channel?: {
            name: string;
            id: string;
            created: number;
            is_archived: boolean;
            is_channel?: boolean | undefined;
            is_group?: boolean | undefined;
            is_im?: boolean | undefined;
            is_mpim?: boolean | undefined;
            is_private?: boolean | undefined;
            is_general?: boolean | undefined;
            unlinked?: number | undefined;
            name_normalized?: string | undefined;
            is_shared?: boolean | undefined;
            is_ext_shared?: boolean | undefined;
            is_org_shared?: boolean | undefined;
            shared_team_ids?: string[] | undefined;
            pending_shared?: string[] | undefined;
            pending_connected_team_ids?: string[] | undefined;
            is_pending_ext_shared?: boolean | undefined;
            is_member?: boolean | undefined;
            is_open?: boolean | undefined;
            topic?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            purpose?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            num_members?: number | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_user_info">;
        ok: z.ZodBoolean;
        user: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            team_id: z.ZodOptional<z.ZodString>;
            name: z.ZodString;
            deleted: z.ZodOptional<z.ZodBoolean>;
            color: z.ZodOptional<z.ZodString>;
            real_name: z.ZodOptional<z.ZodString>;
            tz: z.ZodOptional<z.ZodString>;
            tz_label: z.ZodOptional<z.ZodString>;
            tz_offset: z.ZodOptional<z.ZodNumber>;
            profile: z.ZodOptional<z.ZodObject<{
                title: z.ZodOptional<z.ZodString>;
                phone: z.ZodOptional<z.ZodString>;
                skype: z.ZodOptional<z.ZodString>;
                real_name: z.ZodOptional<z.ZodString>;
                real_name_normalized: z.ZodOptional<z.ZodString>;
                display_name: z.ZodOptional<z.ZodString>;
                display_name_normalized: z.ZodOptional<z.ZodString>;
                fields: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                status_text: z.ZodOptional<z.ZodString>;
                status_emoji: z.ZodOptional<z.ZodString>;
                status_expiration: z.ZodOptional<z.ZodNumber>;
                avatar_hash: z.ZodOptional<z.ZodString>;
                image_original: z.ZodOptional<z.ZodString>;
                is_custom_image: z.ZodOptional<z.ZodBoolean>;
                email: z.ZodOptional<z.ZodString>;
                first_name: z.ZodOptional<z.ZodString>;
                last_name: z.ZodOptional<z.ZodString>;
                image_24: z.ZodOptional<z.ZodString>;
                image_32: z.ZodOptional<z.ZodString>;
                image_48: z.ZodOptional<z.ZodString>;
                image_72: z.ZodOptional<z.ZodString>;
                image_192: z.ZodOptional<z.ZodString>;
                image_512: z.ZodOptional<z.ZodString>;
                image_1024: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                title?: string | undefined;
                email?: string | undefined;
                fields?: Record<string, unknown> | undefined;
                real_name?: string | undefined;
                phone?: string | undefined;
                skype?: string | undefined;
                real_name_normalized?: string | undefined;
                display_name?: string | undefined;
                display_name_normalized?: string | undefined;
                status_text?: string | undefined;
                status_emoji?: string | undefined;
                status_expiration?: number | undefined;
                avatar_hash?: string | undefined;
                image_original?: string | undefined;
                is_custom_image?: boolean | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
                image_24?: string | undefined;
                image_32?: string | undefined;
                image_48?: string | undefined;
                image_72?: string | undefined;
                image_192?: string | undefined;
                image_512?: string | undefined;
                image_1024?: string | undefined;
            }, {
                title?: string | undefined;
                email?: string | undefined;
                fields?: Record<string, unknown> | undefined;
                real_name?: string | undefined;
                phone?: string | undefined;
                skype?: string | undefined;
                real_name_normalized?: string | undefined;
                display_name?: string | undefined;
                display_name_normalized?: string | undefined;
                status_text?: string | undefined;
                status_emoji?: string | undefined;
                status_expiration?: number | undefined;
                avatar_hash?: string | undefined;
                image_original?: string | undefined;
                is_custom_image?: boolean | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
                image_24?: string | undefined;
                image_32?: string | undefined;
                image_48?: string | undefined;
                image_72?: string | undefined;
                image_192?: string | undefined;
                image_512?: string | undefined;
                image_1024?: string | undefined;
            }>>;
            is_admin: z.ZodOptional<z.ZodBoolean>;
            is_owner: z.ZodOptional<z.ZodBoolean>;
            is_primary_owner: z.ZodOptional<z.ZodBoolean>;
            is_restricted: z.ZodOptional<z.ZodBoolean>;
            is_ultra_restricted: z.ZodOptional<z.ZodBoolean>;
            is_bot: z.ZodOptional<z.ZodBoolean>;
            is_app_user: z.ZodOptional<z.ZodBoolean>;
            updated: z.ZodOptional<z.ZodNumber>;
            has_2fa: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            color?: string | undefined;
            team_id?: string | undefined;
            deleted?: boolean | undefined;
            real_name?: string | undefined;
            tz?: string | undefined;
            tz_label?: string | undefined;
            tz_offset?: number | undefined;
            profile?: {
                title?: string | undefined;
                email?: string | undefined;
                fields?: Record<string, unknown> | undefined;
                real_name?: string | undefined;
                phone?: string | undefined;
                skype?: string | undefined;
                real_name_normalized?: string | undefined;
                display_name?: string | undefined;
                display_name_normalized?: string | undefined;
                status_text?: string | undefined;
                status_emoji?: string | undefined;
                status_expiration?: number | undefined;
                avatar_hash?: string | undefined;
                image_original?: string | undefined;
                is_custom_image?: boolean | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
                image_24?: string | undefined;
                image_32?: string | undefined;
                image_48?: string | undefined;
                image_72?: string | undefined;
                image_192?: string | undefined;
                image_512?: string | undefined;
                image_1024?: string | undefined;
            } | undefined;
            is_admin?: boolean | undefined;
            is_owner?: boolean | undefined;
            is_primary_owner?: boolean | undefined;
            is_restricted?: boolean | undefined;
            is_ultra_restricted?: boolean | undefined;
            is_bot?: boolean | undefined;
            is_app_user?: boolean | undefined;
            updated?: number | undefined;
            has_2fa?: boolean | undefined;
        }, {
            name: string;
            id: string;
            color?: string | undefined;
            team_id?: string | undefined;
            deleted?: boolean | undefined;
            real_name?: string | undefined;
            tz?: string | undefined;
            tz_label?: string | undefined;
            tz_offset?: number | undefined;
            profile?: {
                title?: string | undefined;
                email?: string | undefined;
                fields?: Record<string, unknown> | undefined;
                real_name?: string | undefined;
                phone?: string | undefined;
                skype?: string | undefined;
                real_name_normalized?: string | undefined;
                display_name?: string | undefined;
                display_name_normalized?: string | undefined;
                status_text?: string | undefined;
                status_emoji?: string | undefined;
                status_expiration?: number | undefined;
                avatar_hash?: string | undefined;
                image_original?: string | undefined;
                is_custom_image?: boolean | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
                image_24?: string | undefined;
                image_32?: string | undefined;
                image_48?: string | undefined;
                image_72?: string | undefined;
                image_192?: string | undefined;
                image_512?: string | undefined;
                image_1024?: string | undefined;
            } | undefined;
            is_admin?: boolean | undefined;
            is_owner?: boolean | undefined;
            is_primary_owner?: boolean | undefined;
            is_restricted?: boolean | undefined;
            is_ultra_restricted?: boolean | undefined;
            is_bot?: boolean | undefined;
            is_app_user?: boolean | undefined;
            updated?: number | undefined;
            has_2fa?: boolean | undefined;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_user_info";
        ok: boolean;
        user?: {
            name: string;
            id: string;
            color?: string | undefined;
            team_id?: string | undefined;
            deleted?: boolean | undefined;
            real_name?: string | undefined;
            tz?: string | undefined;
            tz_label?: string | undefined;
            tz_offset?: number | undefined;
            profile?: {
                title?: string | undefined;
                email?: string | undefined;
                fields?: Record<string, unknown> | undefined;
                real_name?: string | undefined;
                phone?: string | undefined;
                skype?: string | undefined;
                real_name_normalized?: string | undefined;
                display_name?: string | undefined;
                display_name_normalized?: string | undefined;
                status_text?: string | undefined;
                status_emoji?: string | undefined;
                status_expiration?: number | undefined;
                avatar_hash?: string | undefined;
                image_original?: string | undefined;
                is_custom_image?: boolean | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
                image_24?: string | undefined;
                image_32?: string | undefined;
                image_48?: string | undefined;
                image_72?: string | undefined;
                image_192?: string | undefined;
                image_512?: string | undefined;
                image_1024?: string | undefined;
            } | undefined;
            is_admin?: boolean | undefined;
            is_owner?: boolean | undefined;
            is_primary_owner?: boolean | undefined;
            is_restricted?: boolean | undefined;
            is_ultra_restricted?: boolean | undefined;
            is_bot?: boolean | undefined;
            is_app_user?: boolean | undefined;
            updated?: number | undefined;
            has_2fa?: boolean | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_user_info";
        ok: boolean;
        user?: {
            name: string;
            id: string;
            color?: string | undefined;
            team_id?: string | undefined;
            deleted?: boolean | undefined;
            real_name?: string | undefined;
            tz?: string | undefined;
            tz_label?: string | undefined;
            tz_offset?: number | undefined;
            profile?: {
                title?: string | undefined;
                email?: string | undefined;
                fields?: Record<string, unknown> | undefined;
                real_name?: string | undefined;
                phone?: string | undefined;
                skype?: string | undefined;
                real_name_normalized?: string | undefined;
                display_name?: string | undefined;
                display_name_normalized?: string | undefined;
                status_text?: string | undefined;
                status_emoji?: string | undefined;
                status_expiration?: number | undefined;
                avatar_hash?: string | undefined;
                image_original?: string | undefined;
                is_custom_image?: boolean | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
                image_24?: string | undefined;
                image_32?: string | undefined;
                image_48?: string | undefined;
                image_72?: string | undefined;
                image_192?: string | undefined;
                image_512?: string | undefined;
                image_1024?: string | undefined;
            } | undefined;
            is_admin?: boolean | undefined;
            is_owner?: boolean | undefined;
            is_primary_owner?: boolean | undefined;
            is_restricted?: boolean | undefined;
            is_ultra_restricted?: boolean | undefined;
            is_bot?: boolean | undefined;
            is_app_user?: boolean | undefined;
            updated?: number | undefined;
            has_2fa?: boolean | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_users">;
        ok: z.ZodBoolean;
        members: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            team_id: z.ZodOptional<z.ZodString>;
            name: z.ZodString;
            deleted: z.ZodOptional<z.ZodBoolean>;
            color: z.ZodOptional<z.ZodString>;
            real_name: z.ZodOptional<z.ZodString>;
            tz: z.ZodOptional<z.ZodString>;
            tz_label: z.ZodOptional<z.ZodString>;
            tz_offset: z.ZodOptional<z.ZodNumber>;
            profile: z.ZodOptional<z.ZodObject<{
                title: z.ZodOptional<z.ZodString>;
                phone: z.ZodOptional<z.ZodString>;
                skype: z.ZodOptional<z.ZodString>;
                real_name: z.ZodOptional<z.ZodString>;
                real_name_normalized: z.ZodOptional<z.ZodString>;
                display_name: z.ZodOptional<z.ZodString>;
                display_name_normalized: z.ZodOptional<z.ZodString>;
                fields: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                status_text: z.ZodOptional<z.ZodString>;
                status_emoji: z.ZodOptional<z.ZodString>;
                status_expiration: z.ZodOptional<z.ZodNumber>;
                avatar_hash: z.ZodOptional<z.ZodString>;
                image_original: z.ZodOptional<z.ZodString>;
                is_custom_image: z.ZodOptional<z.ZodBoolean>;
                email: z.ZodOptional<z.ZodString>;
                first_name: z.ZodOptional<z.ZodString>;
                last_name: z.ZodOptional<z.ZodString>;
                image_24: z.ZodOptional<z.ZodString>;
                image_32: z.ZodOptional<z.ZodString>;
                image_48: z.ZodOptional<z.ZodString>;
                image_72: z.ZodOptional<z.ZodString>;
                image_192: z.ZodOptional<z.ZodString>;
                image_512: z.ZodOptional<z.ZodString>;
                image_1024: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                title?: string | undefined;
                email?: string | undefined;
                fields?: Record<string, unknown> | undefined;
                real_name?: string | undefined;
                phone?: string | undefined;
                skype?: string | undefined;
                real_name_normalized?: string | undefined;
                display_name?: string | undefined;
                display_name_normalized?: string | undefined;
                status_text?: string | undefined;
                status_emoji?: string | undefined;
                status_expiration?: number | undefined;
                avatar_hash?: string | undefined;
                image_original?: string | undefined;
                is_custom_image?: boolean | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
                image_24?: string | undefined;
                image_32?: string | undefined;
                image_48?: string | undefined;
                image_72?: string | undefined;
                image_192?: string | undefined;
                image_512?: string | undefined;
                image_1024?: string | undefined;
            }, {
                title?: string | undefined;
                email?: string | undefined;
                fields?: Record<string, unknown> | undefined;
                real_name?: string | undefined;
                phone?: string | undefined;
                skype?: string | undefined;
                real_name_normalized?: string | undefined;
                display_name?: string | undefined;
                display_name_normalized?: string | undefined;
                status_text?: string | undefined;
                status_emoji?: string | undefined;
                status_expiration?: number | undefined;
                avatar_hash?: string | undefined;
                image_original?: string | undefined;
                is_custom_image?: boolean | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
                image_24?: string | undefined;
                image_32?: string | undefined;
                image_48?: string | undefined;
                image_72?: string | undefined;
                image_192?: string | undefined;
                image_512?: string | undefined;
                image_1024?: string | undefined;
            }>>;
            is_admin: z.ZodOptional<z.ZodBoolean>;
            is_owner: z.ZodOptional<z.ZodBoolean>;
            is_primary_owner: z.ZodOptional<z.ZodBoolean>;
            is_restricted: z.ZodOptional<z.ZodBoolean>;
            is_ultra_restricted: z.ZodOptional<z.ZodBoolean>;
            is_bot: z.ZodOptional<z.ZodBoolean>;
            is_app_user: z.ZodOptional<z.ZodBoolean>;
            updated: z.ZodOptional<z.ZodNumber>;
            has_2fa: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            color?: string | undefined;
            team_id?: string | undefined;
            deleted?: boolean | undefined;
            real_name?: string | undefined;
            tz?: string | undefined;
            tz_label?: string | undefined;
            tz_offset?: number | undefined;
            profile?: {
                title?: string | undefined;
                email?: string | undefined;
                fields?: Record<string, unknown> | undefined;
                real_name?: string | undefined;
                phone?: string | undefined;
                skype?: string | undefined;
                real_name_normalized?: string | undefined;
                display_name?: string | undefined;
                display_name_normalized?: string | undefined;
                status_text?: string | undefined;
                status_emoji?: string | undefined;
                status_expiration?: number | undefined;
                avatar_hash?: string | undefined;
                image_original?: string | undefined;
                is_custom_image?: boolean | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
                image_24?: string | undefined;
                image_32?: string | undefined;
                image_48?: string | undefined;
                image_72?: string | undefined;
                image_192?: string | undefined;
                image_512?: string | undefined;
                image_1024?: string | undefined;
            } | undefined;
            is_admin?: boolean | undefined;
            is_owner?: boolean | undefined;
            is_primary_owner?: boolean | undefined;
            is_restricted?: boolean | undefined;
            is_ultra_restricted?: boolean | undefined;
            is_bot?: boolean | undefined;
            is_app_user?: boolean | undefined;
            updated?: number | undefined;
            has_2fa?: boolean | undefined;
        }, {
            name: string;
            id: string;
            color?: string | undefined;
            team_id?: string | undefined;
            deleted?: boolean | undefined;
            real_name?: string | undefined;
            tz?: string | undefined;
            tz_label?: string | undefined;
            tz_offset?: number | undefined;
            profile?: {
                title?: string | undefined;
                email?: string | undefined;
                fields?: Record<string, unknown> | undefined;
                real_name?: string | undefined;
                phone?: string | undefined;
                skype?: string | undefined;
                real_name_normalized?: string | undefined;
                display_name?: string | undefined;
                display_name_normalized?: string | undefined;
                status_text?: string | undefined;
                status_emoji?: string | undefined;
                status_expiration?: number | undefined;
                avatar_hash?: string | undefined;
                image_original?: string | undefined;
                is_custom_image?: boolean | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
                image_24?: string | undefined;
                image_32?: string | undefined;
                image_48?: string | undefined;
                image_72?: string | undefined;
                image_192?: string | undefined;
                image_512?: string | undefined;
                image_1024?: string | undefined;
            } | undefined;
            is_admin?: boolean | undefined;
            is_owner?: boolean | undefined;
            is_primary_owner?: boolean | undefined;
            is_restricted?: boolean | undefined;
            is_ultra_restricted?: boolean | undefined;
            is_bot?: boolean | undefined;
            is_app_user?: boolean | undefined;
            updated?: number | undefined;
            has_2fa?: boolean | undefined;
        }>, "many">>;
        response_metadata: z.ZodOptional<z.ZodObject<{
            next_cursor: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            next_cursor: string;
        }, {
            next_cursor: string;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_users";
        ok: boolean;
        response_metadata?: {
            next_cursor: string;
        } | undefined;
        members?: {
            name: string;
            id: string;
            color?: string | undefined;
            team_id?: string | undefined;
            deleted?: boolean | undefined;
            real_name?: string | undefined;
            tz?: string | undefined;
            tz_label?: string | undefined;
            tz_offset?: number | undefined;
            profile?: {
                title?: string | undefined;
                email?: string | undefined;
                fields?: Record<string, unknown> | undefined;
                real_name?: string | undefined;
                phone?: string | undefined;
                skype?: string | undefined;
                real_name_normalized?: string | undefined;
                display_name?: string | undefined;
                display_name_normalized?: string | undefined;
                status_text?: string | undefined;
                status_emoji?: string | undefined;
                status_expiration?: number | undefined;
                avatar_hash?: string | undefined;
                image_original?: string | undefined;
                is_custom_image?: boolean | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
                image_24?: string | undefined;
                image_32?: string | undefined;
                image_48?: string | undefined;
                image_72?: string | undefined;
                image_192?: string | undefined;
                image_512?: string | undefined;
                image_1024?: string | undefined;
            } | undefined;
            is_admin?: boolean | undefined;
            is_owner?: boolean | undefined;
            is_primary_owner?: boolean | undefined;
            is_restricted?: boolean | undefined;
            is_ultra_restricted?: boolean | undefined;
            is_bot?: boolean | undefined;
            is_app_user?: boolean | undefined;
            updated?: number | undefined;
            has_2fa?: boolean | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_users";
        ok: boolean;
        response_metadata?: {
            next_cursor: string;
        } | undefined;
        members?: {
            name: string;
            id: string;
            color?: string | undefined;
            team_id?: string | undefined;
            deleted?: boolean | undefined;
            real_name?: string | undefined;
            tz?: string | undefined;
            tz_label?: string | undefined;
            tz_offset?: number | undefined;
            profile?: {
                title?: string | undefined;
                email?: string | undefined;
                fields?: Record<string, unknown> | undefined;
                real_name?: string | undefined;
                phone?: string | undefined;
                skype?: string | undefined;
                real_name_normalized?: string | undefined;
                display_name?: string | undefined;
                display_name_normalized?: string | undefined;
                status_text?: string | undefined;
                status_emoji?: string | undefined;
                status_expiration?: number | undefined;
                avatar_hash?: string | undefined;
                image_original?: string | undefined;
                is_custom_image?: boolean | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
                image_24?: string | undefined;
                image_32?: string | undefined;
                image_48?: string | undefined;
                image_72?: string | undefined;
                image_192?: string | undefined;
                image_512?: string | undefined;
                image_1024?: string | undefined;
            } | undefined;
            is_admin?: boolean | undefined;
            is_owner?: boolean | undefined;
            is_primary_owner?: boolean | undefined;
            is_restricted?: boolean | undefined;
            is_ultra_restricted?: boolean | undefined;
            is_bot?: boolean | undefined;
            is_app_user?: boolean | undefined;
            updated?: number | undefined;
            has_2fa?: boolean | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_conversation_history">;
        ok: z.ZodBoolean;
        messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodString;
            ts: z.ZodString;
            user: z.ZodOptional<z.ZodString>;
            bot_id: z.ZodOptional<z.ZodString>;
            bot_profile: z.ZodOptional<z.ZodObject<{
                name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                name?: string | undefined;
            }, {
                name?: string | undefined;
            }>>;
            username: z.ZodOptional<z.ZodString>;
            text: z.ZodOptional<z.ZodString>;
            thread_ts: z.ZodOptional<z.ZodString>;
            parent_user_id: z.ZodOptional<z.ZodString>;
            reply_count: z.ZodOptional<z.ZodNumber>;
            reply_users_count: z.ZodOptional<z.ZodNumber>;
            latest_reply: z.ZodOptional<z.ZodString>;
            reply_users: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            is_locked: z.ZodOptional<z.ZodBoolean>;
            subscribed: z.ZodOptional<z.ZodBoolean>;
            attachments: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            reactions: z.ZodOptional<z.ZodArray<z.ZodObject<{
                name: z.ZodString;
                users: z.ZodArray<z.ZodString, "many">;
                count: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                name: string;
                users: string[];
                count: number;
            }, {
                name: string;
                users: string[];
                count: number;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        }, {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        }>, "many">>;
        has_more: z.ZodOptional<z.ZodBoolean>;
        response_metadata: z.ZodOptional<z.ZodObject<{
            next_cursor: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            next_cursor: string;
        }, {
            next_cursor: string;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_conversation_history";
        ok: boolean;
        response_metadata?: {
            next_cursor: string;
        } | undefined;
        messages?: {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        }[] | undefined;
        has_more?: boolean | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_conversation_history";
        ok: boolean;
        response_metadata?: {
            next_cursor: string;
        } | undefined;
        messages?: {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        }[] | undefined;
        has_more?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_thread_replies">;
        ok: z.ZodBoolean;
        messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodString;
            ts: z.ZodString;
            user: z.ZodOptional<z.ZodString>;
            bot_id: z.ZodOptional<z.ZodString>;
            bot_profile: z.ZodOptional<z.ZodObject<{
                name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                name?: string | undefined;
            }, {
                name?: string | undefined;
            }>>;
            username: z.ZodOptional<z.ZodString>;
            text: z.ZodOptional<z.ZodString>;
            thread_ts: z.ZodOptional<z.ZodString>;
            parent_user_id: z.ZodOptional<z.ZodString>;
            reply_count: z.ZodOptional<z.ZodNumber>;
            reply_users_count: z.ZodOptional<z.ZodNumber>;
            latest_reply: z.ZodOptional<z.ZodString>;
            reply_users: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            is_locked: z.ZodOptional<z.ZodBoolean>;
            subscribed: z.ZodOptional<z.ZodBoolean>;
            attachments: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            reactions: z.ZodOptional<z.ZodArray<z.ZodObject<{
                name: z.ZodString;
                users: z.ZodArray<z.ZodString, "many">;
                count: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                name: string;
                users: string[];
                count: number;
            }, {
                name: string;
                users: string[];
                count: number;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        }, {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        }>, "many">>;
        has_more: z.ZodOptional<z.ZodBoolean>;
        response_metadata: z.ZodOptional<z.ZodObject<{
            next_cursor: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            next_cursor: string;
        }, {
            next_cursor: string;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_thread_replies";
        ok: boolean;
        response_metadata?: {
            next_cursor: string;
        } | undefined;
        messages?: {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        }[] | undefined;
        has_more?: boolean | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_thread_replies";
        ok: boolean;
        response_metadata?: {
            next_cursor: string;
        } | undefined;
        messages?: {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        }[] | undefined;
        has_more?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"update_message">;
        ok: z.ZodBoolean;
        channel: z.ZodOptional<z.ZodString>;
        ts: z.ZodOptional<z.ZodString>;
        text: z.ZodOptional<z.ZodString>;
        message: z.ZodOptional<z.ZodObject<{
            type: z.ZodString;
            ts: z.ZodString;
            user: z.ZodOptional<z.ZodString>;
            bot_id: z.ZodOptional<z.ZodString>;
            bot_profile: z.ZodOptional<z.ZodObject<{
                name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                name?: string | undefined;
            }, {
                name?: string | undefined;
            }>>;
            username: z.ZodOptional<z.ZodString>;
            text: z.ZodOptional<z.ZodString>;
            thread_ts: z.ZodOptional<z.ZodString>;
            parent_user_id: z.ZodOptional<z.ZodString>;
            reply_count: z.ZodOptional<z.ZodNumber>;
            reply_users_count: z.ZodOptional<z.ZodNumber>;
            latest_reply: z.ZodOptional<z.ZodString>;
            reply_users: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            is_locked: z.ZodOptional<z.ZodBoolean>;
            subscribed: z.ZodOptional<z.ZodBoolean>;
            attachments: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            blocks: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
            reactions: z.ZodOptional<z.ZodArray<z.ZodObject<{
                name: z.ZodString;
                users: z.ZodArray<z.ZodString, "many">;
                count: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                name: string;
                users: string[];
                count: number;
            }, {
                name: string;
                users: string[];
                count: number;
            }>, "many">>;
        }, "strip", z.ZodTypeAny, {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        }, {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "update_message";
        ok: boolean;
        message?: {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        } | undefined;
        channel?: string | undefined;
        text?: string | undefined;
        ts?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "update_message";
        ok: boolean;
        message?: {
            type: string;
            ts: string;
            text?: string | undefined;
            username?: string | undefined;
            attachments?: unknown[] | undefined;
            blocks?: unknown[] | undefined;
            thread_ts?: string | undefined;
            user?: string | undefined;
            bot_id?: string | undefined;
            bot_profile?: {
                name?: string | undefined;
            } | undefined;
            parent_user_id?: string | undefined;
            reply_count?: number | undefined;
            reply_users_count?: number | undefined;
            latest_reply?: string | undefined;
            reply_users?: string[] | undefined;
            is_locked?: boolean | undefined;
            subscribed?: boolean | undefined;
            reactions?: {
                name: string;
                users: string[];
                count: number;
            }[] | undefined;
        } | undefined;
        channel?: string | undefined;
        text?: string | undefined;
        ts?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_message">;
        ok: z.ZodBoolean;
        channel: z.ZodOptional<z.ZodString>;
        ts: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_message";
        ok: boolean;
        channel?: string | undefined;
        ts?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "delete_message";
        ok: boolean;
        channel?: string | undefined;
        ts?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"add_reaction">;
        ok: z.ZodBoolean;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "add_reaction";
        ok: boolean;
    }, {
        error: string;
        success: boolean;
        operation: "add_reaction";
        ok: boolean;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"remove_reaction">;
        ok: z.ZodBoolean;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "remove_reaction";
        ok: boolean;
    }, {
        error: string;
        success: boolean;
        operation: "remove_reaction";
        ok: boolean;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"join_channel">;
        ok: z.ZodBoolean;
        channel: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            is_channel: z.ZodOptional<z.ZodBoolean>;
            is_group: z.ZodOptional<z.ZodBoolean>;
            is_im: z.ZodOptional<z.ZodBoolean>;
            is_mpim: z.ZodOptional<z.ZodBoolean>;
            is_private: z.ZodOptional<z.ZodBoolean>;
            created: z.ZodNumber;
            is_archived: z.ZodBoolean;
            is_general: z.ZodOptional<z.ZodBoolean>;
            unlinked: z.ZodOptional<z.ZodNumber>;
            name_normalized: z.ZodOptional<z.ZodString>;
            is_shared: z.ZodOptional<z.ZodBoolean>;
            is_ext_shared: z.ZodOptional<z.ZodBoolean>;
            is_org_shared: z.ZodOptional<z.ZodBoolean>;
            shared_team_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            pending_shared: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            pending_connected_team_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            is_pending_ext_shared: z.ZodOptional<z.ZodBoolean>;
            is_member: z.ZodOptional<z.ZodBoolean>;
            is_open: z.ZodOptional<z.ZodBoolean>;
            topic: z.ZodOptional<z.ZodObject<{
                value: z.ZodString;
                creator: z.ZodString;
                last_set: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                value: string;
                creator: string;
                last_set: number;
            }, {
                value: string;
                creator: string;
                last_set: number;
            }>>;
            purpose: z.ZodOptional<z.ZodObject<{
                value: z.ZodString;
                creator: z.ZodString;
                last_set: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                value: string;
                creator: string;
                last_set: number;
            }, {
                value: string;
                creator: string;
                last_set: number;
            }>>;
            num_members: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            created: number;
            is_archived: boolean;
            is_channel?: boolean | undefined;
            is_group?: boolean | undefined;
            is_im?: boolean | undefined;
            is_mpim?: boolean | undefined;
            is_private?: boolean | undefined;
            is_general?: boolean | undefined;
            unlinked?: number | undefined;
            name_normalized?: string | undefined;
            is_shared?: boolean | undefined;
            is_ext_shared?: boolean | undefined;
            is_org_shared?: boolean | undefined;
            shared_team_ids?: string[] | undefined;
            pending_shared?: string[] | undefined;
            pending_connected_team_ids?: string[] | undefined;
            is_pending_ext_shared?: boolean | undefined;
            is_member?: boolean | undefined;
            is_open?: boolean | undefined;
            topic?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            purpose?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            num_members?: number | undefined;
        }, {
            name: string;
            id: string;
            created: number;
            is_archived: boolean;
            is_channel?: boolean | undefined;
            is_group?: boolean | undefined;
            is_im?: boolean | undefined;
            is_mpim?: boolean | undefined;
            is_private?: boolean | undefined;
            is_general?: boolean | undefined;
            unlinked?: number | undefined;
            name_normalized?: string | undefined;
            is_shared?: boolean | undefined;
            is_ext_shared?: boolean | undefined;
            is_org_shared?: boolean | undefined;
            shared_team_ids?: string[] | undefined;
            pending_shared?: string[] | undefined;
            pending_connected_team_ids?: string[] | undefined;
            is_pending_ext_shared?: boolean | undefined;
            is_member?: boolean | undefined;
            is_open?: boolean | undefined;
            topic?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            purpose?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            num_members?: number | undefined;
        }>>;
        already_in_channel: z.ZodOptional<z.ZodBoolean>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "join_channel";
        ok: boolean;
        channel?: {
            name: string;
            id: string;
            created: number;
            is_archived: boolean;
            is_channel?: boolean | undefined;
            is_group?: boolean | undefined;
            is_im?: boolean | undefined;
            is_mpim?: boolean | undefined;
            is_private?: boolean | undefined;
            is_general?: boolean | undefined;
            unlinked?: number | undefined;
            name_normalized?: string | undefined;
            is_shared?: boolean | undefined;
            is_ext_shared?: boolean | undefined;
            is_org_shared?: boolean | undefined;
            shared_team_ids?: string[] | undefined;
            pending_shared?: string[] | undefined;
            pending_connected_team_ids?: string[] | undefined;
            is_pending_ext_shared?: boolean | undefined;
            is_member?: boolean | undefined;
            is_open?: boolean | undefined;
            topic?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            purpose?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            num_members?: number | undefined;
        } | undefined;
        already_in_channel?: boolean | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "join_channel";
        ok: boolean;
        channel?: {
            name: string;
            id: string;
            created: number;
            is_archived: boolean;
            is_channel?: boolean | undefined;
            is_group?: boolean | undefined;
            is_im?: boolean | undefined;
            is_mpim?: boolean | undefined;
            is_private?: boolean | undefined;
            is_general?: boolean | undefined;
            unlinked?: number | undefined;
            name_normalized?: string | undefined;
            is_shared?: boolean | undefined;
            is_ext_shared?: boolean | undefined;
            is_org_shared?: boolean | undefined;
            shared_team_ids?: string[] | undefined;
            pending_shared?: string[] | undefined;
            pending_connected_team_ids?: string[] | undefined;
            is_pending_ext_shared?: boolean | undefined;
            is_member?: boolean | undefined;
            is_open?: boolean | undefined;
            topic?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            purpose?: {
                value: string;
                creator: string;
                last_set: number;
            } | undefined;
            num_members?: number | undefined;
        } | undefined;
        already_in_channel?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"upload_file">;
        ok: z.ZodBoolean;
        file: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            created: z.ZodNumber;
            timestamp: z.ZodNumber;
            name: z.ZodString;
            title: z.ZodOptional<z.ZodString>;
            mimetype: z.ZodString;
            filetype: z.ZodString;
            pretty_type: z.ZodString;
            user: z.ZodString;
            editable: z.ZodBoolean;
            size: z.ZodNumber;
            mode: z.ZodString;
            is_external: z.ZodBoolean;
            external_type: z.ZodString;
            is_public: z.ZodBoolean;
            public_url_shared: z.ZodBoolean;
            display_as_bot: z.ZodBoolean;
            username: z.ZodString;
            url_private: z.ZodString;
            url_private_download: z.ZodString;
            permalink: z.ZodString;
            permalink_public: z.ZodOptional<z.ZodString>;
            shares: z.ZodOptional<z.ZodObject<{
                public: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodObject<{
                    reply_users: z.ZodArray<z.ZodString, "many">;
                    reply_users_count: z.ZodNumber;
                    reply_count: z.ZodNumber;
                    ts: z.ZodString;
                    channel_name: z.ZodString;
                    team_id: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }>, "many">>>;
                private: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodObject<{
                    reply_users: z.ZodArray<z.ZodString, "many">;
                    reply_users_count: z.ZodNumber;
                    reply_count: z.ZodNumber;
                    ts: z.ZodString;
                    channel_name: z.ZodString;
                    team_id: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }>, "many">>>;
            }, "strip", z.ZodTypeAny, {
                public?: Record<string, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }[]> | undefined;
                private?: Record<string, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }[]> | undefined;
            }, {
                public?: Record<string, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }[]> | undefined;
                private?: Record<string, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }[]> | undefined;
            }>>;
            channels: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            groups: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            ims: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            has_rich_preview: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            username: string;
            user: string;
            timestamp: number;
            id: string;
            created: number;
            mimetype: string;
            filetype: string;
            pretty_type: string;
            editable: boolean;
            size: number;
            mode: string;
            is_external: boolean;
            external_type: string;
            is_public: boolean;
            public_url_shared: boolean;
            display_as_bot: boolean;
            url_private: string;
            url_private_download: string;
            permalink: string;
            title?: string | undefined;
            channels?: string[] | undefined;
            permalink_public?: string | undefined;
            shares?: {
                public?: Record<string, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }[]> | undefined;
                private?: Record<string, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }[]> | undefined;
            } | undefined;
            groups?: string[] | undefined;
            ims?: string[] | undefined;
            has_rich_preview?: boolean | undefined;
        }, {
            name: string;
            username: string;
            user: string;
            timestamp: number;
            id: string;
            created: number;
            mimetype: string;
            filetype: string;
            pretty_type: string;
            editable: boolean;
            size: number;
            mode: string;
            is_external: boolean;
            external_type: string;
            is_public: boolean;
            public_url_shared: boolean;
            display_as_bot: boolean;
            url_private: string;
            url_private_download: string;
            permalink: string;
            title?: string | undefined;
            channels?: string[] | undefined;
            permalink_public?: string | undefined;
            shares?: {
                public?: Record<string, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }[]> | undefined;
                private?: Record<string, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }[]> | undefined;
            } | undefined;
            groups?: string[] | undefined;
            ims?: string[] | undefined;
            has_rich_preview?: boolean | undefined;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "upload_file";
        ok: boolean;
        file?: {
            name: string;
            username: string;
            user: string;
            timestamp: number;
            id: string;
            created: number;
            mimetype: string;
            filetype: string;
            pretty_type: string;
            editable: boolean;
            size: number;
            mode: string;
            is_external: boolean;
            external_type: string;
            is_public: boolean;
            public_url_shared: boolean;
            display_as_bot: boolean;
            url_private: string;
            url_private_download: string;
            permalink: string;
            title?: string | undefined;
            channels?: string[] | undefined;
            permalink_public?: string | undefined;
            shares?: {
                public?: Record<string, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }[]> | undefined;
                private?: Record<string, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }[]> | undefined;
            } | undefined;
            groups?: string[] | undefined;
            ims?: string[] | undefined;
            has_rich_preview?: boolean | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "upload_file";
        ok: boolean;
        file?: {
            name: string;
            username: string;
            user: string;
            timestamp: number;
            id: string;
            created: number;
            mimetype: string;
            filetype: string;
            pretty_type: string;
            editable: boolean;
            size: number;
            mode: string;
            is_external: boolean;
            external_type: string;
            is_public: boolean;
            public_url_shared: boolean;
            display_as_bot: boolean;
            url_private: string;
            url_private_download: string;
            permalink: string;
            title?: string | undefined;
            channels?: string[] | undefined;
            permalink_public?: string | undefined;
            shares?: {
                public?: Record<string, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }[]> | undefined;
                private?: Record<string, {
                    ts: string;
                    reply_count: number;
                    reply_users_count: number;
                    reply_users: string[];
                    team_id: string;
                    channel_name: string;
                }[]> | undefined;
            } | undefined;
            groups?: string[] | undefined;
            ims?: string[] | undefined;
            has_rich_preview?: boolean | undefined;
        } | undefined;
    }>]>;
    static readonly shortDescription = "Slack integration for messaging and workspace management";
    static readonly longDescription = "\n    Comprehensive Slack integration bubble for managing messages, channels, and users.\n    Use cases:\n    - Send messages to channels or direct messages\n    - Retrieve channel information and list channels\n    - Get user information and list workspace members\n    - Manage conversation history and message operations\n    - Add/remove reactions and manage message interactions\n    \n    Security Features:\n    - Token-based authentication\n    - Parameter validation and sanitization\n    - Rate limiting awareness\n    - Comprehensive error handling\n  ";
    static readonly alias = "slack";
    constructor(params?: T, context?: BubbleContext, instanceId?: string);
    protected performAction(context?: BubbleContext): Promise<Extract<SlackResult, {
        operation: T['operation'];
    }>>;
    /**
     * Helper method to resolve channel names to channel IDs.
     * If the input looks like a channel ID (starts with C, G, or D), returns it as-is.
     * Otherwise, searches for a channel with the given name.
     */
    private resolveChannelId;
    private sendMessage;
    private listChannels;
    private getChannelInfo;
    private getUserInfo;
    private listUsers;
    private getConversationHistory;
    private getThreadReplies;
    private updateMessage;
    private deleteMessage;
    private addReaction;
    private removeReaction;
    /**
     * Upload a file to a Slack channel
     * @param params - Upload file parameters including channel and file path
     * @returns Promise resolving to upload result with file metadata
     * @throws AuthenticationError if credentials are invalid
     * @throws ExternalServiceError if Slack API call fails
     */
    private uploadFile;
    private joinChannel;
    /**
     * Choose the appropriate credential for Slack API calls
     * @returns The Slack authentication token
     * @throws AuthenticationError if credentials are not provided
     */
    protected chooseCredential(): string | undefined;
    /**
     * Make an API call to the Slack API
     * @param endpoint - The Slack API endpoint to call
     * @param params - Query parameters or request body
     * @param method - HTTP method (GET or POST)
     * @returns Promise resolving to Slack API response
     * @throws AuthenticationError if authentication token is missing
     * @throws ExternalServiceError if API call fails
     */
    private makeSlackApiCall;
}
export {};
//# sourceMappingURL=slack.d.ts.map