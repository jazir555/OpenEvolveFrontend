import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const TelegramParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"send_message">;
    chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
    text: z.ZodString;
    parse_mode: z.ZodOptional<z.ZodEnum<["HTML", "Markdown", "MarkdownV2"]>>;
    entities: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    disable_web_page_preview: z.ZodOptional<z.ZodBoolean>;
    disable_notification: z.ZodOptional<z.ZodBoolean>;
    protect_content: z.ZodOptional<z.ZodBoolean>;
    reply_to_message_id: z.ZodOptional<z.ZodNumber>;
    allow_sending_without_reply: z.ZodOptional<z.ZodBoolean>;
    reply_markup: z.ZodOptional<z.ZodUnion<[z.ZodObject<{
        inline_keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
            text: z.ZodString;
            url: z.ZodOptional<z.ZodString>;
            callback_data: z.ZodOptional<z.ZodString>;
            web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            login_url: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            switch_inline_query: z.ZodOptional<z.ZodString>;
            switch_inline_query_current_chat: z.ZodOptional<z.ZodString>;
            callback_game: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            pay: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }, {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }>, "many">, "many">;
    }, "strip", z.ZodTypeAny, {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    }, {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    }>, z.ZodObject<{
        keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
            text: z.ZodString;
            request_contact: z.ZodOptional<z.ZodBoolean>;
            request_location: z.ZodOptional<z.ZodBoolean>;
            request_poll: z.ZodOptional<z.ZodObject<{
                type: z.ZodOptional<z.ZodEnum<["quiz", "regular"]>>;
            }, "strip", z.ZodTypeAny, {
                type?: "regular" | "quiz" | undefined;
            }, {
                type?: "regular" | "quiz" | undefined;
            }>>;
            web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }, {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }>, "many">, "many">;
        is_persistent: z.ZodOptional<z.ZodBoolean>;
        resize_keyboard: z.ZodOptional<z.ZodBoolean>;
        one_time_keyboard: z.ZodOptional<z.ZodBoolean>;
        input_field_placeholder: z.ZodOptional<z.ZodString>;
        selective: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        keyboard: {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }[][];
        is_persistent?: boolean | undefined;
        resize_keyboard?: boolean | undefined;
        one_time_keyboard?: boolean | undefined;
        input_field_placeholder?: string | undefined;
        selective?: boolean | undefined;
    }, {
        keyboard: {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }[][];
        is_persistent?: boolean | undefined;
        resize_keyboard?: boolean | undefined;
        one_time_keyboard?: boolean | undefined;
        input_field_placeholder?: string | undefined;
        selective?: boolean | undefined;
    }>]>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "send_message";
    text: string;
    chat_id: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    entities?: unknown[] | undefined;
    parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
    disable_web_page_preview?: boolean | undefined;
    disable_notification?: boolean | undefined;
    protect_content?: boolean | undefined;
    reply_to_message_id?: number | undefined;
    allow_sending_without_reply?: boolean | undefined;
    reply_markup?: {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    } | {
        keyboard: {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }[][];
        is_persistent?: boolean | undefined;
        resize_keyboard?: boolean | undefined;
        one_time_keyboard?: boolean | undefined;
        input_field_placeholder?: string | undefined;
        selective?: boolean | undefined;
    } | undefined;
}, {
    operation: "send_message";
    text: string;
    chat_id: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    entities?: unknown[] | undefined;
    parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
    disable_web_page_preview?: boolean | undefined;
    disable_notification?: boolean | undefined;
    protect_content?: boolean | undefined;
    reply_to_message_id?: number | undefined;
    allow_sending_without_reply?: boolean | undefined;
    reply_markup?: {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    } | {
        keyboard: {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }[][];
        is_persistent?: boolean | undefined;
        resize_keyboard?: boolean | undefined;
        one_time_keyboard?: boolean | undefined;
        input_field_placeholder?: string | undefined;
        selective?: boolean | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"send_photo">;
    chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
    photo: z.ZodUnion<[z.ZodString, z.ZodString]>;
    caption: z.ZodOptional<z.ZodString>;
    parse_mode: z.ZodOptional<z.ZodEnum<["HTML", "Markdown", "MarkdownV2"]>>;
    caption_entities: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    has_spoiler: z.ZodOptional<z.ZodBoolean>;
    disable_notification: z.ZodOptional<z.ZodBoolean>;
    protect_content: z.ZodOptional<z.ZodBoolean>;
    reply_to_message_id: z.ZodOptional<z.ZodNumber>;
    allow_sending_without_reply: z.ZodOptional<z.ZodBoolean>;
    reply_markup: z.ZodOptional<z.ZodUnion<[z.ZodObject<{
        inline_keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
            text: z.ZodString;
            url: z.ZodOptional<z.ZodString>;
            callback_data: z.ZodOptional<z.ZodString>;
            web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            login_url: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            switch_inline_query: z.ZodOptional<z.ZodString>;
            switch_inline_query_current_chat: z.ZodOptional<z.ZodString>;
            callback_game: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            pay: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }, {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }>, "many">, "many">;
    }, "strip", z.ZodTypeAny, {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    }, {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    }>, z.ZodObject<{
        keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
            text: z.ZodString;
            request_contact: z.ZodOptional<z.ZodBoolean>;
            request_location: z.ZodOptional<z.ZodBoolean>;
            request_poll: z.ZodOptional<z.ZodObject<{
                type: z.ZodOptional<z.ZodEnum<["quiz", "regular"]>>;
            }, "strip", z.ZodTypeAny, {
                type?: "regular" | "quiz" | undefined;
            }, {
                type?: "regular" | "quiz" | undefined;
            }>>;
            web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }, {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }>, "many">, "many">;
        is_persistent: z.ZodOptional<z.ZodBoolean>;
        resize_keyboard: z.ZodOptional<z.ZodBoolean>;
        one_time_keyboard: z.ZodOptional<z.ZodBoolean>;
        input_field_placeholder: z.ZodOptional<z.ZodString>;
        selective: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        keyboard: {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }[][];
        is_persistent?: boolean | undefined;
        resize_keyboard?: boolean | undefined;
        one_time_keyboard?: boolean | undefined;
        input_field_placeholder?: string | undefined;
        selective?: boolean | undefined;
    }, {
        keyboard: {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }[][];
        is_persistent?: boolean | undefined;
        resize_keyboard?: boolean | undefined;
        one_time_keyboard?: boolean | undefined;
        input_field_placeholder?: string | undefined;
        selective?: boolean | undefined;
    }>]>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "send_photo";
    photo: string;
    chat_id: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    caption?: string | undefined;
    parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
    disable_notification?: boolean | undefined;
    protect_content?: boolean | undefined;
    reply_to_message_id?: number | undefined;
    allow_sending_without_reply?: boolean | undefined;
    reply_markup?: {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    } | {
        keyboard: {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }[][];
        is_persistent?: boolean | undefined;
        resize_keyboard?: boolean | undefined;
        one_time_keyboard?: boolean | undefined;
        input_field_placeholder?: string | undefined;
        selective?: boolean | undefined;
    } | undefined;
    caption_entities?: unknown[] | undefined;
    has_spoiler?: boolean | undefined;
}, {
    operation: "send_photo";
    photo: string;
    chat_id: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    caption?: string | undefined;
    parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
    disable_notification?: boolean | undefined;
    protect_content?: boolean | undefined;
    reply_to_message_id?: number | undefined;
    allow_sending_without_reply?: boolean | undefined;
    reply_markup?: {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    } | {
        keyboard: {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }[][];
        is_persistent?: boolean | undefined;
        resize_keyboard?: boolean | undefined;
        one_time_keyboard?: boolean | undefined;
        input_field_placeholder?: string | undefined;
        selective?: boolean | undefined;
    } | undefined;
    caption_entities?: unknown[] | undefined;
    has_spoiler?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"send_document">;
    chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
    document: z.ZodUnion<[z.ZodString, z.ZodString]>;
    thumbnail: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodString]>>;
    caption: z.ZodOptional<z.ZodString>;
    parse_mode: z.ZodOptional<z.ZodEnum<["HTML", "Markdown", "MarkdownV2"]>>;
    caption_entities: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    disable_content_type_detection: z.ZodOptional<z.ZodBoolean>;
    disable_notification: z.ZodOptional<z.ZodBoolean>;
    protect_content: z.ZodOptional<z.ZodBoolean>;
    reply_to_message_id: z.ZodOptional<z.ZodNumber>;
    allow_sending_without_reply: z.ZodOptional<z.ZodBoolean>;
    reply_markup: z.ZodOptional<z.ZodUnion<[z.ZodObject<{
        inline_keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
            text: z.ZodString;
            url: z.ZodOptional<z.ZodString>;
            callback_data: z.ZodOptional<z.ZodString>;
            web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            login_url: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            switch_inline_query: z.ZodOptional<z.ZodString>;
            switch_inline_query_current_chat: z.ZodOptional<z.ZodString>;
            callback_game: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            pay: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }, {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }>, "many">, "many">;
    }, "strip", z.ZodTypeAny, {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    }, {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    }>, z.ZodObject<{
        keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
            text: z.ZodString;
            request_contact: z.ZodOptional<z.ZodBoolean>;
            request_location: z.ZodOptional<z.ZodBoolean>;
            request_poll: z.ZodOptional<z.ZodObject<{
                type: z.ZodOptional<z.ZodEnum<["quiz", "regular"]>>;
            }, "strip", z.ZodTypeAny, {
                type?: "regular" | "quiz" | undefined;
            }, {
                type?: "regular" | "quiz" | undefined;
            }>>;
            web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        }, "strip", z.ZodTypeAny, {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }, {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }>, "many">, "many">;
        is_persistent: z.ZodOptional<z.ZodBoolean>;
        resize_keyboard: z.ZodOptional<z.ZodBoolean>;
        one_time_keyboard: z.ZodOptional<z.ZodBoolean>;
        input_field_placeholder: z.ZodOptional<z.ZodString>;
        selective: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        keyboard: {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }[][];
        is_persistent?: boolean | undefined;
        resize_keyboard?: boolean | undefined;
        one_time_keyboard?: boolean | undefined;
        input_field_placeholder?: string | undefined;
        selective?: boolean | undefined;
    }, {
        keyboard: {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }[][];
        is_persistent?: boolean | undefined;
        resize_keyboard?: boolean | undefined;
        one_time_keyboard?: boolean | undefined;
        input_field_placeholder?: string | undefined;
        selective?: boolean | undefined;
    }>]>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "send_document";
    document: string;
    chat_id: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    caption?: string | undefined;
    thumbnail?: string | undefined;
    parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
    disable_notification?: boolean | undefined;
    protect_content?: boolean | undefined;
    reply_to_message_id?: number | undefined;
    allow_sending_without_reply?: boolean | undefined;
    reply_markup?: {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    } | {
        keyboard: {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }[][];
        is_persistent?: boolean | undefined;
        resize_keyboard?: boolean | undefined;
        one_time_keyboard?: boolean | undefined;
        input_field_placeholder?: string | undefined;
        selective?: boolean | undefined;
    } | undefined;
    caption_entities?: unknown[] | undefined;
    disable_content_type_detection?: boolean | undefined;
}, {
    operation: "send_document";
    document: string;
    chat_id: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    caption?: string | undefined;
    thumbnail?: string | undefined;
    parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
    disable_notification?: boolean | undefined;
    protect_content?: boolean | undefined;
    reply_to_message_id?: number | undefined;
    allow_sending_without_reply?: boolean | undefined;
    reply_markup?: {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    } | {
        keyboard: {
            text: string;
            web_app?: Record<string, unknown> | undefined;
            request_contact?: boolean | undefined;
            request_location?: boolean | undefined;
            request_poll?: {
                type?: "regular" | "quiz" | undefined;
            } | undefined;
        }[][];
        is_persistent?: boolean | undefined;
        resize_keyboard?: boolean | undefined;
        one_time_keyboard?: boolean | undefined;
        input_field_placeholder?: string | undefined;
        selective?: boolean | undefined;
    } | undefined;
    caption_entities?: unknown[] | undefined;
    disable_content_type_detection?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"edit_message">;
    chat_id: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodNumber]>>;
    message_id: z.ZodOptional<z.ZodNumber>;
    inline_message_id: z.ZodOptional<z.ZodString>;
    text: z.ZodString;
    parse_mode: z.ZodOptional<z.ZodEnum<["HTML", "Markdown", "MarkdownV2"]>>;
    entities: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    disable_web_page_preview: z.ZodOptional<z.ZodBoolean>;
    reply_markup: z.ZodOptional<z.ZodObject<{
        inline_keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
            text: z.ZodString;
            url: z.ZodOptional<z.ZodString>;
            callback_data: z.ZodOptional<z.ZodString>;
            web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            login_url: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            switch_inline_query: z.ZodOptional<z.ZodString>;
            switch_inline_query_current_chat: z.ZodOptional<z.ZodString>;
            callback_game: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            pay: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }, {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }>, "many">, "many">;
    }, "strip", z.ZodTypeAny, {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    }, {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    }>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "edit_message";
    text: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    message_id?: number | undefined;
    entities?: unknown[] | undefined;
    chat_id?: string | number | undefined;
    parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
    disable_web_page_preview?: boolean | undefined;
    reply_markup?: {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    } | undefined;
    inline_message_id?: string | undefined;
}, {
    operation: "edit_message";
    text: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    message_id?: number | undefined;
    entities?: unknown[] | undefined;
    chat_id?: string | number | undefined;
    parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
    disable_web_page_preview?: boolean | undefined;
    reply_markup?: {
        inline_keyboard: {
            text: string;
            url?: string | undefined;
            callback_data?: string | undefined;
            web_app?: Record<string, unknown> | undefined;
            login_url?: Record<string, unknown> | undefined;
            switch_inline_query?: string | undefined;
            switch_inline_query_current_chat?: string | undefined;
            callback_game?: Record<string, unknown> | undefined;
            pay?: boolean | undefined;
        }[][];
    } | undefined;
    inline_message_id?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_message">;
    chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
    message_id: z.ZodNumber;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_message";
    message_id: number;
    chat_id: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "delete_message";
    message_id: number;
    chat_id: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_me">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_me";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_me";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_chat">;
    chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_chat";
    chat_id: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_chat";
    chat_id: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_updates">;
    offset: z.ZodOptional<z.ZodNumber>;
    limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    timeout: z.ZodOptional<z.ZodNumber>;
    allowed_updates: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_updates";
    limit: number;
    timeout?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    offset?: number | undefined;
    allowed_updates?: string[] | undefined;
}, {
    operation: "get_updates";
    timeout?: number | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    offset?: number | undefined;
    allowed_updates?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"send_chat_action">;
    chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
    action: z.ZodEnum<["typing", "upload_photo", "record_video", "upload_video", "record_voice", "upload_voice", "upload_document", "find_location", "record_video_note", "upload_video_note", "choose_sticker"]>;
    message_thread_id: z.ZodOptional<z.ZodNumber>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "send_chat_action";
    chat_id: string | number;
    action: "typing" | "upload_photo" | "record_video" | "upload_video" | "record_voice" | "upload_voice" | "upload_document" | "find_location" | "record_video_note" | "upload_video_note" | "choose_sticker";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    message_thread_id?: number | undefined;
}, {
    operation: "send_chat_action";
    chat_id: string | number;
    action: "typing" | "upload_photo" | "record_video" | "upload_video" | "record_voice" | "upload_voice" | "upload_document" | "find_location" | "record_video_note" | "upload_video_note" | "choose_sticker";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    message_thread_id?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"set_message_reaction">;
    chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
    message_id: z.ZodNumber;
    reaction: z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodObject<{
        type: z.ZodLiteral<"emoji">;
        emoji: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        type: "emoji";
        emoji: string;
    }, {
        type: "emoji";
        emoji: string;
    }>, z.ZodObject<{
        type: z.ZodLiteral<"custom_emoji">;
        custom_emoji_id: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        type: "custom_emoji";
        custom_emoji_id: string;
    }, {
        type: "custom_emoji";
        custom_emoji_id: string;
    }>]>, "many">>;
    is_big: z.ZodOptional<z.ZodBoolean>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "set_message_reaction";
    message_id: number;
    chat_id: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    reaction?: ({
        type: "emoji";
        emoji: string;
    } | {
        type: "custom_emoji";
        custom_emoji_id: string;
    })[] | undefined;
    is_big?: boolean | undefined;
}, {
    operation: "set_message_reaction";
    message_id: number;
    chat_id: string | number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    reaction?: ({
        type: "emoji";
        emoji: string;
    } | {
        type: "custom_emoji";
        custom_emoji_id: string;
    })[] | undefined;
    is_big?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"set_webhook">;
    url: z.ZodUnion<[z.ZodLiteral<"">, z.ZodString]>;
    ip_address: z.ZodOptional<z.ZodString>;
    max_connections: z.ZodOptional<z.ZodNumber>;
    allowed_updates: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    drop_pending_updates: z.ZodOptional<z.ZodBoolean>;
    secret_token: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    url: string;
    operation: "set_webhook";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    allowed_updates?: string[] | undefined;
    ip_address?: string | undefined;
    max_connections?: number | undefined;
    drop_pending_updates?: boolean | undefined;
    secret_token?: string | undefined;
}, {
    url: string;
    operation: "set_webhook";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    allowed_updates?: string[] | undefined;
    ip_address?: string | undefined;
    max_connections?: number | undefined;
    drop_pending_updates?: boolean | undefined;
    secret_token?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_webhook">;
    drop_pending_updates: z.ZodOptional<z.ZodBoolean>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_webhook";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    drop_pending_updates?: boolean | undefined;
}, {
    operation: "delete_webhook";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    drop_pending_updates?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_webhook_info">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_webhook_info";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_webhook_info";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
export type TelegramParams = z.infer<typeof TelegramParamsSchema>;
export type TelegramParamsParsed = z.output<typeof TelegramParamsSchema>;
export type TelegramParamsInput = z.input<typeof TelegramParamsSchema>;
declare const TelegramResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"send_message">;
    ok: z.ZodBoolean;
    message: z.ZodOptional<z.ZodObject<{
        message_id: z.ZodNumber;
        from: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            is_bot: z.ZodBoolean;
            first_name: z.ZodString;
            last_name: z.ZodOptional<z.ZodString>;
            username: z.ZodOptional<z.ZodString>;
            language_code: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        }, {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        }>>;
        date: z.ZodNumber;
        chat: z.ZodObject<{
            id: z.ZodNumber;
            type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
            title: z.ZodOptional<z.ZodString>;
            username: z.ZodOptional<z.ZodString>;
            first_name: z.ZodOptional<z.ZodString>;
            last_name: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        }, {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        }>;
        text: z.ZodOptional<z.ZodString>;
        photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
            file_id: z.ZodString;
            file_unique_id: z.ZodString;
            width: z.ZodNumber;
            height: z.ZodNumber;
            file_size: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }, {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }>, "many">>;
        document: z.ZodOptional<z.ZodObject<{
            file_id: z.ZodString;
            file_unique_id: z.ZodString;
            file_name: z.ZodOptional<z.ZodString>;
            mime_type: z.ZodOptional<z.ZodString>;
            file_size: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        }, {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    }, {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
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
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "send_message";
    ok: boolean;
    message?: {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"send_photo">;
    ok: z.ZodBoolean;
    message: z.ZodOptional<z.ZodObject<{
        message_id: z.ZodNumber;
        from: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            is_bot: z.ZodBoolean;
            first_name: z.ZodString;
            last_name: z.ZodOptional<z.ZodString>;
            username: z.ZodOptional<z.ZodString>;
            language_code: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        }, {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        }>>;
        date: z.ZodNumber;
        chat: z.ZodObject<{
            id: z.ZodNumber;
            type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
            title: z.ZodOptional<z.ZodString>;
            username: z.ZodOptional<z.ZodString>;
            first_name: z.ZodOptional<z.ZodString>;
            last_name: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        }, {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        }>;
        text: z.ZodOptional<z.ZodString>;
        photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
            file_id: z.ZodString;
            file_unique_id: z.ZodString;
            width: z.ZodNumber;
            height: z.ZodNumber;
            file_size: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }, {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }>, "many">>;
        document: z.ZodOptional<z.ZodObject<{
            file_id: z.ZodString;
            file_unique_id: z.ZodString;
            file_name: z.ZodOptional<z.ZodString>;
            mime_type: z.ZodOptional<z.ZodString>;
            file_size: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        }, {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    }, {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "send_photo";
    ok: boolean;
    message?: {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "send_photo";
    ok: boolean;
    message?: {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"send_document">;
    ok: z.ZodBoolean;
    message: z.ZodOptional<z.ZodObject<{
        message_id: z.ZodNumber;
        from: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            is_bot: z.ZodBoolean;
            first_name: z.ZodString;
            last_name: z.ZodOptional<z.ZodString>;
            username: z.ZodOptional<z.ZodString>;
            language_code: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        }, {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        }>>;
        date: z.ZodNumber;
        chat: z.ZodObject<{
            id: z.ZodNumber;
            type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
            title: z.ZodOptional<z.ZodString>;
            username: z.ZodOptional<z.ZodString>;
            first_name: z.ZodOptional<z.ZodString>;
            last_name: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        }, {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        }>;
        text: z.ZodOptional<z.ZodString>;
        photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
            file_id: z.ZodString;
            file_unique_id: z.ZodString;
            width: z.ZodNumber;
            height: z.ZodNumber;
            file_size: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }, {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }>, "many">>;
        document: z.ZodOptional<z.ZodObject<{
            file_id: z.ZodString;
            file_unique_id: z.ZodString;
            file_name: z.ZodOptional<z.ZodString>;
            mime_type: z.ZodOptional<z.ZodString>;
            file_size: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        }, {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    }, {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "send_document";
    ok: boolean;
    message?: {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "send_document";
    ok: boolean;
    message?: {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"edit_message">;
    ok: z.ZodBoolean;
    message: z.ZodOptional<z.ZodObject<{
        message_id: z.ZodNumber;
        from: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            is_bot: z.ZodBoolean;
            first_name: z.ZodString;
            last_name: z.ZodOptional<z.ZodString>;
            username: z.ZodOptional<z.ZodString>;
            language_code: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        }, {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        }>>;
        date: z.ZodNumber;
        chat: z.ZodObject<{
            id: z.ZodNumber;
            type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
            title: z.ZodOptional<z.ZodString>;
            username: z.ZodOptional<z.ZodString>;
            first_name: z.ZodOptional<z.ZodString>;
            last_name: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        }, {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        }>;
        text: z.ZodOptional<z.ZodString>;
        photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
            file_id: z.ZodString;
            file_unique_id: z.ZodString;
            width: z.ZodNumber;
            height: z.ZodNumber;
            file_size: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }, {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }>, "many">>;
        document: z.ZodOptional<z.ZodObject<{
            file_id: z.ZodString;
            file_unique_id: z.ZodString;
            file_name: z.ZodOptional<z.ZodString>;
            mime_type: z.ZodOptional<z.ZodString>;
            file_size: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        }, {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    }, {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "edit_message";
    ok: boolean;
    message?: {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "edit_message";
    ok: boolean;
    message?: {
        date: number;
        message_id: number;
        chat: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
        };
        text?: string | undefined;
        from?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
        document?: {
            file_id: string;
            file_unique_id: string;
            mime_type?: string | undefined;
            file_size?: number | undefined;
            file_name?: string | undefined;
        } | undefined;
        photo?: {
            file_id: string;
            width: number;
            height: number;
            file_unique_id: string;
            file_size?: number | undefined;
        }[] | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_message">;
    ok: z.ZodBoolean;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_message";
    ok: boolean;
}, {
    error: string;
    success: boolean;
    operation: "delete_message";
    ok: boolean;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_me">;
    ok: z.ZodBoolean;
    user: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        is_bot: z.ZodBoolean;
        first_name: z.ZodString;
        last_name: z.ZodOptional<z.ZodString>;
        username: z.ZodOptional<z.ZodString>;
        language_code: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        id: number;
        first_name: string;
        is_bot: boolean;
        username?: string | undefined;
        last_name?: string | undefined;
        language_code?: string | undefined;
    }, {
        id: number;
        first_name: string;
        is_bot: boolean;
        username?: string | undefined;
        last_name?: string | undefined;
        language_code?: string | undefined;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_me";
    ok: boolean;
    user?: {
        id: number;
        first_name: string;
        is_bot: boolean;
        username?: string | undefined;
        last_name?: string | undefined;
        language_code?: string | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_me";
    ok: boolean;
    user?: {
        id: number;
        first_name: string;
        is_bot: boolean;
        username?: string | undefined;
        last_name?: string | undefined;
        language_code?: string | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_chat">;
    ok: z.ZodBoolean;
    chat: z.ZodOptional<z.ZodObject<{
        id: z.ZodNumber;
        type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
        title: z.ZodOptional<z.ZodString>;
        username: z.ZodOptional<z.ZodString>;
        first_name: z.ZodOptional<z.ZodString>;
        last_name: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        invite_link: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        type: "channel" | "private" | "group" | "supergroup";
        id: number;
        description?: string | undefined;
        title?: string | undefined;
        username?: string | undefined;
        first_name?: string | undefined;
        last_name?: string | undefined;
        invite_link?: string | undefined;
    }, {
        type: "channel" | "private" | "group" | "supergroup";
        id: number;
        description?: string | undefined;
        title?: string | undefined;
        username?: string | undefined;
        first_name?: string | undefined;
        last_name?: string | undefined;
        invite_link?: string | undefined;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_chat";
    ok: boolean;
    chat?: {
        type: "channel" | "private" | "group" | "supergroup";
        id: number;
        description?: string | undefined;
        title?: string | undefined;
        username?: string | undefined;
        first_name?: string | undefined;
        last_name?: string | undefined;
        invite_link?: string | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_chat";
    ok: boolean;
    chat?: {
        type: "channel" | "private" | "group" | "supergroup";
        id: number;
        description?: string | undefined;
        title?: string | undefined;
        username?: string | undefined;
        first_name?: string | undefined;
        last_name?: string | undefined;
        invite_link?: string | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_updates">;
    ok: z.ZodBoolean;
    updates: z.ZodOptional<z.ZodArray<z.ZodObject<{
        update_id: z.ZodNumber;
        message: z.ZodOptional<z.ZodObject<{
            message_id: z.ZodNumber;
            from: z.ZodOptional<z.ZodObject<{
                id: z.ZodNumber;
                is_bot: z.ZodBoolean;
                first_name: z.ZodString;
                last_name: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                language_code: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }>>;
            date: z.ZodNumber;
            chat: z.ZodObject<{
                id: z.ZodNumber;
                type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                title: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                first_name: z.ZodOptional<z.ZodString>;
                last_name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }>;
            text: z.ZodOptional<z.ZodString>;
            photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                width: z.ZodNumber;
                height: z.ZodNumber;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }>, "many">>;
            document: z.ZodOptional<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                file_name: z.ZodOptional<z.ZodString>;
                mime_type: z.ZodOptional<z.ZodString>;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }>>;
        edited_message: z.ZodOptional<z.ZodObject<{
            message_id: z.ZodNumber;
            from: z.ZodOptional<z.ZodObject<{
                id: z.ZodNumber;
                is_bot: z.ZodBoolean;
                first_name: z.ZodString;
                last_name: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                language_code: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }>>;
            date: z.ZodNumber;
            chat: z.ZodObject<{
                id: z.ZodNumber;
                type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                title: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                first_name: z.ZodOptional<z.ZodString>;
                last_name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }>;
            text: z.ZodOptional<z.ZodString>;
            photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                width: z.ZodNumber;
                height: z.ZodNumber;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }>, "many">>;
            document: z.ZodOptional<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                file_name: z.ZodOptional<z.ZodString>;
                mime_type: z.ZodOptional<z.ZodString>;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }>>;
        channel_post: z.ZodOptional<z.ZodObject<{
            message_id: z.ZodNumber;
            from: z.ZodOptional<z.ZodObject<{
                id: z.ZodNumber;
                is_bot: z.ZodBoolean;
                first_name: z.ZodString;
                last_name: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                language_code: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }>>;
            date: z.ZodNumber;
            chat: z.ZodObject<{
                id: z.ZodNumber;
                type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                title: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                first_name: z.ZodOptional<z.ZodString>;
                last_name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }>;
            text: z.ZodOptional<z.ZodString>;
            photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                width: z.ZodNumber;
                height: z.ZodNumber;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }>, "many">>;
            document: z.ZodOptional<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                file_name: z.ZodOptional<z.ZodString>;
                mime_type: z.ZodOptional<z.ZodString>;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }>>;
        edited_channel_post: z.ZodOptional<z.ZodObject<{
            message_id: z.ZodNumber;
            from: z.ZodOptional<z.ZodObject<{
                id: z.ZodNumber;
                is_bot: z.ZodBoolean;
                first_name: z.ZodString;
                last_name: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                language_code: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }>>;
            date: z.ZodNumber;
            chat: z.ZodObject<{
                id: z.ZodNumber;
                type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                title: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                first_name: z.ZodOptional<z.ZodString>;
                last_name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }>;
            text: z.ZodOptional<z.ZodString>;
            photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                width: z.ZodNumber;
                height: z.ZodNumber;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }>, "many">>;
            document: z.ZodOptional<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                file_name: z.ZodOptional<z.ZodString>;
                mime_type: z.ZodOptional<z.ZodString>;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }>>;
        callback_query: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            from: z.ZodObject<{
                id: z.ZodNumber;
                is_bot: z.ZodBoolean;
                first_name: z.ZodString;
                last_name: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                language_code: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }>;
            message: z.ZodOptional<z.ZodObject<{
                message_id: z.ZodNumber;
                from: z.ZodOptional<z.ZodObject<{
                    id: z.ZodNumber;
                    is_bot: z.ZodBoolean;
                    first_name: z.ZodString;
                    last_name: z.ZodOptional<z.ZodString>;
                    username: z.ZodOptional<z.ZodString>;
                    language_code: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                }, {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                }>>;
                date: z.ZodNumber;
                chat: z.ZodObject<{
                    id: z.ZodNumber;
                    type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                    title: z.ZodOptional<z.ZodString>;
                    username: z.ZodOptional<z.ZodString>;
                    first_name: z.ZodOptional<z.ZodString>;
                    last_name: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                }, {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                }>;
                text: z.ZodOptional<z.ZodString>;
                photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    file_id: z.ZodString;
                    file_unique_id: z.ZodString;
                    width: z.ZodNumber;
                    height: z.ZodNumber;
                    file_size: z.ZodOptional<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }, {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }>, "many">>;
                document: z.ZodOptional<z.ZodObject<{
                    file_id: z.ZodString;
                    file_unique_id: z.ZodString;
                    file_name: z.ZodOptional<z.ZodString>;
                    mime_type: z.ZodOptional<z.ZodString>;
                    file_size: z.ZodOptional<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                }, {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                }>>;
            }, "strip", z.ZodTypeAny, {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            }, {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            }>>;
            inline_message_id: z.ZodOptional<z.ZodString>;
            chat_instance: z.ZodString;
            data: z.ZodOptional<z.ZodString>;
            game_short_name: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            from: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            };
            chat_instance: string;
            message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            data?: string | undefined;
            inline_message_id?: string | undefined;
            game_short_name?: string | undefined;
        }, {
            id: string;
            from: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            };
            chat_instance: string;
            message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            data?: string | undefined;
            inline_message_id?: string | undefined;
            game_short_name?: string | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        update_id: number;
        message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        edited_message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        channel_post?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        edited_channel_post?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        callback_query?: {
            id: string;
            from: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            };
            chat_instance: string;
            message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            data?: string | undefined;
            inline_message_id?: string | undefined;
            game_short_name?: string | undefined;
        } | undefined;
    }, {
        update_id: number;
        message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        edited_message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        channel_post?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        edited_channel_post?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        callback_query?: {
            id: string;
            from: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            };
            chat_instance: string;
            message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            data?: string | undefined;
            inline_message_id?: string | undefined;
            game_short_name?: string | undefined;
        } | undefined;
    }>, "many">>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_updates";
    ok: boolean;
    updates?: {
        update_id: number;
        message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        edited_message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        channel_post?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        edited_channel_post?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        callback_query?: {
            id: string;
            from: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            };
            chat_instance: string;
            message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            data?: string | undefined;
            inline_message_id?: string | undefined;
            game_short_name?: string | undefined;
        } | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_updates";
    ok: boolean;
    updates?: {
        update_id: number;
        message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        edited_message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        channel_post?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        edited_channel_post?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
        callback_query?: {
            id: string;
            from: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            };
            chat_instance: string;
            message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            data?: string | undefined;
            inline_message_id?: string | undefined;
            game_short_name?: string | undefined;
        } | undefined;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"send_chat_action">;
    ok: z.ZodBoolean;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "send_chat_action";
    ok: boolean;
}, {
    error: string;
    success: boolean;
    operation: "send_chat_action";
    ok: boolean;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"set_message_reaction">;
    ok: z.ZodBoolean;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "set_message_reaction";
    ok: boolean;
}, {
    error: string;
    success: boolean;
    operation: "set_message_reaction";
    ok: boolean;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"set_webhook">;
    ok: z.ZodBoolean;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "set_webhook";
    ok: boolean;
}, {
    error: string;
    success: boolean;
    operation: "set_webhook";
    ok: boolean;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_webhook">;
    ok: z.ZodBoolean;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_webhook";
    ok: boolean;
}, {
    error: string;
    success: boolean;
    operation: "delete_webhook";
    ok: boolean;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_webhook_info">;
    ok: z.ZodBoolean;
    webhook_info: z.ZodOptional<z.ZodObject<{
        url: z.ZodString;
        has_custom_certificate: z.ZodBoolean;
        pending_update_count: z.ZodNumber;
        ip_address: z.ZodOptional<z.ZodString>;
        last_error_date: z.ZodOptional<z.ZodNumber>;
        last_error_message: z.ZodOptional<z.ZodString>;
        last_synchronization_error_date: z.ZodOptional<z.ZodNumber>;
        max_connections: z.ZodOptional<z.ZodNumber>;
        allowed_updates: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        url: string;
        has_custom_certificate: boolean;
        pending_update_count: number;
        allowed_updates?: string[] | undefined;
        ip_address?: string | undefined;
        max_connections?: number | undefined;
        last_error_date?: number | undefined;
        last_error_message?: string | undefined;
        last_synchronization_error_date?: number | undefined;
    }, {
        url: string;
        has_custom_certificate: boolean;
        pending_update_count: number;
        allowed_updates?: string[] | undefined;
        ip_address?: string | undefined;
        max_connections?: number | undefined;
        last_error_date?: number | undefined;
        last_error_message?: string | undefined;
        last_synchronization_error_date?: number | undefined;
    }>>;
    error: z.ZodString;
    success: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_webhook_info";
    ok: boolean;
    webhook_info?: {
        url: string;
        has_custom_certificate: boolean;
        pending_update_count: number;
        allowed_updates?: string[] | undefined;
        ip_address?: string | undefined;
        max_connections?: number | undefined;
        last_error_date?: number | undefined;
        last_error_message?: string | undefined;
        last_synchronization_error_date?: number | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_webhook_info";
    ok: boolean;
    webhook_info?: {
        url: string;
        has_custom_certificate: boolean;
        pending_update_count: number;
        allowed_updates?: string[] | undefined;
        ip_address?: string | undefined;
        max_connections?: number | undefined;
        last_error_date?: number | undefined;
        last_error_message?: string | undefined;
        last_synchronization_error_date?: number | undefined;
    } | undefined;
}>]>;
export type TelegramResult = z.infer<typeof TelegramResultSchema>;
export declare class TelegramBubble<T extends TelegramParams = TelegramParams> extends ServiceBubble<T, Extract<TelegramResult, {
    operation: T['operation'];
}>> {
    testCredential(): Promise<boolean>;
    static readonly type: "service";
    static readonly service = "telegram";
    static readonly authType: "apikey";
    static readonly bubbleName = "telegram";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"send_message">;
        chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
        text: z.ZodString;
        parse_mode: z.ZodOptional<z.ZodEnum<["HTML", "Markdown", "MarkdownV2"]>>;
        entities: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        disable_web_page_preview: z.ZodOptional<z.ZodBoolean>;
        disable_notification: z.ZodOptional<z.ZodBoolean>;
        protect_content: z.ZodOptional<z.ZodBoolean>;
        reply_to_message_id: z.ZodOptional<z.ZodNumber>;
        allow_sending_without_reply: z.ZodOptional<z.ZodBoolean>;
        reply_markup: z.ZodOptional<z.ZodUnion<[z.ZodObject<{
            inline_keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
                text: z.ZodString;
                url: z.ZodOptional<z.ZodString>;
                callback_data: z.ZodOptional<z.ZodString>;
                web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                login_url: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                switch_inline_query: z.ZodOptional<z.ZodString>;
                switch_inline_query_current_chat: z.ZodOptional<z.ZodString>;
                callback_game: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                pay: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }, {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }>, "many">, "many">;
        }, "strip", z.ZodTypeAny, {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        }, {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        }>, z.ZodObject<{
            keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
                text: z.ZodString;
                request_contact: z.ZodOptional<z.ZodBoolean>;
                request_location: z.ZodOptional<z.ZodBoolean>;
                request_poll: z.ZodOptional<z.ZodObject<{
                    type: z.ZodOptional<z.ZodEnum<["quiz", "regular"]>>;
                }, "strip", z.ZodTypeAny, {
                    type?: "regular" | "quiz" | undefined;
                }, {
                    type?: "regular" | "quiz" | undefined;
                }>>;
                web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            }, "strip", z.ZodTypeAny, {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }, {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }>, "many">, "many">;
            is_persistent: z.ZodOptional<z.ZodBoolean>;
            resize_keyboard: z.ZodOptional<z.ZodBoolean>;
            one_time_keyboard: z.ZodOptional<z.ZodBoolean>;
            input_field_placeholder: z.ZodOptional<z.ZodString>;
            selective: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            keyboard: {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }[][];
            is_persistent?: boolean | undefined;
            resize_keyboard?: boolean | undefined;
            one_time_keyboard?: boolean | undefined;
            input_field_placeholder?: string | undefined;
            selective?: boolean | undefined;
        }, {
            keyboard: {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }[][];
            is_persistent?: boolean | undefined;
            resize_keyboard?: boolean | undefined;
            one_time_keyboard?: boolean | undefined;
            input_field_placeholder?: string | undefined;
            selective?: boolean | undefined;
        }>]>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "send_message";
        text: string;
        chat_id: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        entities?: unknown[] | undefined;
        parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
        disable_web_page_preview?: boolean | undefined;
        disable_notification?: boolean | undefined;
        protect_content?: boolean | undefined;
        reply_to_message_id?: number | undefined;
        allow_sending_without_reply?: boolean | undefined;
        reply_markup?: {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        } | {
            keyboard: {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }[][];
            is_persistent?: boolean | undefined;
            resize_keyboard?: boolean | undefined;
            one_time_keyboard?: boolean | undefined;
            input_field_placeholder?: string | undefined;
            selective?: boolean | undefined;
        } | undefined;
    }, {
        operation: "send_message";
        text: string;
        chat_id: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        entities?: unknown[] | undefined;
        parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
        disable_web_page_preview?: boolean | undefined;
        disable_notification?: boolean | undefined;
        protect_content?: boolean | undefined;
        reply_to_message_id?: number | undefined;
        allow_sending_without_reply?: boolean | undefined;
        reply_markup?: {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        } | {
            keyboard: {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }[][];
            is_persistent?: boolean | undefined;
            resize_keyboard?: boolean | undefined;
            one_time_keyboard?: boolean | undefined;
            input_field_placeholder?: string | undefined;
            selective?: boolean | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"send_photo">;
        chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
        photo: z.ZodUnion<[z.ZodString, z.ZodString]>;
        caption: z.ZodOptional<z.ZodString>;
        parse_mode: z.ZodOptional<z.ZodEnum<["HTML", "Markdown", "MarkdownV2"]>>;
        caption_entities: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        has_spoiler: z.ZodOptional<z.ZodBoolean>;
        disable_notification: z.ZodOptional<z.ZodBoolean>;
        protect_content: z.ZodOptional<z.ZodBoolean>;
        reply_to_message_id: z.ZodOptional<z.ZodNumber>;
        allow_sending_without_reply: z.ZodOptional<z.ZodBoolean>;
        reply_markup: z.ZodOptional<z.ZodUnion<[z.ZodObject<{
            inline_keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
                text: z.ZodString;
                url: z.ZodOptional<z.ZodString>;
                callback_data: z.ZodOptional<z.ZodString>;
                web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                login_url: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                switch_inline_query: z.ZodOptional<z.ZodString>;
                switch_inline_query_current_chat: z.ZodOptional<z.ZodString>;
                callback_game: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                pay: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }, {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }>, "many">, "many">;
        }, "strip", z.ZodTypeAny, {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        }, {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        }>, z.ZodObject<{
            keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
                text: z.ZodString;
                request_contact: z.ZodOptional<z.ZodBoolean>;
                request_location: z.ZodOptional<z.ZodBoolean>;
                request_poll: z.ZodOptional<z.ZodObject<{
                    type: z.ZodOptional<z.ZodEnum<["quiz", "regular"]>>;
                }, "strip", z.ZodTypeAny, {
                    type?: "regular" | "quiz" | undefined;
                }, {
                    type?: "regular" | "quiz" | undefined;
                }>>;
                web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            }, "strip", z.ZodTypeAny, {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }, {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }>, "many">, "many">;
            is_persistent: z.ZodOptional<z.ZodBoolean>;
            resize_keyboard: z.ZodOptional<z.ZodBoolean>;
            one_time_keyboard: z.ZodOptional<z.ZodBoolean>;
            input_field_placeholder: z.ZodOptional<z.ZodString>;
            selective: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            keyboard: {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }[][];
            is_persistent?: boolean | undefined;
            resize_keyboard?: boolean | undefined;
            one_time_keyboard?: boolean | undefined;
            input_field_placeholder?: string | undefined;
            selective?: boolean | undefined;
        }, {
            keyboard: {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }[][];
            is_persistent?: boolean | undefined;
            resize_keyboard?: boolean | undefined;
            one_time_keyboard?: boolean | undefined;
            input_field_placeholder?: string | undefined;
            selective?: boolean | undefined;
        }>]>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "send_photo";
        photo: string;
        chat_id: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        caption?: string | undefined;
        parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
        disable_notification?: boolean | undefined;
        protect_content?: boolean | undefined;
        reply_to_message_id?: number | undefined;
        allow_sending_without_reply?: boolean | undefined;
        reply_markup?: {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        } | {
            keyboard: {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }[][];
            is_persistent?: boolean | undefined;
            resize_keyboard?: boolean | undefined;
            one_time_keyboard?: boolean | undefined;
            input_field_placeholder?: string | undefined;
            selective?: boolean | undefined;
        } | undefined;
        caption_entities?: unknown[] | undefined;
        has_spoiler?: boolean | undefined;
    }, {
        operation: "send_photo";
        photo: string;
        chat_id: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        caption?: string | undefined;
        parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
        disable_notification?: boolean | undefined;
        protect_content?: boolean | undefined;
        reply_to_message_id?: number | undefined;
        allow_sending_without_reply?: boolean | undefined;
        reply_markup?: {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        } | {
            keyboard: {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }[][];
            is_persistent?: boolean | undefined;
            resize_keyboard?: boolean | undefined;
            one_time_keyboard?: boolean | undefined;
            input_field_placeholder?: string | undefined;
            selective?: boolean | undefined;
        } | undefined;
        caption_entities?: unknown[] | undefined;
        has_spoiler?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"send_document">;
        chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
        document: z.ZodUnion<[z.ZodString, z.ZodString]>;
        thumbnail: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodString]>>;
        caption: z.ZodOptional<z.ZodString>;
        parse_mode: z.ZodOptional<z.ZodEnum<["HTML", "Markdown", "MarkdownV2"]>>;
        caption_entities: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        disable_content_type_detection: z.ZodOptional<z.ZodBoolean>;
        disable_notification: z.ZodOptional<z.ZodBoolean>;
        protect_content: z.ZodOptional<z.ZodBoolean>;
        reply_to_message_id: z.ZodOptional<z.ZodNumber>;
        allow_sending_without_reply: z.ZodOptional<z.ZodBoolean>;
        reply_markup: z.ZodOptional<z.ZodUnion<[z.ZodObject<{
            inline_keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
                text: z.ZodString;
                url: z.ZodOptional<z.ZodString>;
                callback_data: z.ZodOptional<z.ZodString>;
                web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                login_url: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                switch_inline_query: z.ZodOptional<z.ZodString>;
                switch_inline_query_current_chat: z.ZodOptional<z.ZodString>;
                callback_game: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                pay: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }, {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }>, "many">, "many">;
        }, "strip", z.ZodTypeAny, {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        }, {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        }>, z.ZodObject<{
            keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
                text: z.ZodString;
                request_contact: z.ZodOptional<z.ZodBoolean>;
                request_location: z.ZodOptional<z.ZodBoolean>;
                request_poll: z.ZodOptional<z.ZodObject<{
                    type: z.ZodOptional<z.ZodEnum<["quiz", "regular"]>>;
                }, "strip", z.ZodTypeAny, {
                    type?: "regular" | "quiz" | undefined;
                }, {
                    type?: "regular" | "quiz" | undefined;
                }>>;
                web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
            }, "strip", z.ZodTypeAny, {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }, {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }>, "many">, "many">;
            is_persistent: z.ZodOptional<z.ZodBoolean>;
            resize_keyboard: z.ZodOptional<z.ZodBoolean>;
            one_time_keyboard: z.ZodOptional<z.ZodBoolean>;
            input_field_placeholder: z.ZodOptional<z.ZodString>;
            selective: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            keyboard: {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }[][];
            is_persistent?: boolean | undefined;
            resize_keyboard?: boolean | undefined;
            one_time_keyboard?: boolean | undefined;
            input_field_placeholder?: string | undefined;
            selective?: boolean | undefined;
        }, {
            keyboard: {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }[][];
            is_persistent?: boolean | undefined;
            resize_keyboard?: boolean | undefined;
            one_time_keyboard?: boolean | undefined;
            input_field_placeholder?: string | undefined;
            selective?: boolean | undefined;
        }>]>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "send_document";
        document: string;
        chat_id: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        caption?: string | undefined;
        thumbnail?: string | undefined;
        parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
        disable_notification?: boolean | undefined;
        protect_content?: boolean | undefined;
        reply_to_message_id?: number | undefined;
        allow_sending_without_reply?: boolean | undefined;
        reply_markup?: {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        } | {
            keyboard: {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }[][];
            is_persistent?: boolean | undefined;
            resize_keyboard?: boolean | undefined;
            one_time_keyboard?: boolean | undefined;
            input_field_placeholder?: string | undefined;
            selective?: boolean | undefined;
        } | undefined;
        caption_entities?: unknown[] | undefined;
        disable_content_type_detection?: boolean | undefined;
    }, {
        operation: "send_document";
        document: string;
        chat_id: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        caption?: string | undefined;
        thumbnail?: string | undefined;
        parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
        disable_notification?: boolean | undefined;
        protect_content?: boolean | undefined;
        reply_to_message_id?: number | undefined;
        allow_sending_without_reply?: boolean | undefined;
        reply_markup?: {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        } | {
            keyboard: {
                text: string;
                web_app?: Record<string, unknown> | undefined;
                request_contact?: boolean | undefined;
                request_location?: boolean | undefined;
                request_poll?: {
                    type?: "regular" | "quiz" | undefined;
                } | undefined;
            }[][];
            is_persistent?: boolean | undefined;
            resize_keyboard?: boolean | undefined;
            one_time_keyboard?: boolean | undefined;
            input_field_placeholder?: string | undefined;
            selective?: boolean | undefined;
        } | undefined;
        caption_entities?: unknown[] | undefined;
        disable_content_type_detection?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"edit_message">;
        chat_id: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodNumber]>>;
        message_id: z.ZodOptional<z.ZodNumber>;
        inline_message_id: z.ZodOptional<z.ZodString>;
        text: z.ZodString;
        parse_mode: z.ZodOptional<z.ZodEnum<["HTML", "Markdown", "MarkdownV2"]>>;
        entities: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        disable_web_page_preview: z.ZodOptional<z.ZodBoolean>;
        reply_markup: z.ZodOptional<z.ZodObject<{
            inline_keyboard: z.ZodArray<z.ZodArray<z.ZodObject<{
                text: z.ZodString;
                url: z.ZodOptional<z.ZodString>;
                callback_data: z.ZodOptional<z.ZodString>;
                web_app: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                login_url: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                switch_inline_query: z.ZodOptional<z.ZodString>;
                switch_inline_query_current_chat: z.ZodOptional<z.ZodString>;
                callback_game: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
                pay: z.ZodOptional<z.ZodBoolean>;
            }, "strip", z.ZodTypeAny, {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }, {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }>, "many">, "many">;
        }, "strip", z.ZodTypeAny, {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        }, {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        }>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "edit_message";
        text: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        message_id?: number | undefined;
        entities?: unknown[] | undefined;
        chat_id?: string | number | undefined;
        parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
        disable_web_page_preview?: boolean | undefined;
        reply_markup?: {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        } | undefined;
        inline_message_id?: string | undefined;
    }, {
        operation: "edit_message";
        text: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        message_id?: number | undefined;
        entities?: unknown[] | undefined;
        chat_id?: string | number | undefined;
        parse_mode?: "HTML" | "Markdown" | "MarkdownV2" | undefined;
        disable_web_page_preview?: boolean | undefined;
        reply_markup?: {
            inline_keyboard: {
                text: string;
                url?: string | undefined;
                callback_data?: string | undefined;
                web_app?: Record<string, unknown> | undefined;
                login_url?: Record<string, unknown> | undefined;
                switch_inline_query?: string | undefined;
                switch_inline_query_current_chat?: string | undefined;
                callback_game?: Record<string, unknown> | undefined;
                pay?: boolean | undefined;
            }[][];
        } | undefined;
        inline_message_id?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_message">;
        chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
        message_id: z.ZodNumber;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_message";
        message_id: number;
        chat_id: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_message";
        message_id: number;
        chat_id: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_me">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_me";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_me";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_chat">;
        chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_chat";
        chat_id: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_chat";
        chat_id: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_updates">;
        offset: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        timeout: z.ZodOptional<z.ZodNumber>;
        allowed_updates: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_updates";
        limit: number;
        timeout?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        offset?: number | undefined;
        allowed_updates?: string[] | undefined;
    }, {
        operation: "get_updates";
        timeout?: number | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        offset?: number | undefined;
        allowed_updates?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"send_chat_action">;
        chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
        action: z.ZodEnum<["typing", "upload_photo", "record_video", "upload_video", "record_voice", "upload_voice", "upload_document", "find_location", "record_video_note", "upload_video_note", "choose_sticker"]>;
        message_thread_id: z.ZodOptional<z.ZodNumber>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "send_chat_action";
        chat_id: string | number;
        action: "typing" | "upload_photo" | "record_video" | "upload_video" | "record_voice" | "upload_voice" | "upload_document" | "find_location" | "record_video_note" | "upload_video_note" | "choose_sticker";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        message_thread_id?: number | undefined;
    }, {
        operation: "send_chat_action";
        chat_id: string | number;
        action: "typing" | "upload_photo" | "record_video" | "upload_video" | "record_voice" | "upload_voice" | "upload_document" | "find_location" | "record_video_note" | "upload_video_note" | "choose_sticker";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        message_thread_id?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"set_message_reaction">;
        chat_id: z.ZodUnion<[z.ZodString, z.ZodNumber]>;
        message_id: z.ZodNumber;
        reaction: z.ZodOptional<z.ZodArray<z.ZodUnion<[z.ZodObject<{
            type: z.ZodLiteral<"emoji">;
            emoji: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type: "emoji";
            emoji: string;
        }, {
            type: "emoji";
            emoji: string;
        }>, z.ZodObject<{
            type: z.ZodLiteral<"custom_emoji">;
            custom_emoji_id: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            type: "custom_emoji";
            custom_emoji_id: string;
        }, {
            type: "custom_emoji";
            custom_emoji_id: string;
        }>]>, "many">>;
        is_big: z.ZodOptional<z.ZodBoolean>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "set_message_reaction";
        message_id: number;
        chat_id: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        reaction?: ({
            type: "emoji";
            emoji: string;
        } | {
            type: "custom_emoji";
            custom_emoji_id: string;
        })[] | undefined;
        is_big?: boolean | undefined;
    }, {
        operation: "set_message_reaction";
        message_id: number;
        chat_id: string | number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        reaction?: ({
            type: "emoji";
            emoji: string;
        } | {
            type: "custom_emoji";
            custom_emoji_id: string;
        })[] | undefined;
        is_big?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"set_webhook">;
        url: z.ZodUnion<[z.ZodLiteral<"">, z.ZodString]>;
        ip_address: z.ZodOptional<z.ZodString>;
        max_connections: z.ZodOptional<z.ZodNumber>;
        allowed_updates: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        drop_pending_updates: z.ZodOptional<z.ZodBoolean>;
        secret_token: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        url: string;
        operation: "set_webhook";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        allowed_updates?: string[] | undefined;
        ip_address?: string | undefined;
        max_connections?: number | undefined;
        drop_pending_updates?: boolean | undefined;
        secret_token?: string | undefined;
    }, {
        url: string;
        operation: "set_webhook";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        allowed_updates?: string[] | undefined;
        ip_address?: string | undefined;
        max_connections?: number | undefined;
        drop_pending_updates?: boolean | undefined;
        secret_token?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_webhook">;
        drop_pending_updates: z.ZodOptional<z.ZodBoolean>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_webhook";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        drop_pending_updates?: boolean | undefined;
    }, {
        operation: "delete_webhook";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        drop_pending_updates?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_webhook_info">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_webhook_info";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_webhook_info";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"send_message">;
        ok: z.ZodBoolean;
        message: z.ZodOptional<z.ZodObject<{
            message_id: z.ZodNumber;
            from: z.ZodOptional<z.ZodObject<{
                id: z.ZodNumber;
                is_bot: z.ZodBoolean;
                first_name: z.ZodString;
                last_name: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                language_code: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }>>;
            date: z.ZodNumber;
            chat: z.ZodObject<{
                id: z.ZodNumber;
                type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                title: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                first_name: z.ZodOptional<z.ZodString>;
                last_name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }>;
            text: z.ZodOptional<z.ZodString>;
            photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                width: z.ZodNumber;
                height: z.ZodNumber;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }>, "many">>;
            document: z.ZodOptional<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                file_name: z.ZodOptional<z.ZodString>;
                mime_type: z.ZodOptional<z.ZodString>;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
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
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "send_message";
        ok: boolean;
        message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"send_photo">;
        ok: z.ZodBoolean;
        message: z.ZodOptional<z.ZodObject<{
            message_id: z.ZodNumber;
            from: z.ZodOptional<z.ZodObject<{
                id: z.ZodNumber;
                is_bot: z.ZodBoolean;
                first_name: z.ZodString;
                last_name: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                language_code: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }>>;
            date: z.ZodNumber;
            chat: z.ZodObject<{
                id: z.ZodNumber;
                type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                title: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                first_name: z.ZodOptional<z.ZodString>;
                last_name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }>;
            text: z.ZodOptional<z.ZodString>;
            photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                width: z.ZodNumber;
                height: z.ZodNumber;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }>, "many">>;
            document: z.ZodOptional<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                file_name: z.ZodOptional<z.ZodString>;
                mime_type: z.ZodOptional<z.ZodString>;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "send_photo";
        ok: boolean;
        message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "send_photo";
        ok: boolean;
        message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"send_document">;
        ok: z.ZodBoolean;
        message: z.ZodOptional<z.ZodObject<{
            message_id: z.ZodNumber;
            from: z.ZodOptional<z.ZodObject<{
                id: z.ZodNumber;
                is_bot: z.ZodBoolean;
                first_name: z.ZodString;
                last_name: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                language_code: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }>>;
            date: z.ZodNumber;
            chat: z.ZodObject<{
                id: z.ZodNumber;
                type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                title: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                first_name: z.ZodOptional<z.ZodString>;
                last_name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }>;
            text: z.ZodOptional<z.ZodString>;
            photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                width: z.ZodNumber;
                height: z.ZodNumber;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }>, "many">>;
            document: z.ZodOptional<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                file_name: z.ZodOptional<z.ZodString>;
                mime_type: z.ZodOptional<z.ZodString>;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "send_document";
        ok: boolean;
        message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "send_document";
        ok: boolean;
        message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"edit_message">;
        ok: z.ZodBoolean;
        message: z.ZodOptional<z.ZodObject<{
            message_id: z.ZodNumber;
            from: z.ZodOptional<z.ZodObject<{
                id: z.ZodNumber;
                is_bot: z.ZodBoolean;
                first_name: z.ZodString;
                last_name: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                language_code: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }, {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            }>>;
            date: z.ZodNumber;
            chat: z.ZodObject<{
                id: z.ZodNumber;
                type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                title: z.ZodOptional<z.ZodString>;
                username: z.ZodOptional<z.ZodString>;
                first_name: z.ZodOptional<z.ZodString>;
                last_name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }, {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            }>;
            text: z.ZodOptional<z.ZodString>;
            photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                width: z.ZodNumber;
                height: z.ZodNumber;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }, {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }>, "many">>;
            document: z.ZodOptional<z.ZodObject<{
                file_id: z.ZodString;
                file_unique_id: z.ZodString;
                file_name: z.ZodOptional<z.ZodString>;
                mime_type: z.ZodOptional<z.ZodString>;
                file_size: z.ZodOptional<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }, {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }, {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "edit_message";
        ok: boolean;
        message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "edit_message";
        ok: boolean;
        message?: {
            date: number;
            message_id: number;
            chat: {
                type: "channel" | "private" | "group" | "supergroup";
                id: number;
                title?: string | undefined;
                username?: string | undefined;
                first_name?: string | undefined;
                last_name?: string | undefined;
            };
            text?: string | undefined;
            from?: {
                id: number;
                first_name: string;
                is_bot: boolean;
                username?: string | undefined;
                last_name?: string | undefined;
                language_code?: string | undefined;
            } | undefined;
            document?: {
                file_id: string;
                file_unique_id: string;
                mime_type?: string | undefined;
                file_size?: number | undefined;
                file_name?: string | undefined;
            } | undefined;
            photo?: {
                file_id: string;
                width: number;
                height: number;
                file_unique_id: string;
                file_size?: number | undefined;
            }[] | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_message">;
        ok: z.ZodBoolean;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_message";
        ok: boolean;
    }, {
        error: string;
        success: boolean;
        operation: "delete_message";
        ok: boolean;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_me">;
        ok: z.ZodBoolean;
        user: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            is_bot: z.ZodBoolean;
            first_name: z.ZodString;
            last_name: z.ZodOptional<z.ZodString>;
            username: z.ZodOptional<z.ZodString>;
            language_code: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        }, {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_me";
        ok: boolean;
        user?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_me";
        ok: boolean;
        user?: {
            id: number;
            first_name: string;
            is_bot: boolean;
            username?: string | undefined;
            last_name?: string | undefined;
            language_code?: string | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_chat">;
        ok: z.ZodBoolean;
        chat: z.ZodOptional<z.ZodObject<{
            id: z.ZodNumber;
            type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
            title: z.ZodOptional<z.ZodString>;
            username: z.ZodOptional<z.ZodString>;
            first_name: z.ZodOptional<z.ZodString>;
            last_name: z.ZodOptional<z.ZodString>;
            description: z.ZodOptional<z.ZodString>;
            invite_link: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            description?: string | undefined;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            invite_link?: string | undefined;
        }, {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            description?: string | undefined;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            invite_link?: string | undefined;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_chat";
        ok: boolean;
        chat?: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            description?: string | undefined;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            invite_link?: string | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_chat";
        ok: boolean;
        chat?: {
            type: "channel" | "private" | "group" | "supergroup";
            id: number;
            description?: string | undefined;
            title?: string | undefined;
            username?: string | undefined;
            first_name?: string | undefined;
            last_name?: string | undefined;
            invite_link?: string | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_updates">;
        ok: z.ZodBoolean;
        updates: z.ZodOptional<z.ZodArray<z.ZodObject<{
            update_id: z.ZodNumber;
            message: z.ZodOptional<z.ZodObject<{
                message_id: z.ZodNumber;
                from: z.ZodOptional<z.ZodObject<{
                    id: z.ZodNumber;
                    is_bot: z.ZodBoolean;
                    first_name: z.ZodString;
                    last_name: z.ZodOptional<z.ZodString>;
                    username: z.ZodOptional<z.ZodString>;
                    language_code: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                }, {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                }>>;
                date: z.ZodNumber;
                chat: z.ZodObject<{
                    id: z.ZodNumber;
                    type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                    title: z.ZodOptional<z.ZodString>;
                    username: z.ZodOptional<z.ZodString>;
                    first_name: z.ZodOptional<z.ZodString>;
                    last_name: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                }, {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                }>;
                text: z.ZodOptional<z.ZodString>;
                photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    file_id: z.ZodString;
                    file_unique_id: z.ZodString;
                    width: z.ZodNumber;
                    height: z.ZodNumber;
                    file_size: z.ZodOptional<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }, {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }>, "many">>;
                document: z.ZodOptional<z.ZodObject<{
                    file_id: z.ZodString;
                    file_unique_id: z.ZodString;
                    file_name: z.ZodOptional<z.ZodString>;
                    mime_type: z.ZodOptional<z.ZodString>;
                    file_size: z.ZodOptional<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                }, {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                }>>;
            }, "strip", z.ZodTypeAny, {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            }, {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            }>>;
            edited_message: z.ZodOptional<z.ZodObject<{
                message_id: z.ZodNumber;
                from: z.ZodOptional<z.ZodObject<{
                    id: z.ZodNumber;
                    is_bot: z.ZodBoolean;
                    first_name: z.ZodString;
                    last_name: z.ZodOptional<z.ZodString>;
                    username: z.ZodOptional<z.ZodString>;
                    language_code: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                }, {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                }>>;
                date: z.ZodNumber;
                chat: z.ZodObject<{
                    id: z.ZodNumber;
                    type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                    title: z.ZodOptional<z.ZodString>;
                    username: z.ZodOptional<z.ZodString>;
                    first_name: z.ZodOptional<z.ZodString>;
                    last_name: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                }, {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                }>;
                text: z.ZodOptional<z.ZodString>;
                photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    file_id: z.ZodString;
                    file_unique_id: z.ZodString;
                    width: z.ZodNumber;
                    height: z.ZodNumber;
                    file_size: z.ZodOptional<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }, {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }>, "many">>;
                document: z.ZodOptional<z.ZodObject<{
                    file_id: z.ZodString;
                    file_unique_id: z.ZodString;
                    file_name: z.ZodOptional<z.ZodString>;
                    mime_type: z.ZodOptional<z.ZodString>;
                    file_size: z.ZodOptional<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                }, {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                }>>;
            }, "strip", z.ZodTypeAny, {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            }, {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            }>>;
            channel_post: z.ZodOptional<z.ZodObject<{
                message_id: z.ZodNumber;
                from: z.ZodOptional<z.ZodObject<{
                    id: z.ZodNumber;
                    is_bot: z.ZodBoolean;
                    first_name: z.ZodString;
                    last_name: z.ZodOptional<z.ZodString>;
                    username: z.ZodOptional<z.ZodString>;
                    language_code: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                }, {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                }>>;
                date: z.ZodNumber;
                chat: z.ZodObject<{
                    id: z.ZodNumber;
                    type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                    title: z.ZodOptional<z.ZodString>;
                    username: z.ZodOptional<z.ZodString>;
                    first_name: z.ZodOptional<z.ZodString>;
                    last_name: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                }, {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                }>;
                text: z.ZodOptional<z.ZodString>;
                photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    file_id: z.ZodString;
                    file_unique_id: z.ZodString;
                    width: z.ZodNumber;
                    height: z.ZodNumber;
                    file_size: z.ZodOptional<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }, {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }>, "many">>;
                document: z.ZodOptional<z.ZodObject<{
                    file_id: z.ZodString;
                    file_unique_id: z.ZodString;
                    file_name: z.ZodOptional<z.ZodString>;
                    mime_type: z.ZodOptional<z.ZodString>;
                    file_size: z.ZodOptional<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                }, {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                }>>;
            }, "strip", z.ZodTypeAny, {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            }, {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            }>>;
            edited_channel_post: z.ZodOptional<z.ZodObject<{
                message_id: z.ZodNumber;
                from: z.ZodOptional<z.ZodObject<{
                    id: z.ZodNumber;
                    is_bot: z.ZodBoolean;
                    first_name: z.ZodString;
                    last_name: z.ZodOptional<z.ZodString>;
                    username: z.ZodOptional<z.ZodString>;
                    language_code: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                }, {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                }>>;
                date: z.ZodNumber;
                chat: z.ZodObject<{
                    id: z.ZodNumber;
                    type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                    title: z.ZodOptional<z.ZodString>;
                    username: z.ZodOptional<z.ZodString>;
                    first_name: z.ZodOptional<z.ZodString>;
                    last_name: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                }, {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                }>;
                text: z.ZodOptional<z.ZodString>;
                photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    file_id: z.ZodString;
                    file_unique_id: z.ZodString;
                    width: z.ZodNumber;
                    height: z.ZodNumber;
                    file_size: z.ZodOptional<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }, {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }>, "many">>;
                document: z.ZodOptional<z.ZodObject<{
                    file_id: z.ZodString;
                    file_unique_id: z.ZodString;
                    file_name: z.ZodOptional<z.ZodString>;
                    mime_type: z.ZodOptional<z.ZodString>;
                    file_size: z.ZodOptional<z.ZodNumber>;
                }, "strip", z.ZodTypeAny, {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                }, {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                }>>;
            }, "strip", z.ZodTypeAny, {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            }, {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            }>>;
            callback_query: z.ZodOptional<z.ZodObject<{
                id: z.ZodString;
                from: z.ZodObject<{
                    id: z.ZodNumber;
                    is_bot: z.ZodBoolean;
                    first_name: z.ZodString;
                    last_name: z.ZodOptional<z.ZodString>;
                    username: z.ZodOptional<z.ZodString>;
                    language_code: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                }, {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                }>;
                message: z.ZodOptional<z.ZodObject<{
                    message_id: z.ZodNumber;
                    from: z.ZodOptional<z.ZodObject<{
                        id: z.ZodNumber;
                        is_bot: z.ZodBoolean;
                        first_name: z.ZodString;
                        last_name: z.ZodOptional<z.ZodString>;
                        username: z.ZodOptional<z.ZodString>;
                        language_code: z.ZodOptional<z.ZodString>;
                    }, "strip", z.ZodTypeAny, {
                        id: number;
                        first_name: string;
                        is_bot: boolean;
                        username?: string | undefined;
                        last_name?: string | undefined;
                        language_code?: string | undefined;
                    }, {
                        id: number;
                        first_name: string;
                        is_bot: boolean;
                        username?: string | undefined;
                        last_name?: string | undefined;
                        language_code?: string | undefined;
                    }>>;
                    date: z.ZodNumber;
                    chat: z.ZodObject<{
                        id: z.ZodNumber;
                        type: z.ZodEnum<["private", "group", "supergroup", "channel"]>;
                        title: z.ZodOptional<z.ZodString>;
                        username: z.ZodOptional<z.ZodString>;
                        first_name: z.ZodOptional<z.ZodString>;
                        last_name: z.ZodOptional<z.ZodString>;
                    }, "strip", z.ZodTypeAny, {
                        type: "channel" | "private" | "group" | "supergroup";
                        id: number;
                        title?: string | undefined;
                        username?: string | undefined;
                        first_name?: string | undefined;
                        last_name?: string | undefined;
                    }, {
                        type: "channel" | "private" | "group" | "supergroup";
                        id: number;
                        title?: string | undefined;
                        username?: string | undefined;
                        first_name?: string | undefined;
                        last_name?: string | undefined;
                    }>;
                    text: z.ZodOptional<z.ZodString>;
                    photo: z.ZodOptional<z.ZodArray<z.ZodObject<{
                        file_id: z.ZodString;
                        file_unique_id: z.ZodString;
                        width: z.ZodNumber;
                        height: z.ZodNumber;
                        file_size: z.ZodOptional<z.ZodNumber>;
                    }, "strip", z.ZodTypeAny, {
                        file_id: string;
                        width: number;
                        height: number;
                        file_unique_id: string;
                        file_size?: number | undefined;
                    }, {
                        file_id: string;
                        width: number;
                        height: number;
                        file_unique_id: string;
                        file_size?: number | undefined;
                    }>, "many">>;
                    document: z.ZodOptional<z.ZodObject<{
                        file_id: z.ZodString;
                        file_unique_id: z.ZodString;
                        file_name: z.ZodOptional<z.ZodString>;
                        mime_type: z.ZodOptional<z.ZodString>;
                        file_size: z.ZodOptional<z.ZodNumber>;
                    }, "strip", z.ZodTypeAny, {
                        file_id: string;
                        file_unique_id: string;
                        mime_type?: string | undefined;
                        file_size?: number | undefined;
                        file_name?: string | undefined;
                    }, {
                        file_id: string;
                        file_unique_id: string;
                        mime_type?: string | undefined;
                        file_size?: number | undefined;
                        file_name?: string | undefined;
                    }>>;
                }, "strip", z.ZodTypeAny, {
                    date: number;
                    message_id: number;
                    chat: {
                        type: "channel" | "private" | "group" | "supergroup";
                        id: number;
                        title?: string | undefined;
                        username?: string | undefined;
                        first_name?: string | undefined;
                        last_name?: string | undefined;
                    };
                    text?: string | undefined;
                    from?: {
                        id: number;
                        first_name: string;
                        is_bot: boolean;
                        username?: string | undefined;
                        last_name?: string | undefined;
                        language_code?: string | undefined;
                    } | undefined;
                    document?: {
                        file_id: string;
                        file_unique_id: string;
                        mime_type?: string | undefined;
                        file_size?: number | undefined;
                        file_name?: string | undefined;
                    } | undefined;
                    photo?: {
                        file_id: string;
                        width: number;
                        height: number;
                        file_unique_id: string;
                        file_size?: number | undefined;
                    }[] | undefined;
                }, {
                    date: number;
                    message_id: number;
                    chat: {
                        type: "channel" | "private" | "group" | "supergroup";
                        id: number;
                        title?: string | undefined;
                        username?: string | undefined;
                        first_name?: string | undefined;
                        last_name?: string | undefined;
                    };
                    text?: string | undefined;
                    from?: {
                        id: number;
                        first_name: string;
                        is_bot: boolean;
                        username?: string | undefined;
                        last_name?: string | undefined;
                        language_code?: string | undefined;
                    } | undefined;
                    document?: {
                        file_id: string;
                        file_unique_id: string;
                        mime_type?: string | undefined;
                        file_size?: number | undefined;
                        file_name?: string | undefined;
                    } | undefined;
                    photo?: {
                        file_id: string;
                        width: number;
                        height: number;
                        file_unique_id: string;
                        file_size?: number | undefined;
                    }[] | undefined;
                }>>;
                inline_message_id: z.ZodOptional<z.ZodString>;
                chat_instance: z.ZodString;
                data: z.ZodOptional<z.ZodString>;
                game_short_name: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                id: string;
                from: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                };
                chat_instance: string;
                message?: {
                    date: number;
                    message_id: number;
                    chat: {
                        type: "channel" | "private" | "group" | "supergroup";
                        id: number;
                        title?: string | undefined;
                        username?: string | undefined;
                        first_name?: string | undefined;
                        last_name?: string | undefined;
                    };
                    text?: string | undefined;
                    from?: {
                        id: number;
                        first_name: string;
                        is_bot: boolean;
                        username?: string | undefined;
                        last_name?: string | undefined;
                        language_code?: string | undefined;
                    } | undefined;
                    document?: {
                        file_id: string;
                        file_unique_id: string;
                        mime_type?: string | undefined;
                        file_size?: number | undefined;
                        file_name?: string | undefined;
                    } | undefined;
                    photo?: {
                        file_id: string;
                        width: number;
                        height: number;
                        file_unique_id: string;
                        file_size?: number | undefined;
                    }[] | undefined;
                } | undefined;
                data?: string | undefined;
                inline_message_id?: string | undefined;
                game_short_name?: string | undefined;
            }, {
                id: string;
                from: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                };
                chat_instance: string;
                message?: {
                    date: number;
                    message_id: number;
                    chat: {
                        type: "channel" | "private" | "group" | "supergroup";
                        id: number;
                        title?: string | undefined;
                        username?: string | undefined;
                        first_name?: string | undefined;
                        last_name?: string | undefined;
                    };
                    text?: string | undefined;
                    from?: {
                        id: number;
                        first_name: string;
                        is_bot: boolean;
                        username?: string | undefined;
                        last_name?: string | undefined;
                        language_code?: string | undefined;
                    } | undefined;
                    document?: {
                        file_id: string;
                        file_unique_id: string;
                        mime_type?: string | undefined;
                        file_size?: number | undefined;
                        file_name?: string | undefined;
                    } | undefined;
                    photo?: {
                        file_id: string;
                        width: number;
                        height: number;
                        file_unique_id: string;
                        file_size?: number | undefined;
                    }[] | undefined;
                } | undefined;
                data?: string | undefined;
                inline_message_id?: string | undefined;
                game_short_name?: string | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            update_id: number;
            message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            edited_message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            channel_post?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            edited_channel_post?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            callback_query?: {
                id: string;
                from: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                };
                chat_instance: string;
                message?: {
                    date: number;
                    message_id: number;
                    chat: {
                        type: "channel" | "private" | "group" | "supergroup";
                        id: number;
                        title?: string | undefined;
                        username?: string | undefined;
                        first_name?: string | undefined;
                        last_name?: string | undefined;
                    };
                    text?: string | undefined;
                    from?: {
                        id: number;
                        first_name: string;
                        is_bot: boolean;
                        username?: string | undefined;
                        last_name?: string | undefined;
                        language_code?: string | undefined;
                    } | undefined;
                    document?: {
                        file_id: string;
                        file_unique_id: string;
                        mime_type?: string | undefined;
                        file_size?: number | undefined;
                        file_name?: string | undefined;
                    } | undefined;
                    photo?: {
                        file_id: string;
                        width: number;
                        height: number;
                        file_unique_id: string;
                        file_size?: number | undefined;
                    }[] | undefined;
                } | undefined;
                data?: string | undefined;
                inline_message_id?: string | undefined;
                game_short_name?: string | undefined;
            } | undefined;
        }, {
            update_id: number;
            message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            edited_message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            channel_post?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            edited_channel_post?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            callback_query?: {
                id: string;
                from: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                };
                chat_instance: string;
                message?: {
                    date: number;
                    message_id: number;
                    chat: {
                        type: "channel" | "private" | "group" | "supergroup";
                        id: number;
                        title?: string | undefined;
                        username?: string | undefined;
                        first_name?: string | undefined;
                        last_name?: string | undefined;
                    };
                    text?: string | undefined;
                    from?: {
                        id: number;
                        first_name: string;
                        is_bot: boolean;
                        username?: string | undefined;
                        last_name?: string | undefined;
                        language_code?: string | undefined;
                    } | undefined;
                    document?: {
                        file_id: string;
                        file_unique_id: string;
                        mime_type?: string | undefined;
                        file_size?: number | undefined;
                        file_name?: string | undefined;
                    } | undefined;
                    photo?: {
                        file_id: string;
                        width: number;
                        height: number;
                        file_unique_id: string;
                        file_size?: number | undefined;
                    }[] | undefined;
                } | undefined;
                data?: string | undefined;
                inline_message_id?: string | undefined;
                game_short_name?: string | undefined;
            } | undefined;
        }>, "many">>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_updates";
        ok: boolean;
        updates?: {
            update_id: number;
            message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            edited_message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            channel_post?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            edited_channel_post?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            callback_query?: {
                id: string;
                from: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                };
                chat_instance: string;
                message?: {
                    date: number;
                    message_id: number;
                    chat: {
                        type: "channel" | "private" | "group" | "supergroup";
                        id: number;
                        title?: string | undefined;
                        username?: string | undefined;
                        first_name?: string | undefined;
                        last_name?: string | undefined;
                    };
                    text?: string | undefined;
                    from?: {
                        id: number;
                        first_name: string;
                        is_bot: boolean;
                        username?: string | undefined;
                        last_name?: string | undefined;
                        language_code?: string | undefined;
                    } | undefined;
                    document?: {
                        file_id: string;
                        file_unique_id: string;
                        mime_type?: string | undefined;
                        file_size?: number | undefined;
                        file_name?: string | undefined;
                    } | undefined;
                    photo?: {
                        file_id: string;
                        width: number;
                        height: number;
                        file_unique_id: string;
                        file_size?: number | undefined;
                    }[] | undefined;
                } | undefined;
                data?: string | undefined;
                inline_message_id?: string | undefined;
                game_short_name?: string | undefined;
            } | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_updates";
        ok: boolean;
        updates?: {
            update_id: number;
            message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            edited_message?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            channel_post?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            edited_channel_post?: {
                date: number;
                message_id: number;
                chat: {
                    type: "channel" | "private" | "group" | "supergroup";
                    id: number;
                    title?: string | undefined;
                    username?: string | undefined;
                    first_name?: string | undefined;
                    last_name?: string | undefined;
                };
                text?: string | undefined;
                from?: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                } | undefined;
                document?: {
                    file_id: string;
                    file_unique_id: string;
                    mime_type?: string | undefined;
                    file_size?: number | undefined;
                    file_name?: string | undefined;
                } | undefined;
                photo?: {
                    file_id: string;
                    width: number;
                    height: number;
                    file_unique_id: string;
                    file_size?: number | undefined;
                }[] | undefined;
            } | undefined;
            callback_query?: {
                id: string;
                from: {
                    id: number;
                    first_name: string;
                    is_bot: boolean;
                    username?: string | undefined;
                    last_name?: string | undefined;
                    language_code?: string | undefined;
                };
                chat_instance: string;
                message?: {
                    date: number;
                    message_id: number;
                    chat: {
                        type: "channel" | "private" | "group" | "supergroup";
                        id: number;
                        title?: string | undefined;
                        username?: string | undefined;
                        first_name?: string | undefined;
                        last_name?: string | undefined;
                    };
                    text?: string | undefined;
                    from?: {
                        id: number;
                        first_name: string;
                        is_bot: boolean;
                        username?: string | undefined;
                        last_name?: string | undefined;
                        language_code?: string | undefined;
                    } | undefined;
                    document?: {
                        file_id: string;
                        file_unique_id: string;
                        mime_type?: string | undefined;
                        file_size?: number | undefined;
                        file_name?: string | undefined;
                    } | undefined;
                    photo?: {
                        file_id: string;
                        width: number;
                        height: number;
                        file_unique_id: string;
                        file_size?: number | undefined;
                    }[] | undefined;
                } | undefined;
                data?: string | undefined;
                inline_message_id?: string | undefined;
                game_short_name?: string | undefined;
            } | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"send_chat_action">;
        ok: z.ZodBoolean;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "send_chat_action";
        ok: boolean;
    }, {
        error: string;
        success: boolean;
        operation: "send_chat_action";
        ok: boolean;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"set_message_reaction">;
        ok: z.ZodBoolean;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "set_message_reaction";
        ok: boolean;
    }, {
        error: string;
        success: boolean;
        operation: "set_message_reaction";
        ok: boolean;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"set_webhook">;
        ok: z.ZodBoolean;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "set_webhook";
        ok: boolean;
    }, {
        error: string;
        success: boolean;
        operation: "set_webhook";
        ok: boolean;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_webhook">;
        ok: z.ZodBoolean;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_webhook";
        ok: boolean;
    }, {
        error: string;
        success: boolean;
        operation: "delete_webhook";
        ok: boolean;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_webhook_info">;
        ok: z.ZodBoolean;
        webhook_info: z.ZodOptional<z.ZodObject<{
            url: z.ZodString;
            has_custom_certificate: z.ZodBoolean;
            pending_update_count: z.ZodNumber;
            ip_address: z.ZodOptional<z.ZodString>;
            last_error_date: z.ZodOptional<z.ZodNumber>;
            last_error_message: z.ZodOptional<z.ZodString>;
            last_synchronization_error_date: z.ZodOptional<z.ZodNumber>;
            max_connections: z.ZodOptional<z.ZodNumber>;
            allowed_updates: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        }, "strip", z.ZodTypeAny, {
            url: string;
            has_custom_certificate: boolean;
            pending_update_count: number;
            allowed_updates?: string[] | undefined;
            ip_address?: string | undefined;
            max_connections?: number | undefined;
            last_error_date?: number | undefined;
            last_error_message?: string | undefined;
            last_synchronization_error_date?: number | undefined;
        }, {
            url: string;
            has_custom_certificate: boolean;
            pending_update_count: number;
            allowed_updates?: string[] | undefined;
            ip_address?: string | undefined;
            max_connections?: number | undefined;
            last_error_date?: number | undefined;
            last_error_message?: string | undefined;
            last_synchronization_error_date?: number | undefined;
        }>>;
        error: z.ZodString;
        success: z.ZodBoolean;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_webhook_info";
        ok: boolean;
        webhook_info?: {
            url: string;
            has_custom_certificate: boolean;
            pending_update_count: number;
            allowed_updates?: string[] | undefined;
            ip_address?: string | undefined;
            max_connections?: number | undefined;
            last_error_date?: number | undefined;
            last_error_message?: string | undefined;
            last_synchronization_error_date?: number | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_webhook_info";
        ok: boolean;
        webhook_info?: {
            url: string;
            has_custom_certificate: boolean;
            pending_update_count: number;
            allowed_updates?: string[] | undefined;
            ip_address?: string | undefined;
            max_connections?: number | undefined;
            last_error_date?: number | undefined;
            last_error_message?: string | undefined;
            last_synchronization_error_date?: number | undefined;
        } | undefined;
    }>]>;
    static readonly shortDescription = "Telegram Bot API integration for messaging and bot management";
    static readonly longDescription = "\n    Comprehensive Telegram Bot API integration bubble for managing messages, chats, and bot operations.\n    Use cases:\n    - Send text messages, photos, and documents to chats\n    - Edit and delete messages\n    - Get bot and chat information\n    - Receive updates via polling or webhooks\n    - Support for inline keyboards and reply keyboards\n    \n    Security Features:\n    - Bot token-based authentication\n    - Parameter validation and sanitization\n    - Rate limiting awareness\n    - Comprehensive error handling\n  ";
    static readonly alias = "telegram";
    constructor(params?: T, context?: BubbleContext, instanceId?: string);
    protected chooseCredential(): string | undefined;
    protected performAction(context?: BubbleContext): Promise<Extract<TelegramResult, {
        operation: T['operation'];
    }>>;
    /**
     * Make an API call to the Telegram Bot API
     */
    private makeTelegramApiCall;
    private sendMessage;
    private sendPhoto;
    private sendDocument;
    private editMessage;
    private deleteMessage;
    private getMe;
    private getChat;
    private getUpdates;
    private sendChatAction;
    private setMessageReaction;
    private setWebhook;
    private deleteWebhook;
    private getWebhookInfo;
}
export {};
//# sourceMappingURL=telegram.d.ts.map