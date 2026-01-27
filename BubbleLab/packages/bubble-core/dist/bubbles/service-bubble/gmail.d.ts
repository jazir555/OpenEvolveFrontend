import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const GmailParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"send_email">;
    to: z.ZodArray<z.ZodString, "many">;
    cc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    bcc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    subject: z.ZodString;
    body_text: z.ZodOptional<z.ZodString>;
    body_html: z.ZodOptional<z.ZodString>;
    reply_to: z.ZodOptional<z.ZodString>;
    thread_id: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "send_email";
    to: string[];
    subject: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cc?: string[] | undefined;
    bcc?: string[] | undefined;
    reply_to?: string | undefined;
    body_text?: string | undefined;
    body_html?: string | undefined;
    thread_id?: string | undefined;
}, {
    operation: "send_email";
    to: string[];
    subject: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cc?: string[] | undefined;
    bcc?: string[] | undefined;
    reply_to?: string | undefined;
    body_text?: string | undefined;
    body_html?: string | undefined;
    thread_id?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_emails">;
    query: z.ZodOptional<z.ZodString>;
    label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    include_spam_trash: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    page_token: z.ZodOptional<z.ZodString>;
    include_details: z.ZodDefault<z.ZodBoolean>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_emails";
    max_results: number;
    include_spam_trash: boolean;
    include_details: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    query?: string | undefined;
    label_ids?: string[] | undefined;
    page_token?: string | undefined;
}, {
    operation: "list_emails";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    query?: string | undefined;
    max_results?: number | undefined;
    label_ids?: string[] | undefined;
    include_spam_trash?: boolean | undefined;
    page_token?: string | undefined;
    include_details?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_email">;
    message_id: z.ZodString;
    format: z.ZodDefault<z.ZodOptional<z.ZodEnum<["minimal", "full", "raw", "metadata"]>>>;
    metadata_headers: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    format: "minimal" | "full" | "raw" | "metadata";
    operation: "get_email";
    message_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata_headers?: string[] | undefined;
}, {
    operation: "get_email";
    message_id: string;
    format?: "minimal" | "full" | "raw" | "metadata" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    metadata_headers?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"search_emails">;
    query: z.ZodString;
    max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    include_spam_trash: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    query: string;
    operation: "search_emails";
    max_results: number;
    include_spam_trash: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    query: string;
    operation: "search_emails";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    max_results?: number | undefined;
    include_spam_trash?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"mark_as_read">;
    message_ids: z.ZodArray<z.ZodString, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "mark_as_read";
    message_ids: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "mark_as_read";
    message_ids: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"mark_as_unread">;
    message_ids: z.ZodArray<z.ZodString, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "mark_as_unread";
    message_ids: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "mark_as_unread";
    message_ids: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_draft">;
    to: z.ZodArray<z.ZodString, "many">;
    cc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    bcc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    subject: z.ZodString;
    body_text: z.ZodOptional<z.ZodString>;
    body_html: z.ZodOptional<z.ZodString>;
    reply_to: z.ZodOptional<z.ZodString>;
    thread_id: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "create_draft";
    to: string[];
    subject: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cc?: string[] | undefined;
    bcc?: string[] | undefined;
    reply_to?: string | undefined;
    body_text?: string | undefined;
    body_html?: string | undefined;
    thread_id?: string | undefined;
}, {
    operation: "create_draft";
    to: string[];
    subject: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cc?: string[] | undefined;
    bcc?: string[] | undefined;
    reply_to?: string | undefined;
    body_text?: string | undefined;
    body_html?: string | undefined;
    thread_id?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"send_draft">;
    draft_id: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "send_draft";
    draft_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "send_draft";
    draft_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_drafts">;
    query: z.ZodOptional<z.ZodString>;
    max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    page_token: z.ZodOptional<z.ZodString>;
    include_spam_trash: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_drafts";
    max_results: number;
    include_spam_trash: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    query?: string | undefined;
    page_token?: string | undefined;
}, {
    operation: "list_drafts";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    query?: string | undefined;
    max_results?: number | undefined;
    include_spam_trash?: boolean | undefined;
    page_token?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_email">;
    message_id: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "delete_email";
    message_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "delete_email";
    message_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"trash_email">;
    message_id: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "trash_email";
    message_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "trash_email";
    message_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_threads">;
    query: z.ZodOptional<z.ZodString>;
    label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    include_spam_trash: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    page_token: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_threads";
    max_results: number;
    include_spam_trash: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    query?: string | undefined;
    label_ids?: string[] | undefined;
    page_token?: string | undefined;
}, {
    operation: "list_threads";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    query?: string | undefined;
    max_results?: number | undefined;
    label_ids?: string[] | undefined;
    include_spam_trash?: boolean | undefined;
    page_token?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_labels">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "list_labels";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "list_labels";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_label">;
    name: z.ZodString;
    label_list_visibility: z.ZodDefault<z.ZodOptional<z.ZodEnum<["labelShow", "labelShowIfUnread", "labelHide"]>>>;
    message_list_visibility: z.ZodDefault<z.ZodOptional<z.ZodEnum<["show", "hide"]>>>;
    background_color: z.ZodOptional<z.ZodString>;
    text_color: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    operation: "create_label";
    label_list_visibility: "labelShow" | "labelShowIfUnread" | "labelHide";
    message_list_visibility: "show" | "hide";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    background_color?: string | undefined;
    text_color?: string | undefined;
}, {
    name: string;
    operation: "create_label";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    label_list_visibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
    message_list_visibility?: "show" | "hide" | undefined;
    background_color?: string | undefined;
    text_color?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"modify_message_labels">;
    message_id: z.ZodString;
    add_label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    remove_label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "modify_message_labels";
    message_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    add_label_ids?: string[] | undefined;
    remove_label_ids?: string[] | undefined;
}, {
    operation: "modify_message_labels";
    message_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    add_label_ids?: string[] | undefined;
    remove_label_ids?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"modify_thread_labels">;
    thread_id: z.ZodString;
    add_label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    remove_label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "modify_thread_labels";
    thread_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    add_label_ids?: string[] | undefined;
    remove_label_ids?: string[] | undefined;
}, {
    operation: "modify_thread_labels";
    thread_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    add_label_ids?: string[] | undefined;
    remove_label_ids?: string[] | undefined;
}>]>;
declare const GmailResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"send_email">;
    success: z.ZodBoolean;
    message_id: z.ZodOptional<z.ZodString>;
    thread_id: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "send_email";
    thread_id?: string | undefined;
    message_id?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "send_email";
    thread_id?: string | undefined;
    message_id?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_emails">;
    success: z.ZodBoolean;
    messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        threadId: z.ZodOptional<z.ZodString>;
        labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        snippet: z.ZodOptional<z.ZodString>;
        textContent: z.ZodOptional<z.ZodString>;
        historyId: z.ZodOptional<z.ZodString>;
        internalDate: z.ZodOptional<z.ZodString>;
        sizeEstimate: z.ZodOptional<z.ZodNumber>;
        raw: z.ZodOptional<z.ZodString>;
        payload: z.ZodOptional<z.ZodObject<{
            mimeType: z.ZodOptional<z.ZodString>;
            headers: z.ZodOptional<z.ZodArray<z.ZodObject<{
                name: z.ZodString;
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
                name: string;
            }, {
                value: string;
                name: string;
            }>, "many">>;
            body: z.ZodOptional<z.ZodObject<{
                data: z.ZodOptional<z.ZodString>;
                size: z.ZodOptional<z.ZodNumber>;
                attachmentId: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            }, {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            }>>;
            parts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        }, "strip", z.ZodTypeAny, {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        }, {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        raw?: string | undefined;
        threadId?: string | undefined;
        labelIds?: string[] | undefined;
        snippet?: string | undefined;
        textContent?: string | undefined;
        historyId?: string | undefined;
        internalDate?: string | undefined;
        sizeEstimate?: number | undefined;
        payload?: {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        } | undefined;
    }, {
        id: string;
        raw?: string | undefined;
        threadId?: string | undefined;
        labelIds?: string[] | undefined;
        snippet?: string | undefined;
        textContent?: string | undefined;
        historyId?: string | undefined;
        internalDate?: string | undefined;
        sizeEstimate?: number | undefined;
        payload?: {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        } | undefined;
    }>, "many">>;
    next_page_token: z.ZodOptional<z.ZodString>;
    result_size_estimate: z.ZodOptional<z.ZodNumber>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_emails";
    messages?: {
        id: string;
        raw?: string | undefined;
        threadId?: string | undefined;
        labelIds?: string[] | undefined;
        snippet?: string | undefined;
        textContent?: string | undefined;
        historyId?: string | undefined;
        internalDate?: string | undefined;
        sizeEstimate?: number | undefined;
        payload?: {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        } | undefined;
    }[] | undefined;
    next_page_token?: string | undefined;
    result_size_estimate?: number | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_emails";
    messages?: {
        id: string;
        raw?: string | undefined;
        threadId?: string | undefined;
        labelIds?: string[] | undefined;
        snippet?: string | undefined;
        textContent?: string | undefined;
        historyId?: string | undefined;
        internalDate?: string | undefined;
        sizeEstimate?: number | undefined;
        payload?: {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        } | undefined;
    }[] | undefined;
    next_page_token?: string | undefined;
    result_size_estimate?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_email">;
    success: z.ZodBoolean;
    message: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        threadId: z.ZodOptional<z.ZodString>;
        labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        snippet: z.ZodOptional<z.ZodString>;
        textContent: z.ZodOptional<z.ZodString>;
        historyId: z.ZodOptional<z.ZodString>;
        internalDate: z.ZodOptional<z.ZodString>;
        sizeEstimate: z.ZodOptional<z.ZodNumber>;
        raw: z.ZodOptional<z.ZodString>;
        payload: z.ZodOptional<z.ZodObject<{
            mimeType: z.ZodOptional<z.ZodString>;
            headers: z.ZodOptional<z.ZodArray<z.ZodObject<{
                name: z.ZodString;
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
                name: string;
            }, {
                value: string;
                name: string;
            }>, "many">>;
            body: z.ZodOptional<z.ZodObject<{
                data: z.ZodOptional<z.ZodString>;
                size: z.ZodOptional<z.ZodNumber>;
                attachmentId: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            }, {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            }>>;
            parts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        }, "strip", z.ZodTypeAny, {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        }, {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        raw?: string | undefined;
        threadId?: string | undefined;
        labelIds?: string[] | undefined;
        snippet?: string | undefined;
        textContent?: string | undefined;
        historyId?: string | undefined;
        internalDate?: string | undefined;
        sizeEstimate?: number | undefined;
        payload?: {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        } | undefined;
    }, {
        id: string;
        raw?: string | undefined;
        threadId?: string | undefined;
        labelIds?: string[] | undefined;
        snippet?: string | undefined;
        textContent?: string | undefined;
        historyId?: string | undefined;
        internalDate?: string | undefined;
        sizeEstimate?: number | undefined;
        payload?: {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        } | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_email";
    message?: {
        id: string;
        raw?: string | undefined;
        threadId?: string | undefined;
        labelIds?: string[] | undefined;
        snippet?: string | undefined;
        textContent?: string | undefined;
        historyId?: string | undefined;
        internalDate?: string | undefined;
        sizeEstimate?: number | undefined;
        payload?: {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        } | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_email";
    message?: {
        id: string;
        raw?: string | undefined;
        threadId?: string | undefined;
        labelIds?: string[] | undefined;
        snippet?: string | undefined;
        textContent?: string | undefined;
        historyId?: string | undefined;
        internalDate?: string | undefined;
        sizeEstimate?: number | undefined;
        payload?: {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        } | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"search_emails">;
    success: z.ZodBoolean;
    messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        threadId: z.ZodOptional<z.ZodString>;
        labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        snippet: z.ZodOptional<z.ZodString>;
        textContent: z.ZodOptional<z.ZodString>;
        historyId: z.ZodOptional<z.ZodString>;
        internalDate: z.ZodOptional<z.ZodString>;
        sizeEstimate: z.ZodOptional<z.ZodNumber>;
        raw: z.ZodOptional<z.ZodString>;
        payload: z.ZodOptional<z.ZodObject<{
            mimeType: z.ZodOptional<z.ZodString>;
            headers: z.ZodOptional<z.ZodArray<z.ZodObject<{
                name: z.ZodString;
                value: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                value: string;
                name: string;
            }, {
                value: string;
                name: string;
            }>, "many">>;
            body: z.ZodOptional<z.ZodObject<{
                data: z.ZodOptional<z.ZodString>;
                size: z.ZodOptional<z.ZodNumber>;
                attachmentId: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            }, {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            }>>;
            parts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
        }, "strip", z.ZodTypeAny, {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        }, {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        raw?: string | undefined;
        threadId?: string | undefined;
        labelIds?: string[] | undefined;
        snippet?: string | undefined;
        textContent?: string | undefined;
        historyId?: string | undefined;
        internalDate?: string | undefined;
        sizeEstimate?: number | undefined;
        payload?: {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        } | undefined;
    }, {
        id: string;
        raw?: string | undefined;
        threadId?: string | undefined;
        labelIds?: string[] | undefined;
        snippet?: string | undefined;
        textContent?: string | undefined;
        historyId?: string | undefined;
        internalDate?: string | undefined;
        sizeEstimate?: number | undefined;
        payload?: {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        } | undefined;
    }>, "many">>;
    result_size_estimate: z.ZodOptional<z.ZodNumber>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "search_emails";
    messages?: {
        id: string;
        raw?: string | undefined;
        threadId?: string | undefined;
        labelIds?: string[] | undefined;
        snippet?: string | undefined;
        textContent?: string | undefined;
        historyId?: string | undefined;
        internalDate?: string | undefined;
        sizeEstimate?: number | undefined;
        payload?: {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        } | undefined;
    }[] | undefined;
    result_size_estimate?: number | undefined;
}, {
    error: string;
    success: boolean;
    operation: "search_emails";
    messages?: {
        id: string;
        raw?: string | undefined;
        threadId?: string | undefined;
        labelIds?: string[] | undefined;
        snippet?: string | undefined;
        textContent?: string | undefined;
        historyId?: string | undefined;
        internalDate?: string | undefined;
        sizeEstimate?: number | undefined;
        payload?: {
            mimeType?: string | undefined;
            headers?: {
                value: string;
                name: string;
            }[] | undefined;
            body?: {
                data?: string | undefined;
                size?: number | undefined;
                attachmentId?: string | undefined;
            } | undefined;
            parts?: any[] | undefined;
        } | undefined;
    }[] | undefined;
    result_size_estimate?: number | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"mark_as_read">;
    success: z.ZodBoolean;
    modified_messages: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "mark_as_read";
    modified_messages?: string[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "mark_as_read";
    modified_messages?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"mark_as_unread">;
    success: z.ZodBoolean;
    modified_messages: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "mark_as_unread";
    modified_messages?: string[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "mark_as_unread";
    modified_messages?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_draft">;
    success: z.ZodBoolean;
    draft: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        message: z.ZodObject<{
            id: z.ZodString;
            threadId: z.ZodOptional<z.ZodString>;
            labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            snippet: z.ZodOptional<z.ZodString>;
            textContent: z.ZodOptional<z.ZodString>;
            historyId: z.ZodOptional<z.ZodString>;
            internalDate: z.ZodOptional<z.ZodString>;
            sizeEstimate: z.ZodOptional<z.ZodNumber>;
            raw: z.ZodOptional<z.ZodString>;
            payload: z.ZodOptional<z.ZodObject<{
                mimeType: z.ZodOptional<z.ZodString>;
                headers: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    name: z.ZodString;
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                    name: string;
                }, {
                    value: string;
                    name: string;
                }>, "many">>;
                body: z.ZodOptional<z.ZodObject<{
                    data: z.ZodOptional<z.ZodString>;
                    size: z.ZodOptional<z.ZodNumber>;
                    attachmentId: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                }, {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                }>>;
                parts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
            }, "strip", z.ZodTypeAny, {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            }, {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }, {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        message: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        };
        id: string;
    }, {
        message: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        };
        id: string;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_draft";
    draft?: {
        message: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        };
        id: string;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_draft";
    draft?: {
        message: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        };
        id: string;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"send_draft">;
    success: z.ZodBoolean;
    message_id: z.ZodOptional<z.ZodString>;
    thread_id: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "send_draft";
    thread_id?: string | undefined;
    message_id?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "send_draft";
    thread_id?: string | undefined;
    message_id?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_drafts">;
    success: z.ZodBoolean;
    drafts: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        message: z.ZodObject<{
            id: z.ZodString;
            threadId: z.ZodOptional<z.ZodString>;
            labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            snippet: z.ZodOptional<z.ZodString>;
            textContent: z.ZodOptional<z.ZodString>;
            historyId: z.ZodOptional<z.ZodString>;
            internalDate: z.ZodOptional<z.ZodString>;
            sizeEstimate: z.ZodOptional<z.ZodNumber>;
            raw: z.ZodOptional<z.ZodString>;
            payload: z.ZodOptional<z.ZodObject<{
                mimeType: z.ZodOptional<z.ZodString>;
                headers: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    name: z.ZodString;
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                    name: string;
                }, {
                    value: string;
                    name: string;
                }>, "many">>;
                body: z.ZodOptional<z.ZodObject<{
                    data: z.ZodOptional<z.ZodString>;
                    size: z.ZodOptional<z.ZodNumber>;
                    attachmentId: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                }, {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                }>>;
                parts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
            }, "strip", z.ZodTypeAny, {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            }, {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }, {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        message: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        };
        id: string;
    }, {
        message: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        };
        id: string;
    }>, "many">>;
    next_page_token: z.ZodOptional<z.ZodString>;
    result_size_estimate: z.ZodOptional<z.ZodNumber>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_drafts";
    next_page_token?: string | undefined;
    result_size_estimate?: number | undefined;
    drafts?: {
        message: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        };
        id: string;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_drafts";
    next_page_token?: string | undefined;
    result_size_estimate?: number | undefined;
    drafts?: {
        message: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        };
        id: string;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"delete_email">;
    success: z.ZodBoolean;
    deleted_message_id: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "delete_email";
    deleted_message_id?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "delete_email";
    deleted_message_id?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"trash_email">;
    success: z.ZodBoolean;
    trashed_message_id: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "trash_email";
    trashed_message_id?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "trash_email";
    trashed_message_id?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_threads">;
    success: z.ZodBoolean;
    threads: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        historyId: z.ZodOptional<z.ZodString>;
        messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            threadId: z.ZodOptional<z.ZodString>;
            labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            snippet: z.ZodOptional<z.ZodString>;
            textContent: z.ZodOptional<z.ZodString>;
            historyId: z.ZodOptional<z.ZodString>;
            internalDate: z.ZodOptional<z.ZodString>;
            sizeEstimate: z.ZodOptional<z.ZodNumber>;
            raw: z.ZodOptional<z.ZodString>;
            payload: z.ZodOptional<z.ZodObject<{
                mimeType: z.ZodOptional<z.ZodString>;
                headers: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    name: z.ZodString;
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                    name: string;
                }, {
                    value: string;
                    name: string;
                }>, "many">>;
                body: z.ZodOptional<z.ZodObject<{
                    data: z.ZodOptional<z.ZodString>;
                    size: z.ZodOptional<z.ZodNumber>;
                    attachmentId: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                }, {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                }>>;
                parts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
            }, "strip", z.ZodTypeAny, {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            }, {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }, {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }>, "many">>;
        snippet: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        messages?: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }[] | undefined;
        snippet?: string | undefined;
        historyId?: string | undefined;
    }, {
        id: string;
        messages?: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }[] | undefined;
        snippet?: string | undefined;
        historyId?: string | undefined;
    }>, "many">>;
    next_page_token: z.ZodOptional<z.ZodString>;
    result_size_estimate: z.ZodOptional<z.ZodNumber>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_threads";
    next_page_token?: string | undefined;
    result_size_estimate?: number | undefined;
    threads?: {
        id: string;
        messages?: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }[] | undefined;
        snippet?: string | undefined;
        historyId?: string | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_threads";
    next_page_token?: string | undefined;
    result_size_estimate?: number | undefined;
    threads?: {
        id: string;
        messages?: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }[] | undefined;
        snippet?: string | undefined;
        historyId?: string | undefined;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"list_labels">;
    success: z.ZodBoolean;
    labels: z.ZodOptional<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        type: z.ZodOptional<z.ZodEnum<["system", "user"]>>;
        messageListVisibility: z.ZodOptional<z.ZodEnum<["show", "hide"]>>;
        labelListVisibility: z.ZodOptional<z.ZodEnum<["labelShow", "labelShowIfUnread", "labelHide"]>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        type?: "user" | "system" | undefined;
        messageListVisibility?: "show" | "hide" | undefined;
        labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
    }, {
        name: string;
        id: string;
        type?: "user" | "system" | undefined;
        messageListVisibility?: "show" | "hide" | undefined;
        labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
    }>, "many">>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "list_labels";
    labels?: {
        name: string;
        id: string;
        type?: "user" | "system" | undefined;
        messageListVisibility?: "show" | "hide" | undefined;
        labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
    }[] | undefined;
}, {
    error: string;
    success: boolean;
    operation: "list_labels";
    labels?: {
        name: string;
        id: string;
        type?: "user" | "system" | undefined;
        messageListVisibility?: "show" | "hide" | undefined;
        labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
    }[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"create_label">;
    success: z.ZodBoolean;
    label: z.ZodOptional<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        type: z.ZodOptional<z.ZodEnum<["system", "user"]>>;
        messageListVisibility: z.ZodOptional<z.ZodEnum<["show", "hide"]>>;
        labelListVisibility: z.ZodOptional<z.ZodEnum<["labelShow", "labelShowIfUnread", "labelHide"]>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        type?: "user" | "system" | undefined;
        messageListVisibility?: "show" | "hide" | undefined;
        labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
    }, {
        name: string;
        id: string;
        type?: "user" | "system" | undefined;
        messageListVisibility?: "show" | "hide" | undefined;
        labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
    }>>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "create_label";
    label?: {
        name: string;
        id: string;
        type?: "user" | "system" | undefined;
        messageListVisibility?: "show" | "hide" | undefined;
        labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
    } | undefined;
}, {
    error: string;
    success: boolean;
    operation: "create_label";
    label?: {
        name: string;
        id: string;
        type?: "user" | "system" | undefined;
        messageListVisibility?: "show" | "hide" | undefined;
        labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
    } | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"modify_message_labels">;
    success: z.ZodBoolean;
    message_id: z.ZodOptional<z.ZodString>;
    label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "modify_message_labels";
    label_ids?: string[] | undefined;
    message_id?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "modify_message_labels";
    label_ids?: string[] | undefined;
    message_id?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"modify_thread_labels">;
    success: z.ZodBoolean;
    thread_id: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "modify_thread_labels";
    thread_id?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "modify_thread_labels";
    thread_id?: string | undefined;
}>]>;
type GmailResult = z.output<typeof GmailResultSchema>;
type GmailParams = z.input<typeof GmailParamsSchema>;
export type GmailOperationResult<T extends GmailParams['operation']> = Extract<GmailResult, {
    operation: T;
}>;
export type GmailParamsInput = z.input<typeof GmailParamsSchema>;
export declare class GmailBubble<T extends GmailParams = GmailParams> extends ServiceBubble<T, Extract<GmailResult, {
    operation: T['operation'];
}>> {
    static readonly type: "service";
    static readonly service = "gmail";
    static readonly authType: "oauth";
    static readonly bubbleName = "gmail";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"send_email">;
        to: z.ZodArray<z.ZodString, "many">;
        cc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        bcc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        subject: z.ZodString;
        body_text: z.ZodOptional<z.ZodString>;
        body_html: z.ZodOptional<z.ZodString>;
        reply_to: z.ZodOptional<z.ZodString>;
        thread_id: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "send_email";
        to: string[];
        subject: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cc?: string[] | undefined;
        bcc?: string[] | undefined;
        reply_to?: string | undefined;
        body_text?: string | undefined;
        body_html?: string | undefined;
        thread_id?: string | undefined;
    }, {
        operation: "send_email";
        to: string[];
        subject: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cc?: string[] | undefined;
        bcc?: string[] | undefined;
        reply_to?: string | undefined;
        body_text?: string | undefined;
        body_html?: string | undefined;
        thread_id?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_emails">;
        query: z.ZodOptional<z.ZodString>;
        label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        include_spam_trash: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        page_token: z.ZodOptional<z.ZodString>;
        include_details: z.ZodDefault<z.ZodBoolean>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_emails";
        max_results: number;
        include_spam_trash: boolean;
        include_details: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        query?: string | undefined;
        label_ids?: string[] | undefined;
        page_token?: string | undefined;
    }, {
        operation: "list_emails";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        query?: string | undefined;
        max_results?: number | undefined;
        label_ids?: string[] | undefined;
        include_spam_trash?: boolean | undefined;
        page_token?: string | undefined;
        include_details?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_email">;
        message_id: z.ZodString;
        format: z.ZodDefault<z.ZodOptional<z.ZodEnum<["minimal", "full", "raw", "metadata"]>>>;
        metadata_headers: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        format: "minimal" | "full" | "raw" | "metadata";
        operation: "get_email";
        message_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata_headers?: string[] | undefined;
    }, {
        operation: "get_email";
        message_id: string;
        format?: "minimal" | "full" | "raw" | "metadata" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        metadata_headers?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"search_emails">;
        query: z.ZodString;
        max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        include_spam_trash: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        query: string;
        operation: "search_emails";
        max_results: number;
        include_spam_trash: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        query: string;
        operation: "search_emails";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        max_results?: number | undefined;
        include_spam_trash?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"mark_as_read">;
        message_ids: z.ZodArray<z.ZodString, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "mark_as_read";
        message_ids: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "mark_as_read";
        message_ids: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"mark_as_unread">;
        message_ids: z.ZodArray<z.ZodString, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "mark_as_unread";
        message_ids: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "mark_as_unread";
        message_ids: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_draft">;
        to: z.ZodArray<z.ZodString, "many">;
        cc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        bcc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        subject: z.ZodString;
        body_text: z.ZodOptional<z.ZodString>;
        body_html: z.ZodOptional<z.ZodString>;
        reply_to: z.ZodOptional<z.ZodString>;
        thread_id: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "create_draft";
        to: string[];
        subject: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cc?: string[] | undefined;
        bcc?: string[] | undefined;
        reply_to?: string | undefined;
        body_text?: string | undefined;
        body_html?: string | undefined;
        thread_id?: string | undefined;
    }, {
        operation: "create_draft";
        to: string[];
        subject: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cc?: string[] | undefined;
        bcc?: string[] | undefined;
        reply_to?: string | undefined;
        body_text?: string | undefined;
        body_html?: string | undefined;
        thread_id?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"send_draft">;
        draft_id: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "send_draft";
        draft_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "send_draft";
        draft_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_drafts">;
        query: z.ZodOptional<z.ZodString>;
        max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        page_token: z.ZodOptional<z.ZodString>;
        include_spam_trash: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_drafts";
        max_results: number;
        include_spam_trash: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        query?: string | undefined;
        page_token?: string | undefined;
    }, {
        operation: "list_drafts";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        query?: string | undefined;
        max_results?: number | undefined;
        include_spam_trash?: boolean | undefined;
        page_token?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_email">;
        message_id: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "delete_email";
        message_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "delete_email";
        message_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"trash_email">;
        message_id: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "trash_email";
        message_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "trash_email";
        message_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_threads">;
        query: z.ZodOptional<z.ZodString>;
        label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        include_spam_trash: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        page_token: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_threads";
        max_results: number;
        include_spam_trash: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        query?: string | undefined;
        label_ids?: string[] | undefined;
        page_token?: string | undefined;
    }, {
        operation: "list_threads";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        query?: string | undefined;
        max_results?: number | undefined;
        label_ids?: string[] | undefined;
        include_spam_trash?: boolean | undefined;
        page_token?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_labels">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "list_labels";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "list_labels";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_label">;
        name: z.ZodString;
        label_list_visibility: z.ZodDefault<z.ZodOptional<z.ZodEnum<["labelShow", "labelShowIfUnread", "labelHide"]>>>;
        message_list_visibility: z.ZodDefault<z.ZodOptional<z.ZodEnum<["show", "hide"]>>>;
        background_color: z.ZodOptional<z.ZodString>;
        text_color: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        operation: "create_label";
        label_list_visibility: "labelShow" | "labelShowIfUnread" | "labelHide";
        message_list_visibility: "show" | "hide";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        background_color?: string | undefined;
        text_color?: string | undefined;
    }, {
        name: string;
        operation: "create_label";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        label_list_visibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
        message_list_visibility?: "show" | "hide" | undefined;
        background_color?: string | undefined;
        text_color?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"modify_message_labels">;
        message_id: z.ZodString;
        add_label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        remove_label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "modify_message_labels";
        message_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        add_label_ids?: string[] | undefined;
        remove_label_ids?: string[] | undefined;
    }, {
        operation: "modify_message_labels";
        message_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        add_label_ids?: string[] | undefined;
        remove_label_ids?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"modify_thread_labels">;
        thread_id: z.ZodString;
        add_label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        remove_label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "modify_thread_labels";
        thread_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        add_label_ids?: string[] | undefined;
        remove_label_ids?: string[] | undefined;
    }, {
        operation: "modify_thread_labels";
        thread_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        add_label_ids?: string[] | undefined;
        remove_label_ids?: string[] | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"send_email">;
        success: z.ZodBoolean;
        message_id: z.ZodOptional<z.ZodString>;
        thread_id: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "send_email";
        thread_id?: string | undefined;
        message_id?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "send_email";
        thread_id?: string | undefined;
        message_id?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_emails">;
        success: z.ZodBoolean;
        messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            threadId: z.ZodOptional<z.ZodString>;
            labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            snippet: z.ZodOptional<z.ZodString>;
            textContent: z.ZodOptional<z.ZodString>;
            historyId: z.ZodOptional<z.ZodString>;
            internalDate: z.ZodOptional<z.ZodString>;
            sizeEstimate: z.ZodOptional<z.ZodNumber>;
            raw: z.ZodOptional<z.ZodString>;
            payload: z.ZodOptional<z.ZodObject<{
                mimeType: z.ZodOptional<z.ZodString>;
                headers: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    name: z.ZodString;
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                    name: string;
                }, {
                    value: string;
                    name: string;
                }>, "many">>;
                body: z.ZodOptional<z.ZodObject<{
                    data: z.ZodOptional<z.ZodString>;
                    size: z.ZodOptional<z.ZodNumber>;
                    attachmentId: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                }, {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                }>>;
                parts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
            }, "strip", z.ZodTypeAny, {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            }, {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }, {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }>, "many">>;
        next_page_token: z.ZodOptional<z.ZodString>;
        result_size_estimate: z.ZodOptional<z.ZodNumber>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_emails";
        messages?: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }[] | undefined;
        next_page_token?: string | undefined;
        result_size_estimate?: number | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_emails";
        messages?: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }[] | undefined;
        next_page_token?: string | undefined;
        result_size_estimate?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_email">;
        success: z.ZodBoolean;
        message: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            threadId: z.ZodOptional<z.ZodString>;
            labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            snippet: z.ZodOptional<z.ZodString>;
            textContent: z.ZodOptional<z.ZodString>;
            historyId: z.ZodOptional<z.ZodString>;
            internalDate: z.ZodOptional<z.ZodString>;
            sizeEstimate: z.ZodOptional<z.ZodNumber>;
            raw: z.ZodOptional<z.ZodString>;
            payload: z.ZodOptional<z.ZodObject<{
                mimeType: z.ZodOptional<z.ZodString>;
                headers: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    name: z.ZodString;
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                    name: string;
                }, {
                    value: string;
                    name: string;
                }>, "many">>;
                body: z.ZodOptional<z.ZodObject<{
                    data: z.ZodOptional<z.ZodString>;
                    size: z.ZodOptional<z.ZodNumber>;
                    attachmentId: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                }, {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                }>>;
                parts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
            }, "strip", z.ZodTypeAny, {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            }, {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }, {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_email";
        message?: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_email";
        message?: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"search_emails">;
        success: z.ZodBoolean;
        messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            threadId: z.ZodOptional<z.ZodString>;
            labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            snippet: z.ZodOptional<z.ZodString>;
            textContent: z.ZodOptional<z.ZodString>;
            historyId: z.ZodOptional<z.ZodString>;
            internalDate: z.ZodOptional<z.ZodString>;
            sizeEstimate: z.ZodOptional<z.ZodNumber>;
            raw: z.ZodOptional<z.ZodString>;
            payload: z.ZodOptional<z.ZodObject<{
                mimeType: z.ZodOptional<z.ZodString>;
                headers: z.ZodOptional<z.ZodArray<z.ZodObject<{
                    name: z.ZodString;
                    value: z.ZodString;
                }, "strip", z.ZodTypeAny, {
                    value: string;
                    name: string;
                }, {
                    value: string;
                    name: string;
                }>, "many">>;
                body: z.ZodOptional<z.ZodObject<{
                    data: z.ZodOptional<z.ZodString>;
                    size: z.ZodOptional<z.ZodNumber>;
                    attachmentId: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                }, {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                }>>;
                parts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
            }, "strip", z.ZodTypeAny, {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            }, {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }, {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }>, "many">>;
        result_size_estimate: z.ZodOptional<z.ZodNumber>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "search_emails";
        messages?: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }[] | undefined;
        result_size_estimate?: number | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "search_emails";
        messages?: {
            id: string;
            raw?: string | undefined;
            threadId?: string | undefined;
            labelIds?: string[] | undefined;
            snippet?: string | undefined;
            textContent?: string | undefined;
            historyId?: string | undefined;
            internalDate?: string | undefined;
            sizeEstimate?: number | undefined;
            payload?: {
                mimeType?: string | undefined;
                headers?: {
                    value: string;
                    name: string;
                }[] | undefined;
                body?: {
                    data?: string | undefined;
                    size?: number | undefined;
                    attachmentId?: string | undefined;
                } | undefined;
                parts?: any[] | undefined;
            } | undefined;
        }[] | undefined;
        result_size_estimate?: number | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"mark_as_read">;
        success: z.ZodBoolean;
        modified_messages: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "mark_as_read";
        modified_messages?: string[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "mark_as_read";
        modified_messages?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"mark_as_unread">;
        success: z.ZodBoolean;
        modified_messages: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "mark_as_unread";
        modified_messages?: string[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "mark_as_unread";
        modified_messages?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_draft">;
        success: z.ZodBoolean;
        draft: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            message: z.ZodObject<{
                id: z.ZodString;
                threadId: z.ZodOptional<z.ZodString>;
                labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
                snippet: z.ZodOptional<z.ZodString>;
                textContent: z.ZodOptional<z.ZodString>;
                historyId: z.ZodOptional<z.ZodString>;
                internalDate: z.ZodOptional<z.ZodString>;
                sizeEstimate: z.ZodOptional<z.ZodNumber>;
                raw: z.ZodOptional<z.ZodString>;
                payload: z.ZodOptional<z.ZodObject<{
                    mimeType: z.ZodOptional<z.ZodString>;
                    headers: z.ZodOptional<z.ZodArray<z.ZodObject<{
                        name: z.ZodString;
                        value: z.ZodString;
                    }, "strip", z.ZodTypeAny, {
                        value: string;
                        name: string;
                    }, {
                        value: string;
                        name: string;
                    }>, "many">>;
                    body: z.ZodOptional<z.ZodObject<{
                        data: z.ZodOptional<z.ZodString>;
                        size: z.ZodOptional<z.ZodNumber>;
                        attachmentId: z.ZodOptional<z.ZodString>;
                    }, "strip", z.ZodTypeAny, {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    }, {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    }>>;
                    parts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
                }, "strip", z.ZodTypeAny, {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                }, {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                }>>;
            }, "strip", z.ZodTypeAny, {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            }, {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            }>;
        }, "strip", z.ZodTypeAny, {
            message: {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            };
            id: string;
        }, {
            message: {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            };
            id: string;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_draft";
        draft?: {
            message: {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            };
            id: string;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_draft";
        draft?: {
            message: {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            };
            id: string;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"send_draft">;
        success: z.ZodBoolean;
        message_id: z.ZodOptional<z.ZodString>;
        thread_id: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "send_draft";
        thread_id?: string | undefined;
        message_id?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "send_draft";
        thread_id?: string | undefined;
        message_id?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_drafts">;
        success: z.ZodBoolean;
        drafts: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            message: z.ZodObject<{
                id: z.ZodString;
                threadId: z.ZodOptional<z.ZodString>;
                labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
                snippet: z.ZodOptional<z.ZodString>;
                textContent: z.ZodOptional<z.ZodString>;
                historyId: z.ZodOptional<z.ZodString>;
                internalDate: z.ZodOptional<z.ZodString>;
                sizeEstimate: z.ZodOptional<z.ZodNumber>;
                raw: z.ZodOptional<z.ZodString>;
                payload: z.ZodOptional<z.ZodObject<{
                    mimeType: z.ZodOptional<z.ZodString>;
                    headers: z.ZodOptional<z.ZodArray<z.ZodObject<{
                        name: z.ZodString;
                        value: z.ZodString;
                    }, "strip", z.ZodTypeAny, {
                        value: string;
                        name: string;
                    }, {
                        value: string;
                        name: string;
                    }>, "many">>;
                    body: z.ZodOptional<z.ZodObject<{
                        data: z.ZodOptional<z.ZodString>;
                        size: z.ZodOptional<z.ZodNumber>;
                        attachmentId: z.ZodOptional<z.ZodString>;
                    }, "strip", z.ZodTypeAny, {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    }, {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    }>>;
                    parts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
                }, "strip", z.ZodTypeAny, {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                }, {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                }>>;
            }, "strip", z.ZodTypeAny, {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            }, {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            }>;
        }, "strip", z.ZodTypeAny, {
            message: {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            };
            id: string;
        }, {
            message: {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            };
            id: string;
        }>, "many">>;
        next_page_token: z.ZodOptional<z.ZodString>;
        result_size_estimate: z.ZodOptional<z.ZodNumber>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_drafts";
        next_page_token?: string | undefined;
        result_size_estimate?: number | undefined;
        drafts?: {
            message: {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            };
            id: string;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_drafts";
        next_page_token?: string | undefined;
        result_size_estimate?: number | undefined;
        drafts?: {
            message: {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            };
            id: string;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"delete_email">;
        success: z.ZodBoolean;
        deleted_message_id: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "delete_email";
        deleted_message_id?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "delete_email";
        deleted_message_id?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"trash_email">;
        success: z.ZodBoolean;
        trashed_message_id: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "trash_email";
        trashed_message_id?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "trash_email";
        trashed_message_id?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_threads">;
        success: z.ZodBoolean;
        threads: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            historyId: z.ZodOptional<z.ZodString>;
            messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
                id: z.ZodString;
                threadId: z.ZodOptional<z.ZodString>;
                labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
                snippet: z.ZodOptional<z.ZodString>;
                textContent: z.ZodOptional<z.ZodString>;
                historyId: z.ZodOptional<z.ZodString>;
                internalDate: z.ZodOptional<z.ZodString>;
                sizeEstimate: z.ZodOptional<z.ZodNumber>;
                raw: z.ZodOptional<z.ZodString>;
                payload: z.ZodOptional<z.ZodObject<{
                    mimeType: z.ZodOptional<z.ZodString>;
                    headers: z.ZodOptional<z.ZodArray<z.ZodObject<{
                        name: z.ZodString;
                        value: z.ZodString;
                    }, "strip", z.ZodTypeAny, {
                        value: string;
                        name: string;
                    }, {
                        value: string;
                        name: string;
                    }>, "many">>;
                    body: z.ZodOptional<z.ZodObject<{
                        data: z.ZodOptional<z.ZodString>;
                        size: z.ZodOptional<z.ZodNumber>;
                        attachmentId: z.ZodOptional<z.ZodString>;
                    }, "strip", z.ZodTypeAny, {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    }, {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    }>>;
                    parts: z.ZodOptional<z.ZodArray<z.ZodAny, "many">>;
                }, "strip", z.ZodTypeAny, {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                }, {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                }>>;
            }, "strip", z.ZodTypeAny, {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            }, {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            }>, "many">>;
            snippet: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            id: string;
            messages?: {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            }[] | undefined;
            snippet?: string | undefined;
            historyId?: string | undefined;
        }, {
            id: string;
            messages?: {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            }[] | undefined;
            snippet?: string | undefined;
            historyId?: string | undefined;
        }>, "many">>;
        next_page_token: z.ZodOptional<z.ZodString>;
        result_size_estimate: z.ZodOptional<z.ZodNumber>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_threads";
        next_page_token?: string | undefined;
        result_size_estimate?: number | undefined;
        threads?: {
            id: string;
            messages?: {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            }[] | undefined;
            snippet?: string | undefined;
            historyId?: string | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_threads";
        next_page_token?: string | undefined;
        result_size_estimate?: number | undefined;
        threads?: {
            id: string;
            messages?: {
                id: string;
                raw?: string | undefined;
                threadId?: string | undefined;
                labelIds?: string[] | undefined;
                snippet?: string | undefined;
                textContent?: string | undefined;
                historyId?: string | undefined;
                internalDate?: string | undefined;
                sizeEstimate?: number | undefined;
                payload?: {
                    mimeType?: string | undefined;
                    headers?: {
                        value: string;
                        name: string;
                    }[] | undefined;
                    body?: {
                        data?: string | undefined;
                        size?: number | undefined;
                        attachmentId?: string | undefined;
                    } | undefined;
                    parts?: any[] | undefined;
                } | undefined;
            }[] | undefined;
            snippet?: string | undefined;
            historyId?: string | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"list_labels">;
        success: z.ZodBoolean;
        labels: z.ZodOptional<z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            type: z.ZodOptional<z.ZodEnum<["system", "user"]>>;
            messageListVisibility: z.ZodOptional<z.ZodEnum<["show", "hide"]>>;
            labelListVisibility: z.ZodOptional<z.ZodEnum<["labelShow", "labelShowIfUnread", "labelHide"]>>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            type?: "user" | "system" | undefined;
            messageListVisibility?: "show" | "hide" | undefined;
            labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
        }, {
            name: string;
            id: string;
            type?: "user" | "system" | undefined;
            messageListVisibility?: "show" | "hide" | undefined;
            labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
        }>, "many">>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "list_labels";
        labels?: {
            name: string;
            id: string;
            type?: "user" | "system" | undefined;
            messageListVisibility?: "show" | "hide" | undefined;
            labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
        }[] | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "list_labels";
        labels?: {
            name: string;
            id: string;
            type?: "user" | "system" | undefined;
            messageListVisibility?: "show" | "hide" | undefined;
            labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
        }[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"create_label">;
        success: z.ZodBoolean;
        label: z.ZodOptional<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            type: z.ZodOptional<z.ZodEnum<["system", "user"]>>;
            messageListVisibility: z.ZodOptional<z.ZodEnum<["show", "hide"]>>;
            labelListVisibility: z.ZodOptional<z.ZodEnum<["labelShow", "labelShowIfUnread", "labelHide"]>>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            id: string;
            type?: "user" | "system" | undefined;
            messageListVisibility?: "show" | "hide" | undefined;
            labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
        }, {
            name: string;
            id: string;
            type?: "user" | "system" | undefined;
            messageListVisibility?: "show" | "hide" | undefined;
            labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
        }>>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "create_label";
        label?: {
            name: string;
            id: string;
            type?: "user" | "system" | undefined;
            messageListVisibility?: "show" | "hide" | undefined;
            labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
        } | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "create_label";
        label?: {
            name: string;
            id: string;
            type?: "user" | "system" | undefined;
            messageListVisibility?: "show" | "hide" | undefined;
            labelListVisibility?: "labelShow" | "labelShowIfUnread" | "labelHide" | undefined;
        } | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"modify_message_labels">;
        success: z.ZodBoolean;
        message_id: z.ZodOptional<z.ZodString>;
        label_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "modify_message_labels";
        label_ids?: string[] | undefined;
        message_id?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "modify_message_labels";
        label_ids?: string[] | undefined;
        message_id?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"modify_thread_labels">;
        success: z.ZodBoolean;
        thread_id: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "modify_thread_labels";
        thread_id?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "modify_thread_labels";
        thread_id?: string | undefined;
    }>]>;
    static readonly shortDescription = "Gmail integration for email management";
    static readonly longDescription = "\n    Gmail service integration for comprehensive email management and automation.\n    Use cases:\n    - Send and receive emails with rich formatting\n    - Search and filter emails with advanced queries\n    - Manage drafts and email threads\n    - Mark messages as read/unread\n    - Organize emails with labels and folders\n    - Handle email attachments and metadata\n  ";
    static readonly alias = "gmail";
    constructor(params?: T, context?: BubbleContext);
    testCredential(): Promise<boolean>;
    private makeGmailApiRequest;
    /**
     * Extract clean, readable text content from a Gmail message
     */
    private extractEmailTextContent;
    /**
     * Decode base64url encoded content to UTF-8 string
     */
    private decodeBase64;
    /**
     * Clean up email content by removing forwarded/replied content and excessive whitespace
     */
    private cleanEmailContent;
    /**
     * Clean up a body part by removing base64 data fields
     */
    private cleanBodyPart;
    /**
     * Filter headers to only keep essential ones that users care about
     */
    private filterEssentialHeaders;
    /**
     * Clean up payload by removing base64 data fields to reduce response size
     */
    private cleanPayloadData;
    /**
     * Process and clean a Gmail message by extracting text content and removing heavy fields
     */
    private processAndCleanMessage;
    protected performAction(context?: BubbleContext): Promise<Extract<GmailResult, {
        operation: T['operation'];
    }>>;
    private createEmailMessage;
    private sendEmail;
    private listEmails;
    private getEmail;
    private searchEmails;
    private markAsRead;
    private markAsUnread;
    private createDraft;
    private sendDraft;
    private listDrafts;
    private deleteEmail;
    private trashEmail;
    private listThreads;
    private listLabels;
    private createLabel;
    private modifyMessageLabels;
    private modifyThreadLabels;
    protected chooseCredential(): string | undefined;
}
export {};
//# sourceMappingURL=gmail.d.ts.map