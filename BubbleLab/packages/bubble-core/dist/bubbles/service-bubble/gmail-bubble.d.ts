import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const GmailBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"sendEmail">;
    to: z.ZodArray<z.ZodString, "many">;
    subject: z.ZodString;
    body: z.ZodString;
    cc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    bcc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    isHtml: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    attachments: z.ZodOptional<z.ZodArray<z.ZodObject<{
        filename: z.ZodString;
        content: z.ZodString;
        contentType: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        content: string;
        filename: string;
        contentType?: string | undefined;
    }, {
        content: string;
        filename: string;
        contentType?: string | undefined;
    }>, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "sendEmail";
    to: string[];
    subject: string;
    body: string;
    isHtml: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    attachments?: {
        content: string;
        filename: string;
        contentType?: string | undefined;
    }[] | undefined;
    cc?: string[] | undefined;
    bcc?: string[] | undefined;
}, {
    operation: "sendEmail";
    to: string[];
    subject: string;
    body: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    attachments?: {
        content: string;
        filename: string;
        contentType?: string | undefined;
    }[] | undefined;
    cc?: string[] | undefined;
    bcc?: string[] | undefined;
    isHtml?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"readEmail">;
    emailId: z.ZodString;
    format: z.ZodDefault<z.ZodOptional<z.ZodEnum<["full", "metadata", "minimal"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    format: "minimal" | "full" | "metadata";
    operation: "readEmail";
    emailId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "readEmail";
    emailId: string;
    format?: "minimal" | "full" | "metadata" | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"listEmails">;
    labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    maxResults: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    pageToken: z.ZodOptional<z.ZodString>;
    includeSpamTrash: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "listEmails";
    maxResults: number;
    includeSpamTrash: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    labelIds?: string[] | undefined;
    pageToken?: string | undefined;
}, {
    operation: "listEmails";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    labelIds?: string[] | undefined;
    maxResults?: number | undefined;
    pageToken?: string | undefined;
    includeSpamTrash?: boolean | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"searchEmails">;
    query: z.ZodString;
    maxResults: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    pageToken: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    query: string;
    operation: "searchEmails";
    maxResults: number;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    pageToken?: string | undefined;
}, {
    query: string;
    operation: "searchEmails";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    maxResults?: number | undefined;
    pageToken?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteEmail">;
    emailId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "deleteEmail";
    emailId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "deleteEmail";
    emailId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"markAsRead">;
    emailIds: z.ZodArray<z.ZodString, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "markAsRead";
    emailIds: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "markAsRead";
    emailIds: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"markAsUnread">;
    emailIds: z.ZodArray<z.ZodString, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "markAsUnread";
    emailIds: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "markAsUnread";
    emailIds: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"modifyLabels">;
    emailId: z.ZodString;
    addLabelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    removeLabelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "modifyLabels";
    emailId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    addLabelIds?: string[] | undefined;
    removeLabelIds?: string[] | undefined;
}, {
    operation: "modifyLabels";
    emailId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    addLabelIds?: string[] | undefined;
    removeLabelIds?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getAttachment">;
    emailId: z.ZodString;
    attachmentId: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getAttachment";
    attachmentId: string;
    emailId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getAttachment";
    attachmentId: string;
    emailId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createDraft">;
    to: z.ZodArray<z.ZodString, "many">;
    subject: z.ZodString;
    body: z.ZodString;
    cc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    bcc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    isHtml: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "createDraft";
    to: string[];
    subject: string;
    body: string;
    isHtml: boolean;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cc?: string[] | undefined;
    bcc?: string[] | undefined;
}, {
    operation: "createDraft";
    to: string[];
    subject: string;
    body: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    cc?: string[] | undefined;
    bcc?: string[] | undefined;
    isHtml?: boolean | undefined;
}>]>;
type GmailBubbleParams = z.input<typeof GmailBubbleParamsSchema>;
declare const GmailBubbleResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodUnknown;
    error: z.ZodString;
    meta: z.ZodObject<{
        operation: z.ZodString;
        emailId: z.ZodOptional<z.ZodString>;
        emailCount: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        emailId?: string | undefined;
        emailCount?: number | undefined;
    }, {
        operation: string;
        emailId?: string | undefined;
        emailCount?: number | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        emailId?: string | undefined;
        emailCount?: number | undefined;
    };
    data?: unknown;
}, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        emailId?: string | undefined;
        emailCount?: number | undefined;
    };
    data?: unknown;
}>;
type GmailBubbleResult = z.output<typeof GmailBubbleResultSchema>;
export declare class GmailBubble extends ServiceBubble<GmailBubbleParams, GmailBubbleResult> {
    static readonly service = "gmail";
    static readonly authType: "oauth";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"sendEmail">;
        to: z.ZodArray<z.ZodString, "many">;
        subject: z.ZodString;
        body: z.ZodString;
        cc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        bcc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        isHtml: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        attachments: z.ZodOptional<z.ZodArray<z.ZodObject<{
            filename: z.ZodString;
            content: z.ZodString;
            contentType: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            content: string;
            filename: string;
            contentType?: string | undefined;
        }, {
            content: string;
            filename: string;
            contentType?: string | undefined;
        }>, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "sendEmail";
        to: string[];
        subject: string;
        body: string;
        isHtml: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        attachments?: {
            content: string;
            filename: string;
            contentType?: string | undefined;
        }[] | undefined;
        cc?: string[] | undefined;
        bcc?: string[] | undefined;
    }, {
        operation: "sendEmail";
        to: string[];
        subject: string;
        body: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        attachments?: {
            content: string;
            filename: string;
            contentType?: string | undefined;
        }[] | undefined;
        cc?: string[] | undefined;
        bcc?: string[] | undefined;
        isHtml?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"readEmail">;
        emailId: z.ZodString;
        format: z.ZodDefault<z.ZodOptional<z.ZodEnum<["full", "metadata", "minimal"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        format: "minimal" | "full" | "metadata";
        operation: "readEmail";
        emailId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "readEmail";
        emailId: string;
        format?: "minimal" | "full" | "metadata" | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"listEmails">;
        labelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        maxResults: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        pageToken: z.ZodOptional<z.ZodString>;
        includeSpamTrash: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "listEmails";
        maxResults: number;
        includeSpamTrash: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        labelIds?: string[] | undefined;
        pageToken?: string | undefined;
    }, {
        operation: "listEmails";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        labelIds?: string[] | undefined;
        maxResults?: number | undefined;
        pageToken?: string | undefined;
        includeSpamTrash?: boolean | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"searchEmails">;
        query: z.ZodString;
        maxResults: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        pageToken: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        query: string;
        operation: "searchEmails";
        maxResults: number;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        pageToken?: string | undefined;
    }, {
        query: string;
        operation: "searchEmails";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        maxResults?: number | undefined;
        pageToken?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteEmail">;
        emailId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "deleteEmail";
        emailId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "deleteEmail";
        emailId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"markAsRead">;
        emailIds: z.ZodArray<z.ZodString, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "markAsRead";
        emailIds: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "markAsRead";
        emailIds: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"markAsUnread">;
        emailIds: z.ZodArray<z.ZodString, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "markAsUnread";
        emailIds: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "markAsUnread";
        emailIds: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"modifyLabels">;
        emailId: z.ZodString;
        addLabelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        removeLabelIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "modifyLabels";
        emailId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        addLabelIds?: string[] | undefined;
        removeLabelIds?: string[] | undefined;
    }, {
        operation: "modifyLabels";
        emailId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        addLabelIds?: string[] | undefined;
        removeLabelIds?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getAttachment">;
        emailId: z.ZodString;
        attachmentId: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getAttachment";
        attachmentId: string;
        emailId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getAttachment";
        attachmentId: string;
        emailId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createDraft">;
        to: z.ZodArray<z.ZodString, "many">;
        subject: z.ZodString;
        body: z.ZodString;
        cc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        bcc: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        isHtml: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "createDraft";
        to: string[];
        subject: string;
        body: string;
        isHtml: boolean;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cc?: string[] | undefined;
        bcc?: string[] | undefined;
    }, {
        operation: "createDraft";
        to: string[];
        subject: string;
        body: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        cc?: string[] | undefined;
        bcc?: string[] | undefined;
        isHtml?: boolean | undefined;
    }>]>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodUnknown;
        error: z.ZodString;
        meta: z.ZodObject<{
            operation: z.ZodString;
            emailId: z.ZodOptional<z.ZodString>;
            emailCount: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            emailId?: string | undefined;
            emailCount?: number | undefined;
        }, {
            operation: string;
            emailId?: string | undefined;
            emailCount?: number | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            emailId?: string | undefined;
            emailCount?: number | undefined;
        };
        data?: unknown;
    }, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            emailId?: string | undefined;
            emailCount?: number | undefined;
        };
        data?: unknown;
    }>;
    static readonly shortDescription = "Email service by Google";
    static readonly longDescription = "\n    Gmail Bubble for email management and automation.\n\n    Features:\n    - Send emails with attachments\n    - Read and search emails\n    - Manage labels and folders\n    - Mark as read/unread\n    - Handle attachments\n    - Draft management\n    - Thread support\n\n    Use cases:\n    - Automated notifications\n    - Email processing workflows\n    - Customer support automation\n    - Email analytics\n    - Attachment processing\n    - Newsletter management\n  ";
    static readonly alias = "email";
    private accessToken;
    private baseUrl;
    constructor(params: GmailBubbleParams, context?: BubbleContext, instanceId?: string);
    protected getCredentialType(): CredentialType;
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    private getToken;
    protected performAction(context?: BubbleContext): Promise<GmailBubbleResult>;
    private makeRequest;
    private encodeEmail;
    private sendEmail;
    private readEmail;
    private getHeader;
    private listEmails;
    private searchEmails;
    private deleteEmail;
    private markAsRead;
    private markAsUnread;
    private modifyLabels;
    private getAttachment;
    private createDraft;
}
export {};
//# sourceMappingURL=gmail-bubble.d.ts.map