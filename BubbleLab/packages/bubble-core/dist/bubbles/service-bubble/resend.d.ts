import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';
declare const ResendParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"send_email">;
    from: z.ZodDefault<z.ZodString>;
    to: z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>;
    cc: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>>;
    bcc: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>>;
    subject: z.ZodString;
    text: z.ZodOptional<z.ZodString>;
    html: z.ZodOptional<z.ZodString>;
    reply_to: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>>;
    scheduled_at: z.ZodOptional<z.ZodString>;
    attachments: z.ZodOptional<z.ZodArray<z.ZodObject<{
        filename: z.ZodString;
        content: z.ZodUnion<[z.ZodString, z.ZodType<Buffer<ArrayBufferLike>, z.ZodTypeDef, Buffer<ArrayBufferLike>>]>;
        content_type: z.ZodOptional<z.ZodString>;
        path: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        content: string | Buffer<ArrayBufferLike>;
        filename: string;
        path?: string | undefined;
        content_type?: string | undefined;
    }, {
        content: string | Buffer<ArrayBufferLike>;
        filename: string;
        path?: string | undefined;
        content_type?: string | undefined;
    }>, "many">>;
    tags: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        value: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        value: string;
        name: string;
    }, {
        value: string;
        name: string;
    }>, "many">>;
    headers: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "send_email";
    from: string;
    to: string | string[];
    subject: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    text?: string | undefined;
    attachments?: {
        content: string | Buffer<ArrayBufferLike>;
        filename: string;
        path?: string | undefined;
        content_type?: string | undefined;
    }[] | undefined;
    cc?: string | string[] | undefined;
    bcc?: string | string[] | undefined;
    html?: string | undefined;
    reply_to?: string | string[] | undefined;
    scheduled_at?: string | undefined;
    tags?: {
        value: string;
        name: string;
    }[] | undefined;
    headers?: Record<string, string> | undefined;
}, {
    operation: "send_email";
    to: string | string[];
    subject: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    text?: string | undefined;
    attachments?: {
        content: string | Buffer<ArrayBufferLike>;
        filename: string;
        path?: string | undefined;
        content_type?: string | undefined;
    }[] | undefined;
    from?: string | undefined;
    cc?: string | string[] | undefined;
    bcc?: string | string[] | undefined;
    html?: string | undefined;
    reply_to?: string | string[] | undefined;
    scheduled_at?: string | undefined;
    tags?: {
        value: string;
        name: string;
    }[] | undefined;
    headers?: Record<string, string> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_email_status">;
    email_id: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "get_email_status";
    email_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "get_email_status";
    email_id: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
declare const ResendResultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"send_email">;
    success: z.ZodBoolean;
    email_id: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "send_email";
    email_id?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "send_email";
    email_id?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"get_email_status">;
    success: z.ZodBoolean;
    status: z.ZodOptional<z.ZodString>;
    created_at: z.ZodOptional<z.ZodString>;
    last_event: z.ZodOptional<z.ZodString>;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "get_email_status";
    status?: string | undefined;
    created_at?: string | undefined;
    last_event?: string | undefined;
}, {
    error: string;
    success: boolean;
    operation: "get_email_status";
    status?: string | undefined;
    created_at?: string | undefined;
    last_event?: string | undefined;
}>]>;
type ResendResult = z.output<typeof ResendResultSchema>;
type ResendParams = z.input<typeof ResendParamsSchema>;
export type ResendOperationResult<T extends ResendParams['operation']> = Extract<ResendResult, {
    operation: T;
}>;
export declare class ResendBubble<T extends ResendParams = ResendParams> extends ServiceBubble<T, Extract<ResendResult, {
    operation: T['operation'];
}>> {
    static readonly type: "service";
    static readonly service = "resend";
    static readonly authType: "apikey";
    static readonly bubbleName = "resend";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"send_email">;
        from: z.ZodDefault<z.ZodString>;
        to: z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>;
        cc: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>>;
        bcc: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>>;
        subject: z.ZodString;
        text: z.ZodOptional<z.ZodString>;
        html: z.ZodOptional<z.ZodString>;
        reply_to: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>>;
        scheduled_at: z.ZodOptional<z.ZodString>;
        attachments: z.ZodOptional<z.ZodArray<z.ZodObject<{
            filename: z.ZodString;
            content: z.ZodUnion<[z.ZodString, z.ZodType<Buffer<ArrayBufferLike>, z.ZodTypeDef, Buffer<ArrayBufferLike>>]>;
            content_type: z.ZodOptional<z.ZodString>;
            path: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            content: string | Buffer<ArrayBufferLike>;
            filename: string;
            path?: string | undefined;
            content_type?: string | undefined;
        }, {
            content: string | Buffer<ArrayBufferLike>;
            filename: string;
            path?: string | undefined;
            content_type?: string | undefined;
        }>, "many">>;
        tags: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            value: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            value: string;
            name: string;
        }, {
            value: string;
            name: string;
        }>, "many">>;
        headers: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "send_email";
        from: string;
        to: string | string[];
        subject: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        text?: string | undefined;
        attachments?: {
            content: string | Buffer<ArrayBufferLike>;
            filename: string;
            path?: string | undefined;
            content_type?: string | undefined;
        }[] | undefined;
        cc?: string | string[] | undefined;
        bcc?: string | string[] | undefined;
        html?: string | undefined;
        reply_to?: string | string[] | undefined;
        scheduled_at?: string | undefined;
        tags?: {
            value: string;
            name: string;
        }[] | undefined;
        headers?: Record<string, string> | undefined;
    }, {
        operation: "send_email";
        to: string | string[];
        subject: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        text?: string | undefined;
        attachments?: {
            content: string | Buffer<ArrayBufferLike>;
            filename: string;
            path?: string | undefined;
            content_type?: string | undefined;
        }[] | undefined;
        from?: string | undefined;
        cc?: string | string[] | undefined;
        bcc?: string | string[] | undefined;
        html?: string | undefined;
        reply_to?: string | string[] | undefined;
        scheduled_at?: string | undefined;
        tags?: {
            value: string;
            name: string;
        }[] | undefined;
        headers?: Record<string, string> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_email_status">;
        email_id: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "get_email_status";
        email_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "get_email_status";
        email_id: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"send_email">;
        success: z.ZodBoolean;
        email_id: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "send_email";
        email_id?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "send_email";
        email_id?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"get_email_status">;
        success: z.ZodBoolean;
        status: z.ZodOptional<z.ZodString>;
        created_at: z.ZodOptional<z.ZodString>;
        last_event: z.ZodOptional<z.ZodString>;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "get_email_status";
        status?: string | undefined;
        created_at?: string | undefined;
        last_event?: string | undefined;
    }, {
        error: string;
        success: boolean;
        operation: "get_email_status";
        status?: string | undefined;
        created_at?: string | undefined;
        last_event?: string | undefined;
    }>]>;
    static readonly shortDescription = "Email sending service via Resend API";
    static readonly longDescription = "\n    Resend email service integration for sending transactional emails.\n    Use cases:\n    - Send transactional emails with HTML and text content\n    - Track email delivery status and metrics\n    - Manage email attachments and custom headers\n    \n    Security Features:\n    - API key-based authentication\n    - Email address validation\n    - Domain enforcement for user credentials (validates that sender domain is verified in Resend)\n    - System credentials (bubblelab.ai domain) skip domain validation\n    - Content sanitization\n    - Rate limiting awareness\n  ";
    static readonly alias = "resend";
    private resend?;
    private verifiedDomains?;
    constructor(params?: T, context?: BubbleContext);
    /**
     * Extracts the domain from an email address.
     * Handles both formats: "Name <email@domain.com>" and "email@domain.com"
     */
    private extractDomainFromEmail;
    /**
     * Checks if the email domain is a system domain (BubbleLab's default domain).
     * System credentials use bubblelab.ai domain and should skip domain validation.
     */
    private isSystemDomain;
    /**
     * Fetches and caches the list of verified domains from Resend API
     */
    private getVerifiedDomains;
    /**
     * Validates that the from email domain is verified in Resend.
     * Skips validation for system domains (bubblelab.ai) as they use system credentials.
     * Only enforces validation for user-provided credentials.
     */
    private validateFromDomain;
    testCredential(): Promise<boolean>;
    protected performAction(context?: BubbleContext): Promise<Extract<ResendResult, {
        operation: T['operation'];
    }>>;
    private sendEmail;
    private getEmailStatus;
    protected chooseCredential(): string | undefined;
}
export {};
//# sourceMappingURL=resend.d.ts.map