import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const SendGridBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"sendEmail">;
    to: z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>;
    from: z.ZodString;
    subject: z.ZodString;
    text: z.ZodOptional<z.ZodString>;
    html: z.ZodOptional<z.ZodString>;
    attachments: z.ZodOptional<z.ZodArray<z.ZodObject<{
        filename: z.ZodString;
        content: z.ZodString;
        type: z.ZodOptional<z.ZodString>;
        disposition: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        content: string;
        filename: string;
        type?: string | undefined;
        disposition?: string | undefined;
    }, {
        content: string;
        filename: string;
        type?: string | undefined;
        disposition?: string | undefined;
    }>, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "sendEmail";
    from: string;
    to: string | string[];
    subject: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    text?: string | undefined;
    attachments?: {
        content: string;
        filename: string;
        type?: string | undefined;
        disposition?: string | undefined;
    }[] | undefined;
    html?: string | undefined;
}, {
    operation: "sendEmail";
    from: string;
    to: string | string[];
    subject: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    text?: string | undefined;
    attachments?: {
        content: string;
        filename: string;
        type?: string | undefined;
        disposition?: string | undefined;
    }[] | undefined;
    html?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"sendBulkEmails">;
    messages: z.ZodArray<z.ZodObject<{
        to: z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>;
        from: z.ZodString;
        subject: z.ZodString;
        text: z.ZodOptional<z.ZodString>;
        html: z.ZodOptional<z.ZodString>;
        attachments: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    }, "strip", z.ZodTypeAny, {
        from: string;
        to: string | string[];
        subject: string;
        text?: string | undefined;
        attachments?: unknown[] | undefined;
        html?: string | undefined;
    }, {
        from: string;
        to: string | string[];
        subject: string;
        text?: string | undefined;
        attachments?: unknown[] | undefined;
        html?: string | undefined;
    }>, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "sendBulkEmails";
    messages: {
        from: string;
        to: string | string[];
        subject: string;
        text?: string | undefined;
        attachments?: unknown[] | undefined;
        html?: string | undefined;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "sendBulkEmails";
    messages: {
        from: string;
        to: string | string[];
        subject: string;
        text?: string | undefined;
        attachments?: unknown[] | undefined;
        html?: string | undefined;
    }[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"sendTemplate">;
    to: z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>;
    from: z.ZodString;
    templateId: z.ZodString;
    dynamicData: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "sendTemplate";
    from: string;
    to: string | string[];
    templateId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    dynamicData?: Record<string, unknown> | undefined;
}, {
    operation: "sendTemplate";
    from: string;
    to: string | string[];
    templateId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    dynamicData?: Record<string, unknown> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"addContact">;
    email: z.ZodString;
    firstName: z.ZodOptional<z.ZodString>;
    lastName: z.ZodOptional<z.ZodString>;
    customFields: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    listIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    email: string;
    operation: "addContact";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    firstName?: string | undefined;
    lastName?: string | undefined;
    customFields?: Record<string, unknown> | undefined;
    listIds?: string[] | undefined;
}, {
    email: string;
    operation: "addContact";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    firstName?: string | undefined;
    lastName?: string | undefined;
    customFields?: Record<string, unknown> | undefined;
    listIds?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getContact">;
    email: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    email: string;
    operation: "getContact";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    email: string;
    operation: "getContact";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"deleteContact">;
    email: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    email: string;
    operation: "deleteContact";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    email: string;
    operation: "deleteContact";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createList">;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    operation: "createList";
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    name: string;
    operation: "createList";
    description?: string | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"addToList">;
    listId: z.ZodString;
    emails: z.ZodArray<z.ZodString, "many">;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "addToList";
    emails: string[];
    listId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "addToList";
    emails: string[];
    listId: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>]>;
type SendGridBubbleParams = z.input<typeof SendGridBubbleParamsSchema>;
declare const SendGridBubbleResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodUnknown;
    error: z.ZodString;
    meta: z.ZodObject<{
        operation: z.ZodString;
        messageId: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        messageId?: string | undefined;
    }, {
        operation: string;
        messageId?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        messageId?: string | undefined;
    };
    data?: unknown;
}, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        messageId?: string | undefined;
    };
    data?: unknown;
}>;
type SendGridBubbleResult = z.output<typeof SendGridBubbleResultSchema>;
export declare class SendGridBubble extends ServiceBubble<SendGridBubbleParams, SendGridBubbleResult> {
    static readonly service = "sendgrid";
    static readonly authType: "apikey";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"sendEmail">;
        to: z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>;
        from: z.ZodString;
        subject: z.ZodString;
        text: z.ZodOptional<z.ZodString>;
        html: z.ZodOptional<z.ZodString>;
        attachments: z.ZodOptional<z.ZodArray<z.ZodObject<{
            filename: z.ZodString;
            content: z.ZodString;
            type: z.ZodOptional<z.ZodString>;
            disposition: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            content: string;
            filename: string;
            type?: string | undefined;
            disposition?: string | undefined;
        }, {
            content: string;
            filename: string;
            type?: string | undefined;
            disposition?: string | undefined;
        }>, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "sendEmail";
        from: string;
        to: string | string[];
        subject: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        text?: string | undefined;
        attachments?: {
            content: string;
            filename: string;
            type?: string | undefined;
            disposition?: string | undefined;
        }[] | undefined;
        html?: string | undefined;
    }, {
        operation: "sendEmail";
        from: string;
        to: string | string[];
        subject: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        text?: string | undefined;
        attachments?: {
            content: string;
            filename: string;
            type?: string | undefined;
            disposition?: string | undefined;
        }[] | undefined;
        html?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"sendBulkEmails">;
        messages: z.ZodArray<z.ZodObject<{
            to: z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>;
            from: z.ZodString;
            subject: z.ZodString;
            text: z.ZodOptional<z.ZodString>;
            html: z.ZodOptional<z.ZodString>;
            attachments: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        }, "strip", z.ZodTypeAny, {
            from: string;
            to: string | string[];
            subject: string;
            text?: string | undefined;
            attachments?: unknown[] | undefined;
            html?: string | undefined;
        }, {
            from: string;
            to: string | string[];
            subject: string;
            text?: string | undefined;
            attachments?: unknown[] | undefined;
            html?: string | undefined;
        }>, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "sendBulkEmails";
        messages: {
            from: string;
            to: string | string[];
            subject: string;
            text?: string | undefined;
            attachments?: unknown[] | undefined;
            html?: string | undefined;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "sendBulkEmails";
        messages: {
            from: string;
            to: string | string[];
            subject: string;
            text?: string | undefined;
            attachments?: unknown[] | undefined;
            html?: string | undefined;
        }[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"sendTemplate">;
        to: z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString, "many">]>;
        from: z.ZodString;
        templateId: z.ZodString;
        dynamicData: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "sendTemplate";
        from: string;
        to: string | string[];
        templateId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        dynamicData?: Record<string, unknown> | undefined;
    }, {
        operation: "sendTemplate";
        from: string;
        to: string | string[];
        templateId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        dynamicData?: Record<string, unknown> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"addContact">;
        email: z.ZodString;
        firstName: z.ZodOptional<z.ZodString>;
        lastName: z.ZodOptional<z.ZodString>;
        customFields: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        listIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        email: string;
        operation: "addContact";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        firstName?: string | undefined;
        lastName?: string | undefined;
        customFields?: Record<string, unknown> | undefined;
        listIds?: string[] | undefined;
    }, {
        email: string;
        operation: "addContact";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        firstName?: string | undefined;
        lastName?: string | undefined;
        customFields?: Record<string, unknown> | undefined;
        listIds?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getContact">;
        email: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        email: string;
        operation: "getContact";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        email: string;
        operation: "getContact";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"deleteContact">;
        email: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        email: string;
        operation: "deleteContact";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        email: string;
        operation: "deleteContact";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createList">;
        name: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        operation: "createList";
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        name: string;
        operation: "createList";
        description?: string | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"addToList">;
        listId: z.ZodString;
        emails: z.ZodArray<z.ZodString, "many">;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "addToList";
        emails: string[];
        listId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "addToList";
        emails: string[];
        listId: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>]>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodUnknown;
        error: z.ZodString;
        meta: z.ZodObject<{
            operation: z.ZodString;
            messageId: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            messageId?: string | undefined;
        }, {
            operation: string;
            messageId?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            messageId?: string | undefined;
        };
        data?: unknown;
    }, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            messageId?: string | undefined;
        };
        data?: unknown;
    }>;
    static readonly shortDescription = "Email delivery and marketing automation platform";
    static readonly longDescription = "\n    SendGrid Bubble for transactional and marketing emails.\n\n    Features:\n    - Send single and bulk emails\n    - Template-based emails with dynamic content\n    - Contact management and segmentation\n    - Contact lists for campaigns\n    - Attachments and HTML content\n    - High deliverability rates\n\n    Use cases:\n    - Transactional emails (passwords, notifications)\n    - Marketing campaigns\n    - Newsletter distribution\n    - User onboarding emails\n    - Automated email sequences\n  ";
    static readonly alias = "email";
    constructor(params: SendGridBubbleParams, context?: BubbleContext, instanceId?: string);
    protected getCredentialType(): CredentialType;
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    private getApiKey;
    protected performAction(context?: BubbleContext): Promise<SendGridBubbleResult>;
    private sendEmail;
    private sendBulkEmails;
    private sendTemplate;
    private addContact;
    private getContact;
    private deleteContact;
    private createList;
    private addToList;
}
export {};
//# sourceMappingURL=sendgrid-bubble.d.ts.map