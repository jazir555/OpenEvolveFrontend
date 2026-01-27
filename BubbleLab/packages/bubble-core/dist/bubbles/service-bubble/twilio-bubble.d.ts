import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const TwilioBubbleParamsSchema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
    operation: z.ZodLiteral<"sendSMS">;
    to: z.ZodString;
    from: z.ZodString;
    body: z.ZodString;
    statusCallback: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "sendSMS";
    from: string;
    to: string;
    body: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    statusCallback?: string | undefined;
}, {
    operation: "sendSMS";
    from: string;
    to: string;
    body: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    statusCallback?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"makeCall">;
    to: z.ZodString;
    from: z.ZodString;
    url: z.ZodString;
    statusCallback: z.ZodOptional<z.ZodString>;
    method: z.ZodDefault<z.ZodOptional<z.ZodEnum<["GET", "POST"]>>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    url: string;
    operation: "makeCall";
    from: string;
    to: string;
    method: "GET" | "POST";
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    statusCallback?: string | undefined;
}, {
    url: string;
    operation: "makeCall";
    from: string;
    to: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    method?: "GET" | "POST" | undefined;
    statusCallback?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"sendWhatsApp">;
    to: z.ZodString;
    from: z.ZodString;
    body: z.ZodString;
    mediaUrl: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "sendWhatsApp";
    from: string;
    to: string;
    body: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    mediaUrl?: string[] | undefined;
}, {
    operation: "sendWhatsApp";
    from: string;
    to: string;
    body: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    mediaUrl?: string[] | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"lookupNumber">;
    phoneNumber: z.ZodString;
    type: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodEnum<["carrier", "caller-name", "phone-type"]>, "many">>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    type: ("carrier" | "caller-name" | "phone-type")[];
    operation: "lookupNumber";
    phoneNumber: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "lookupNumber";
    phoneNumber: string;
    type?: ("carrier" | "caller-name" | "phone-type")[] | undefined;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"createMessage">;
    to: z.ZodString;
    from: z.ZodString;
    body: z.ZodString;
    scheduleTime: z.ZodOptional<z.ZodString>;
    statusCallback: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "createMessage";
    from: string;
    to: string;
    body: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    statusCallback?: string | undefined;
    scheduleTime?: string | undefined;
}, {
    operation: "createMessage";
    from: string;
    to: string;
    body: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    statusCallback?: string | undefined;
    scheduleTime?: string | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getMessage">;
    messageSid: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getMessage";
    messageSid: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getMessage";
    messageSid: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"getMedia">;
    messageSid: z.ZodString;
    mediaSid: z.ZodString;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "getMedia";
    messageSid: string;
    mediaSid: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}, {
    operation: "getMedia";
    messageSid: string;
    mediaSid: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
}>, z.ZodObject<{
    operation: z.ZodLiteral<"validateNumber">;
    phoneNumber: z.ZodString;
    countryCode: z.ZodOptional<z.ZodString>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "validateNumber";
    phoneNumber: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    countryCode?: string | undefined;
}, {
    operation: "validateNumber";
    phoneNumber: string;
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    countryCode?: string | undefined;
}>]>;
type TwilioBubbleParams = z.input<typeof TwilioBubbleParamsSchema>;
declare const TwilioBubbleResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    data: z.ZodUnknown;
    error: z.ZodString;
    meta: z.ZodObject<{
        operation: z.ZodString;
        sid: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        operation: string;
        sid?: string | undefined;
    }, {
        operation: string;
        sid?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        sid?: string | undefined;
    };
    data?: unknown;
}, {
    error: string;
    success: boolean;
    meta: {
        operation: string;
        sid?: string | undefined;
    };
    data?: unknown;
}>;
type TwilioBubbleResult = z.output<typeof TwilioBubbleResultSchema>;
export declare class TwilioBubble extends ServiceBubble<TwilioBubbleParams, TwilioBubbleResult> {
    static readonly service = "twilio";
    static readonly authType: "apikey";
    static readonly bubbleName: BubbleName;
    static readonly type: "service";
    static readonly schema: z.ZodDiscriminatedUnion<"operation", [z.ZodObject<{
        operation: z.ZodLiteral<"sendSMS">;
        to: z.ZodString;
        from: z.ZodString;
        body: z.ZodString;
        statusCallback: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "sendSMS";
        from: string;
        to: string;
        body: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        statusCallback?: string | undefined;
    }, {
        operation: "sendSMS";
        from: string;
        to: string;
        body: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        statusCallback?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"makeCall">;
        to: z.ZodString;
        from: z.ZodString;
        url: z.ZodString;
        statusCallback: z.ZodOptional<z.ZodString>;
        method: z.ZodDefault<z.ZodOptional<z.ZodEnum<["GET", "POST"]>>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        url: string;
        operation: "makeCall";
        from: string;
        to: string;
        method: "GET" | "POST";
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        statusCallback?: string | undefined;
    }, {
        url: string;
        operation: "makeCall";
        from: string;
        to: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        method?: "GET" | "POST" | undefined;
        statusCallback?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"sendWhatsApp">;
        to: z.ZodString;
        from: z.ZodString;
        body: z.ZodString;
        mediaUrl: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "sendWhatsApp";
        from: string;
        to: string;
        body: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        mediaUrl?: string[] | undefined;
    }, {
        operation: "sendWhatsApp";
        from: string;
        to: string;
        body: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        mediaUrl?: string[] | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"lookupNumber">;
        phoneNumber: z.ZodString;
        type: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodEnum<["carrier", "caller-name", "phone-type"]>, "many">>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        type: ("carrier" | "caller-name" | "phone-type")[];
        operation: "lookupNumber";
        phoneNumber: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "lookupNumber";
        phoneNumber: string;
        type?: ("carrier" | "caller-name" | "phone-type")[] | undefined;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"createMessage">;
        to: z.ZodString;
        from: z.ZodString;
        body: z.ZodString;
        scheduleTime: z.ZodOptional<z.ZodString>;
        statusCallback: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "createMessage";
        from: string;
        to: string;
        body: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        statusCallback?: string | undefined;
        scheduleTime?: string | undefined;
    }, {
        operation: "createMessage";
        from: string;
        to: string;
        body: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        statusCallback?: string | undefined;
        scheduleTime?: string | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getMessage">;
        messageSid: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getMessage";
        messageSid: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getMessage";
        messageSid: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"getMedia">;
        messageSid: z.ZodString;
        mediaSid: z.ZodString;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "getMedia";
        messageSid: string;
        mediaSid: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }, {
        operation: "getMedia";
        messageSid: string;
        mediaSid: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
    }>, z.ZodObject<{
        operation: z.ZodLiteral<"validateNumber">;
        phoneNumber: z.ZodString;
        countryCode: z.ZodOptional<z.ZodString>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "validateNumber";
        phoneNumber: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        countryCode?: string | undefined;
    }, {
        operation: "validateNumber";
        phoneNumber: string;
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        countryCode?: string | undefined;
    }>]>;
    static readonly resultSchema: z.ZodObject<{
        success: z.ZodBoolean;
        data: z.ZodUnknown;
        error: z.ZodString;
        meta: z.ZodObject<{
            operation: z.ZodString;
            sid: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            operation: string;
            sid?: string | undefined;
        }, {
            operation: string;
            sid?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            sid?: string | undefined;
        };
        data?: unknown;
    }, {
        error: string;
        success: boolean;
        meta: {
            operation: string;
            sid?: string | undefined;
        };
        data?: unknown;
    }>;
    static readonly shortDescription = "SMS, voice, and WhatsApp messaging platform";
    static readonly longDescription = "\n    Twilio Bubble for programmable communication.\n\n    Features:\n    - Send SMS and MMS messages\n    - Make and receive voice calls\n    - WhatsApp messaging\n    - Phone number lookup and validation\n    - Media handling for MMS\n    - Scheduled messaging\n\n    Use cases:\n    - SMS notifications and alerts\n    - Two-factor authentication\n    - Voice call automation\n    - WhatsApp business messaging\n    - Phone number verification\n    - Appointment reminders\n  ";
    static readonly alias = "sms";
    private client;
    constructor(params: TwilioBubbleParams, context?: BubbleContext, instanceId?: string);
    protected getCredentialType(): CredentialType;
    protected chooseCredential(): string | undefined;
    testCredential(): Promise<boolean>;
    private getClient;
    protected performAction(context?: BubbleContext): Promise<TwilioBubbleResult>;
    private sendSMS;
    private makeCall;
    private sendWhatsApp;
    private lookupNumber;
    private createMessage;
    private getMessage;
    private getMedia;
    private validateNumber;
}
export {};
//# sourceMappingURL=twilio-bubble.d.ts.map